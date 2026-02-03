"""
Autoencoder model with no RBM as a prior.

Inherits from AutoEncoderBase and implements a different KL divergence loss.
"""

import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy_with_logits
from model.gumbel import GumbelMod
from model.encoder.encoderhierarchybase import HierarchicalEncoder
from model.decoder.decoder import Decoder
from model.decoder.decoderhierarchybase import DecoderHierarchyBase, DecoderHierarchyBaseV2
from model.rbm.rbm import RBM
from torch.distributions import Bernoulli


#logging module with handmade settings.
from CaloQuVAE import logging
logger = logging.getLogger(__name__)

from model.autoencoder.autoencoderbase import AutoEncoderBase, AutoEncoderHidden
from utils.HLF.atlasgeo import AtlasGeometry, DifferentiableFeatureExtractor
from utils.HLF.mmd import compute_cmmd, ConditionNormalizer


class AutoEncoderSeparate(AutoEncoderBase):
    def __init__(self, cfg):
        super(AutoEncoderSeparate, self).__init__(cfg)
        # Initialize Geometry and Feature Extractor ---
        # Assuming cfg contains the path to the geometry file
        geo_file = self._config.data.binning_path        
        self.geo = AtlasGeometry(geo_file)
        self.feature_extractor = DifferentiableFeatureExtractor(self.geo)
        self.cond_normalizer = ConditionNormalizer(method='log_minmax', max_val=300000.0)


    def posterior_entropy(self, post_logits, is_training=True):
        """
        Calculate the posterior entropy for the given logits

        Replaces KL divergence in generic autoencoder base class.
        
        """
        # logits = torch.cat(post_logits, dim=1)

        # p_z = torch.sigmoid(logits).detach()
        # epsilon = 1e-8
        # entropy_per_node = - self._bce_loss(logits, p_z)
        # entropy_per_z = torch.sum(entropy_per_node, dim=1)

        # batch_average_entropy = torch.mean(entropy_per_z, dim=0)

        # return batch_average_entropy

        logits = torch.cat(post_logits, dim=1)        
        if torch.isnan(logits).any():
            logger.warning("NaN detected in post_logits, returning zero entropy.")
            return None

        dist = Bernoulli(logits=logits)
        entropy_per_node = dist.entropy()
        entropy_per_z = torch.sum(entropy_per_node, dim=1)
        batch_average_entropy = torch.mean(entropy_per_z, dim=0)
        
        return batch_average_entropy
    
    def pos_energy(self, post_samples):
        """
        Compute positive phase (energy expval under posterior variables)
        """
        pos_energy = self.prior.energy_exp_cond(post_samples[0],post_samples[1],post_samples[2],post_samples[3]).mean()
        return pos_energy


    def decode(self, post_samples, x, x0, beta=5, act_fct_slope=0.02):
            """
            Overridden decode method. 
            Always returns raw components in training to allow physics losses
            to backpropagate through the hit mask.
            """
            output_hits, output_activations = self.decoder(torch.cat(post_samples, 1), x0)
            
            # Initialize placeholders
            activations_raw = None
            hit_mask_attached = None

            if self.training:
                # 1. Get the Soft (Gumbel) Mask with gradients attached
                # This allows physics losses to push the mask towards "1" if energy is needed
                hit_mask_attached = self._hit_smoothing_dist_mod(output_hits, beta=beta)
        
                output_activations = self._activation_fct(act_fct_slope, output_activations) * torch.where(x > 0, 1., 0.)
                
                activations_raw = self._activation_fct(act_fct_slope, output_activations)
            else:
                # Evaluation mode: Hard masking for cleaner evaluation
                output_activations = self._activation_fct(0.0, output_activations) * self._hit_smoothing_dist_mod(output_hits)
        
            return output_hits, output_activations, activations_raw, hit_mask_attached

    def loss(self, input_data, x0, args):
            """
            Child class loss with Focal Loss + Differentiable Physics Loss.
            """
            # Unpack the 4 values returned by the new decode method
            beta, post_logits, post_samples, output_activations, output_hits, activations_raw, hit_mask_attached = args

            # --- [1. Setup Spatial Weights] ---
            r_weight_coeff = getattr(self._config.model.loss_coeff, "r_loss_coeff", 0.0)
            if r_weight_coeff > 0.0:
                spatial_map = self.feature_extractor.r_grid_norm.view(1, -1)
                pixel_weights = 1.0 + (r_weight_coeff * spatial_map)
            else:
                pixel_weights = 1.0

            # --- [2. Standard Reconstruction Loss] ---
            # output_activations uses the ground truth mask
            squared_diff = torch.pow((input_data - output_activations), 2)
            arg = self._config.model.mse_weight * input_data
            # Clamp to avoid overflow (approx 80 is safe for float32)
            energy_weighting = torch.exp(torch.clamp(arg, max=80))            
            ae_loss = squared_diff * energy_weighting * pixel_weights
            ae_loss = torch.clamp(torch.mean(torch.sum(ae_loss, dim=1), dim=0) * self._config.model.coefficient, max=1e30)

            # --- [3. Hit Loss with Focal Support] ---
            targets = torch.where(input_data > 0, 1., 0.)
            bce_raw = binary_cross_entropy_with_logits(output_hits, targets, reduction='none')

            if hasattr(self._config.model.loss_coeff, "focal_alpha") and hasattr(self._config.model.loss_coeff, "focal_gamma"):
                alpha = self._config.model.loss_coeff.focal_alpha
                gamma = self._config.model.loss_coeff.focal_gamma
                pt = torch.exp(-bce_raw)
                alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
                focal_loss = alpha_t * (1 - pt).pow(gamma) * bce_raw
                hit_loss = torch.mean(torch.sum(focal_loss * pixel_weights, dim=1), dim=0)
            else:
                hit_loss = torch.mean(torch.sum(bce_raw * pixel_weights, dim=1), dim=0)

            # --- [4. Feature Extraction Setup] ---
            mmd_weight = getattr(self._config.model.loss_coeff, "mmd_loss", 0.0)
            mae_weight = getattr(self._config.model.loss_coeff, "feature_mae", 0.0) 

            mmd_loss_total = torch.tensor(0.0, device=input_data.device)
            mae_loss_total = torch.tensor(0.0, device=input_data.device)

            if mmd_weight > 0.0 or mae_weight > 0.0:
                # CRITICAL CHANGE: Construct reconstruction with gradients flowing to mask
                if activations_raw is not None and hit_mask_attached is not None:
                    # This product allows gradients to flow: Loss -> Mask -> Hits Head
                    physics_recon = activations_raw * hit_mask_attached
                else:
                    # Fallback for Eval mode (activations_raw is None)
                    physics_recon = output_activations

                feat_gt = self.feature_extractor(input_data)
                feat_recon = self.feature_extractor(physics_recon)

                # --- [5. Conditional MMD Feature Loss] ---
                if mmd_weight > 0.0:
                    norm_energy = self.cond_normalizer(x0)
                    for key in feat_gt.keys():
                        val_gt = feat_gt[key].view(input_data.size(0), -1)
                        val_recon = feat_recon[key].view(input_data.size(0), -1)
                        mmd_loss_total += compute_cmmd(val_gt, norm_energy, val_recon, norm_energy)

                # --- [6. Physics Feature MAE Loss] ---
                if mae_weight > 0.0:
                    for key in feat_gt.keys():
                        val_gt = feat_gt[key].view(input_data.size(0), -1)
                        val_recon = feat_recon[key].view(input_data.size(0), -1)
                        if "E_" in key:
                            val_gt = torch.log1p(val_gt)
                            val_recon = torch.log1p(torch.nn.functional.relu(val_recon))
                        scale = torch.clamp(torch.mean(torch.abs(val_gt)).detach(), min=1e-5)
                        mae_loss_total += torch.abs(val_gt - val_recon).mean() / scale

            # --- [7. Aggregate and Return] ---
            total_loss_dict = {
                "ae_loss": ae_loss,
                "hit_loss": hit_loss,
            }
            if hasattr(self._config.model.loss_coeff, "mmd_loss"):
                total_loss_dict["mmd_loss"] = mmd_loss_total
            if hasattr(self._config.model.loss_coeff, "feature_mae"):
                total_loss_dict["feature_mae"] = mae_loss_total

            if hasattr(self._config.model.loss_coeff, 'pos_energy') and hasattr(self._config.model.loss_coeff, 'logit_distance'):
                l_dist = torch.pow(torch.cat(post_logits, 1) - torch.cat(self.logit_distance(post_samples, post_logits), 1), 2).mean()
                pos_energy = self.pos_energy(post_samples)
                entropy_loss = -1 * self.posterior_entropy(post_logits)        

                total_loss_dict.update({
                    "entropy": entropy_loss,
                    "pos_energy": pos_energy,
                    "logit_distance": l_dist
                })
            if torch.isnan(ae_loss):
                print("FAIL: AE Loss is NaN. Check energy_weighting exp() overflow.")
                
            if torch.isnan(mae_loss_total):
                print("FAIL: MAE Loss is NaN. Check for negative inputs to log1p.")

            if torch.isnan(mmd_loss_total):
                print("FAIL: MMD Loss is NaN. Check kernel bandwidth or division by zero in cmmd.")

            return total_loss_dict
            
    def forward(self, xx, beta_latent=5, beta_hits=5, act_fct_slope=0.02):
        """
        - Overrides forward in autoencoderbase to unpack  and return more values for loss calculation.
        """
        logger.debug("VAE_forward")
        
        x, x0 = xx
        
        beta, post_logits, post_samples = self.encoder(x, x0, beta_latent)

        output_hits, output_activations, activations_raw, hit_mask_attached = self.decode(post_samples, x, x0, beta_hits, act_fct_slope)

        return beta, post_logits, post_samples, output_activations, output_hits, activations_raw, hit_mask_attached




class AutoEncoderSeparateHidden(AutoEncoderSeparate):
    def __init__(self, cfg):
        super(AutoEncoderSeparateHidden, self).__init__(cfg)

    def pos_energy(self, post_samples):
        """
        Compute positive phase (energy expval under posterior variables)
        """
        p3 = self.prior.sigmoid_C_k(self.prior.weight_dict['03'],   self.prior.weight_dict['13'],   self.prior.weight_dict['23'], 
                              post_samples[0],post_samples[1],post_samples[2], self.prior.bias_dict['3'])
        pos_energy = self.prior.energy_exp_cond(post_samples[0],post_samples[1],post_samples[2], p3).mean()
        return pos_energy