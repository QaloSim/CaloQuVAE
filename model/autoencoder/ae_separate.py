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
        Overridden decode method from autoencoderbase.
        Returns raw components needed for differentiable physics loss.
        """
        output_hits, output_activations = self.decoder(torch.cat(post_samples, 1), x0)
        
        # Initialize placeholders for the new returns
        activations_raw = None
        hit_mask_attached = None

        if self.training:
            if hasattr(self._config, "separate_hits") and self._config.separate_hits:
                # 1. Get the Soft (Gumbel) Mask with gradients attached
                hit_mask_attached = self._hit_smoothing_dist_mod(output_hits, beta=beta)
        
                # 2. Get Raw Activations
                activations_raw = self._activation_fct(act_fct_slope, output_activations)
                
                # 3. Create the "Safe" output for MSE (Detached Mask)
                # This prevents the MSE loss from collapsing the mask to zero
                output_activations = activations_raw * hit_mask_attached.detach() 
            else:
                # Fallback for standard training (fixed geometry or no separate hits)
                output_activations = self._activation_fct(act_fct_slope, output_activations) * torch.where(x > 0, 1., 0.)
        else:
            # Evaluation mode
            output_activations = self._activation_fct(0.0, output_activations) * self._hit_smoothing_dist_mod(output_hits)
       
        # Return 4 values now (update your training loop unpacking to match this!)
        return output_hits, output_activations, activations_raw, hit_mask_attached

    def loss(self, input_data, x0, args):
        """
        Child class loss with Focal Loss + Differentiable Physics Loss.
        """
        logger.debug("loss")
        
        # Unpack the 4 values returned by the new decode method
        beta, post_logits, post_samples, output_activations, output_hits, activations_raw, hit_mask_attached = args

        # 1. --- Standard Reconstruction Loss ---
        # Uses output_activations (which has the detached mask).
        ae_loss = torch.pow((input_data - output_activations), 2) * torch.exp(self._config.model.mse_weight * input_data)
        ae_loss = torch.mean(torch.sum(ae_loss, dim=1), dim=0) * self._config.model.coefficient

        # 2. --- Hit Loss with Focal Support ---
        targets = torch.where(input_data > 0, 1., 0.)
        
        # Calculate raw unreduced BCE first. This is -log(pt).
        bce_raw = binary_cross_entropy_with_logits(output_hits, targets, reduction='none')

        # Check if Focal Loss parameters exist in config
        if hasattr(self._config.model, "focal_alpha") and hasattr(self._config.model, "focal_gamma"):
            alpha = self._config.model.focal_alpha
            gamma = self._config.model.focal_gamma
            
            # Calculate pt (probability of the true class)
            pt = torch.exp(-bce_raw)
            
            # Calculate alpha_t: alpha for class 1, (1-alpha) for class 0
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            
            # Apply Focal Loss factors
            focal_loss = alpha_t * (1 - pt).pow(gamma) * bce_raw
            hit_loss = torch.mean(torch.sum(focal_loss, dim=1), dim=0)
        else:
            # Revert to standard BCE reduction
            hit_loss = torch.mean(torch.sum(bce_raw, dim=1), dim=0)

        # 3. --- Feature Extraction Setup (The Fix) ---
        mmd_weight = getattr(self._config.model.loss_coeff, "mmd_loss", 0.0)
        mae_weight = getattr(self._config.model.loss_coeff, "feature_mae", 0.0) 

        feat_gt = None
        feat_recon = None
        
        if mmd_weight > 0.0 or mae_weight > 0.0:
            # Construct the Differentiable Reconstruction
            if activations_raw is not None and hit_mask_attached is not None:
                # This path allows gradients to flow into the mask (hits head)
                physics_recon = activations_raw * hit_mask_attached
            else:
                # Fallback if raw components aren't available
                physics_recon = output_activations

            feat_gt = self.feature_extractor(input_data)
            feat_recon = self.feature_extractor(physics_recon)

        # 4. --- Conditional MMD Feature Loss ---
        mmd_loss_total = torch.tensor(0.0, device=input_data.device)            
        if mmd_weight > 0.0:
            norm_energy = self.cond_normalizer(x0)
            for key in feat_gt.keys():
                val_gt = feat_gt[key].view(input_data.size(0), -1)
                val_recon = feat_recon[key].view(input_data.size(0), -1)
                mmd_loss_total += compute_cmmd(val_gt, norm_energy, val_recon, norm_energy)

        # 5. --- Physics Feature MAE Loss ---
        mae_loss_total = torch.tensor(0.0, device=input_data.device)
        if mae_weight > 0.0:
            for key in feat_gt.keys():
                val_gt = feat_gt[key].view(input_data.size(0), -1)
                val_recon = feat_recon[key].view(input_data.size(0), -1)
                mae_loss_total += torch.abs(val_gt - val_recon).mean()

        # 6. --- Aggregate and Return ---
        total_loss_dict = {
            "ae_loss": ae_loss,
            "hit_loss": hit_loss,
            "mmd_loss": mmd_loss_total,
            "feature_mae": mae_loss_total
        }

        # Add RBM specific terms if configured (Standard Boilerplate)
        if hasattr(self._config.model.loss_coeff, 'pos_energy') and hasattr(self._config.model.loss_coeff, 'logit_distance'):
            l_dist = torch.pow(torch.cat(post_logits, 1) - torch.cat(self.logit_distance(post_samples, post_logits), 1), 2).mean()
            pos_energy = self.pos_energy(post_samples)
            entropy_loss = -1 * self.posterior_entropy(post_logits)        

            total_loss_dict.update({
                "entropy": entropy_loss,
                "pos_energy": pos_energy,
                "logit_distance": l_dist
            })

        return total_loss_dict

    def forward(self, xx, beta_smoothing_fct=5, act_fct_slope=0.02):
        """
        - Overrides forward in autoencoderbase to unpack  and return more values for loss calculation.
        """
        logger.debug("VAE_forward")
        
        x, x0 = xx
        
        beta, post_logits, post_samples = self.encoder(x, x0, beta_smoothing_fct)

        output_hits, output_activations, activations_raw, hit_mask_attached = self.decode(post_samples, x, x0, beta, act_fct_slope)

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