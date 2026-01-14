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

    def loss(self, input_data, x0, args):
            """
            Extended loss function with MMD term.
            """
            logger.debug("loss")
            beta, post_logits, post_samples, output_activations, output_hits = args

            # 1. Standard Reconstruction Loss
            ae_loss = torch.pow((input_data - output_activations),2) * torch.exp(self._config.model.mse_weight*input_data)
            ae_loss = torch.mean(torch.sum(ae_loss, dim=1), dim=0) * self._config.model.coefficient

            # 2. Hit Loss (BCE)
            hit_loss = binary_cross_entropy_with_logits(output_hits, torch.where(input_data > 0, 1., 0.), 
                        reduction='none')
            hit_loss = torch.mean(torch.sum(hit_loss, dim=1), dim=0)

            # 3. --- Conditional MMD Feature Loss ---
            # We only compute MMD if the weight is > 0 to save compute
            
            mmd_loss_total = torch.tensor(0.0, device=input_data.device)            
            mmd_weight = getattr(self._config.model.loss_coeff, "mmd_loss", 0.0)
            
            if mmd_weight > 0.0:
                # A. Extract Physics Features
                feat_gt = self.feature_extractor(input_data)
                feat_recon = self.feature_extractor(output_activations)
                # B. Normalize Conditions (Energy)
                norm_energy = self.cond_normalizer(x0)

                # C. Compute C-MMD for each feature
                for key in feat_gt.keys():
                    val_gt = feat_gt[key].view(input_data.size(0), -1)
                    val_recon = feat_recon[key].view(input_data.size(0), -1)
                    
                    # In reconstruction, the target condition is the SAME as the source condition.
                    mmd_loss_total += compute_cmmd(val_gt, norm_energy, val_recon, norm_energy)
            # 4. Aggregate
            total_loss_dict = {
                "ae_loss": ae_loss,
                "hit_loss": hit_loss,
                "mmd_loss": mmd_loss_total
            }

            # 5. Add RBM specific terms if configured
            if hasattr(self._config.model.loss_coeff, 'pos_energy') and hasattr(self._config.model.loss_coeff, 'logit_distance'):
                l_dist = torch.pow(torch.cat(post_logits,1) - torch.cat(self.logit_distance(post_samples, post_logits),1),2).mean()
                pos_energy = self.pos_energy(post_samples)
                entropy_loss = -1 * self.posterior_entropy(post_logits)        

                total_loss_dict.update({
                    "entropy": entropy_loss,
                    "pos_energy": pos_energy,
                    "logit_distance": l_dist
                })

            return total_loss_dict



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