"""
Separate Autoencoder conditioned on layer energies (u)
Inherits from AutoencoderSeparate
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.autoencoder.ae_separate import AutoEncoderSeparate
from model.encoder.encoderhierarchybase import HierarchicalEncoderLayers
from model.decoder.decoder_layers import DecoderLayers, UpsamplingDecoderLayers

from CaloQuVAE import logging
logger = logging.getLogger(__name__)


class AutoencoderLayers(AutoEncoderSeparate):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.register_buffer('feature_min', torch.zeros(5))
        self.register_buffer('feature_max', torch.ones(5))

    def _create_encoder(self):
        if self._config.model.encoder == "hierarchicalencoderlayers":
            return HierarchicalEncoderLayers(self._config)
        else:
            raise ValueError(f"Unknown encoder type: {self._config.model.encoder}")
    
    def _create_decoder(self):
        logger.debug("::_create_decoder")
        if self._config.model.decoder == "decoderlayers":
            return DecoderLayers(self._config)
        elif self._config.model.decoder == "decoderlayersupsample":
            return UpsamplingDecoderLayers(self._config)
        else:
            raise ValueError(f"Unknown decoder type: {self._config.model.decoder}")



    def decode(self, post_samples, x, x0, u, act_fct_slope=0.02, beta_hits=3.0):
        """
        Decodes through self.decoder
        Args:
            post_samples: binary samples from the latent space, shape (b, n_hierarchy, n_latent_per_p)
            x: shower voxels, shape (b, n_voxels)
            x0: incident energy, shape (b, 1)
            u: Layer energies, shape (b, n_l)
        """
        output_hits, output_activations = self.decoder(torch.cat(post_samples, dim=1), x0, u)
        if self.training:
            output_activations = self._activation_fct(act_fct_slope, output_activations) * torch.where(x > 0, 1., 0.)
        else:
            output_activations = self._activation_fct(0.0, output_activations) * self._hit_smoothing(output_hits)
        return output_hits, output_activations

    def forward(self, inputs, beta_latent=1.0, beta_hits=1.0, act_fct_slope=0.02):
        """
        Forward pass through the model
        Args:
            inputs: tuple of (x, x0, u)
                x: shower voxels, shape (b, n_voxels)
                x0: incident energy, shape (b, 1)
                u: Layer energies, shape (b, n_l)
            beta_latent: beta for gumbel mod in encoder
            beta_hits: beta for gumbel mod in decoder hits head
        """
        x, x0, u = inputs
        post_logits, post_samples = self.encoder(x, x0, u, beta_smoothing_fct=beta_latent)
        output_hits, output_activations = self.decode(post_samples, x, x0, u, act_fct_slope, beta_hits)
        return post_logits, post_samples, output_hits, output_activations

    def loss(self, x, output_hits, output_activations):
        """
        Computes loss between input and output
        Args:
            x: shower voxels, shape (b, n_voxels)
            x0: incident energy, shape (b, 1)
            u: Layer energies, shape (b, n_l)
            output_hits: predicted hits, shape (b, n_voxels)
            output_activations: predicted activations, shape (b, n_voxels)
        Returns:
            loss_dict: dictionary of loss components
        """
        squared_diff = torch.pow((x - output_activations), 2)
        arg = self._config.model.mse_weight * x
        energy_weighting = torch.exp(torch.clamp(arg, max=20))            
        ae_loss = squared_diff * energy_weighting
        ae_loss = torch.clamp(torch.mean(torch.sum(ae_loss, dim=1), dim=0) * self._config.model.coefficient, max=1e30)
        target_hits = (x > 0).float()
        bce_raw = F.binary_cross_entropy_with_logits(output_hits, target_hits, reduction='none')
        bce_loss = torch.mean(torch.sum(bce_raw, dim=1), dim=0)
        loss_dict = {
            "ae_loss": ae_loss,
            "hit_loss": bce_loss,
        }
        return loss_dict


class AutoencoderLayersBCE(AutoencoderLayers):
    """
    uses a BCE loss on activations after layer-normalized transformation to voxels
    """
    def decode(self, post_samples, x, x0, u, act_fct_slope=0.02, beta_hits=3.0, temperature=1.0):
        """
        Decodes through self.decoder using a Masked Softmax for the continuous head.
        """
        # output_activations are now raw, unnormalized logits
        output_hits, output_activations = self.decoder(torch.cat(post_samples, dim=1), x0, u)
        
        B = x.shape[0]
        z = self._config.data.z
        logits_3d = output_activations.view(B, z, -1)
        
        if self.training:
            # Teacher forcing: mask using the TRUE hits
            mask_3d = (x > 0).view(B, z, -1)
        else:
            # Inference: mask using the PREDICTED hits
            mask_3d = (self._hit_smoothing(output_hits) > 0).view(B, z, -1)
            
        # Push masked logits to -inf
        masked_logits = logits_3d.masked_fill(~mask_3d, float('-inf'))
        masked_logits /= temperature  # Apply temperature scaling
        
        # Softmax over the voxels-per-layer dimension
        probs_3d = F.softmax(masked_logits, dim=-1)
        
        # If an entire layer is masked out, softmax outputs NaNs. Zero them out safely.
        probs_3d = torch.nan_to_num(probs_3d, nan=0.0)
        output_activations_frac = probs_3d.view(B, -1)
        
        return output_hits, output_activations_frac

    def forward(self, inputs, beta_latent=1.0, beta_hits=1.0, act_fct_slope=0.02, temperature=1.0):
        """
        Forward pass through the model
        Args:
            inputs: tuple of (x, x0, u)
                x: shower voxels, shape (b, n_voxels)
                x0: incident energy, shape (b, 1)
                u: Layer energies, shape (b, n_l)
            beta_latent: beta for gumbel mod in encoder
            beta_hits: beta for gumbel mod in decoder hits head
        """
        x, x0, u = inputs
        post_logits, post_samples = self.encoder(x, x0, u, beta_smoothing_fct=beta_latent)
        output_hits, output_activations = self.decode(post_samples, x, x0, u, act_fct_slope, beta_hits, temperature)
        return post_logits, post_samples, output_hits, output_activations

    def mutual_information_penalty(self, post_logits, eps=1e-8):
        """
        Calculates the Mutual Information penalty for binary latent nodes.
        post_logits: Tensor of shape [batch_size, num_latent_nodes]
        """
        probs = torch.sigmoid(post_logits)
        
        #  Calculate Conditional Entropy H(Z|X)
        # We want to MINIMIZE this (push probs towards 0 or 1 for each sample)
        # Shape: [batch_size, num_latent_nodes] -> reduce to scalar
        h_z_given_x = - (probs * torch.log(probs + eps) + (1 - probs) * torch.log(1 - probs + eps))
        expected_h_z_given_x = h_z_given_x.sum(dim=-1).mean() 
        
        #  Calculate Marginal Entropy H(Z)
        # We want to MAXIMIZE this (push average activation across batch towards 0.5)
        # Shape: [num_latent_nodes]
        p_marginal = probs.mean(dim=0)
        h_z = - (p_marginal * torch.log(p_marginal + eps) + (1 - p_marginal) * torch.log(1 - p_marginal + eps))
        expected_h_z = h_z.sum()
        
        #  Mutual Information Penalty
        # We want to maximize MI, so we minimize negative MI: H(Z|X) - H(Z)
        mi_loss = expected_h_z_given_x - expected_h_z
        
        return mi_loss

    def geom_loss(self, x, output_hits):
        """
        Geometric loss that only applies to the hits head
        MAE loss that punishes deviations in sparsity and shower moments
        """
        target_hits = (x > 0).float()
        model_hits = torch.sigmoid(output_hits)

        feat_gt = self.feature_extractor(target_hits)
        feat_pred = self.feature_extractor(model_hits)

        geom_loss_total = 0.0
        for key in feat_gt.keys():
            val_gt = feat_gt[key].view(x.size(0), -1)
            val_pred = feat_pred[key].view(x.size(0), -1)
            
            scale = getattr(self, f"geom_{key}")
            safe_scale = torch.clamp(scale, min=1e-6)
            
            # Calculate scaled MAE
            loss_components = torch.abs(val_gt - val_pred) / safe_scale
            geom_loss_total += loss_components.mean()    
        return geom_loss_total

    def loss(self, x, output_hits, output_activations, post_logits=None):
        """
        Computes BCE for the hits head and Cross-Entropy for the continuous fractions.
        x is expected to be the layer fractions here (the output of engine._reduceBCE)
        """
        target_hits = (x > 0).float()
        
        # Hits Head Loss (BCE)
        bce_raw = F.binary_cross_entropy_with_logits(output_hits, target_hits, reduction='none')
        alpha = self._config.model.loss_coeff.focal_alpha
        gamma = self._config.model.loss_coeff.focal_gamma
        pt = torch.exp(-bce_raw)
        alpha_t = alpha * target_hits + (1 - alpha) * (1 - target_hits)
        focal_loss = alpha_t * (1 - pt).pow(gamma) * bce_raw
        hit_loss = torch.mean(torch.sum(focal_loss, dim=1), dim=0)
        
        # Continuous Head Loss (Cross-Entropy)
        # Formula: - sum( target_prob * log(predicted_prob) )
        ce_raw = - x * torch.log(output_activations + 1e-12)
        ae_loss = torch.mean(torch.sum(ce_raw, dim=1), dim=0)
        
        loss_dict = {
            "ae_loss": ae_loss,
            "hit_loss": hit_loss,
        }
        latent_mi_coeff = getattr(self._config.model.loss_coeff, "latent_mi_loss", 0.0)
        if post_logits is not None and latent_mi_coeff > 0:
            mi_loss = self.mutual_information_penalty(post_logits)
            loss_dict["latent_mi_loss"] = mi_loss
        
        if hasattr(self._config.model.loss_coeff, "geom_loss") and self._config.model.loss_coeff.geom_loss > 0:
            geom_loss = self.geom_loss(x, output_hits)
            loss_dict["geom_loss"] = geom_loss
        return loss_dict

