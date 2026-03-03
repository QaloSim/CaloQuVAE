"""
Separate Autoencoder conditioned on layer energies (u)
Inherits from AutoencoderSeparate
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.autoencoder.ae_separate import AutoEncoderSeparate
from model.encoder.encoderhierarchybase import HierarchicalEncoderLayers
from model.decoder.decoder_layers import DecoderLayers

from CaloQuVAE import logging
logger = logging.getLogger(__name__)


class AutoencoderLayers(AutoEncoderSeparate):
    def __init__(self, cfg):
        super().__init__(cfg)

    def _create_encoder(self):
        if self._config.model.encoder == "hierarchicalencoderlayers":
            return HierarchicalEncoderLayers(self._config)
        else:
            raise ValueError(f"Unknown encoder type: {self._config.model.encoder}")
    
    def _create_decoder(self):
        logger.debug("::_create_decoder")
        if self._config.model.decoder == "decoderlayers":
            return DecoderLayers(self._config)



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




