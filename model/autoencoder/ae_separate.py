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

from model.autoencoder.autoencoderbase import AutoEncoderBase


class AutoEncoderSeparate(AutoEncoderBase):
    def __init__(self, cfg):
        super(AutoEncoderSeparate, self).__init__(cfg)

    def create_networks(self):
        """Override to create encoder and decoder networks without RBM prior."""
        logger.debug("Creating Network Structures")
        self.encoder = self._create_encoder()
        self.decoder = self._create_decoder()

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
    
    def loss(self, input_data, args):
        """
        Override autoencoderbase loss function

        Removes KL divergence term and uses posterior entropy instead.
        """
        logger.debug("loss")
        beta, post_logits, post_samples, output_activations, output_hits = args

        entropy_loss = -1 * self.posterior_entropy(post_logits)        

        
        ae_loss = torch.pow((input_data - output_activations),2) * torch.exp(self._config.model.mse_weight*input_data)
        ae_loss = torch.mean(torch.sum(ae_loss, dim=1), dim=0) * self._config.model.coefficient

        hit_loss = binary_cross_entropy_with_logits(output_hits, torch.where(input_data > 0, 1., 0.), reduction='none')
        hit_loss = torch.mean(torch.sum(hit_loss, dim=1), dim=0)

        return {
            "ae_loss": ae_loss,
            "hit_loss": hit_loss,
            "entropy": entropy_loss,
        }

