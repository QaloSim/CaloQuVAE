"""
Base class for Autoencoder frameworks.

Defines basic common methods and variables shared between models.
Each model overwrites as needed. 
This class inherits from torch.nn.Module, ensuring that network parameters
are registered properly. 
"""
import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy_with_logits
from model.gumbel import GumbelMod
from model.encoder.encoderhierarchybase import HierarchicalEncoder
from model.decoder.decoder import Decoder
from model.rbm.zephyr import ZephyrRBM

#logging module with handmade settings.
from CaloQVAE import logging
logger = logging.getLogger(__name__)

# Base Class for Autoencoder models
class AutoEncoderBase(nn.Module):
    def __init__(self, cfg):
        super(AutoEncoderBase,self).__init__()
        self._config=cfg
        self._hit_smoothing_dist_mod = GumbelMod()
        self._bce_loss = nn.BCEWithLogitsLoss(reduction="none")

    def _activation_fct(self, slope):
        return nn.LeakyReLU(slope)

    def type(self):
        """String identifier for current model.

        Returns:
            model_type: "AE", "VAE", etc.
        """
        return self._model_name

    def _create_encoder(self):
        logger.debug("::_create_encoder")
        if self._config.model.encoder == "hierachicalencoder":
            return HierarchicalEncoder(self._config)

    def _create_decoder(self):
        logger.debug("::_create_decoder")
        if self._config.model.decoder == "decoder":
            return Decoder(self._config)
        
    def _create_prior(self):
        logger.debug("::_create_prior")
        return ZephyrRBM(self._config)

    def _create_sampler(self):
        """
            Define the sampler to be used for sampling from the RBM.

        Returns:
            Instance of baseSampler.
        """
        raise NotImplementedError
    
    def create_networks(self):
        logger.debug("Creating Network Structures")
        self.encoder=self._create_encoder()
        self.decoder=self._create_decoder()
        self.prior=self._create_prior()
        # self.sampler = self._create_sampler()
        # self.stater = self._create_stat()
        
        # self._qpu_sampler = self.prior._qpu_sampler
        # self.sampling_time_qpu = []
        # self.sampling_time_gpu = []
    
    # def generate_samples(self):
    #     raise NotImplementedError

    # def __repr__(self):
    #     parameter_string="\n".join([str(par.shape) if isinstance(par,torch.Tensor) else str(par)  for par in self.__dict__.items()])
    #     return parameter_string
    
    def forward(self, xx, beta_smoothing_fct=5, act_fct_slope=0.02):
        """
        - Overrides forward in GumBoltCaloV5.py
        
        Returns:
            out: output container 
        """
        logger.debug("VAE_forward")
        
        x, x0 = xx
        
        beta, post_logits, post_samples = self.encoder(x, x0, beta_smoothing_fct)
        
        output_hits, output_activations = self.decoder(torch.cat(post_samples,1), x0)

        beta = torch.tensor(self._config.model.output_smoothing_fct, dtype=torch.float, device=output_hits.device, requires_grad=False)
        
        if self.training:
            output_activations = self._activation_fct(act_fct_slope)(output_activations) * torch.where(x > 0, 1., 0.)
        else:
            output_activations = self._activation_fct(0.0)(output_activations) * self._hit_smoothing_dist_mod(output_hits, beta)
       
        return beta, post_logits, post_samples, output_activations, output_hits
    
    def loss(self, input_data, args):
        """
        - Overrides loss in gumboltCaloV5.py
        """
        logger.debug("loss")
        beta, post_logits, post_samples, output_activations, output_hits = args

        # kl_loss, entropy, pos_energy, neg_energy = self.kl_divergence(post_logits, post_samples)
        kl_loss, entropy, pos_energy, neg_energy = 0,0,0,0
        
        ae_loss = torch.pow((input_data - output_activations),2) * torch.exp(self._config.model.mse_weight*input_data)
        ae_loss = torch.mean(torch.sum(ae_loss, dim=1), dim=0) * self._config.model.coefficient

        hit_loss = binary_cross_entropy_with_logits(output_hits, torch.where(input_data > 0, 1., 0.), reduction='none')
        hit_loss = torch.mean(torch.sum(hit_loss, dim=1), dim=0)

        return {"ae_loss":ae_loss, "kl_loss":kl_loss, "hit_loss":hit_loss,
                "entropy":entropy, "pos_energy":pos_energy, "neg_energy":neg_energy}
    
    # def kl_divergence(self, post_logits, post_samples, is_training=True):
    #     """Overrides kl_divergence in GumBolt.py

    #     :param post_logits (list) : List of f(logit_i|x, e) for each hierarchy
    #                                 layer i. Each element is a tensor of size
    #                                 (batch_size * n_nodes_per_hierarchy_layer)
    #     :param post_zetas (list) : List of q(zeta_i|x, e) for each hierarchy
    #                                layer i. Each element is a tensor of size
    #                                (batch_size * n_nodes_per_hierarchy_layer)
    #     """
    #     # Concatenate all hierarchy levels
    #     # logits_q_z = torch.cat(post_logits, 1)
    #     # post_zetas = torch.cat(post_samples, 1)

    #     # Compute cross-entropy b/w post_logits and post_samples
    #     entropy = - self._bce_loss(torch.cat(post_logits, 1), torch.cat(post_samples, 1)[:,self._config.model.n_latent_nodes:])
    #     entropy = torch.mean(torch.sum(entropy, dim=1), dim=0)

    #     # Compute positive phase (energy expval under posterior variables) 
    #     n_nodes_p = self.prior.nodes_per_partition
    #     pos_energy = self.energy_exp_cond(post_zetas[0],post_zetas[1],post_zetas[2],post_zetas[3])

    #     # Compute gradient computation of the logZ term
    #     p0_state, p1_state, p2_state, p3_state \
    #         = self.sampler.block_gibbs_sampling_cond(post_zetas[0],post_zetas[1],post_zetas[2],post_zetas[3], method=self._config.model.rbmMethod)
        
    #     # neg_energy = - self.energy_exp(p0_state, p1_state, p2_state, p3_state)
    #     neg_energy = - self.energy_exp_cond(p0_state, p1_state, p2_state, p3_state)

    #     # Estimate of the kl-divergence
    #     kl_loss = entropy + pos_energy + neg_energy
    #     return kl_loss, entropy, pos_energy, neg_energy

    def print_model_info(self):
        for key,par in self.__dict__.items():
            if isinstance(par,torch.Tensor):
                logger.info("{0}: {1}".format(key, par.shape))
            else:
                logger.debug("{0}: {1}".format(key, par))
        
if __name__=="__main__":
    logger.info("Running autoencoderbase.py directly") 
    logger.info("Success")