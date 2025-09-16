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
from model.encoder.encoderhierarchybase import HierarchicalEncoder, HierarchicalEncoderHidden
from model.decoder.decoder import Decoder
from model.decoder.decoderhierarchybase import DecoderHierarchyBase, DecoderHierarchyBaseV2, DecoderHierarchyBaseV3, DecoderHierarchyBaseV4, DecoderHierarchyBaseV5
from model.decoder.decoder_hier_geo import DecoderHierarchyGeometry
from model.decoder.decoder_full_geo import DecoderFullGeo
from model.decoder.decoderhierarchy0 import DecoderHierarchy0, DecoderHierarchyv3, DecoderHierarchy0Hidden
from model.decoder.decoderhierarchy0ca import DecoderHierarchy0CA
from model.decoder.decoderhierarchytf import DecoderHierarchyTF, DecoderHierarchyTFv2
from model.decoder.decoder_ATLAS_new import DecoderATLASNew, DecoderFullGeoATLASNew
from model.rbm.rbm import RBM, RBM_Hidden
from model.rbm.rbm_torch import RBMtorch, RBM_Hiddentorch

#logging module with handmade settings.
from CaloQuVAE import logging
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
        elif self._config.model.encoder == "hierarchicalencoderhidden":
            return HierarchicalEncoderHidden(self._config)

    def _create_decoder(self):
        logger.debug("::_create_decoder")
        if self._config.model.decoder == "decoder":
            return Decoder(self._config)
        elif self._config.model.decoder == "hierachicaldecoder":
            return DecoderHierarchyBase(self._config)
        elif self._config.model.decoder == "hierarchicaldecoderv2":
            return DecoderHierarchyBaseV2(self._config)
        elif self._config.model.decoder == "hierarchicaldecoderv3":
            return DecoderHierarchyBaseV3(self._config)
        elif self._config.model.decoder == "decoderhiergeo":
            return DecoderHierarchyGeometry(self._config)
        elif self._config.model.decoder == "decoderhierarchyv4":
            return DecoderHierarchyBaseV4(self._config)        
        elif self._config.model.decoder == "decoderhierarchyv5":
            return DecoderHierarchyBaseV5(self._config)
        elif self._config.model.decoder == "decoderhierachy0":
            return DecoderHierarchy0(self._config)
        elif self._config.model.decoder == "decoderhierachyv3":
            return DecoderHierarchyv3(self._config)
        elif self._config.model.decoder == "decoderhierachytf":
            return DecoderHierarchyTF(self._config)
        elif self._config.model.decoder == "decoderfullgeo":
            return DecoderFullGeo(self._config)
        elif self._config.model.decoder == "decoderatlasnew":
            return DecoderATLASNew(self._config)
        elif self._config.model.decoder == "decoderfullgeoatlasnew":
            return DecoderFullGeoATLASNew(self._config)
        elif self._config.model.decoder == "decoderhierachytfv2":
            return DecoderHierarchyTFv2(self._config)
        elif self._config.model.decoder == "decoderhierachy0ca":
            return DecoderHierarchy0CA(self._config)

    def _create_prior(self):
        logger.debug("::_create_prior")
        return RBMtorch(self._config)

    def create_networks(self):
        logger.debug("Creating Network Structures")
        self.encoder=self._create_encoder()
        self.prior=self._create_prior()
        self.decoder=self._create_decoder()
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

        output_hits, output_activations = self.decode(post_samples, x, x0, beta, act_fct_slope)

        return beta, post_logits, post_samples, output_activations, output_hits

    def decode(self, post_samples, x, x0, beta=5, act_fct_slope=0.02):
        
        output_hits, output_activations = self.decoder(torch.cat(post_samples,1), x0)
        
        if self.training:
            output_activations = self._activation_fct(act_fct_slope)(output_activations) * torch.where(x > 0, 1., 0.)
        else:
            output_activations = self._activation_fct(0.0)(output_activations) * self._hit_smoothing_dist_mod(output_hits)
       
        return output_hits, output_activations
    
    def loss(self, input_data, args):
        """
        - Overrides loss in gumboltCaloV5.py
        """
        logger.debug("loss")
        beta, post_logits, post_samples, output_activations, output_hits = args

        kl_loss, entropy, pos_energy, neg_energy = self.kl_divergence(post_logits, post_samples)
        # kl_loss, entropy, pos_energy, neg_energy = 0,0,0,0
        
        ae_loss = torch.pow((input_data - output_activations),2) * torch.exp(self._config.model.mse_weight*input_data)
        ae_loss = torch.mean(torch.sum(ae_loss, dim=1), dim=0) * self._config.model.coefficient

        hit_loss = binary_cross_entropy_with_logits(output_hits, torch.where(input_data > 0, 1., 0.),
                        reduction='none')
        # weight= (1+input_data).pow(self._config.model.bce_weights_power)
        hit_loss = torch.mean(torch.sum(hit_loss, dim=1), dim=0)
        l_dist = torch.pow(torch.cat(post_logits,1) - torch.cat(self.logit_distance(post_samples, post_logits),1),2).mean()

        return {"ae_loss":ae_loss, "kl_loss":kl_loss, "hit_loss":hit_loss,
                "entropy":entropy, "pos_energy":pos_energy, "neg_energy":neg_energy, "logit_distance":l_dist}
    
    def kl_divergence(self, post_logits, post_samples, is_training=True):
        """Overrides kl_divergence in GumBolt.py

        :param post_logits (list) : List of f(logit_i|x, e) for each hierarchy
                                    layer i. Each element is a tensor of size
                                    (batch_size * n_nodes_per_hierarchy_layer)
        :param post_zetas (list) : List of q(zeta_i|x, e) for each hierarchy
                                   layer i. Each element is a tensor of size
                                   (batch_size * n_nodes_per_hierarchy_layer)
        """
        # Concatenate all hierarchy levels
        # logits_q_z = torch.cat(post_logits, 1)
        # post_zetas = torch.cat(post_samples, 1)

        # Compute cross-entropy b/w post_logits and post_samples
        entropy = - self._bce_loss(torch.cat(post_logits, 1), torch.cat(post_samples, 1)[:,self._config.rbm.latent_nodes_per_p:])
        entropy = torch.mean(torch.sum(entropy, dim=1), dim=0)

        # Compute positive phase (energy expval under posterior variables) 
        pos_energy = self.prior.energy_exp_cond(post_samples[0].detach(),post_samples[1],post_samples[2],post_samples[3]).mean()

        # Compute gradient computation of the logZ term
        p0_state, p1_state, p2_state, p3_state \
            = self.prior.block_gibbs_sampling_cond(post_samples[0].detach(),post_samples[1].detach(),post_samples[2].detach(),post_samples[3].detach())
        
        # neg_energy = - self.energy_exp(p0_state, p1_state, p2_state, p3_state)
        neg_energy = - self.prior.energy_exp_cond(p0_state, p1_state, p2_state, p3_state).mean()

        # Estimate of the kl-divergence
        kl_loss = entropy + pos_energy + neg_energy
        return kl_loss, entropy, pos_energy, neg_energy
    
    def logit_distance(self, post_samples, post_logits):
        p0 = post_samples[0]
        p1 = post_logits[0]
        p2 = post_logits[1]
        p3 = post_logits[2]

        W01 = self.prior.weight_dict['01']
        W02 = self.prior.weight_dict['02']
        W03 = self.prior.weight_dict['03']

        W12 = self.prior.weight_dict['12']
        W13 = self.prior.weight_dict['13']
        W23 = self.prior.weight_dict['23']
        # precompute the needed transposes only once
        W12_T = W12.T
        W13_T = W13.T
        W23_T = W23.T

        b1 = self.prior.bias_dict['1']
        b2 = self.prior.bias_dict['2']
        b3 = self.prior.bias_dict['3']

        p3_ans = self.logit_mcmc(W03,   W13,   W23,   p0, torch.sigmoid(p1), torch.sigmoid(p2), b3)
        p2_ans = self.logit_mcmc(W02,   W12, W23_T,   p0, torch.sigmoid(p1), torch.sigmoid(p3), b2)
        p1_ans = self.logit_mcmc(W01, W12_T, W13_T,   p0, torch.sigmoid(p2), torch.sigmoid(p3), b1)

        return p1_ans, p2_ans, p3_ans
        return torch.pow(torch.cat(post_logits,1) - torch.cat([p1_ans, p2_ans, p3_ans],1),2).mean()

    def logit_mcmc(self, weights_ax, weights_bx, weights_cx,
                 pa_state, pb_state, pc_state, bias_x) -> torch.Tensor:
        """partition_state()

        :param weights_a (torch.Tensor) : (n_nodes_a, n_nodes_x)
        :param weights_b (torch.Tensor) : (n_nodes_b, n_nodes_x)
        :param weights_c (torch.Tensor) : (n_nodes_c, n_nodes_x)
        :param pa_state (torch.Tensor) : (batch_size, n_nodes_a)
        :param pb_state (torch.Tensor) : (batch_size, n_nodes_b)
        :param pc_state (torch.Tensor) : (batch_size, n_nodes_c)
        :param bias_x (torch.Tensor) : (n_nodes_x)
        """
        p_activations = (torch.matmul(pa_state, weights_ax) +
                            torch.matmul(pb_state, weights_bx) +
                            torch.matmul(pc_state, weights_cx) + bias_x)
        return p_activations.detach()

    def print_model_info(self):
        for key,par in self.__dict__.items():
            if isinstance(par,torch.Tensor):
                logger.info("{0}: {1}".format(key, par.shape))
            else:
                logger.debug("{0}: {1}".format(key, par))

class AutoEncoderHidden(AutoEncoderBase):
    def __init__(self, cfg):
        super(AutoEncoderHidden,self).__init__(cfg)

    def _create_encoder(self):
        logger.debug("::_create_encoder")
        if self._config.model.encoder == "hierarchicalencoderhidden":
            return HierarchicalEncoderHidden(self._config)
        
    def _create_prior(self):
        logger.debug("::_create_prior")
        # return RBM_Hidden(self._config)
        return RBM_Hiddentorch(self._config)
    
    def _create_decoder(self):
        logger.debug("::_create_decoder")
        if self._config.model.decoder == "decoderhierachy0hidden":
            return DecoderHierarchy0Hidden(self._config)
        elif self._config.model.decoder == "decoderfullgeoatlasnew":
            return DecoderFullGeoATLASNew(self._config)

        
        
    def kl_divergence(self, post_logits, post_samples, is_training=True):
        """Overrides kl_divergence in GumBolt.py

        :param post_logits (list) : List of f(logit_i|x, e) for each hierarchy
                                    layer i. Each element is a tensor of size
                                    (batch_size * n_nodes_per_hierarchy_layer)
        :param post_zetas (list) : List of q(zeta_i|x, e) for each hierarchy
                                   layer i. Each element is a tensor of size
                                   (batch_size * n_nodes_per_hierarchy_layer)
        """
        # Concatenate all hierarchy levels
        # logits_q_z = torch.cat(post_logits, 1)
        # post_zetas = torch.cat(post_samples, 1)

        # Compute cross-entropy b/w post_logits and post_samples
        entropy = - self._bce_loss(torch.cat(post_logits, 1), torch.cat(post_samples, 1)[:,self._config.rbm.latent_nodes_per_p:])
        entropy = torch.mean(torch.sum(entropy, dim=1), dim=0)

        # Compute positive phase (energy expval under posterior variables) 
        p3 = self.prior.sigmoid_C_k(self.prior.weight_dict['03'],   self.prior.weight_dict['13'],   self.prior.weight_dict['23'], 
                              post_samples[0],post_samples[1],post_samples[2], self.prior.bias_dict['3'])
        pos_energy = self.prior.energy_exp_cond(post_samples[0],post_samples[1],post_samples[2], p3).mean()

        # Compute gradient computation of the logZ term
        p0_state, p1_state, p2_state, p3_state \
            = self.prior.block_gibbs_sampling_cond(post_samples[0].detach(),post_samples[1].detach(),post_samples[2].detach())
        
        # neg_energy = - self.energy_exp(p0_state, p1_state, p2_state, p3_state)
        neg_energy = - self.prior.energy_exp_cond(p0_state, p1_state, p2_state, p3_state).mean()

        # Estimate of the kl-divergence
        kl_loss = entropy + pos_energy + neg_energy
        return kl_loss, entropy, pos_energy, neg_energy

if __name__=="__main__":
    logger.info("Running autoencoderbase.py directly") 
    logger.info("Success")