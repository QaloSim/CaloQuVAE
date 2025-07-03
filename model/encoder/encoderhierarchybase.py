"""
This model is specifically tailored for Atlas Reg.

Authors: The CaloQVAE
Year: 2025
"""

import torch.nn as nn
import torch
from model.gumbel import GumbelMod
import torch.nn.functional as F
import numpy as np

class HierarchicalEncoder(nn.Module):
    def __init__(self, cfg):
        super(HierarchicalEncoder, self).__init__()
        self.smoothing_dist_mod = GumbelMod()
        self._config = cfg

        self.n_latent_hierarchy_lvls=self._config.model.latent_hierarchy_lvls

        self.n_latent_nodes=self._config.rbm.latent_nodes_per_p

        self._networks=nn.ModuleList([])
        
        for lvl in range(self.n_latent_hierarchy_lvls-1):
            network=self._create_hierarchy_network(level=lvl)
            self._networks.append(network)

    def _create_hierarchy_network(self, level=0):

        if self._config.model.encoderblock == "AtlasReg":
            return EncoderBlockPBH3Dv3Reg(self._config)

    def forward(self, x, x0, beta_smoothing_fct=5):
        """ This function defines a hierarchical approximate posterior distribution. The length of the output is equal 
            to n_latent_hierarchy_lvls and each element in the list is a DistUtil object containing posterior distribution 
            for the group of latent nodes in each hierarchy level. 

        Args:
            input: a tensor containing input tensor.
            is_training: A boolean indicating whether we are building a training graph or evaluation graph.

        Returns:
            posterior: a list of DistUtil objects containing posterior parameters.
            post_samples: A list of samples from all the levels in the hierarchy, i.e. q(z_k| z_{0<i<k}, x).
        """
        
        post_samples = []
        post_logits = []
        
        post_samples.append(self.binary_energy(x0))
        
        for lvl in range(self.n_latent_hierarchy_lvls-1):
            
            current_net=self._networks[lvl]
            current_input = x

            # Clamping logit values
            logits=torch.clamp(current_net(current_input, x0, post_samples), min=-88., max=88.)

            post_logits.append(logits)

            beta = torch.tensor(beta_smoothing_fct,
                                dtype=torch.float, device=logits.device,
                                requires_grad=False)

            samples=self.smoothing_dist_mod(logits, beta)

            post_samples.append(samples)
              
        return beta, post_logits, post_samples
    
    def binary(self, x, bits):
        mask = 2**torch.arange(bits).to(x.device, x.dtype)
        return x.bitwise_and(mask).ne(0).byte().to(dtype=torch.float)
    
    def binary_energy(self, x, lin_bits=20, sqrt_bits=20, log_bits=20):
        reps = int(np.floor(self.n_latent_nodes/(lin_bits+sqrt_bits+log_bits)))
        residual = self.n_latent_nodes - reps*(lin_bits+sqrt_bits+log_bits)
        x = torch.cat((self.binary(x.int(),lin_bits), 
                       self.binary((x.sqrt() * torch.sqrt(torch.tensor(10))).int(),sqrt_bits), 
                       self.binary((x.log() * torch.tensor(10).exp()).int(),log_bits)), 1)
        return torch.cat((x.repeat(1,reps), torch.zeros(x.shape[0],residual).to(x.device, x.dtype)), 1)
    

class EncoderBlockPBH3Dv3Reg(nn.Module):
    def __init__(self, cfg=None):
        super(EncoderBlockPBH3Dv3Reg, self).__init__()
        self._config = cfg
        self.n_latent_nodes = self._config.rbm.latent_nodes_per_p
        self.z = self._config.data.z #45
        self.r = self._config.data.r #9
        self.phi = self._config.data.phi #16
        
        self.seq1 = nn.Sequential(
    
                PeriodicConv3d(1, 32, (1,3,3), (1,1,2), 1),
                nn.BatchNorm3d(32),
                nn.PReLU(32, 0.02),
    
                PeriodicConv3d(32, 128, (2,2,3), (1,2,2), 1),
                nn.BatchNorm3d(128),
                nn.PReLU(128, 0.02),
                )

        self.seq2 = nn.Sequential(
                        PeriodicConv3d(129, 256, (3,3,3), (1,2,1), 0),
                        nn.BatchNorm3d(256),
                        nn.PReLU(256, 0.02),

                        PeriodicConv3d(256, self.n_latent_nodes, (3,3,3), (2,2,2), 0),
                        nn.PReLU(self.n_latent_nodes, 1.0),
                        nn.Flatten(),
                        )
        

    def forward(self, x, x0, post_samples):
        # 1 channel of a 3d object / shower
        x = x.reshape(x.shape[0], 1, self.z, self.phi, self.r) 
        pos_enc_samples = self._pos_enc(post_samples)
        x = x + pos_enc_samples.unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat(1,1,torch.tensor(x.shape[-3:-2]).item(),torch.tensor(x.shape[-2:-1]).item(), torch.tensor(x.shape[-1:]).item())
        x = self.seq1(x)
            
        x0 = self.trans_energy(x0)
        x = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat(1,1,torch.tensor(x.shape[-3:-2]).item(),torch.tensor(x.shape[-2:-1]).item(), torch.tensor(x.shape[-1:]).item())), 1)
        x = self.seq2(x)
        
        return x
    
    def _pos_enc(self, post_samples):
        post_samples = torch.cat(post_samples,1)
        M = post_samples.shape[1]

        pres = [(torch.arange(0,M).multiply(np.pi/M).cos().to(post_samples.device) * post_samples + torch.arange(0,M).multiply(np.pi/M).sin().to(post_samples.device) *(1 - post_samples).abs()).divide(np.sqrt(M)).unsqueeze(2) for i in np.arange(1,M/4-1,1)]
        pos_enc = torch.cat(pres,2).transpose(1,2);
        res = pos_enc.sum([1,2])/(M-1)
        return res.unsqueeze(1)
    
    def trans_energy(self, x0, log_e_max=16.0, log_e_min=5.0, s_map = 1.0):
        # s_map = max(scaled voxel energy u_i) * (incidence energy / slope of total energy in shower) of the dataset
        return ((torch.log(x0) - log_e_min)/(log_e_max - log_e_min)) * s_map


class PeriodicConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(PeriodicConv3d, self).__init__()
        self.padding = padding
        # try 3x3x3 cubic convolution
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=dilation, groups=groups, bias=bias)
    def forward(self, x):
        # Pad input tensor with periodic boundary and circle-center conditions
        if self.padding == 1:
            mid = x.shape[-2] // 2
            shift = torch.cat((x[..., mid:, [0]], x[..., :mid, [0]]), -2)
            x = torch.cat((shift,x), dim=-1)
        x = F.pad(x, (0, 0, self.padding, self.padding, 0, 0), mode='circular')
        # Apply convolution
        x = self.conv(x)
        return x