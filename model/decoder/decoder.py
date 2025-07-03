"""
This model is specifically tailored for Atlas Reg.

Authors: The CaloQVAE
Year: 2025
"""

import torch.nn as nn
import torch
import torch.nn.functional as F

class Decoder(nn.Module):
    def __init__(self, cfg):
        super(Decoder, self).__init__()
        self._config = cfg

        self.n_latent_hierarchy_lvls=self._config.rbm.partitions

        self.n_latent_nodes=self._config.rbm.latent_nodes_per_p * self._config.rbm.partitions

        self.z = self._config.data.z
        self.r = self._config.data.r
        self.phi = self._config.data.phi
        
        self._layers =  nn.Sequential(
                   nn.Unflatten(1, (self.n_latent_nodes, 1, 1, 1)),

                   PeriodicConvTranspose3d(self.n_latent_nodes, 512, (3,3,3), (2,2,2), 0),
                   nn.BatchNorm3d(512),
                   # self.dropout,
                   nn.PReLU(512, 0.02),
                   

                   PeriodicConvTranspose3d(512, 128, (3,3,3), (1,2,1), 0),
                   nn.BatchNorm3d(128),
                   nn.PReLU(128, 0.02),
                                   )
        
        self._layers2 = nn.Sequential(
                   PeriodicConvTranspose3d(129, 64, (2,3,2), (1,1,2), 1),
                   nn.BatchNorm3d(64),
                   nn.PReLU(64, 0.02),

                   PeriodicConvTranspose3d(64, 1, (2,2,3), (1,1,2), 1),
                   nn.PReLU(1, 1.0),
                                   )
        
        self._layers3 = nn.Sequential(
                   PeriodicConvTranspose3d(129, 64, (2,3,2), (1,1,2), 1),
                   nn.BatchNorm3d(64),
                   nn.PReLU(64, 0.02),

                   PeriodicConvTranspose3d(64, 1, (2,2,3), (1,1,2), 1),
                   nn.PReLU(1, 0.02),
                                   )
        
    def forward(self, x, x0):
                
        x = self._layers(x)
        x0 = self.trans_energy(x0)
        xx0 = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat(1,1,torch.tensor(x.shape[-3:-2]).item(),torch.tensor(x.shape[-2:-1]).item(), torch.tensor(x.shape[-1:]).item())), 1)
        x1 = self._layers2(xx0) #hits
        x2 = self._layers3(xx0)
        return x1.reshape(x1.shape[0],self.z*self.r*self.phi), x2.reshape(x1.shape[0],self.z*self.r*self.phi)
    
    def trans_energy(self, x0, log_e_max=16.0, log_e_min=5.0, s_map = 1.0):
        return ((torch.log(x0) - log_e_min)/(log_e_max - log_e_min)) * s_map
    

class PeriodicConvTranspose3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(PeriodicConvTranspose3d, self).__init__()
        self.padding = padding
        self.conv = nn.ConvTranspose3d(in_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        # Apply convolution
        x = self.conv(x)
        # Pad input tensor with periodic boundary conditions
        if self.padding == 1:
            mid = x.shape[-2] // 2
            shift = torch.cat((x[..., mid:, [0]], x[..., :mid, [0]]), -2)
            x = torch.cat((shift,x), dim=-1)
            x = F.pad(x, (0, 0, self.padding, self.padding, 0, 0), mode='circular')
        return x