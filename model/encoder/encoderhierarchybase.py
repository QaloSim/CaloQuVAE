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
from model.encoder.balancedgraycodes import BalancedGrayCodeCodec
from model.encoder.offsetgraycodes import GrayCodeOffset
from model.encoder.graycodes import GrayCode
from typing import List, Tuple

class HierarchicalEncoder(nn.Module):
    def __init__(self, cfg):
        super(HierarchicalEncoder, self).__init__()
        self.smoothing_dist_mod = GumbelMod()
        self._config = cfg

        self.n_latent_hierarchy_lvls=self._config.rbm.partitions

        self.n_latent_nodes=self._config.rbm.latent_nodes_per_p

        self._networks=nn.ModuleList([])
        if hasattr(self._config.model, 'cond_p_size'):
            self.cond_p_size = self._config.model.cond_p_size
            if hasattr(self._config.model, 'u_bits'):
                self.e_inc_p_size = self.cond_p_size - self._config.model.u_bits * self._config.data.z

        for lvl in range(self.n_latent_hierarchy_lvls-1):
            network=self._create_hierarchy_network(level=lvl)
            self._networks.append(network)
        
        self.gray_codec = GrayCode()
        # set encoding function based on config
        if hasattr(self._config, "use_u") and self._config.use_u and self._config.model.lin_bits<13:
            self.energy_encoding_fct = self.gray_einc_with_u_compact
        elif hasattr(self._config, "use_u") and self._config.use_u:
            self.energy_encoding_fct = self.gray_einc_with_u
        elif hasattr(self._config, "use_gray_code_compact") and self._config.use_gray_code_compact:
            self.energy_encoding_fct = self.gray_encoding_compact
        elif hasattr(self._config, "use_gray_code") and self._config.use_gray_code:
            self.energy_encoding_fct = self.gray_energy_encoding
        elif hasattr(self._config, "refactor_binary_energy") and self._config.refactor_binary_energy:
            self.energy_encoding_fct = self.binary_energy_refactored
        else:
            self.energy_encoding_fct = self.binary_energy
        
        if getattr(self._config.model, "u_cdf", False):
            self.gray_encoding_fct = self.gray_u_cdf
        else:
            self.gray_encoding_fct = self.gray_u
        

    def _create_hierarchy_network(self, level=0):

        if self._config.model.encoderblock == "AtlasReg":
            return EncoderBlockPBH3Dv3Reg(self._config)
        elif self._config.model.encoderblock == "AtlasNew":
            return EncoderBlockATLASNew(self._config)
        elif self._config.model.encoderblock == "CaloChallenge2":
            return EncoderBlockPBH3Dv3(self._config)

    def forward(self, x, x0, beta_smoothing_fct=5):
            """ This function defines a hierarchical approximate posterior distribution. """
            
            post_samples = []
            post_logits = []
            
            post_samples.append(self.energy_encoding_fct(x0))
            
            for lvl in range(self.n_latent_hierarchy_lvls-1):
                
                current_net = self._networks[lvl]
                current_input = x

                # Clamping logit values
                logits = torch.clamp(current_net(current_input, x0, post_samples), min=-88., max=88.)

                post_logits.append(logits)

                beta = torch.tensor(beta_smoothing_fct,
                                    dtype=torch.float, device=logits.device,
                                    requires_grad=False)

                samples = self.smoothing_dist_mod(logits, beta)

                post_samples.append(samples)
                
            return beta, post_logits, post_samples

        # --- Gray Code Encoding Method ---

    def gray_energy_encoding(self, x, lin_bits=19, sqrt_bits=17, log_bits=15):
        """
        Encodes incidence energy using standard Gray Codes
        Replicates the structure/repetition logic of binary_energy_refactored.
        """
        # Override defaults if present in config
        if hasattr(self._config.model, 'lin_bits'):
            lin_bits = self._config.model.lin_bits
        if hasattr(self._config.model, 'sqrt_bits'):
            sqrt_bits = self._config.model.sqrt_bits
        if hasattr(self._config.model, 'log_bits'):
            log_bits = self._config.model.log_bits

        # 1. Get the encoded parts using the codec
        # Linear: direct int cast
        lin_enc = self.gray_codec.encode(x.int(), lin_bits)
        
        # Sqrt: * 200 scaling
        sqrt_enc = self.gray_codec.encode((x.sqrt() * 200).int(), sqrt_bits)
        
        # Log: * 5000 scaling
        # log_enc = self.gray_codec.encode((x.log() * 5000).int()-2**15, log_bits)
        log_enc = self.gray_codec.encode((x.log() * 5000).int(), log_bits)


        x_encoded = torch.cat((lin_enc, sqrt_enc, log_enc), dim=1)

        # 2. Handle Repetitions (Same logic as binary_energy_refactored)
        total_bits_per_rep = lin_bits + sqrt_bits + log_bits
        
        reps = int(np.floor(self.cond_p_size / total_bits_per_rep))
        residual = self.cond_p_size - reps * total_bits_per_rep

        # 3. Repeat and Pad
        padding = torch.zeros(x.shape[0], residual, device=x.device, dtype=x.dtype)
        
        return torch.cat((x_encoded.repeat(1, reps), padding), 1)
    
    def gray_encoding_compact(self, x, lin_bits=19, sqrt_bits=16, log_bits=14):
        """
        Encodes incidence energy using standard Gray Codes
        Compacter version with smaller scaling factors to ensure 1 MeV precision at 1 GeV
        """
        # 1. Get the encoded parts using the codec
        # Linear: direct int cast
        if hasattr(self._config.model, 'lin_bits'):
            lin_bits = self._config.model.lin_bits
        if hasattr(self._config.model, 'sqrt_bits'):
            sqrt_bits = self._config.model.sqrt_bits
        if hasattr(self._config.model, 'log_bits'):
            log_bits = self._config.model.log_bits

        lin_enc = self.gray_codec.encode(x.int(), lin_bits)
        
        sqrt_enc = self.gray_codec.encode((x.sqrt() * 64).int(), sqrt_bits)
        
        log_enc = self.gray_codec.encode((x.log() * 1000).int(), log_bits)

        x_encoded = torch.cat((lin_enc, sqrt_enc, log_enc), dim=2)
        x_encoded = x_encoded.view(x_encoded.shape[0], x_encoded.shape[2]) # second dimension is 1

        total_bits_per_rep = lin_bits + sqrt_bits + log_bits
        
        reps = int(np.floor(self.cond_p_size / total_bits_per_rep))
        residual = self.cond_p_size - reps * total_bits_per_rep

        # 3. Repeat and Pad
        padding = torch.zeros(x.shape[0], residual, device=x.device, dtype=x.dtype)

        return torch.cat((x_encoded.repeat(1, reps), padding), 1)

    def gray_einc_with_u(self, x, lin_bits=13, sqrt_bits=8, log_bits=6):
        """
        Encodes incidence energy using standard Gray Codes
        More compact to allow for additional conditioning on layer energy vector u
        """
        if hasattr(self._config.model, 'lin_bits'):
            lin_bits = self._config.model.lin_bits
        if hasattr(self._config.model, 'sqrt_bits'):
            sqrt_bits = self._config.model.sqrt_bits
        if hasattr(self._config.model, 'log_bits'):
            log_bits = self._config.model.log_bits
        
        lin_enc = self.gray_codec.encode((x / 64.0).int(), lin_bits)
        sqrt_enc = self.gray_codec.encode((x.sqrt() * 4/9.0).int(), sqrt_bits)
        log_enc = self.gray_codec.encode(((x / 1000.0).log() * 32/3.0).int(), log_bits)
        x_encoded = torch.cat((lin_enc, sqrt_enc, log_enc), dim=2)
        x_encoded = x_encoded.view(x_encoded.shape[0], x_encoded.shape[2]) # second dimension is 1

        return x_encoded
    
    def gray_einc_with_u_compact(self, x, lin_bits=9, sqrt_bits=6, log_bits=6):
        """
        Encodes incidence energy using standard Gray Codes
        More compact version with smaller scaling factors
        """
        if hasattr(self._config.model, 'lin_bits'):
            lin_bits = self._config.model.lin_bits
        if hasattr(self._config.model, 'sqrt_bits'):
            sqrt_bits = self._config.model.sqrt_bits
        if hasattr(self._config.model, 'log_bits'):
            log_bits = self._config.model.log_bits

        lin_enc = self.gray_codec.encode((x / 588.0).int(), lin_bits)
        sqrt_enc = self.gray_codec.encode((x.sqrt() * 4.0 / 35.0).round().int(), sqrt_bits)
        log_enc = self.gray_codec.encode(((x / 1000.0).log() * 32/3.0).int(), log_bits)
        x_encoded = torch.cat((lin_enc, sqrt_enc, log_enc), dim=2)
        x_encoded = x_encoded.view(x_encoded.shape[0], x_encoded.shape[2]) # second dimension is 1

        return x_encoded


    def gray_u(self, u):
        """
        Encodes each u_i in u using u_bits of Gray Code
        u_i are real valued in between [0, 1]
        """
        u_bits = self._config.model.u_bits
        u_scaled_bits = (u * (2**u_bits - 1)).round().clamp(0, 2**u_bits - 1).int()
        u_encoded = self.gray_codec.encode(u_scaled_bits, u_bits)
        return u_encoded.view(u_encoded.shape[0], u_encoded.shape[1]*u_encoded.shape[2])
    
    def gray_u_cdf(self, u):
        """
        Encodes each u_i in u using u_bits of Gray Code via CDF Quantile Binning
        u_i are real valued in [0, 1]
        """
        u_bits = self._config.model.u_bits
        
        bin_edges = self.u_bin_edges.to(u.device)
        
        quantized_list = []
        for i in range(u.shape[1]):
            # bucketize returns an integer bin index from 0 to 2**u_bits - 1
            q = torch.bucketize(u[:, i], bin_edges[i])
            quantized_list.append(q)
                
        u_scaled_bits = torch.stack(quantized_list, dim=1).int()
            
        u_scaled_bits = u_scaled_bits.clamp(0, 2**u_bits - 1)
        
        # Encode using the existing GrayCode codec
        u_encoded = self.gray_codec.encode(u_scaled_bits, u_bits)
        
        return u_encoded.view(u_encoded.shape[0], u_encoded.shape[1] * u_encoded.shape[2])


        
    def binary(self, x, bits):
        mask = 2**torch.arange(bits).to(x.device, x.dtype)
        return x.bitwise_and(mask).ne(0).byte().to(dtype=torch.float)
    
    def binary_energy(self, x, lin_bits=15, sqrt_bits=15, log_bits=15):
        reps = int(np.floor(self.n_latent_nodes/(lin_bits+sqrt_bits+log_bits)))
        residual = self.n_latent_nodes - reps*(lin_bits+sqrt_bits+log_bits)
        x = torch.cat((self.binary(x.int(),lin_bits), 
                       self.binary((x.sqrt() * torch.sqrt(torch.tensor(10))).int(),sqrt_bits), 
                       self.binary((x.log() * torch.tensor(10).exp()).int(),log_bits)), 1)
        return torch.cat((x.repeat(1,reps), torch.zeros(x.shape[0],residual).to(x.device, x.dtype)), 1)

    def binary_energy_refactored(self, x, lin_bits=19, sqrt_bits=17, log_bits=17):
        if hasattr(self._config.model, 'lin_bits'):
            lin_bits = self._config.model.lin_bits
        if hasattr(self._config.model, 'sqrt_bits'):
            sqrt_bits = self._config.model.sqrt_bits
        if hasattr(self._config.model, 'log_bits'):
            log_bits = self._config.model.log_bits

        total_bits_per_rep = lin_bits + sqrt_bits + log_bits
        
        reps = int(np.floor(self.cond_p_size / total_bits_per_rep))
        residual = self.cond_p_size - reps * (total_bits_per_rep)

        x_encoded = torch.cat((
            self.binary(x.int(), lin_bits), 
            self.binary((x.sqrt() * 200).int(), sqrt_bits), 
            self.binary((x.log() * 1e4).int(), log_bits)
        ), 1)
        
        return torch.cat((x_encoded.repeat(1, reps), torch.zeros(x.shape[0], residual).to(x.device, x.dtype)), 1)


class HierarchicalEncoderLayers(HierarchicalEncoder):
    def __init__(self, cfg):
        super().__init__(cfg)
        u_bits = self._config.model.u_bits
        num_edges = (2 ** u_bits) - 1
        num_u_vars = self._config.data.z
        
        # Register a correctly shaped dummy buffer so copy_() works later
        dummy_edges = torch.zeros((num_u_vars, num_edges), dtype=torch.float32)
        self.register_buffer('u_bin_edges', dummy_edges)
    
    def _create_hierarchy_network(self, level=0):

        if self._config.model.encoderblock == "EncoderLayers":
            return EncoderLayers(cfg=self._config, level=level)
        if self._config.model.encoderblock == "EncoderLayersZRP":
            return EncoderLayersZRP(cfg=self._config, level=level)
        else:
            raise ValueError(f"Unknown encoder block type: {self._config.model.encoderblock}")


    def forward(self, x, x0, u, beta_smoothing_fct=1):
        """
        Forward method for the layer-conditioned hierarchical encoder.
        Args: x: input voxels, shape (b, n_voxels), x0: incident energy, shape (b, 1), u: layer energies, shape (b, n_l)
        Returns: post_logits, post_samples
        """
        post_samples = []
        post_logits = []
        
        post_samples.append(torch.cat((self.energy_encoding_fct(x0), self.gray_encoding_fct(u)), dim=1))
        
        for lvl in range(self.n_latent_hierarchy_lvls-1):
            current_net = self._networks[lvl]
            current_input = x

            logits = torch.clamp(current_net(current_input, x0, u, post_samples), min=-88., max=88.)
            post_logits.append(logits)

            beta = torch.tensor(beta_smoothing_fct, dtype=torch.float, device=logits.device, requires_grad=False)
            samples = self.smoothing_dist_mod(logits, beta)

            post_samples.append(samples)
            
        return post_logits, post_samples


class HierarchicalEncoderHidden(HierarchicalEncoder):
    def __init__(self, cfg):
        super(HierarchicalEncoderHidden, self).__init__(cfg)
        self.smoothing_dist_mod = GumbelMod()
        self._config = cfg

        self.n_latent_hierarchy_lvls=self._config.rbm.partitions - self._config.model.hidden_layer
        # self.n_latent_hierarchy_lvls=self._config.rbm.partitions - self._config.rbm.hidden_layer

        self.n_latent_nodes=self._config.rbm.latent_nodes_per_p

        self._networks=nn.ModuleList([])
        
        for lvl in range(self.n_latent_hierarchy_lvls-1):
            network=self._create_hierarchy_network(level=lvl)
            self._networks.append(network)


####################Encoder blocks
##################################
    

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
        pos_enc = torch.cat(pres,2).transpose(1,2)
        res = pos_enc.sum([1,2])/(M-1)
        return res.unsqueeze(1)
    
    def trans_energy(self, x0, log_e_max=16.0, log_e_min=5.0, s_map = 1.0):
        # s_map = max(scaled voxel energy u_i) * (incidence energy / slope of total energy in shower) of the dataset
        return ((torch.log(x0) - log_e_min)/(log_e_max - log_e_min)) * s_map

class EncoderBlockPBH3Dv3(nn.Module):
    def __init__(self, cfg=None):
        super(EncoderBlockPBH3Dv3, self).__init__()
        self._config = cfg
        self.n_latent_nodes = self._config.rbm.latent_nodes_per_p
        self.z = self._config.data.z #45
        self.r = self._config.data.r #9
        self.phi = self._config.data.phi #16
        
        self.seq1 = nn.Sequential(
    
                   PeriodicConv3d(1, 32, (3,3,3), (2,1,1), 1),
                   nn.BatchNorm3d(32),
                   nn.PReLU(32, 0.02),
    
                   PeriodicConv3d(32, 64, (3,3,3), (2,1,1), 1),
                   nn.BatchNorm3d(64),
                   nn.PReLU(64, 0.02),

                   PeriodicConv3d(64, 128, (3,3,3), (1,2,1), 1),
                   nn.BatchNorm3d(128),
                   nn.PReLU(128, 0.02),
                )

        self.seq2 = nn.Sequential(
                           PeriodicConv3d(129, 256, (3,3,3), (2,2,1), 0),
                           nn.BatchNorm3d(256),
                           nn.PReLU(256, 0.02),

                           PeriodicConv3d(256, self.n_latent_nodes, (3,3,3), (1,2,2), 0),
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
        pos_enc = torch.cat(pres,2).transpose(1,2)
        res = pos_enc.sum([1,2])/(M-1)
        return res.unsqueeze(1)
    
    def trans_energy(self, x0, log_e_max=14.0, log_e_min=6.0, s_map = 1.0):
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

class EncoderBlockATLASNew(EncoderBlockPBH3Dv3Reg):
    def __init__(self, cfg=None):
        super(EncoderBlockATLASNew, self).__init__(cfg)
        self._config = cfg
        self.n_latent_nodes = self._config.rbm.latent_nodes_per_p
        self.z = self._config.data.z #5
        self.r = self._config.data.r #14
        self.phi = self._config.data.phi #24
        
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

                        PeriodicConv3d(256, self.n_latent_nodes, (2,3,3), (2,2,2), 0),
                        nn.PReLU(self.n_latent_nodes, 1.0),
                        nn.Flatten(),
                        )


class PeriodicConv3dPadding(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):   
        super().__init__()     
        # Parse padding into a tuple (pad_z, pad_phi, pad_r)
        if isinstance(padding, int):
            self.pad_z = self.pad_phi = self.pad_r = padding
        elif isinstance(padding, tuple) and len(padding) == 3:
            self.pad_z, self.pad_phi, self.pad_r = padding
        else:
            raise ValueError("padding must be an int or a 3-tuple (pad_z, pad_phi, pad_r)")

        # Internal Conv3d does absolutely no padding
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        # Z-axis Padding: Zero padding on both ends
        if self.pad_z > 0:
            # F.pad format for 3D is (left_R, right_R, left_Phi, right_Phi, left_Z, right_Z)
            x = F.pad(x, (0, 0, 0, 0, self.pad_z, self.pad_z), mode='constant', value=0)
            
        # Phi-axis (Angular) Padding: Circular padding
        if self.pad_phi > 0:
            x = F.pad(x, (0, 0, self.pad_phi, self.pad_phi, 0, 0), mode='circular')
            
        # R-axis (Radial) Padding: Outer zero pad + Origin crossing
        if self.pad_r > 0:
            # First, pad the outer edge of the cylinder (R_max) with zeros
            x = F.pad(x, (0, self.pad_r, 0, 0, 0, 0), mode='constant', value=0)
            
            # Second, handle the origin (R=0) crossing
            inner_r = x[..., :self.pad_r]
            
            # Shift the angular dimension by pi
            mid = inner_r.shape[-2] // 2
            shift = torch.cat((inner_r[..., mid:, :], inner_r[..., :mid, :]), dim=-2)
            
            # Reverse the radial order
            shift = torch.flip(shift, dims=[-1])
            
            # Prepend to the radial dimension
            x = torch.cat((shift, x), dim=-1)

        # Apply convolution
        x = self.conv(x)
        return x


class EncoderLayers(nn.Module):
    """
    Subencoder block for layer-conditioned AE.
    In addition to layer-wise conditioning, uses upgraded periodic cylindrical convolutions with custom padding,
    FiLM for previous post_samples, and vectorized operations
    """
    def __init__(self, cfg=None, level=0):
        super().__init__()
        self._config = cfg
        self.n_latent_nodes = self._config.rbm.latent_nodes_per_p
        self.z = self._config.data.z #5
        self.r = self._config.data.r #24
        self.phi = self._config.data.phi #14

        self.seq1_out_channels = 64
        self.level = level
    

        # len of previous post_samples: conditioning nodes plus previous hierarchy levels' latent nodes
        self.len_post_samples = self._config.model.cond_p_size + self.n_latent_nodes * self.level
        self.context_dim = self.len_post_samples + 3 + self.z # +3 for energy conditioning, +z for layer-wise conditioning

        self.film_mlp_seq1 = nn.Sequential(
            nn.Linear(self.context_dim, self.seq1_out_channels * 2),
            nn.SiLU(),
            nn.Linear(self.seq1_out_channels * 2, self.seq1_out_channels * 2) # Multiplied by 2 to output both gamma and beta
        )

        self.film_mlp_seq2 = nn.Sequential(
            nn.Linear(self.context_dim, self.n_latent_nodes * 2),
            nn.SiLU(),
            nn.Linear(2*self.n_latent_nodes, 2 * self.n_latent_nodes)
        )

        self.seq1 = nn.Sequential(
            # (5, 24, 14) -> (5, 12, 7)
            PeriodicConv3dPadding(1, 32, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.SiLU(),
            # (5, 12, 7) -> (5, 6, 4)
            PeriodicConv3dPadding(32, self.seq1_out_channels, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(self.seq1_out_channels),
            nn.SiLU(),
        )
        self.seq2 = nn.Sequential(
            # (5, 6, 4) -> (3, 3, 3)
            PeriodicConv3dPadding(self.seq1_out_channels, 128, (3, 4, 2), stride=(1, 1, 1), padding=(0, 0, 0)),
            nn.BatchNorm3d(128),
            nn.SiLU(),
            # (3, 3, 3) -> (1, 1, 1)
            PeriodicConv3dPadding(128, self.n_latent_nodes, (3, 3, 3), stride=(1, 1, 1), padding=(0, 0, 0)),
            nn.Flatten(),
        )

            
    def _get_film_params(self, mlp: nn.Module, context: torch.Tensor, is_spatial: bool) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generates gamma and beta parameters from the combined context vector.
        """
        film_params = mlp(context)
        gamma, beta = torch.chunk(film_params, 2, dim=1)
        
        if is_spatial:
            # Reshape for 5D tensor broadcasting: (Batch, Channels, Z, Phi, R)
            gamma = gamma.view(-1, gamma.shape[1], 1, 1, 1)
            beta = beta.view(-1, beta.shape[1], 1, 1, 1)
            
        return gamma, beta

    
    def forward(self, x: torch.Tensor, x0: torch.Tensor, u: torch.Tensor, post_samples: List[torch.Tensor]) -> torch.Tensor:
        """
        Forward method of subencoder, conditioned on x0 (incidence energy), u (layer-wise energy vector), 
        and post_samples (previous hierarchy levels' latent samples).
        """

        x = x.view(x.shape[0], 1, self.z, self.r, self.phi) 
        
        history = torch.cat(post_samples, dim=1)
        x0_transformed = self.trans_energy_multibasis(x0)
        
        
        global_context = torch.cat([history, x0_transformed, u], dim=1)
        assert global_context.shape[1] == self.context_dim, f"Expected context dimension {self.context_dim}, got {global_context.shape[1]} at level {self.level}"
        
        x = self.seq1(x)
        
        # Apply Spatial FiLM (Intermediate conditioning)
        gamma1, beta1 = self._get_film_params(self.film_mlp_seq1, global_context, is_spatial=True)
        x = (1 + gamma1) * x + beta1
        
        x = self.seq2(x)
        gamma2, beta2 = self._get_film_params(self.film_mlp_seq2, global_context, is_spatial=False)
        x = (1 + gamma2) * x + beta2
        
        return x

    def trans_energy_multibasis(self, x0, energy_min=900.0, energy_max=310000.0):
        """
        Encodes incidence energy into a (batch_size, 3) tensor with Linear, Sqrt, and Log bases.
        All components are min-max normalized to approx [0, 1] range based on input bounds.
        
        Args:
            x0 (torch.Tensor): Input energy in MeV.
            energy_min (float): Min expected energy (default 1 GeV = 1000 MeV).
            energy_max (float): Max expected energy (default 300 GeV = 300000 MeV).
        
        Returns:
            torch.Tensor: Shape (batch_size, 3)
                - Index 0: Normalized Linear
                - Index 1: Normalized Sqrt
                - Index 2: Normalized Log
        """
        # Ensure x0 is float for division/log
        x0 = x0.float()
        if x0.dim() > 1:
            x0 = x0.squeeze()
        
        # 1. Linear Scaling: (x - min) / (max - min)
        lin_norm = (x0 - energy_min) / (energy_max - energy_min)
        
        # 2. Square Root Scaling: (sqrt(x) - sqrt(min)) / (sqrt(max) - sqrt(min))
        # Pre-calculate bounds for efficiency
        sqrt_min = torch.sqrt(torch.tensor(energy_min))
        sqrt_max = torch.sqrt(torch.tensor(energy_max))
        sqrt_norm = (torch.sqrt(x0) - sqrt_min) / (sqrt_max - sqrt_min)
        
        # 3. Log Scaling: (log(x) - log(min)) / (log(max) - log(min))
        log_min = torch.log(torch.tensor(energy_min))
        log_max = torch.log(torch.tensor(energy_max))
        log_norm = (torch.log(x0) - log_min) / (log_max - log_min)
        
        # Stack along the last dimension to create (batch_size, 3)
        return torch.stack([lin_norm, sqrt_norm, log_norm], dim=-1)

class PeriodicConv3dPaddingZRP(nn.Module):
    """
    Corrected padding class strictly for (Z, R, Phi) tensor layouts.
    PyTorch F.pad pads from the last dimension backwards: (Phi, R, Z)
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):   
        super().__init__()     
        
        # Parse padding into a tuple (pad_z, pad_r, pad_phi)
        if isinstance(padding, int):
            self.pad_z = self.pad_r = self.pad_phi = padding
        elif isinstance(padding, tuple) and len(padding) == 3:
            self.pad_z, self.pad_r, self.pad_phi = padding
        else:
            raise ValueError("padding must be an int or a 3-tuple (pad_z, pad_r, pad_phi)")

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=dilation, groups=groups, bias=bias)

    def forward(self, x):
        #  Z-axis Padding: Zero padding on both ends
        # F.pad for 3D maps to: (left_Phi, right_Phi, left_R, right_R, left_Z, right_Z)
        if self.pad_z > 0:
            x = F.pad(x, (0, 0, 0, 0, self.pad_z, self.pad_z), mode='constant', value=0)
            
        # Phi-axis (Angular) Padding: Circular padding
        if self.pad_phi > 0:
            # Applies to the LAST dimension (Phi)
            x = F.pad(x, (self.pad_phi, self.pad_phi, 0, 0, 0, 0), mode='circular')
            
        # R-axis (Radial) Padding: Outer zero pad + Origin crossing
        if self.pad_r > 0:
            # First, pad the outer edge of the cylinder (right side of R dimension) with zeros
            x = F.pad(x, (0, 0, 0, self.pad_r, 0, 0), mode='constant', value=0)
            
            # Second, handle the origin (R=0) crossing
            # Slice the inner-most radii along the R dimension (2nd to last dim)
            inner_r = x[..., :self.pad_r, :]
            
            # Shift the angular dimension (last dim) by pi
            mid = inner_r.shape[-1] // 2
            shift = torch.cat((inner_r[..., mid:], inner_r[..., :mid]), dim=-1)
            
            # Reverse the radial order (2nd to last dim)
            shift = torch.flip(shift, dims=[-2])
            
            # Prepend to the radial dimension (2nd to last dim)
            x = torch.cat((shift, x), dim=-2)

        # Apply convolution
        x = self.conv(x)
        return x

class EncoderLayersZRP(nn.Module):
    """
    Fixed Encoder block that correctly handles (Z, R, Phi) layout 
    using the corrected PeriodicConv3dPaddingZRP class.
    """
    def __init__(self, cfg=None, level=0):
        super().__init__()
        self._config = cfg
        self.n_latent_nodes = self._config.rbm.latent_nodes_per_p
        self.z = self._config.data.z 
        self.r = self._config.data.r 
        self.phi = self._config.data.phi 

        self.seq1_out_channels = 64
        self.level = level
    
        self.len_post_samples = self._config.model.cond_p_size + self.n_latent_nodes * self.level
        self.context_dim = self.len_post_samples + 3 + self.z 

        self.film_mlp_seq1 = nn.Sequential(
            nn.Linear(self.context_dim, self.seq1_out_channels * 2),
            nn.SiLU(),
            nn.Linear(self.seq1_out_channels * 2, self.seq1_out_channels * 2) 
        )

        self.film_mlp_seq2 = nn.Sequential(
            nn.Linear(self.context_dim, self.n_latent_nodes * 2),
            nn.SiLU(),
            nn.Linear(2*self.n_latent_nodes, 2 * self.n_latent_nodes)
        )

        self.seq1 = nn.Sequential(
            PeriodicConv3dPaddingZRP(1, 32, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(32),
            nn.SiLU(),
            PeriodicConv3dPaddingZRP(32, self.seq1_out_channels, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(self.seq1_out_channels),
            nn.SiLU(),
        )
        self.seq2 = nn.Sequential(
            PeriodicConv3dPaddingZRP(self.seq1_out_channels, 128, (3, 4, 2), stride=(1, 1, 1), padding=(0, 0, 0)),
            nn.BatchNorm3d(128),
            nn.SiLU(),
            PeriodicConv3dPaddingZRP(128, self.n_latent_nodes, (3, 3, 3), stride=(1, 1, 1), padding=(0, 0, 0)),
            nn.Flatten(),
        )
        nn.init.zeros_(self.film_mlp_seq1[-1].weight)
        nn.init.zeros_(self.film_mlp_seq1[-1].bias)
        nn.init.zeros_(self.film_mlp_seq2[-1].weight)
        nn.init.zeros_(self.film_mlp_seq2[-1].bias)
            
    def _get_film_params(self, mlp: nn.Module, context: torch.Tensor, is_spatial: bool):
        film_params = mlp(context)
        gamma, beta = torch.chunk(film_params, 2, dim=1)
        
        if is_spatial:
            gamma = gamma.view(-1, gamma.shape[1], 1, 1, 1)
            beta = beta.view(-1, beta.shape[1], 1, 1, 1)
            
        return gamma, beta

    def forward(self, x: torch.Tensor, x0: torch.Tensor, u: torch.Tensor, post_samples):
        # Explicitly enforce the layout
        x = x.view(x.shape[0], 1, self.z, self.r, self.phi) 
        
        history = torch.cat(post_samples, dim=1)
        x0_transformed = self.trans_energy_multibasis(x0)
        
        global_context = torch.cat([history, x0_transformed, u], dim=1)
        
        x = self.seq1(x)
        gamma1, beta1 = self._get_film_params(self.film_mlp_seq1, global_context, is_spatial=True)
        x = (1 + gamma1) * x + beta1
        
        x = self.seq2(x)
        gamma2, beta2 = self._get_film_params(self.film_mlp_seq2, global_context, is_spatial=False)
        x = (1 + gamma2) * x + beta2
        
        return x

    def trans_energy_multibasis(self, x0, energy_min=900.0, energy_max=310000.0):
        x0 = x0.float()
        if x0.dim() > 1:
            x0 = x0.squeeze()
        
        lin_norm = (x0 - energy_min) / (energy_max - energy_min)
        sqrt_min = torch.sqrt(torch.tensor(energy_min))
        sqrt_max = torch.sqrt(torch.tensor(energy_max))
        sqrt_norm = (torch.sqrt(x0) - sqrt_min) / (sqrt_max - sqrt_min)
        log_min = torch.log(torch.tensor(energy_min))
        log_max = torch.log(torch.tensor(energy_max))
        log_norm = (torch.log(x0) - log_min) / (log_max - log_min)
        
        return torch.stack([lin_norm, sqrt_norm, log_norm], dim=-1)




