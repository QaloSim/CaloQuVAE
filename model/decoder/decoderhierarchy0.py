# This model is the decoder for the hierarchical structure for CaloChallenge.
# It is designed to work with the hierarchical encoder and is tailored for CaloChallenge.
# Authors: The CaloQVAE (2025)

import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange

#########Decoder Hierarchy Class#######################
#######################################################

class DecoderHierarchy0(nn.Module):
    def __init__(self, cfg):
        super(DecoderHierarchy0, self).__init__()
        self._config = cfg
        self._create_hierarchy_network()
        self._create_skipcon_decoders()

    def _create_hierarchy_network(self):
        self.latent_nodes = self._config.rbm.latent_nodes_per_p * self._config.rbm.partitions
        self.hierarchical_lvls = self._config.rbm.partitions

        inp_layers = self._config.model.decoder_input
        out_layers = self._config.model.decoder_output

        self.moduleLayers = nn.ModuleList([])
        for i in range(len(inp_layers)):
            # self.moduleLayers.append(Decoder(self._config, inp_layers[i], out_layers[i]))   
            self.moduleLayers.append(DecoderLinAtt(self._config, inp_layers[i], out_layers[i]))   

    def _create_skipcon_decoders(self):
        latent_inp = 2 * self._config.rbm.latent_nodes_per_p
        self.subdecs = nn.ModuleList([])
        for i in range(len(self._config.model.decoder_output)-1):
            recon_out = self.latent_nodes + self._config.model.decoder_output[i]
            self.subdecs.append(nn.Conv3d(latent_inp, recon_out, kernel_size=1, stride=1, padding=0))
    
    def forward(self, x, x0):
        x_lat = x
        self.x1, self.x2 = torch.tensor([]).to(x.device), torch.tensor([]).to(x.device) # store hits and activation tensors
        for lvl in range(len(self.moduleLayers)):
            cur_net = self.moduleLayers[lvl]
            output_hits, output_activations = cur_net(x, x0)
            z = output_hits * output_activations
            if lvl == len(self.moduleLayers) - 1:
                self.x1 = output_hits
                self.x2 = output_activations
            else:
                partition_ind_start = (len(self.moduleLayers) - 1 - lvl) * self._config.rbm.latent_nodes_per_p
                partition_ind_end = (len(self.moduleLayers) - lvl) * self._config.rbm.latent_nodes_per_p
                enc_z = torch.cat((x[:,0:self._config.rbm.latent_nodes_per_p], x[:,partition_ind_start:partition_ind_end]), dim=1)
                enc_z = torch.unflatten(enc_z, 1, (2 * self._config.rbm.latent_nodes_per_p, 1, 1, 1))
                enc_z = self.subdecs[lvl](enc_z).view(enc_z.size(0), -1)
                xz = torch.cat((x_lat, z), dim=1)
                x = enc_z + xz
        return self.x1, self.x2

class DecoderHierarchyv3(DecoderHierarchy0):
        def __init__(self, cfg):
            super(DecoderHierarchyv3, self).__init__(cfg)

        def _create_skipcon_decoders(self):
            """
            Create skip connections for the hierarchical decoder.
            The ith skip connection feeds the latent nodes of the ith RBM partition to the ith subdecoder,
            as well as the conditioned incident energy stored in partition 0 of the RBM.
            """
            self.skip_connections = nn.ModuleList()
            input_size = 2*self._config.rbm.latent_nodes_per_p
            output_size = self.latent_nodes  # Each skip connection outputs the input size of the subdecoders minus the shower size
            for i in range(self.hierarchical_lvls-1):
                skip_connection = nn.Sequential(
                    nn.Conv3d(
                        in_channels=input_size,
                        out_channels=input_size + self._config.rbm.latent_nodes_per_p,
                        kernel_size=(1, 1, 1),
                        stride=(1, 1, 1)),
                    nn.Conv3d(
                        in_channels=input_size + self._config.rbm.latent_nodes_per_p,
                        out_channels=output_size,
                        kernel_size=(1, 1, 1),
                        stride=(1, 1, 1))
                )
                self.skip_connections.append(skip_connection)

        def forward(self, x, x0):
            x_lat = x
            output_hits, output_activations = None, None

            for lvl in range(self.hierarchical_lvls):
                cur_decoder = self.moduleLayers[lvl]
                output_hits, output_activations = cur_decoder(x, x0)
                if lvl < self.hierarchical_lvls - 1: # If not the last level, prepare input for the next level
                    outputs = output_hits * output_activations
                    z = outputs
                    # Concatenate the output of the current decoder with the latent nodes of the next RBM partition
                    partition_idx_start = (self.hierarchical_lvls - lvl - 1) * self._config.rbm.latent_nodes_per_p #start index for the current RBM partition
                    partition_idx_end = partition_idx_start + self._config.rbm.latent_nodes_per_p  # end index for the current RBM partition
                    enc_z = torch.cat((x_lat[:, 0:self._config.rbm.latent_nodes_per_p], x_lat[:, partition_idx_start:partition_idx_end]), dim=1)  # concatenate the incident energy and the latent nodes of the current RBM partition
                    enc_z = torch.unflatten(enc_z, 1, (self._config.rbm.latent_nodes_per_p*2, 1, 1, 1))
                    # Apply skip connection
                    enc_z = self.skip_connections[lvl](enc_z).view(enc_z.size(0), -1)  # Flatten the output of the skip connection

                    x = torch.cat((enc_z, z), dim=1)  # Concatenate the output of the skip connection with the output of the current decoder
            # If the last level, just return the output            
            return output_hits, output_activations



###############Decoder Class#######################
###################################################

class Decoder(nn.Module):
    def __init__(self, cfg, input_size, output_size):
        super(Decoder, self).__init__()
        self._config = cfg

        self.n_latent_hierarchy_lvls=self._config.rbm.partitions

        self.n_latent_nodes=self._config.rbm.latent_nodes_per_p * self._config.rbm.partitions

        self.z = self._config.data.z
        self.r = self._config.data.r
        self.phi = self._config.data.phi

        output_size_z = int( output_size / (self.r * self.phi))

        self._layers =  nn.Sequential(
                   nn.Unflatten(1, (input_size, 1, 1, 1)),

                   PeriodicConvTranspose3d(input_size, 512, (3,3,2), (2,1,1), 0),
                   nn.BatchNorm3d(512),
                   nn.PReLU(512, 0.02),
                   

                   PeriodicConvTranspose3d(512, 128, (5,4,2), (2,1,1), 0),
                   nn.BatchNorm3d(128),
                   nn.PReLU(128, 0.02),
                                   )
        
        self._layers2 = nn.Sequential(
                   PeriodicConvTranspose3d(129, 64, (3,3,2), (2,1,1), 1),
                   nn.BatchNorm3d(64),
                   nn.PReLU(64, 0.02),

                   PeriodicConvTranspose3d(64, 32, (5,3,2), (2,1,1), 1),
                   nn.BatchNorm3d(32),
                   nn.PReLU(32, 1.0),

                   PeriodicConvTranspose3d(32, 1, (5,3,3), (1,1,1), 0),
                   PeriodicConv3d(1, 1, (self.z - output_size_z + 1, 1, 1), (1,1,1), 0),
                   nn.PReLU(1, 1.0)
                                   )
        
        self._layers3 = nn.Sequential(
                   PeriodicConvTranspose3d(129, 64, (3,3,2), (2,1,1), 1),
                   nn.BatchNorm3d(64),
                   nn.PReLU(64, 0.02),

                   PeriodicConvTranspose3d(64, 32, (5,3,2), (2,1,1), 1),
                   nn.BatchNorm3d(32),
                   nn.PReLU(32, 0.02),

                   PeriodicConvTranspose3d(32, 1, (5,3,3), (1,1,1), 0),
                   PeriodicConv3d(1, 1, (self.z - output_size_z + 1, 1, 1), (1,1,1), 0),
                   nn.PReLU(1, 0.02),
                                   )
        
    def forward(self, x, x0):
                
        x = self._layers(x)
        x0 = self.trans_energy(x0)
        xx0 = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat(1,1,torch.tensor(x.shape[-3:-2]).item(),torch.tensor(x.shape[-2:-1]).item(), torch.tensor(x.shape[-1:]).item())), 1)
        x1 = self._layers2(xx0) #hits
        x2 = self._layers3(xx0)
        return x1.reshape(x1.shape[0],-1), x2.reshape(x1.shape[0],-1)
    
    def trans_energy(self, x0, log_e_max=14.0, log_e_min=6.0, s_map = 1.0):
        return ((torch.log(x0) - log_e_min)/(log_e_max - log_e_min)) * s_map
    

class DecoderLinAtt(nn.Module):
    def __init__(self, cfg, input_size, output_size):
        super(DecoderLinAtt, self).__init__()

        self._config = cfg

        self.n_latent_hierarchy_lvls=self._config.rbm.partitions

        self.n_latent_nodes=self._config.rbm.latent_nodes_per_p * self._config.rbm.partitions

        self.z = self._config.data.z
        self.r = self._config.data.r
        self.phi = self._config.data.phi

        output_size_z = int( output_size / (self.r * self.phi))

        self._layers =  nn.Sequential(
                   nn.Unflatten(1, (input_size, 1, 1, 1)),

                   PeriodicConvTranspose3d(input_size, 512, (3,3,2), (2,1,1), 0),
                   nn.BatchNorm3d(512),
                   nn.PReLU(512, 0.02),
                   

                   PeriodicConvTranspose3d(512, 128, (5,4,2), (2,1,1), 0),
                   nn.BatchNorm3d(128),
                   nn.PReLU(128, 0.02),
                                   )
        
        self._layers2 = nn.Sequential(
                   PeriodicConvTranspose3d(129, 64, (3,3,2), (2,1,1), 1),
                   nn.BatchNorm3d(64),
                   nn.PReLU(64, 0.02),

                   PeriodicConvTranspose3d(64, 32, (5,3,2), (2,1,1), 1),
                   nn.BatchNorm3d(32),
                   nn.PReLU(32, 1.0),

                   PeriodicConvTranspose3d(32, 1, (5,3,3), (1,1,1), 0),
                   PeriodicConv3d(1, 1, (self.z - output_size_z + 1, 1, 1), (1,1,1), 0),
                   nn.PReLU(1, 1.0)
                                   )
        
        self._layers3 = nn.Sequential(
                   PeriodicConvTranspose3d(129, 64, (3,3,2), (2,1,1), 1),
                   nn.GroupNorm(1,64),
                   nn.SiLU(64),
                   LinearAttention(64, cylindrical = False),

                   PeriodicConvTranspose3d(64, 32, (5,3,2), (2,1,1), 1),
                   nn.GroupNorm(1,32),
                   nn.SiLU(32),
                   LinearAttention(32, cylindrical = False),

                   PeriodicConvTranspose3d(32, 1, (5,3,3), (1,1,1), 0),
                   PeriodicConv3d(1, 1, (self.z - output_size_z + 1, 1, 1), (1,1,1), 0),
                   nn.SiLU(1),
                                   )
        
    def forward(self, x, x0):
                
        x = self._layers(x)
        x0 = self.trans_energy(x0)
        xx0 = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat(1,1,torch.tensor(x.shape[-3:-2]).item(),torch.tensor(x.shape[-2:-1]).item(), torch.tensor(x.shape[-1:]).item())), 1)
        x1 = self._layers2(xx0) #hits
        x2 = self._layers3(xx0)
        return x1.reshape(x1.shape[0],-1), x2.reshape(x1.shape[0],-1)
    
    def trans_energy(self, x0, log_e_max=14.0, log_e_min=6.0, s_map = 1.0):
        return ((torch.log(x0) - log_e_min)/(log_e_max - log_e_min)) * s_map
    
##############Auxiliary Classes########################
#######################################################

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
    
class PeriodicConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(PeriodicConv3d, self).__init__()
        self.padding = padding
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=0, dilation=dilation, groups=groups, bias=bias)
    def forward(self, x):
        # Pad input tensor with periodic boundary and circle-center conditions
        if self.padding == 1:
            mid = x.shape[-1] // 2
            shift = torch.cat((x[..., [-1], mid:], x[..., [-1], :mid]), -1)
            x = torch.cat((x, shift), dim=-2)
        x = F.pad(x, (self.padding, self.padding, 0, 0, 0, 0), mode='circular')
        # Apply convolution
        x = self.conv(x)
        return x
    
class LinearAttention(nn.Module):
    def __init__(self, dim, heads=1, dim_head=32, cylindrical = False):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        if(cylindrical):
            # self.to_qkv = CylindricalConv(dim, hidden_dim * 3, kernel_size = 1, bias=False)
            # self.to_out = nn.Sequential(CylindricalConv(hidden_dim, dim, kernel_size = 1), nn.GroupNorm(1,dim))
            self.to_qkv = PeriodicConv3d(dim, hidden_dim * 3, kernel_size = 1, bias=False)
            self.to_out = nn.Sequential(PeriodicConv3d(hidden_dim, dim, kernel_size = 1), nn.GroupNorm(1,dim))
        else: 
            self.to_qkv = nn.Conv3d(dim, hidden_dim * 3, kernel_size = 1, bias=False)
            self.to_out = nn.Sequential(nn.Conv3d(hidden_dim, dim, kernel_size = 1), nn.GroupNorm(1,dim))

    def forward(self, x):
        b, c, l, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, "b (h c) x y z -> b h c (x y z)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y z) -> b (h c) x y z", h=self.heads, x=l, y=h, z = w)
        return self.to_out(out)


