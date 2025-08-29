import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LeakyReLU, ReLU
from model.gumbel import GumbelMod
from einops import rearrange
from model.decoder.decoderhierarchybase import PeriodicConvTranspose3d, LinearAttention, CropLayer

class DecoderFullGeo(nn.Module):
    def __init__(self, cfg):
        """
        Hierarchical decoder that uses 4 subdecoders to sample from the 4 RBM partitions. 
        The ith subdecoder takes as input the output of the (i-1)th subdecoder,
        and the output of the ith skip connection corresponding to the latent nodes in the ith RBM partition.
        Shower is a cuboid that gets upsampled to (z, r, phi) by the first subdecoder.
        The first subdecoder takes as input the latent nodes of the first RBM partition.
        """
        super(DecoderFullGeo, self).__init__()
        self._config = cfg
        if hasattr(self._config.model, 'hidden_layer') and self._config.model.hidden_layer:
            print("Using hidden layer in decoder")
            self.n_latent_nodes = self._config.rbm.latent_nodes_per_p * (self._config.rbm.partitions - 1)
            self.p_size = int(self._config.rbm.latent_nodes_per_p * 3/4)
        else:
            self.n_latent_nodes = self._config.rbm.latent_nodes_per_p * self._config.rbm.partitions
            self.p_size = self._config.rbm.latent_nodes_per_p

        self.n_latent_hierarchy_lvls = self._config.rbm.partitions

        self.z = self._config.data.z
        self.r = self._config.data.r
        self.phi = self._config.data.phi

        self.input_shapes = [
            (1, 1, 1),
            (self.z, self.phi, self.r),
            (self.z, self.phi, self.r),
            (self.z, self.phi, self.r),
        ]
        

        self._create_hierarchy_networks()
        self._create_skip_connections()




    def _create_hierarchy_networks(self):
        self.subdecoders = nn.ModuleList()
        for i in range(self.n_latent_hierarchy_lvls):
            if i == 0:
                subdecoder = FirstSubDecoder(self._config)
            else:
                subdecoder = SubDecoder(self._config, last_subdecoder=(i == self.n_latent_hierarchy_lvls - 1))
            self.subdecoders.append(subdecoder)

    def trans_energy(self, x0, log_e_max=14.0, log_e_min=6.0, s_map = 1.0):
        return ((torch.log(x0) - log_e_min)/(log_e_max - log_e_min)) * s_map
    
    def _create_skip_connections(self):
        self.skip_connections = nn.ModuleList()
        for i in range(self.n_latent_hierarchy_lvls-1):
            skip_connection = nn.Sequential(
                nn.ConvTranspose3d(self.p_size * (2+i), 64, (3, 5, 7), (1, 1, 1), padding=0),
                nn.BatchNorm3d(64),
                nn.PReLU(64, 0.02),
                # upscales to (64, 3, 5, 7)
                nn.ConvTranspose3d(64, 32, (3, 5, 7), (1, 1, 2), padding=0),
                nn.BatchNorm3d(32),
                nn.PReLU(32, 0.02),
                # upscales to (32, 5, 8, 12)
                nn.ConvTranspose3d(32, 1, (3, 6, 6), (1, 1, 1), padding=0),
            ) #outputs (1, 7, 14, 24)
            self.skip_connections.append(skip_connection)
        

    
    def forward(self, x, x0):
        x_lat = x
        x0 = self.trans_energy(x0)
        x0_reshaped = x0.view(x0.shape[0], 1, 1, 1, 1)
        x = x.view(x.shape[0], self.n_latent_nodes, 1, 1, 1)  # Reshape x to match the input shape of the first subdecoder
        prev_output = None
        partition_idx_start = (self.n_latent_hierarchy_lvls-1) * self.p_size  # start index for the z3 RBM partition
        partition_idx_end = partition_idx_start + self.p_size # end index for the z3 RBM partition


        for lvl in range(self.n_latent_hierarchy_lvls):
            curr_subdecoder = self.subdecoders[lvl]
            x0_broadcasted = x0_reshaped.expand(x.shape[0], 1, *self.input_shapes[lvl])

            decoder_input = torch.cat((x, x0_broadcasted), dim=1)  # Concatenate along the channel dimension
            # print(decoder_input.shape)
            
            if lvl < self.n_latent_hierarchy_lvls - 1:
                output = curr_subdecoder(decoder_input, x0)
                if prev_output is not None:
                    output += prev_output  # add/refine the previous subdecoder output
                prev_output = output
                enc_z = torch.cat((x_lat[:, 0:self.p_size], x_lat[:, partition_idx_start:partition_idx_end]), dim=1)  # concatenate the incident energy and the latent nodes of the current RBM partition
                enc_z = torch.unflatten(enc_z, 1, (self.p_size*(2+lvl), 1, 1, 1))
                # Apply skip connection
                enc_z = self.skip_connections[lvl](enc_z)
                partition_idx_start -= self.p_size  # start index for the current RBM partition, moves one partition back every level
                # print(output.shape, enc_z.shape)
                x = torch.cat((output, enc_z), dim=1)  # concatenate the output of the current subdecoder and the skip connection output

            else:  # last level
                output_hits, output_activations = curr_subdecoder(decoder_input, x0)
                output_hits = output_hits.reshape(output_hits.shape[0], self.z*self.phi*self.r)
                output_activations = output_activations.reshape(output_activations.shape[0], self.z*self.phi*self.r)
                return output_hits, output_activations



            
class FirstSubDecoder(nn.Module):
    def __init__(self, cfg):
        super(FirstSubDecoder, self).__init__()
        self._config = cfg
        if hasattr(self._config.model, 'hidden_layer') and self._config.model.hidden_layer:
            self.n_latent_nodes = self._config.rbm.latent_nodes_per_p * (self._config.rbm.partitions - 1)
        else:
            self.n_latent_nodes = self._config.rbm.latent_nodes_per_p * self._config.rbm.partitions
        print(self.n_latent_nodes, "first subdecoder")
        self.shower_size = (self._config.data.z, self._config.data.phi, self._config.data.r)

        self._layers1 = nn.Sequential(
            PeriodicConvTranspose3d(self.n_latent_nodes+1, 512, (3, 3, 3), stride=(1, 1, 1), padding=0),
            nn.BatchNorm3d(512),
            nn.PReLU(512, 0.02),
            # upscales to (512, 3, 3, 3)
            PeriodicConvTranspose3d(512, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=0),
            nn.BatchNorm3d(256),
            nn.PReLU(256, 0.02),
            # upscales to (256, 5, 5, 5)
            PeriodicConvTranspose3d(256, 128, (3, 3, 3), stride=(1, 1, 1), padding=0),
            nn.BatchNorm3d(128),
            nn.PReLU(128, 0.02),
            # upscales to (128, 7, 7, 7)
        )
        self._layers2 = nn.Sequential(
            # layer for activations
            nn.ConvTranspose3d(129, 64, (3, 3, 5), stride=(1, 2, 3), padding=(1, 0, 0)),
            nn.GroupNorm(1, 64),
            nn.SiLU(),
            LinearAttention(64, cylindrical=False),
            # upscales to  (64, 7, 15, 23)
            nn.ConvTranspose3d(64, 32, (3, 2, 2), stride=(1, 1, 1), padding=(1, 0, 0)),
            nn.GroupNorm(1, 32),
            nn.SiLU(),
            LinearAttention(32, cylindrical=False),
            # upscales to (32, 7, 16, 24)
            CropLayer(),
        )
        self._layers2_hits = nn.Sequential(
            # layer for hits
            nn.ConvTranspose3d(129, 64, (3, 3, 5), stride=(1, 2, 3), padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.PReLU(64, 0.02),
            # upscales to  (64, 7, 15, 23)
            nn.ConvTranspose3d(64, 32, (3, 2, 2), stride=(1, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(32),
            nn.PReLU(32, 0.02),
            # upscales to (32, 7, 16, 24)
            nn.PReLU(1, 1.0),
            CropLayer(),
        )


    def forward(self,x, x0):
        x = self._layers1(x)
        d1, d2, d3 = x.shape[-3], x.shape[-2], x.shape[-1]
        xx0 = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat(1, 1, d1, d2, d3)), dim=1)  # Concatenate the incident energy to the output of the first layer
        x1 = self._layers2(xx0).reshape(xx0.shape[0], 32, self.shower_size[0], self.shower_size[1], self.shower_size[2])
        x2 = self._layers2_hits(xx0).reshape(xx0.shape[0], 32, self.shower_size[0], self.shower_size[1], self.shower_size[2])
        return x1 * x2

        

class SubDecoder(nn.Module):
    def __init__(self, cfg, first_subdecoder=False, last_subdecoder=False):
        super(SubDecoder, self).__init__()
        self._config = cfg
        self.n_latent_nodes = self._config.rbm.latent_nodes_per_p * self._config.rbm.partitions
        self.shower_size = (self._config.data.z, self._config.data.r, self._config.data.phi)

        self.first_subdecoder = first_subdecoder
        self.last_subdecoder = last_subdecoder

        if self.last_subdecoder:
            self._subdecoder_layers = nn.Sequential(
                # maintains same shape throughout
                nn.ConvTranspose3d(34, 32, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.GroupNorm(1, 32),
                nn.SiLU(),
                LinearAttention(32, cylindrical=False),

                nn.ConvTranspose3d(32, 32, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.GroupNorm(1, 32),
                nn.SiLU(),
                LinearAttention(32, cylindrical=False),

                nn.ConvTranspose3d(32, 32, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.GroupNorm(1, 32),
                nn.SiLU(),
                LinearAttention(32, cylindrical=False),

                nn.ConvTranspose3d(32, 1, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.PReLU(1, 1.0),
            )
            self._last_subdecoder_hits = nn.Sequential(
                #same as activations, but with different activation functions
                nn.ConvTranspose3d(34, 32, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.BatchNorm3d(32),
                nn.PReLU(32, 0.02),
                LinearAttention(32, cylindrical=False),

                nn.ConvTranspose3d(32, 32, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.BatchNorm3d(32),
                nn.PReLU(32, 0.02),
                LinearAttention(32, cylindrical=False),

                nn.ConvTranspose3d(32, 32, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.BatchNorm3d(32),
                nn.PReLU(32, 0.02),
                LinearAttention(32, cylindrical=False),

                nn.ConvTranspose3d(32, 1, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.PReLU(1, 1.0),
            )
        else:
            self._subdecoder_layers = nn.Sequential(
                # maintains same shape throughout
                nn.ConvTranspose3d(34, 32, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.GroupNorm(1, 32),
                nn.SiLU(),
                LinearAttention(32, cylindrical=False),

                nn.ConvTranspose3d(32, 32, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.GroupNorm(1, 32),
                nn.SiLU(),
                LinearAttention(32, cylindrical=False),

                nn.ConvTranspose3d(32, 32, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1)),
                nn.GroupNorm(1, 32),
                nn.SiLU(),
                LinearAttention(32, cylindrical=False),
            )
    def forward(self, x, x0):
        if self.first_subdecoder:
            x = self._first_subdecoder_layers(x)
        elif self.last_subdecoder:
            activations = self._subdecoder_layers(x)
            hits = self._last_subdecoder_hits(x)
            return hits, activations
        else:
            x = self._subdecoder_layers(x)
        return x