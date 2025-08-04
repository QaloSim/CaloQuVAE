import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LeakyReLU, ReLU
from model.gumbel import GumbelMod
from einops import rearrange
from model.decoder.decoderhierarchybase import PeriodicConvTranspose3d, LinearAttention

class DecoderHierarchyGeometry(nn.Module):
    def __init__(self, cfg):
        """
        Hierarchical decoder that uses 4 subdecoders to sample from the 4 RBM partitions. 
        The ith subdecoder takes as input the output of the (i-1)th subdecoder,
        and the output of the ith skip connection corresponding to the latent nodes in the ith RBM partition.
        Shower is a cuboid that gets upsampled to (z, r, phi) dimensions gradually by each subdecoder.
        The first subdecoder takes as input the latent nodes of the first RBM partition.
        """

        super(DecoderHierarchyGeometry, self).__init__()
        self._config = cfg

        self.n_latent_hierarchy_lvls = self._config.rbm.partitions
        self.n_latent_nodes = self._config.rbm.latent_nodes_per_p * self._config.rbm.partitions

        self.z = self._config.data.z
        self.r = self._config.data.r
        self.phi = self._config.data.phi

        self.input_shapes = [
            (1, 1, 1),  # Input shape for the first subdecoder
            (7, 7, 7),  # Input shape for the second subdecoder
            (7, 10, 12),  # Input shape for the third subdecoder
            (7, 12, 18)   # Input shape for the fourth subdecoder
        ]
        
        self._create_hierarchy_networks()
        self._create_skip_connections()
        self._create_residual_projections()

    def _create_hierarchy_networks(self):
        self.subdecoders = nn.ModuleList()
        self.subdecoders.append(nn.Sequential(
            nn.ConvTranspose3d(self.n_latent_nodes+1, 1024, (3, 3, 3), stride=(1, 1, 1), padding=0),
            nn.BatchNorm3d(1024),
            nn.PReLU(1024, 0.02),

            nn.ConvTranspose3d(1024, 512, (3, 3, 3), stride=(1, 1, 1), padding=0),
            nn.BatchNorm3d(512),
            nn.PReLU(512, 0.02),

            nn.ConvTranspose3d(512, 256, (3, 3, 3), stride=(1, 1, 1), padding=0),
            nn.BatchNorm3d(256),
            nn.PReLU(256, 0.02),
        )) # First subdecoder outputs 7x7x7 cuboid
        self.subdecoders.append(nn.Sequential(
            nn.ConvTranspose3d(258, 128, (1, 2, 2), stride =(1, 1, 1), padding=0),
            nn.BatchNorm3d(128),
            nn.SiLU(),
            LinearAttention(128, cylindrical=False),

            nn.ConvTranspose3d(128, 64, (3, 3, 3), stride=(1, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.SiLU(),
            LinearAttention(64, cylindrical=False),

            nn.ConvTranspose3d(64, 32, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 0)),
            nn.BatchNorm3d(32),
            nn.SiLU(),
            LinearAttention(32, cylindrical=False),
        )) # Second subdecoder outputs 7x10x12 cuboid
        self.subdecoders.append(nn.Sequential(
            nn.ConvTranspose3d(34, 32, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 0)),
            nn.BatchNorm3d(32),
            nn.SiLU(),
            LinearAttention(32, cylindrical=False),

            nn.ConvTranspose3d(32, 32, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 0)),
            nn.BatchNorm3d(32),
            nn.SiLU(),
            LinearAttention(32, cylindrical=False),

            nn.ConvTranspose3d(32, 16, (3, 3, 3), stride=(1, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(16),
            nn.SiLU(),
            LinearAttention(16, cylindrical=False),
        )) # Third subdecoder outputs 7x12x18 cuboid
        self.subdecoders.append(LastSubdecoder()) # Fourth subdecoder outputs 7x14x24 full shower
    
    def _create_skip_connections(self):
        """
        Create skip connections for the subdecoders.
        The skip connections transforms partition z0 and z_(3-i) to be passed to the ith subdecoder.
        """
        self.skip_connections = nn.ModuleList()
        self.skip_connections.append(nn.Sequential(
            nn.ConvTranspose3d(2*self._config.rbm.latent_nodes_per_p, 32, (3, 3, 3), stride=(1, 1, 1), padding=0),
            nn.ConvTranspose3d(32, 1, (5, 5, 5), stride=(1, 1, 1), padding=0),
        )) # Outputs 7x7x7 cuboid
        self.skip_connections.append(nn.Sequential(
            nn.ConvTranspose3d(2*self._config.rbm.latent_nodes_per_p, 32, (3, 5, 5), stride=(1, 1, 1), padding=0),
            nn.ConvTranspose3d(32, 1, (5, 6, 4), stride=(1, 1, 2), padding=0),
        )) # Outputs 7x10x12 cuboid
        self.skip_connections.append(nn.Sequential(
            nn.ConvTranspose3d(2*self._config.rbm.latent_nodes_per_p, 32, (3, 5, 7), stride=(1, 1, 1), padding=0),
            nn.ConvTranspose3d(32, 1, (5, 4, 6), stride=(1, 2, 2), padding=0),
        )) # Outputs 7x12x18 cuboid
    
    def _create_residual_projections(self):
        self.residual_projections = nn.ModuleList()
        self.residual_projections.append(nn.Conv3d(256, 32, kernel_size=1)) # Projects subdec 0 (256 channels) to subdec 1 (32 channels)
        self.residual_projections.append(nn.Conv3d(32, 16, kernel_size=1)) # Projects subdec 1 (32 channels) to subdec 2 (16 channels)
        self.residual_projections.append(nn.Conv3d(16, 8, kernel_size=1)) # Projects subdec 2 (16 channels) to subdec 3 (8 channels before heads)

    def trans_energy(self, x0, log_e_max=14.0, log_e_min=6.0, s_map = 1.0):
        return ((torch.log(x0) - log_e_min)/(log_e_max - log_e_min)) * s_map

    
    def forward(self, x, x0):
        x_lat = x
        x = x.view(x.shape[0], self.n_latent_nodes, 1, 1, 1)  # Reshape x to match the input shape of the first subdecoder
        prev_outputs = None

        for lvl in range(self.n_latent_hierarchy_lvls):
            x0_trans = self.trans_energy(x0) #transform incidence energy
            x0_reshaped = x0.view(x0_trans.shape[0], 1, 1, 1, 1)
            x0_broadcast = x0_reshaped.expand(-1, -1, *self.input_shapes[lvl]) #broadcast to match shape of subdecoder input

            curr_decoder = self.subdecoders[lvl]

            #print(x.shape, x0_broadcast.shape)
            decoder_input = torch.cat((x, x0_broadcast), dim=1)  # concatenate the inputs and the transformed incident energy

            if lvl < self.n_latent_hierarchy_lvls - 1:
                outputs = curr_decoder(decoder_input)  # pass through the subdecoder

                if prev_outputs is not None:
                    # Add residual connection from previous subdecoder
                    target_shape = outputs.shape[2:]  # Get the spatial dimensions of the current outputs
                    prev_output_upsampled = F.interpolate(prev_outputs, size=target_shape, mode='trilinear', align_corners=False) #match spatial dimensions
                    prev_output_projected = self.residual_projections[lvl-1](prev_output_upsampled) #match channels
                    outputs += prev_output_projected  # Add the upsampled previous output to the current output
                
                prev_outputs = outputs  # Store the current outputs for the next iteration

                partition_idx_start = (self.n_latent_hierarchy_lvls - lvl - 1) * self._config.rbm.latent_nodes_per_p #start index for the current RBM partition
                partition_idx_end = partition_idx_start + self._config.rbm.latent_nodes_per_p  # end index for the current RBM partition

                enc_z = torch.cat((x_lat[:, 0:self._config.rbm.latent_nodes_per_p], x_lat[:, partition_idx_start:partition_idx_end]), dim=1)  # concatenate the incident energy and the latent nodes of the current RBM partition
                enc_z = torch.unflatten(enc_z, 1, (self._config.rbm.latent_nodes_per_p*2, 1, 1, 1))
                enc_z = self.skip_connections[lvl](enc_z)  # apply the skip connection to the concatenated latent nodes
                #print("shapes:", enc_z.shape, outputs.shape)
                x = torch.cat((outputs, enc_z), dim=1)  # concatenate the outputs of the subdecoder and the skip connection
            else:
                target_shape = (7, 14, 20)
                residual_upsampled = F.interpolate(prev_outputs, size=target_shape, mode='trilinear', align_corners=False)
                projected_residual = self.residual_projections[lvl-1](residual_upsampled)

                # Pass the processed residual to the final decoder
                output_hits, output_activations = curr_decoder(decoder_input, projected_residual=projected_residual)
                #print("Final output shapes:", output_hits.shape, output_activations.shape)
                # Flatten the outputs to match showers
                output_hits = output_hits.reshape(output_hits.shape[0], self.z*self.r*self.phi)
                output_activations = output_activations.reshape(output_activations.shape[0], self.z*self.r*self.phi)
                return output_hits, output_activations




class LastSubdecoder(nn.Module):
    """
    The last subdecoder that outputs the final shower cuboid.
    Two headed: separate heads for hits and activations
    """
    def __init__(self):
        super(LastSubdecoder, self).__init__()

        # Input is 7x12x18 cuboid
        self._layers1 = nn.Sequential(
            nn.ConvTranspose3d(18, 8, (3, 3, 3), stride=(1, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(8),
            nn.SiLU(),
            LinearAttention(8, cylindrical=False),
        ) # Outputs 7x14x20

        self._layers2 = nn.Sequential(
            nn.ConvTranspose3d(8, 4, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 0)),
            nn.GroupNorm(1, 4),
            nn.PReLU(4, 0.02),
            LinearAttention(4, cylindrical=False),

            nn.ConvTranspose3d(4, 2, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 0)),
            nn.GroupNorm(1, 2),
            nn.PReLU(2, 0.02),
            LinearAttention(2, cylindrical=False),

            nn.ConvTranspose3d(2, 1, (1, 1, 1), stride=(1, 1, 1), padding=0),
            nn.PReLU(1, 0.02),

        ) # Outputs 7x14x24 cuboid

        self._layers3 = nn.Sequential(
            nn.ConvTranspose3d(8, 4, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 0)),
            nn.GroupNorm(1, 4),
            nn.SiLU(),
            LinearAttention(4, cylindrical=False),

            nn.ConvTranspose3d(4, 2, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 0)),
            nn.GroupNorm(1, 2),
            nn.SiLU(),
            LinearAttention(2, cylindrical=False),

            nn.ConvTranspose3d(2, 1, (1, 1, 1), stride=(1, 1, 1), padding=0),
            nn.SiLU(),

        ) # Outputs 7x14x24 cuboid
    
    def forward(self, x, projected_residual=None):
        x = self._layers1(x)

        if projected_residual is not None:
            x += projected_residual

        hits = self._layers2(x)
        activations = self._layers3(x)
        return hits, activations

        