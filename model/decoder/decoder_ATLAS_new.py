import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LeakyReLU, ReLU
from model.gumbel import GumbelMod
from einops import rearrange

from model.decoder.decoderhierarchybase import *
from model.decoder.decoder_full_geo import *


class DecoderATLASNew(DecoderHierarchyBaseV4):
    """
    Decoder for ATLAS New model.
    This class extends the DecoderHierarchyBase to work for the dimensions of the new ATLAS dataset.
    """

    def _create_hierarchy_network(self):
        """
        Create the hierarchy network. Each subdecoder has identical architecture, but different input and outputs.
        """
        self.latent_nodes = self._config.rbm.latent_nodes_per_p * self._config.rbm.partitions # Number of latent nodes in total
        self.hierarchical_levels = self._config.rbm.partitions

        self.shower_size = self._config.data.z * self._config.data.phi * self._config.data.r  # size of the shower output by each subdecoder

        # List of input sizes for each subdecoder
        self.input_sizes = [self.latent_nodes + self.shower_size] * self.hierarchical_levels
        self.input_sizes[0] = self.latent_nodes  # First subdecoder only takes latent nodes as input

        # List of output sizes for each subdecoder
        self.output_sizes = [self.shower_size] * self.hierarchical_levels

        # Create the subdecoders
        self.subdecoders = nn.ModuleList()
        for i in range(self.hierarchical_levels):
            self.subdecoders.append(SubDecoderATLASNew(num_input_nodes=self.input_sizes[i],num_output_nodes=self.output_sizes[i]))

class SubDecoderATLASNew(DecoderCNNPB3Dv4_HEMOD):
    """
    Subdecoder for ATLAS New model.
    This class extends the DecoderCNNPB3Dv4_HEMOD to handle the specific dimensions of the ATLAS New dataset.
    """

    def __init__(self, num_input_nodes, num_output_nodes):
        """
        Initialize the subdecoder with the given input and output sizes.
        """    
        super(SubDecoderATLASNew, self).__init__(num_input_nodes, num_output_nodes)
        self._layers1 = nn.Sequential(
            nn.Unflatten(1, (num_input_nodes, 1, 1, 1)),  # Assuming input is flattened, reshape to (batch_size, num_input_nodes, 1, 1, 1)
            PeriodicConvTranspose3d(num_input_nodes, 512, (3, 3, 3), stride=(1, 1, 1), padding=0),
            nn.BatchNorm3d(512),
            nn.PReLU(512, 0.02),
            # upscales to (batch_size, 512, 3, 3, 3)
            PeriodicConvTranspose3d(512, 128, kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=0),
            nn.BatchNorm3d(128),
            nn.PReLU(128, 0.02),
            # upscales to (batch_size, 128, 5, 7, 7)
            )
        
        self._layers2 = nn.Sequential(
            # layer for hits
            PeriodicConvTranspose3d(129, 64, (1, 3, 5), stride=(1, 2, 3), padding=0),
            nn.BatchNorm3d(64),
            nn.PReLU(64, 0.02),
            # upscales to (batch_size, 64, 5, 15, 23)

            PeriodicConvTranspose3d(64, 32, (1, 2, 2), stride=(1, 1, 1), padding=0),
            nn.BatchNorm3d(32),
            nn.PReLU(32, 1.0),
            # upscales to (batch_size, 32, 5, 16, 24)

            PeriodicConvTranspose3d(32, 1, (1, 1, 1), stride=(1, 1, 1), padding=0),
            CropLayer(), # Crop to (batch_size, 1, 5, 14, 24)
            
            PeriodicConv3d(1, 1, (1, 1, 1), stride=(1, 1, 1), padding=0),
            nn.PReLU(1, 1.0),

                    )
        
        self._layers3 = nn.Sequential(
            # layer for activations, identical upscaling to hits but with linear attention
            PeriodicConvTranspose3d(129, 64, (1, 3, 5), stride=(1, 2, 3), padding=0),
            nn.GroupNorm(1, 64),
            nn.SiLU(),
            LinearAttention(64, cylindrical=False),

            PeriodicConvTranspose3d(64, 32, (1, 2, 2), stride=(1, 1, 1), padding=0),
            nn.GroupNorm(1, 32),
            nn.SiLU(),
            LinearAttention(32, cylindrical=False),

            PeriodicConvTranspose3d(32, 1, (1, 1, 1), stride=(1, 1, 1), padding=0),
            CropLayer(),  # Crop to (batch_size, 1, 5, 14, 24)
            PeriodicConv3d(1, 1, (1, 1, 1), stride=(1, 1, 1), padding=0),
            nn.SiLU(),

                )

class DecoderFullGeoATLASNew(DecoderFullGeo):

    def _create_hierarchy_networks(self):
        self.subdecoders = nn.ModuleList()
        for i in range(self.n_latent_hierarchy_lvls):
            if i == 0:
                subdecoder = FirstSubDecoderATLASNew(self._config)
            else:
                subdecoder = SubDecoder(self._config, last_subdecoder=(i == self.n_latent_hierarchy_lvls - 1))
            self.subdecoders.append(subdecoder)
    
    def _create_skip_connections(self):
        self.skip_connections = nn.ModuleList()
        if hasattr(self, 'cond_p_size'):
            start = self.cond_p_size + self.p_size
        else:
            start = self.p_size * 2
        for i in range(self.n_latent_hierarchy_lvls-1):
            skip_connection = nn.Sequential(
                nn.ConvTranspose3d(start + i * self.p_size, 64, (3, 5, 7), (1, 1, 1), padding=0),
                nn.BatchNorm3d(64),
                nn.PReLU(64, 0.02),
                # upscales to (64, 3, 5, 7)
                nn.ConvTranspose3d(64, 32, (3, 5, 7), (1, 1, 2), padding=0),
                nn.BatchNorm3d(32),
                nn.PReLU(32, 0.02),
                # upscales to (32, 5, 8, 12)
                nn.ConvTranspose3d(32, 1, (3, 6, 6), (1, 1, 1), padding=(1, 0, 0)),
            ) #outputs (1, 5, 14, 24)
            self.skip_connections.append(skip_connection)

    


class FirstSubDecoderATLASNew(FirstSubDecoder):
    def __init__(self, cfg):
        super(FirstSubDecoderATLASNew, self).__init__(cfg)
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
            nn.ConvTranspose3d(256, 128, (3, 3, 3), stride=(1, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(128),
            nn.PReLU(128, 0.02),
            # upscales to (128, 5, 7, 7)
        )
        self._layers2 = nn.Sequential(
            # layer for activations
            nn.ConvTranspose3d(129, 64, (3, 5, 5), stride=(1, 1, 2), padding=(1, 0, 0)),
            nn.GroupNorm(1, 64),
            nn.SiLU(),
            LinearAttention(64, cylindrical=False),
            # upscales to  (64, 5, 11, 17)
            nn.ConvTranspose3d(64, 64, (3, 3, 5), stride=(1, 1, 1), padding=(1, 0, 0)),
            nn.GroupNorm(1, 64),
            nn.SiLU(),
            LinearAttention(64, cylindrical=False),
            # upscales to (64, 5, 13, 21)

            nn.ConvTranspose3d(64, 32, (3, 2, 4), stride=(1, 1, 1), padding=(1, 0, 0)),
            nn.GroupNorm(1, 32),
            nn.SiLU(),
            LinearAttention(32, cylindrical=False),
        )
        self._layers2_hits = nn.Sequential(
            # layer for hits
            nn.ConvTranspose3d(129, 64, (3, 5, 5), stride=(1, 1, 2), padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.PReLU(64, 0.02),
            # upscales to  (64, 5, 15, 23)
            nn.ConvTranspose3d(64, 64, (3, 3, 5), stride=(1, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.PReLU(64, 0.02),
            # upscales to (64, 5, 11, 17)
            nn.ConvTranspose3d(64, 32, (3, 2, 4), stride=(1, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(32),
            nn.PReLU(32, 1.0),
        )

