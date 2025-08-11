import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LeakyReLU, ReLU
from model.gumbel import GumbelMod
from einops import rearrange

from model.decoder.decoderhierarchybase import *


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

