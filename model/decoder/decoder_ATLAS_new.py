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

class FirstSubdecoderAtlasClean(FirstSubDecoder):
    def __init__(self, cfg, energy_channels=1):
        super().__init__(cfg)
        self.shower_size = (self._config.data.z, self._config.data.phi, self._config.data.r)

        # 1. Update _layers1 input size (Already done by you, correct)
        self._layers1 = nn.Sequential(
            PeriodicConvTranspose3d(self.n_latent_nodes + energy_channels, 512, (3, 3, 3), stride=(1, 1, 1), padding=0),
            nn.BatchNorm3d(512),
            nn.SiLU(), 
            
            # Upscales to (512, 3, 3, 3)
            PeriodicConvTranspose3d(512, 256, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=0),
            nn.BatchNorm3d(256),
            nn.SiLU(),
            # Upscales to (256, 5, 5, 5)
            nn.ConvTranspose3d(256, 128, (3, 3, 3), stride=(1, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(128),
            # Upscales to (128, 5, 7, 7)
        )

        # 2. FIX: Update input channels from 129 -> 128 + energy_channels
        # 128 comes from the previous layer, energy_channels comes from the skip connection
        in_channels_L2 = 128 + energy_channels 

        self._layers2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels_L2, 64, (3, 5, 5), stride=(1, 1, 2), padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.SiLU(),
            LinearAttention(64, cylindrical=False),
            
            # Upscales to (64, 5, 11, 17)
            nn.ConvTranspose3d(64, 64, (3, 3, 5), stride=(1, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.SiLU(),
            LinearAttention(64, cylindrical=False),
            
            # Upscales to (64, 5, 13, 21)
            nn.ConvTranspose3d(64, 32, (3, 2, 4), stride=(1, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(32),
            nn.SiLU(),
            LinearAttention(32, cylindrical=False),
        )

        self._layers2_hits = nn.Sequential(
            # 3. FIX: Update input channels here too
            nn.ConvTranspose3d(in_channels_L2, 64, (3, 5, 5), stride=(1, 1, 2), padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.SiLU(),
            
            # Upscales to (64, 5, 15, 23)
            nn.ConvTranspose3d(64, 64, (3, 3, 5), stride=(1, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.SiLU(),
            
            # Upscales to (64, 5, 11, 17)
            nn.ConvTranspose3d(64, 32, (3, 2, 4), stride=(1, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(32),
        )

    # 4. FIX: Override forward to handle multi-energy channel broadcasting
    def forward(self, x, x0):
        # x input shape: (Batch, n_latent + energy_channels, 1, 1, 1)
        x = self._layers1(x) 
        # x output shape: (Batch, 128, 5, 7, 7)
        
        d1, d2, d3 = x.shape[-3], x.shape[-2], x.shape[-1]
        
        # Broadcast x0 (Batch, energy_channels) to (Batch, energy_channels, d1, d2, d3)
        # We assume x0 is passed in as (Batch, energy_channels) or (Batch, energy_channels, 1, 1, 1)
        x0_broadcast = x0.view(x0.shape[0], x0.shape[1], 1, 1, 1).expand(-1, -1, d1, d2, d3)

        # Concatenate: (Batch, 128, ...) + (Batch, energy_channels, ...) = (Batch, 128 + energy_channels, ...)
        xx0 = torch.cat((x, x0_broadcast), dim=1) 
        
        x1 = self._layers2(xx0).reshape(xx0.shape[0], 32, self.shower_size[0], self.shower_size[1], self.shower_size[2])
        x2 = self._layers2_hits(xx0).reshape(xx0.shape[0], 32, self.shower_size[0], self.shower_size[1], self.shower_size[2])
        
        return x1 * x2


class DecoderFullGeoATLASClean(DecoderFullGeo):

    def _create_hierarchy_networks(self):
        self.subdecoders = nn.ModuleList()
        for i in range(self.n_latent_hierarchy_lvls):
            if i == 0:
                subdecoder = FirstSubdecoderAtlasClean(self._config)
            else:
                subdecoder = SubdecoderClean(self._config, last_subdecoder=(i == self.n_latent_hierarchy_lvls - 1))
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
                nn.SiLU(),
                # upscales to (64, 3, 5, 7)
                nn.ConvTranspose3d(64, 32, (3, 5, 7), (1, 1, 2), padding=0),
                nn.BatchNorm3d(32),
                nn.SiLU(),
                # upscales to (32, 5, 8, 12)
                nn.ConvTranspose3d(32, 1, (3, 6, 6), (1, 1, 1), padding=(1, 0, 0)),
                nn.SiLU(),
            ) #outputs (1, 5, 14, 24)
            self.skip_connections.append(skip_connection)

class DecoderFullGeoATLASCompact(DecoderFullGeoATLASClean):

    def _create_hierarchy_networks(self):
        self.subdecoders = nn.ModuleList()
        for i in range(self.n_latent_hierarchy_lvls):
            if i == 0:
                subdecoder = FirstSubdecoderAtlasClean(self._config, energy_channels=3)
            else:
                subdecoder = SubdecoderClean(self._config, last_subdecoder=(i == self.n_latent_hierarchy_lvls - 1), energy_channels=3)
            self.subdecoders.append(subdecoder)

    
    def trans_energy_multibasis(self, x0, energy_min=1000.0, energy_max=300000.0):
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

    def forward(self, x, x0):
        x_lat = x
        x0 = self.trans_energy_multibasis(x0)
        x0_reshaped = x0.view(x0.shape[0], 3, 1, 1, 1)
        x = x.view(x.shape[0], self.n_latent_nodes, 1, 1, 1)  # Reshape x to match the input shape of the first subdecoder
        prev_output = None
        partition_idx_start = self.n_latent_nodes - self.p_size  # start index for the z3 RBM partition
        partition_idx_end = partition_idx_start + self.p_size # end index for the z3 RBM partition


        for lvl in range(self.n_latent_hierarchy_lvls):
            curr_subdecoder = self.subdecoders[lvl]
            x0_broadcasted = x0_reshaped.expand(x.shape[0], 3, *self.input_shapes[lvl])

            decoder_input = torch.cat((x, x0_broadcasted), dim=1)  # Concatenate along the channel dimension
            # print(decoder_input.shape)
            
            if lvl < self.n_latent_hierarchy_lvls - 1:
                output = curr_subdecoder(decoder_input, x0)
                if prev_output is not None:
                    output += prev_output  # add/refine the previous subdecoder output
                prev_output = output
                enc_z = torch.cat((x_lat[:, 0:self.cond_p_size], x_lat[:, partition_idx_start:partition_idx_end]), dim=1)  # concatenate the incident energy and the latent nodes of the current RBM partition
                enc_z = torch.unflatten(enc_z, 1, (self.cond_p_size + self.p_size*(1+lvl), 1, 1, 1))
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




class CylindricalTranspose3D(nn.ConvTranspose3d):
    """
    A 3D Transpose Convolution that enforces circular (cylindrical) topology 
    on the Phi dimension (Dimension 3: D, H, W -> Z, Phi, R).
    
    Robust to cases where padding > dimension size.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.phi_axis = 3 # (Batch, C, Z, Phi, R) -> Index 3
        self.kernel_phi = self.kernel_size[1]
        self.stride_phi = self.stride[1]
        
        # Store original intended padding 
        self.pad_z = self.padding[0]
        self.pad_r = self.padding[2]
        self.user_pad_phi = self.padding[1]

    def forward(self, x):
        # x shape: (Batch, Channel, Z, Phi, R)
        
        # --- 1. Robust Circular Padding ---
        # We need to pad 'k' pixels on left and right. 
        # F.pad(mode='circular') fails if k > dim_size. 
        # We use index_select to manually wrap.
        
        k = self.kernel_phi
        dim_size = x.shape[self.phi_axis]
        
        # Calculate indices for left and right padding
        # Left pad: takes from the end (wrapping around)
        # Right pad: takes from the start (wrapping around)
        idx_left = torch.arange(-k, 0, device=x.device) % dim_size
        idx_right = torch.arange(0, k, device=x.device) % dim_size
        
        # Extract the padding slices
        left_pad = x.index_select(self.phi_axis, idx_left)
        right_pad = x.index_select(self.phi_axis, idx_right)
        
        # Concatenate: [Left_Pad, Original, Right_Pad]
        x_padded = torch.cat([left_pad, x, right_pad], dim=self.phi_axis)
        
        # --- 2. Adjust internal layer padding ---
        # We want Z and R to use standard padding, but Phi to have 0 (handled manually above)
        original_padding = self.padding
        self.padding = (self.pad_z, 0, self.pad_r)
        
        # --- 3. Perform Transpose Conv ---
        out = super().forward(x_padded)
        
        # --- 4. Restore internal state ---
        self.padding = original_padding
        
        # --- 5. Crop the output ---
        # Calculate expected output size based on original input size
        in_phi = x.shape[self.phi_axis]
        
        # Standard Transpose Conv output size formula
        expected_phi = (in_phi - 1) * self.stride_phi - 2 * self.user_pad_phi + self.kernel_phi + self.output_padding[1]
        
        # Center crop on dimension 3
        curr_phi = out.shape[self.phi_axis]
        start = (curr_phi - expected_phi) // 2
        
        # Safety check: ensure we don't slice out of bounds if something is off (rare)
        if start < 0: start = 0
        
        out = out[:, :, :, start : start + expected_phi, :]
        
        return out
        

class FirstSubdecoderAtlasCylinder(FirstSubdecoderAtlasClean):
    def __init__(self, cfg):
        # We initialize the parent to get standard attributes, 
        # but we will immediately overwrite the layers.
        super().__init__(cfg)

        # _layers1 is kept as is (PeriodicConvTranspose3d is already cylindrical-aware presumably).
        # We redefine _layers2 to use CylindricalTranspose3D
        self._layers2 = nn.Sequential(
            # Input 129 implies concatenation happened before this block
            CylindricalTranspose3D(129, 64, (3, 5, 5), stride=(1, 1, 2), padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.SiLU(),
            # UPDATED: cylindrical=True
            LinearAttention(64, cylindrical=True),
            
            # Upscales to (64, 5, 11, 17)
            CylindricalTranspose3D(64, 64, (3, 3, 5), stride=(1, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.SiLU(),
            LinearAttention(64, cylindrical=True),
            
            # Upscales to (64, 5, 13, 21)
            CylindricalTranspose3D(64, 32, (3, 2, 4), stride=(1, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(32),
            nn.SiLU(),
            LinearAttention(32, cylindrical=True),
        )

        self._layers2_hits = nn.Sequential(
            # Layer for hits
            CylindricalTranspose3D(129, 64, (3, 5, 5), stride=(1, 1, 2), padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.SiLU(),
            
            # Upscales to (64, 5, 15, 23)
            CylindricalTranspose3D(64, 64, (3, 3, 5), stride=(1, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(64),
            nn.SiLU(),
            
            # Upscales to (64, 5, 11, 17)
            CylindricalTranspose3D(64, 32, (3, 2, 4), stride=(1, 1, 1), padding=(1, 0, 0)),
            nn.BatchNorm3d(32),
        )


class DecoderFullGeoATLASCylinder(DecoderFullGeoATLASClean):
    def _create_hierarchy_networks(self):
        self.subdecoders = nn.ModuleList()
        for i in range(self.n_latent_hierarchy_lvls):
            if i == 0:
                # Use the new Cylindrical First Subdecoder
                subdecoder = FirstSubdecoderAtlasCylinder(self._config)
            else:
                # Assuming SubdecoderClean handles standard logic, 
                # or you might need a Cylindrical version of SubdecoderClean too
                # if the later stages also need cylindrical transpose.
                # For now, we use the standard one as requested for the hierarchy.
                subdecoder = SubdecoderClean(self._config, last_subdecoder=(i == self.n_latent_hierarchy_lvls - 1))
            self.subdecoders.append(subdecoder)
    
    def _create_skip_connections(self):
        self.skip_connections = nn.ModuleList()
        if hasattr(self, 'cond_p_size'):
            start = self.cond_p_size + self.p_size
        else:
            start = self.p_size * 2
            
        for i in range(self.n_latent_hierarchy_lvls-1):
            # Updated to use CylindricalTranspose3D
            skip_connection = nn.Sequential(
                CylindricalTranspose3D(start + i * self.p_size, 64, (3, 5, 7), (1, 1, 1), padding=0),
                nn.BatchNorm3d(64),
                nn.SiLU(),
                # upscales to (64, 3, 5, 7)
                CylindricalTranspose3D(64, 32, (3, 5, 7), (1, 1, 2), padding=0),
                nn.BatchNorm3d(32),
                nn.SiLU(),
                # upscales to (32, 5, 8, 12)
                CylindricalTranspose3D(32, 1, (3, 6, 6), (1, 1, 1), padding=(1, 0, 0)),
                nn.SiLU(),
            ) 
            self.skip_connections.append(skip_connection)