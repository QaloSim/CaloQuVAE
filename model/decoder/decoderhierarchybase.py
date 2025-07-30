import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import LeakyReLU, ReLU
from model.gumbel import GumbelMod
from einops import rearrange


class DecoderHierarchyBase(nn.Module):
    def __init__(self, cfg):
        """
        Hierarchical decoder that uses 4 subdecoders to sample from the 4 RBM partitions. 
        The ith subdecoder takes as input the output of the (i-1)th subdecoder,
        and the output of the ith skip connection corresponding to the latent nodes in the ith RBM partition.
        Outputs a shower of shape (7, 24, 14)
        """
        super(DecoderHierarchyBase, self).__init__()
        self.smoothing_dist_mod = GumbelMod()
        self._config = cfg

        self._create_hierarchy_network()
        self._create_skip_connections()

    def _create_hierarchy_network(self):
        """
        Create the hierarchy network. Each subdecoder has identical architecture, but different input and outputs.
        """
        self.latent_nodes = self._config.rbm.latent_nodes_per_p * self._config.rbm.partitions # Number of latent nodes in total
        self.hierarchical_levels = self._config.rbm.partitions

        self.shower_size = 7 * 24 * 14  # size of the shower output by each subdecoder

        # List of input sizes for each subdecoder
        self.input_sizes = [self.latent_nodes + self.shower_size] * self.hierarchical_levels
        self.input_sizes[0] = self.latent_nodes  # First subdecoder only takes latent nodes as input

        # List of output sizes for each subdecoder
        self.output_sizes = [self.shower_size] * self.hierarchical_levels

        # Create the subdecoders
        self.subdecoders = nn.ModuleList()
        for i in range(self.hierarchical_levels):
            self.subdecoders.append(DecoderCNNPB3Dv4_HEMOD(num_input_nodes=self.input_sizes[i],num_output_nodes=self.output_sizes[i]))

    def _create_skip_connections(self):
        """
        Create skip connections for the hierarchical decoder.
        The ith skip connection feeds the latent nodes of the ith RBM partition to the ith subdecoder,
        as well as the conditioned incident energy stored in partition 0 of the RBM.
        """
        self.skip_connections = nn.ModuleList()
        input_size = 2*self._config.rbm.latent_nodes_per_p  # Each skip connection takes the latent nodes and the incident energy as input
        output_size = self.shower_size + self.latent_nodes # Each skip connection outputs the input size of the subdecoders
        for i in range(self.hierarchical_levels-1):
            skip_connection = nn.Conv3d(
                in_channels=input_size,
                out_channels=output_size,
                kernel_size=(1, 1, 1),
                stride=(1, 1, 1))
            self.skip_connections.append(skip_connection)

    def forward(self, x, x0):
        x_lat = x
        self.x1, self.x2 = torch.tensor([]).to(x.device), torch.tensor([]).to(x.device) #initialize tensors for activations and hits

        for lvl in range(self.hierarchical_levels):
            cur_decoder = self.subdecoders[lvl]
            output_hits, output_activations = cur_decoder(x, x0)
            outputs = output_hits * output_activations
            z = outputs
            if lvl < self.hierarchical_levels - 1: # If not the last level, prepare input for the next level
                # Concatenate the output of the current decoder with the latent nodes of the next RBM partition
                partition_idx_start = (self.hierarchical_levels - lvl - 1) * self._config.rbm.latent_nodes_per_p #start index for the current RBM partition
                partition_idx_end = partition_idx_start + self._config.rbm.latent_nodes_per_p  # end index for the current RBM partition
                enc_z = torch.cat((x[:, 0:self._config.rbm.latent_nodes_per_p], x[:, partition_idx_start:partition_idx_end]), dim=1)  # concatenate the incident energy and the latent nodes of the current RBM partition
                enc_z = torch.unflatten(enc_z, 1, (self._config.rbm.latent_nodes_per_p*2, 1, 1, 1))
                # Apply skip connection
                enc_z = self.skip_connections[lvl](enc_z).view(enc_z.size(0), -1)  # Flatten the output of the skip connection
                xz = torch.cat((x_lat, z), dim=1) # Concatenate the latent nodes and the output of the current decoder
                x = xz + enc_z  # Add the output of the skip connection to the output of the current decoder

            else:  # If the last level, just return the output
                self.x1 = output_hits
                self.x2 = output_activations
            
        return self.x1, self.x2  # Return the output of the last decoder, which is the shower output


class DecoderHierarchyBaseV2(DecoderHierarchyBase):
    def __init__(self, cfg):
        """
        Hierarchical decoder that uses 4 subdecoders to sample from the 4 RBM partitions.
        The ith subdecoder takes as input the output of the (i-1)th subdecoder,
        and the output of the ith skip connection corresponding to the latent nodes in the ith RBM partition.
        Outputs a shower of shape (7, 24, 14)
        """
        super(DecoderHierarchyBaseV2, self).__init__(cfg)
    
    def _create_skip_connections(self):
        self.skip_connections = nn.ModuleList()
        input_size = 2*self._config.rbm.latent_nodes_per_p  # Each skip connection takes the latent nodes and the incident energy as input
        output_size = self.latent_nodes # Each skip connection outputs the input size of the subdecoders minus the shower size
        for i in range(self.hierarchical_levels-1):
            skip_connection = nn.Conv3d(
                in_channels=input_size,
                out_channels=output_size,
                kernel_size=(1, 1, 1),
                stride=(1, 1, 1))
            self.skip_connections.append(skip_connection)


    
    def forward(self, x, x0):
        x_lat = x
        output_hits, output_activations = None, None

        for lvl in range(self.hierarchical_levels):
            cur_decoder = self.subdecoders[lvl]
            output_hits, output_activations = cur_decoder(x, x0)
            if lvl < self.hierarchical_levels - 1: # If not the last level, prepare input for the next level
                outputs = output_hits * output_activations
                z = outputs
                # Concatenate the output of the current decoder with the latent nodes of the next RBM partition
                partition_idx_start = (self.hierarchical_levels - lvl - 1) * self._config.rbm.latent_nodes_per_p #start index for the current RBM partition
                partition_idx_end = partition_idx_start + self._config.rbm.latent_nodes_per_p  # end index for the current RBM partition
                enc_z = torch.cat((x_lat[:, 0:self._config.rbm.latent_nodes_per_p], x_lat[:, partition_idx_start:partition_idx_end]), dim=1)  # concatenate the incident energy and the latent nodes of the current RBM partition
                enc_z = torch.unflatten(enc_z, 1, (self._config.rbm.latent_nodes_per_p*2, 1, 1, 1))
                # Apply skip connection
                enc_z = self.skip_connections[lvl](enc_z).view(enc_z.size(0), -1)  # Flatten the output of the skip connection

                x = torch.cat((enc_z, z), dim=1)  # Concatenate the output of the skip connection with the output of the current decoder
        # If the last level, just return the output            
        return output_hits, output_activations # Return the output of the last decoder, which is the shower output

class DecoderHierarchyBaseV3(DecoderHierarchyBaseV2):
    def _create_skip_connections(self):
        """
        Create skip connections for the hierarchical decoder.
        The ith skip connection feeds the latent nodes of the ith RBM partition to the ith subdecoder,
        as well as the conditioned incident energy stored in partition 0 of the RBM.
        """
        self.skip_connections = nn.ModuleList()
        input_size = 2*self._config.rbm.latent_nodes_per_p
        output_size = self.latent_nodes  # Each skip connection outputs the input size of the subdecoders minus the shower size
        for i in range(self.hierarchical_levels-1):
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

class DecoderHierarchyBaseV4(DecoderHierarchyBase):

    def _create_skip_connections(self):
        self.skip_connections = nn.ModuleList()
        input_sizes = [(2+i)*self._config.rbm.latent_nodes_per_p for i in range(self.hierarchical_levels-1)]  # Each skip connection takes an increasing number of partitions
        for i in range(self.hierarchical_levels-1):
            skip_connection = nn.Sequential(
                nn.Conv3d(
                    in_channels=input_sizes[i],
                    out_channels=input_sizes[i]+ self._config.rbm.latent_nodes_per_p,
                    kernel_size=(1, 1, 1),
                    stride=(1, 1, 1)),
                nn.Conv3d(
                    in_channels=input_sizes[i]+ self._config.rbm.latent_nodes_per_p,
                    out_channels=self.latent_nodes,
                    kernel_size=(1, 1, 1),
                    stride=(1, 1, 1))
            )
            self.skip_connections.append(skip_connection)



    def forward(self, x, x0):
        x_lat = x
        output_hits, output_activations = None, None
        partition_idx_start = (self.hierarchical_levels-1) * self._config.rbm.latent_nodes_per_p  # start index for the z3 RBM partition
        partition_idx_end = partition_idx_start + self._config.rbm.latent_nodes_per_p  # end index for the z3 RBM partition

        for lvl in range(self.hierarchical_levels):
            cur_decoder = self.subdecoders[lvl]
            output_hits, output_activations = cur_decoder(x, x0)
            if lvl < self.hierarchical_levels - 1: # If not the last level, prepare input for the next level
                outputs = output_hits * output_activations
                z = outputs
                # Concatenate the output of the current decoder with the latent nodes of the next RBM partition
                enc_z = torch.cat((x_lat[:, 0:self._config.rbm.latent_nodes_per_p], x_lat[:, partition_idx_start:partition_idx_end]), dim=1)  # concatenate the incident energy and the latent nodes of the current RBM partition
                enc_z = torch.unflatten(enc_z, 1, (self._config.rbm.latent_nodes_per_p*(2+lvl), 1, 1, 1))
                # Apply skip connection
                enc_z = self.skip_connections[lvl](enc_z).view(enc_z.size(0), -1)  # Flatten the output of the skip connection                partition_idx_start -= self._config.rbm.latent_nodes_per_p  # start index for the current RBM partition, moves one partition back every level
                partition_idx_start -= self._config.rbm.latent_nodes_per_p  # start index for the current RBM partition, moves one partition back every level

                x = torch.cat((enc_z, z), dim=1)  # Concatenate the output of the skip connection with the output of the current decoder
        # If the last level, just return the output            
        return output_hits, output_activations # Return the output of the last decoder, which is the shower output
    

        


class PeriodicConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(PeriodicConv3d, self).__init__()
        self.padding = padding
        # try 3x3x3 cubic convolution
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

class CropLayer(nn.Module):
    def forward(self, x):
        return x[:, :, :, 1:15, :]

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


class DecoderCNNPB3Dv4_HEMOD(nn.Module):
    def __init__(self, num_input_nodes, num_output_nodes, output_activation_fct=nn.Identity()):
        super(DecoderCNNPB3Dv4_HEMOD, self).__init__()
        self.num_input_nodes = num_input_nodes
        self.num_output_nodes = num_output_nodes
        self.output_activation_fct = output_activation_fct

        self._layers1 = nn.Sequential(
            nn.Unflatten(1, (num_input_nodes, 1, 1, 1)),  # Assuming input is flattened, reshape to (batch_size, num_input_nodes, 1, 1, 1)
            PeriodicConvTranspose3d(num_input_nodes, 512, (3, 3, 3), stride=(1, 1, 1), padding=0),
            nn.BatchNorm3d(512),
            nn.PReLU(512, 0.02),
            # upscales to (batch_size, 512, 3, 3, 3)
            PeriodicConvTranspose3d(512, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=0),
            nn.BatchNorm3d(128),
            nn.PReLU(128, 0.02),
            # upscales to (batch_size, 128, 7, 7, 7)
                    )

        self._layers2 = nn.Sequential(
            # layer for hits
            PeriodicConvTranspose3d(129, 64, (1, 3, 5), stride=(1, 2, 3), padding=0),
            nn.BatchNorm3d(64),
            nn.PReLU(64, 0.02),
            # upscales to (batch_size, 64, 7, 15, 23)

            PeriodicConvTranspose3d(64, 32, (1, 2, 2), stride=(1, 1, 1), padding=0),
            nn.BatchNorm3d(32),
            nn.PReLU(32, 1.0),
            # upscales to (batch_size, 32, 7, 16, 24)

            PeriodicConvTranspose3d(32, 1, (1, 1, 1), stride=(1, 1, 1), padding=0),
            CropLayer(), # Crop to (batch_size, 1, 7, 14, 24)
            
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
            CropLayer(),  # Crop to (batch_size, 1, 7, 14, 24)
            PeriodicConv3d(1, 1, (1, 1, 1), stride=(1, 1, 1), padding=0),
            nn.SiLU(),

                )

    def trans_energy(self, x0, log_e_max=14.0, log_e_min=6.0, s_map = 1.0):
        return ((torch.log(x0) - log_e_min)/(log_e_max - log_e_min)) * s_map
    
    def forward(self, x, x0):
        """
        Forward pass of the decoder.
        :param x: Input tensor of shape (batch_size, num_input_nodes)
        :param x0: Input tensor for incident energy, shape (batch_size, 1)
        :return: Tuple of output hits and activations
        """

        x = self._layers1(x)
        x0 = self.trans_energy(x0)
        # Concatenate the incident energy to the output of the first layer
        d1, d2, d3 = x.shape[-3], x.shape[-2], x.shape[-1]
        xx0 = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat(1,1, d1, d2, d3)), dim=1)  # Concatenate the incident energy to the output of the first layer
        x1 = self._layers2(xx0).reshape(xx0.shape[0], self.num_output_nodes) # Output hits
        x2 = self._layers3(xx0).reshape(xx0.shape[0], self.num_output_nodes)  # Output activations

        return x1, x2
        