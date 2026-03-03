import torch
import torch.nn as nn
from model.decoder.decoderhierarchybase import LinearAttention
from model.decoder.decoder_ATLAS_new import DecoderFullGeoATLASCompact


class FiLMLayer(nn.Module):
    """Generates layer-specific gamma and beta directly from the condition vector."""
    def __init__(self, cond_channels, feature_channels):
        super().__init__()
        # Maps the condition directly to the required number of parameters (gamma + beta)
        self.film_gen = nn.Linear(cond_channels, 2 * feature_channels)
        
        # Initialize to identity transform (gamma=0, beta=0) so it starts unperturbed
        nn.init.zeros_(self.film_gen.weight)
        nn.init.zeros_(self.film_gen.bias)

    def forward(self, x, cond):
        film_params = self.film_gen(cond)
        gamma, beta = torch.chunk(film_params, 2, dim=1)
        
        # Reshape for 3D broadcasting: (Batch, Channels, D, H, W)
        gamma = gamma.view(-1, gamma.shape[1], 1, 1, 1)
        beta = beta.view(-1, beta.shape[1], 1, 1, 1)
        
        return (1 + gamma) * x + beta

class IrregularFiLMBlock(nn.Module):
    """A flexible block to handle varying convs, activations, and attention."""
    def __init__(self, conv_layer, out_channels, cond_channels, use_act=True, use_attn=False):
        super().__init__()
        self.conv = conv_layer
        self.norm = nn.BatchNorm3d(out_channels)
        self.film = FiLMLayer(cond_channels, out_channels)
        
        self.act = nn.SiLU() if use_act else nn.Identity()
        self.attn = LinearAttention(out_channels, cylindrical=False) if use_attn else nn.Identity()

    def forward(self, x, cond):
        x = self.conv(x)
        x = self.norm(x)
        x = self.film(x, cond) # Apply unique FiLM parameters
        x = self.act(x)
        x = self.attn(x)
        return x

class FirstSubdecoderLayers(nn.Module):
    def __init__(self, cfg, cond_channels=1):
        super().__init__()
        self._config = cfg
        self.shower_size = (self._config.data.z, self._config.data.phi, self._config.data.r)
        self.n_latent_nodes =self._config.model.cond_p_size + (self._config.rbm.partitions - 1) * self._config.rbm.latent_nodes_per_p



        self.layer1_1 = IrregularFiLMBlock(
            nn.ConvTranspose3d(self.n_latent_nodes, 512, (3, 3, 3), stride=(1, 1, 1), padding=0),
            out_channels=512, cond_channels=cond_channels, use_act=True
        )
        self.layer1_2 = IrregularFiLMBlock(
            nn.ConvTranspose3d(512, 256, (3, 3, 3), stride=(1, 1, 1), padding=0),
            out_channels=256, cond_channels=cond_channels, use_act=True
        )
        self.layer1_3 = IrregularFiLMBlock(
            nn.ConvTranspose3d(256, 128, (3, 3, 3), stride=(1, 1, 1), padding=(1, 0, 0)),
            out_channels=128, cond_channels=cond_channels, use_act=False
        )

        # Removed spatial concatenation, so input channels stay at 128
        in_channels_L2 = 128 

        self.layer2_act_1 = IrregularFiLMBlock(
            nn.ConvTranspose3d(in_channels_L2, 64, (3, 5, 5), stride=(1, 1, 2), padding=(1, 0, 0)),
            out_channels=64, cond_channels=cond_channels, use_act=True, use_attn=True
        )
        self.layer2_act_2 = IrregularFiLMBlock(
            nn.ConvTranspose3d(64, 64, (3, 3, 5), stride=(1, 1, 1), padding=(1, 0, 0)),
            out_channels=64, cond_channels=cond_channels, use_act=True, use_attn=True
        )
        self.layer2_act_3 = IrregularFiLMBlock(
            nn.ConvTranspose3d(64, 32, (3, 2, 4), stride=(1, 1, 1), padding=(1, 0, 0)),
            out_channels=32, cond_channels=cond_channels, use_act=True, use_attn=True
        )

        self.layer2_hits_1 = IrregularFiLMBlock(
            nn.ConvTranspose3d(in_channels_L2, 64, (3, 5, 5), stride=(1, 1, 2), padding=(1, 0, 0)),
            out_channels=64, cond_channels=cond_channels, use_act=True
        )
        self.layer2_hits_2 = IrregularFiLMBlock(
            nn.ConvTranspose3d(64, 64, (3, 3, 5), stride=(1, 1, 1), padding=(1, 0, 0)),
            out_channels=64, cond_channels=cond_channels, use_act=True
        )
        self.layer2_hits_3 = IrregularFiLMBlock(
            nn.ConvTranspose3d(64, 32, (3, 2, 4), stride=(1, 1, 1), padding=(1, 0, 0)),
            out_channels=32, cond_channels=cond_channels, use_act=False
        )

    def forward(self, x, x0):
        """
        Forward pass through the first subdecoder, applying FiLM conditioning at each block
        Args:
            x: Input feature map, shape (Batch, Channels, D, H, W)
            x0: Conditioning variables, shape (Batch, 8) including incidence and layer-wise energies
        """

        # We assume x0 is strictly (Batch, cond_channels) 
        if x0.dim() > 2:
            x0 = x0.view(x0.shape[0], -1)

        x = self.layer1_1(x, x0)
        x = self.layer1_2(x, x0)
        x = self.layer1_3(x, x0)

        # Process Activations Branch
        x1 = self.layer2_act_1(x, x0)
        x1 = self.layer2_act_2(x1, x0)
        x1 = self.layer2_act_3(x1, x0)
        x1 = x1.reshape(x1.shape[0], 32, self.shower_size[0], self.shower_size[1], self.shower_size[2])

        # Process Hits Branch
        x2 = self.layer2_hits_1(x, x0)
        x2 = self.layer2_hits_2(x2, x0)
        x2 = self.layer2_hits_3(x2, x0)
        x2 = x2.reshape(x2.shape[0], 32, self.shower_size[0], self.shower_size[1], self.shower_size[2])

        return x1 * x2



class FiLMBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, cond_channels=32):
        super().__init__()
        self.conv = nn.ConvTranspose3d(in_channels, out_channels, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.norm = nn.BatchNorm3d(out_channels)
        
        # Layer-specific FiLM projection from the shared conditioning embedding
        self.film_proj = nn.Linear(cond_channels, 2 * out_channels)
        
        # Initialize to identity transform (gamma=0, beta=0)
        nn.init.zeros_(self.film_proj.weight)
        nn.init.zeros_(self.film_proj.bias)
        
        self.act = nn.SiLU()
        self.attn = LinearAttention(out_channels, cylindrical=False)

    def forward(self, x, cond):
        x = self.conv(x)
        x = self.norm(x)
        
        # 1. Generate unique gamma and beta for THIS specific layer
        film_params = self.film_proj(cond)
        gamma, beta = torch.chunk(film_params, 2, dim=1)
        
        # 2. Reshape for 3D broadcasting: (Batch, Channels, D, H, W)
        gamma = gamma.view(-1, gamma.shape[1], 1, 1, 1)
        beta = beta.view(-1, beta.shape[1], 1, 1, 1)
        
        # 3. Apply FiLM conditioning
        x = (1 + gamma) * x + beta
        
        x = self.act(x)
        x = self.attn(x)
        return x


class SubdecoderLayers(nn.Module):
    def __init__(self, cfg, last_subdecoder=False):
        super().__init__()
        self._config = cfg
        self.last_subdecoder = last_subdecoder


        # Base channels: 32 latent + 1 skip connection
        in_ch = 32 + 1
        
        # Maps the raw conditioning vector to a rich (Batch, 32) representation
        self.cond_embedding = nn.Sequential(
            nn.Linear(8, 32),
            nn.SiLU()
        )

        if self.last_subdecoder:
            # Head 1: Activations
            self.act_blocks = nn.ModuleList([
                FiLMBlock3D(in_ch, 32, cond_channels=32),
                FiLMBlock3D(32, 32, cond_channels=32),
                FiLMBlock3D(32, 32, cond_channels=32)
            ])
            self.act_final = nn.ConvTranspose3d(32, 1, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
            
            # Head 2: Hits
            self.hit_blocks = nn.ModuleList([
                FiLMBlock3D(in_ch, 32, cond_channels=32),
                FiLMBlock3D(32, 32, cond_channels=32),
                FiLMBlock3D(32, 32, cond_channels=32)
            ])
            self.hit_final = nn.ConvTranspose3d(32, 1, (3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
            
        else:
            self.blocks = nn.ModuleList([
                FiLMBlock3D(in_ch, 32, cond_channels=32),
                FiLMBlock3D(32, 32, cond_channels=32),
                FiLMBlock3D(32, 32, cond_channels=32)
            ])

    def forward(self, x, x0):
        """
        Forward pass through the subdecoder, applying FiLM conditioning at each block
        Args:
            x: Input feature map, shape (Batch, Channels, D, H, W)
            x0: Conditioning variables, shape (Batch, 8) including incidence and layer-wise energies
        """
        # 1. Generate the shared conditioning embedding
        cond = self.cond_embedding(x0) # (Batch, 32)

        # 2. Process through blocks, passing the shared condition to each
        if self.last_subdecoder:
            # Process Activations
            act_x = x
            for block in self.act_blocks:
                act_x = block(act_x, cond)
            activations = self.act_final(act_x)
            
            # Process Hits
            hit_x = x
            for block in self.hit_blocks:
                hit_x = block(hit_x, cond)
            hits = self.hit_final(hit_x)
            
            return hits, activations
            
        else:
            for block in self.blocks:
                x = block(x, cond)
            return x

class DecoderLayers(DecoderFullGeoATLASCompact):

    def _create_hierarchy_networks(self):
        self.subdecoders = nn.ModuleList()
        # x0 is 3 dims, u is 5 dims -> total condition dimension is 8
        cond_dim = 8 
        
        for i in range(self.n_latent_hierarchy_lvls):
            if i == 0:
                self.subdecoders.append(
                    FirstSubdecoderLayers(self._config, cond_channels=cond_dim)
                )
            else:
                self.subdecoders.append(
                    SubdecoderLayers(self._config, last_subdecoder=(i == self.n_latent_hierarchy_lvls - 1))
                )

    def forward(self, x, x0, u):
        """
        Forward pass of the hierarchical decoder.
        Extends previous decoders by adding layer-wise conditioning and FiLM at every block.
        Args:
            x: latent representation, shape (Batch, n_latent_nodes)
            x0: incident energy, shape (Batch, 1)
            u: layer-wise energies, shape (Batch, n_l)
        Returns:
            output_hits: predicted hits, shape (Batch, n_voxels)
            output_activations: predicted activations, shape (Batch, n_voxels)
        """
        x_lat = x
        x0 = self.trans_energy_multibasis(x0) # Shape: (b, 3)
        
        # Fuse the incidence energy and layer-wise energies into one condition vector
        cond_vec = torch.cat([x0, u], dim=1) # Shape: (b, 8)
        
        x = x.view(x.shape[0], self.n_latent_nodes, 1, 1, 1)  
        
        prev_output = None
        partition_idx_start = self.n_latent_nodes - self.p_size 
        partition_idx_end = partition_idx_start + self.p_size 

        for lvl in range(self.n_latent_hierarchy_lvls):
            curr_subdecoder = self.subdecoders[lvl]
            
            if lvl < self.n_latent_hierarchy_lvls - 1:
                output = curr_subdecoder(x, cond_vec)
                
                if prev_output is not None:
                    output += prev_output  
                prev_output = output
                
                enc_z = torch.cat((x_lat[:, 0:self.cond_p_size], x_lat[:, partition_idx_start:partition_idx_end]), dim=1)  
                enc_z = torch.unflatten(enc_z, 1, (self.cond_p_size + self.p_size*(1+lvl), 1, 1, 1))
                
                # Apply skip connection
                enc_z = self.skip_connections[lvl](enc_z)
                partition_idx_start -= self.p_size  
                
                x = torch.cat((output, enc_z), dim=1)  

            else:  # last level
                output_hits, output_activations = curr_subdecoder(x, cond_vec)
                output_hits = output_hits.reshape(output_hits.shape[0], self.z*self.phi*self.r)
                output_activations = output_activations.reshape(output_activations.shape[0], self.z*self.phi*self.r)
                return output_hits, output_activations