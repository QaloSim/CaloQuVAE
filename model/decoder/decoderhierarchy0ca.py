# This model is the decoder for the hierarchical structure for CaloChallenge.
# It is designed to work with the hierarchical encoder and is tailored for CaloChallenge.
# Authors: The CaloQVAE (2025)

import torch.nn as nn
import torch
import torch.nn.functional as F
from einops import rearrange
from model.decoder.decoderhierarchy0 import DecoderLinAtt

#########Decoder Hierarchy Class#######################
#######################################################

class DecoderHierarchy0CA(nn.Module):
    def __init__(self, cfg):
        super(DecoderHierarchy0CA, self).__init__()
        self._config = cfg
        self._create_hierarchy_network()
        self._create_skipcon_decoders()
        self._create_cross_attention()

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

    def _create_cross_attention(self):
        # self.crossAtt = nn.ModuleList([])
        self.crossAtt = CrossAtt(self._config)
        self.attn_scale = [nn.Parameter(torch.tensor(0.1)) for _ in range(len(self._config.model.decoder_output)-1)]
        # for i in range(len(self._config.model.decoder_output)-1):
            # self.crossAtt.append(CrossAtt(self._config))

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
                # x = enc_z + xz
                x = xz + self.attn_scale[lvl] * self.crossAtt(enc_z, xz)
        return self.x1, self.x2
    
########################################
##################CrossAtt
########################################

class CrossAtt(nn.Module):
    def __init__(self, cfg):
        super(CrossAtt, self).__init__()
        self._config = cfg
        self.head_size = self._config.model.head_size

        self.query = nn.Linear(1, self.head_size, bias=False)
        self.value = nn.Linear(1, self.head_size, bias=False)
        self.key = nn.Linear(1, self.head_size, bias=False)

        self.linear = nn.Linear(self.head_size, 1, bias=False)

    def forward(self, z_skip, z):
        z,z_skip = z.unsqueeze(-1), z_skip.unsqueeze(-1)
        query = self.query(z_skip)
        value = self.value(z_skip)
        key = self.key(z)

        wei = query @ key.transpose(-2, -1) * self.head_size**-0.5
        wei = F.softmax(wei, dim=-1)
        out = nn.Flatten()(self.linear(wei @ value))
        # return query, value, key, out
        return out