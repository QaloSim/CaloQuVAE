import torch.nn as nn
from einops import rearrange
import torch.nn.functional as F
import torch
from model.decoder.decoderhierarchy0 import PeriodicConvTranspose3d

class DecoderHierarchyTF(nn.Module):
    def __init__(self, cfg):
        super(DecoderHierarchyTF, self).__init__()
        self._config = cfg
        self.head_size = self._config.model.head_size
        self._create_hierarchy_network()
        self._create_skipcon_decoders()

    def _create_hierarchy_network(self):
        self.latent_nodes = self._config.rbm.latent_nodes_per_p * self._config.rbm.partitions
        self.hierarchical_lvls = self._config.rbm.partitions

        inp_layers = self._config.model.decoder_input

        self.moduleLayers = nn.ModuleList([])
        for i in range(self.hierarchical_lvls-1):
            self.moduleLayers.append(DecoderAtt(self._config, inp_layers[i])) 
        self.moduleLayers.append(Decoder(self._config, inp_layers[-1]))   

    def _create_skipcon_decoders(self):
        self.lnpp = self._config.rbm.latent_nodes_per_p
        self.subdecs = nn.ModuleList([])
        for i in range(self.hierarchical_lvls-1):
            self.subdecs.append(Skip(self._config))
    
    def forward(self, z, x0):
        z_prime = z
        for i in range(len(self.moduleLayers)-1):
            x1, x2 = self.moduleLayers[i](z_prime, x0)
            keys = x1 * x2
            z_skip = torch.cat((z_prime[:,:self.lnpp], z_prime[:,self.lnpp*(3-i):self.lnpp*(4-i)]), dim=1)

            out = self.subdecs[i](z_skip, keys)
            # z_prime = torch.cat((z_prime,z),dim=1)
            z_prime = torch.cat((out,z),dim=1)
                
        x1, x2 = self.moduleLayers[-1](z_prime, x0)
        return x1,x2 #,out,z_prime
    
####################################################
################Decoder Classes##################
####################################################

class DecoderAtt(nn.Module):
    def __init__(self, cfg, input_size):
        super(DecoderAtt, self).__init__()
        self._config = cfg

        self.n_latent_hierarchy_lvls=self._config.rbm.partitions

        self.n_latent_nodes=self._config.rbm.latent_nodes_per_p * self._config.rbm.partitions

        self.z = self._config.data.z
        self.r = self._config.data.r
        self.phi = self._config.data.phi

        # output_size_z = int( output_size / (self.r * self.phi))

        self._layers =  nn.Sequential(
                   nn.Unflatten(1, (input_size, 1, 1, 1)),

                   PeriodicConvTranspose3d(input_size, 512, (3,3,2), (1,1,1), 0),
                   nn.BatchNorm3d(512),
                   nn.PReLU(512, 0.02),
                   

                   PeriodicConvTranspose3d(512, 128, (3,3,2), (1,1,1), 0),
                   nn.BatchNorm3d(128),
                   nn.PReLU(128, 0.02),
                                   )
        
        self._layers2 = nn.Sequential(
                   PeriodicConvTranspose3d(129, 64, (3,3,2), (1,1,1), 1),
                   nn.BatchNorm3d(64),
                   nn.PReLU(64, 0.02),

                   PeriodicConvTranspose3d(64, 32, (2,2,2), (1,1,1), 1),
                   nn.BatchNorm3d(32),
                   nn.PReLU(32, 1.0),

                #    PeriodicConvTranspose3d(32, 1, (5,3,3), (1,1,1), 0),
                #    PeriodicConv3d(1, 1, (self.z - output_size_z + 1, 1, 1), (1,1,1), 0),
                   nn.PReLU(1, 1.0)
                                   )
        
        self._layers3 = nn.Sequential(
                   PeriodicConvTranspose3d(129, 64, (3,3,2), (1,1,1), 1),
                   nn.BatchNorm3d(64),
                   nn.PReLU(64, 0.02),

                   PeriodicConvTranspose3d(64, 32, (2,2,2), (1,1,1), 1),
                   nn.BatchNorm3d(32),
                   nn.PReLU(32, 0.02),

                #    PeriodicConvTranspose3d(32, 1, (5,3,3), (1,1,1), 0),
                #    PeriodicConv3d(1, 1, (self.z - output_size_z + 1, 1, 1), (1,1,1), 0),
                   nn.PReLU(1, 0.02),
                                   )
        
    def forward(self, x, x0):
                
        x = self._layers(x)
        x0 = self.trans_energy(x0)
        xx0 = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat(1,1,torch.tensor(x.shape[-3:-2]).item(),torch.tensor(x.shape[-2:-1]).item(), torch.tensor(x.shape[-1:]).item())), 1)
        x1 = self._layers2(xx0) #hits
        x2 = self._layers3(xx0)
        return rearrange(x1, "b c l h w -> b (l h w) c"), rearrange(x1, "b c l h w -> b (l h w) c")
    
    def trans_energy(self, x0, log_e_max=14.0, log_e_min=6.0, s_map = 1.0):
        return ((torch.log(x0) - log_e_min)/(log_e_max - log_e_min)) * s_map
    
class Skip(nn.Module):
    def __init__(self, cfg):
        super(Skip, self).__init__()
        self._config = cfg
        self.head_size = self._config.model.head_size
        self.seq = nn.Sequential(
            nn.Unflatten(1, (self._config.rbm.latent_nodes_per_p*2,1,1,1)),
            PeriodicConvTranspose3d(self._config.rbm.latent_nodes_per_p*2, self.head_size,(3,3,3),(1,1,1),0),
        )
        self.query = nn.Linear(27,self._config.model.skip_output_size, bias=False)
        self.value = nn.Linear(27,self._config.model.skip_output_size, bias=False)
        self.linear = nn.Linear(self.head_size, 1, bias=False)

    def forward(self, x, keys):
        x = self.seq(x)
        x = rearrange(x, "b c l h w -> b c (l h w)")
        x_query = self.query(x).transpose(-2,-1)
        x_value = self.value(x).transpose(-2,-1)

        wei = x_query @ keys.transpose(-2,-1) * self.head_size**-0.5
        wei = F.softmax(wei,dim=-1)
        out = self.linear(wei @ x_value).reshape(-1,self._config.model.skip_output_size)

        
        return out
    
class Decoder(nn.Module):
    def __init__(self, cfg, input_size):
        super(Decoder, self).__init__()
        self._config = cfg

        self.n_latent_hierarchy_lvls=self._config.rbm.partitions

        self.n_latent_nodes=self._config.rbm.latent_nodes_per_p * self._config.rbm.partitions

        self.z = self._config.data.z
        self.r = self._config.data.r
        self.phi = self._config.data.phi
        
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
                   nn.PReLU(1, 0.02),
                                   )
        
    def forward(self, x, x0):
                
        x = self._layers(x)
        x0 = self.trans_energy(x0)
        xx0 = torch.cat((x, x0.unsqueeze(2).unsqueeze(3).unsqueeze(4).repeat(1,1,torch.tensor(x.shape[-3:-2]).item(),torch.tensor(x.shape[-2:-1]).item(), torch.tensor(x.shape[-1:]).item())), 1)
        x1 = self._layers2(xx0) #hits
        x2 = self._layers3(xx0)
        return x1.reshape(x1.shape[0],self.z*self.r*self.phi), x2.reshape(x1.shape[0],self.z*self.r*self.phi)
    
    def trans_energy(self, x0, log_e_max=16.0, log_e_min=5.0, s_map = 1.0):
        return ((torch.log(x0) - log_e_min)/(log_e_max - log_e_min)) * s_map