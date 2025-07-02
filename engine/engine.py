"""
Base Class of Engines. Defines properties and methods.
"""

import torch
import numpy as np

# Weights and Biases
import wandb

from CaloQuVAE import logging
logger = logging.getLogger(__name__)

class Engine():

    def __init__(self, cfg=None, **kwargs):
        super(Engine,self).__init__()

        self._config = cfg
        
        self._model = None
        self._optimiser = None
        self._optimiser_c = None
        self._data_mgr = None
        self._device = None

    @property
    def model(self):
        return self._model
    
    @model.setter   
    def model(self,model):
        self._model=model
        
    @property
    def critic(self):
        return self._critic
    
    @critic.setter   
    def critic(self,critic):
        self._critic=critic
        
    @property
    def critic_2(self):
        return self._critic_2
    
    @critic_2.setter   
    def critic_2(self,critic):
        self._critic_2=critic

    @property
    def optimiser(self):
        return self._optimiser
    
    @optimiser.setter   
    def optimiser(self,optimiser):
        self._optimiser=optimiser
        
    @property
    def optimiser_c(self):
        return self._optimiser_c
    
    @optimiser_c.setter   
    def optimiser_c(self,optimiser_c):
        self._optimiser_c=optimiser_c
        
    @property
    def optimiser_c_2(self):
        return self._optimiser_c_2
    
    @optimiser_c_2.setter   
    def optimiser_c_2(self,optimiser_c):
        self._optimiser_c_2=optimiser_c
    
    @property
    def data_mgr(self):
        return self._data_mgr
    
    @data_mgr.setter   
    def data_mgr(self,data_mgr):
        assert data_mgr is not None, "Empty Data Manager"
        self._data_mgr=data_mgr
        
    @property
    def device(self):
        return self._device
    
    @device.setter
    def device(self, device):
        self._device=device
    
    def generate_samples(self):
        raise NotImplementedError

    def fit(self, epoch):
        log_batch_idx = max(len(self.data_mgr.train_loader)//self._config.engine.n_batches_log_train, 1)
        self.model.train()
        for i, (x, x0) in enumerate(self.data_mgr.train_loader):
            x = x.to(self.device).to(dtype=torch.float32)
            x0 = x0.to(self.device).to(dtype=torch.float32)
            x = self._reduce(x, x0)
            # Forward pass
            output = self.model((x, x0))
            # Compute loss
            loss_dict = self.model.loss(x, output)
            loss_dict["loss"] = loss_dict["ae_loss"] + \
                loss_dict["kl_loss"] + loss_dict["hit_loss"]
            # Backward pass and optimization
            self.optimiser.zero_grad()
            loss_dict["loss"].backward()
            self.optimiser.step()

            if (i % log_batch_idx) == 0:
                    logger.info('Epoch: {} [{}/{} ({:.0f}%)]\t Batch Loss: {:.4f}'.format(epoch,
                        i, len(self.data_mgr.train_loader),100.*i/len(self.data_mgr.train_loader),
                        loss_dict["loss"]))
                    wandb.log(loss_dict)
            if i == 0:
                break
    
    def evaluate(self, data_loader, epoch):
        log_batch_idx = max(len(data_loader)//self._config.engine.n_batches_log_val, 1)
        self.model.eval()
        with torch.no_grad():
            bs = [batch[0].shape[0] for batch in data_loader]
            ar_size = np.sum(bs)
            ar_input_size = self._config.data.z * self._config.data.r * self._config.data.phi
            ar_latent_size = self._config.rbm.latent_nodes_per_p
            
            incident_energy = torch.zeros((ar_size, 1), dtype=torch.float32)
            showers = torch.zeros((ar_size, ar_input_size), dtype=torch.float32)
            showers_reduce = torch.zeros((ar_size, ar_input_size), dtype=torch.float32)
            showers_recon = torch.zeros((ar_size, ar_input_size), dtype=torch.float32)
            showers_reduce_recon = torch.zeros((ar_size, ar_input_size), dtype=torch.float32)
            post_samples = torch.zeros((ar_size, ar_latent_size * 4), dtype=torch.float32)
            post_logits = torch.zeros((ar_size, ar_latent_size * 3), dtype=torch.float32)
            prior_samples = torch.zeros((ar_size, ar_latent_size * 4), dtype=torch.float32)

            for i, (x, x0) in enumerate(data_loader):
                x = x.to(self.device).to(dtype=torch.float32)
                x0 = x0.to(self.device).to(dtype=torch.float32)
                x_reduce = self._reduce(x, x0)
                # Forward pass
                output = self.model((x_reduce, x0))
                # Compute loss
                loss_dict = self.model.loss(x_reduce, output)
                loss_dict["loss"] = loss_dict["ae_loss"] + \
                    loss_dict["kl_loss"] + loss_dict["hit_loss"]
                
                idx1, idx2 = int(np.sum(bs[:i])), int(np.sum(bs[:i+1]))
                incident_energy[idx1:idx2,:] = x0.cpu()
                showers[idx1:idx2,:] = x.cpu()
                showers_reduce[idx1:idx2,:] = x_reduce.cpu()
                showers_recon[idx1:idx2,:] = self._reduceinv(output[3], x0).cpu()
                showers_reduce_recon[idx1:idx2,:] = output[3].cpu()
                post_samples[idx1:idx2,:] = torch.cat(output[2],dim=1).cpu()
                post_logits[idx1:idx2,:] = torch.cat(output[1],dim=1).cpu()
                prior_samples[idx1:idx2,:] = torch.cat(self.model.prior.block_gibbs_sampling_cond(p0 = output[2][0]),dim=1).cpu()

                if (i % log_batch_idx) == 0:
                        logger.info('Epoch: {} [{}/{} ({:.0f}%)]\t Batch Loss: {:.4f}'.format(epoch,
                            i, len(data_loader),100.*i/len(data_loader),
                            loss_dict["loss"]))
                        wandb.log(loss_dict)
                if i == 0:
                    break
    
    @property
    def model_creator(self):
        return self._model_creator
    
    @model_creator.setter
    def model_creator(self, model_creator):
        assert model_creator is not None
        self._model_creator = model_creator

    def _save_model(self, name="blank"):
        config_string = "_".join(str(i) for i in [self._config.model.model_name,f'{name}'])
        self._model_creator.save_state(config_string)

    def _reduce(self, in_data, true_energy, R=1e-7):
        """
        CaloDiff Transformation Scheme
        """
        ϵ = in_data/true_energy #*self.e_scale
        x = R + (1-2*R)*ϵ
        u = torch.log(x*(1-R)/(R*(1-x)))
        return u

        
    def _reduceinv(self, in_data, true_energy, R=1e-7):
        """
        CaloDiff Transformation Scheme
        """
        
        x = (torch.sigmoid(in_data + torch.log(torch.tensor([R/(1-R)]).to(in_data.device)) ) - R)/(1-2*R) * true_energy 
        x[torch.isclose(x, torch.tensor([0]).to(dtype=x.dtype, device=x.device)) ] = 0.0
        
        return x