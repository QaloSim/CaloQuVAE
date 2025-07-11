"""
Base Class of Engines. Defines properties and methods.
"""

import torch
import numpy as np

# Weights and Biases
import wandb

# Plotting
from utils.plots import vae_plots
from utils.atlas_plots import plot_calorimeter_shower
from utils.rbm_plots import plot_rbm_histogram

from collections import defaultdict

from CaloQuVAE import logging
logger = logging.getLogger(__name__)

class Engine():

    def __init__(self, cfg=None, **kwargs):
        super(Engine,self).__init__()

        self._config = cfg
        self.beta = self._config.engine.beta_gumbel_start
        self.slope = self._config.engine.slope_act_fct_start 
        
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
    
    def _anneal_params(self, num_batches, batch_idx, epoch):
        if epoch > self._config.engine.beta_gumbel_epoch_start:
            delta_beta = self._config.engine.beta_gumbel_end - self._config.engine.beta_gumbel_start
            delta_slope = 0.0 - self._config.engine.slope_act_fct_start

            delta = (self._config.engine.beta_gumbel_epoch_end - self._config.engine.beta_gumbel_epoch_start)*num_batches

            self.beta = min(self._config.engine.beta_gumbel_start + delta_beta/delta * ((epoch-1)*num_batches + batch_idx), self._config.engine.beta_gumbel_end)
            self.slope = max(self._config.engine.slope_act_fct_start + delta_slope/delta * ((epoch-1)*num_batches + batch_idx), 0.0)

    def fit_vae(self, epoch):
        log_batch_idx = max(len(self.data_mgr.train_loader)//self._config.engine.n_batches_log_train, 1)
        self.model.train()
        for i, (x, x0) in enumerate(self.data_mgr.train_loader):
            # Anneal parameters
            self._anneal_params(len(self.data_mgr.train_loader), i, epoch)
            x = x.to(self.device)
            x0 = x0.to(self.device)
            x = self._reduce(x, x0)
            # Forward pass
            output = self.model((x, x0), self.beta, self.slope)
            # Compute loss
            loss_dict = self.model.loss(x, output)
            loss_dict["loss"] = torch.stack([loss_dict[key] * self._config.model.loss_coeff[key]  for key in loss_dict.keys() if "loss" != key]).sum()
            self.model.prior.gradient_rbm_centered(output[2])
            self.model.prior.update_params()
            
            # Backward pass and optimization
            self.optimiser.zero_grad()
            loss_dict["loss"].backward()
            self.optimiser.step()

            if (i % log_batch_idx) == 0:
                    logger.info('Epoch: {} [{}/{} ({:.0f}%)]\t beta: {:.3f}, slope: {:.3f} \t Batch Loss: {:.4f}'.format(epoch,
                        i, len(self.data_mgr.train_loader),100.*i/len(self.data_mgr.train_loader),
                        self.beta, self.slope, loss_dict["loss"]))
                    wandb.log(loss_dict)

                    
    def fit_ae(self, epoch):
        log_batch_idx = max(len(self.data_mgr.train_loader)//self._config.engine.n_batches_log_train, 1)
        self.model.train()
        for i, (x, x0) in enumerate(self.data_mgr.train_loader):
            # Anneal parameters
            self._anneal_params(len(self.data_mgr.train_loader), i, epoch)
            x = x.to(self.device).to(dtype=torch.float32)
            x0 = x0.to(self.device).to(dtype=torch.float32)
            x = self._reduce(x, x0)
            # Forward pass
            output = self.model((x, x0), self.beta, self.slope)
            # Compute loss
            loss_dict = self.model.loss(x, output)
            loss_dict["loss"] = torch.stack([loss_dict[key] * self._config.model.loss_coeff[key]  for key in loss_dict.keys() if "loss" != key]).sum()
            
            # Backward pass and optimization
            self.optimiser.zero_grad()
            loss_dict["loss"].backward()
            self.optimiser.step()

            if (i % log_batch_idx) == 0:
                    logger.info('Epoch: {} [{}/{} ({:.0f}%)]\t beta: {:.3f}, slope: {:.3f} \t Batch Loss: {:.4f}'.format(epoch,
                        i, len(self.data_mgr.train_loader),100.*i/len(self.data_mgr.train_loader),
                        self.beta, self.slope, loss_dict["loss"]))
                    wandb.log(loss_dict)
    
    def evaluate_vae(self, data_loader, epoch):
        log_batch_idx = max(len(data_loader)//self._config.engine.n_batches_log_val, 1)
        self.model.eval()
        total_loss_dict = {}
        with torch.no_grad():
            bs = [batch[0].shape[0] for batch in data_loader]
            ar_size = np.sum(bs)
            ar_input_size = self._config.data.z * self._config.data.r * self._config.data.phi
            ar_latent_size = self._config.rbm.latent_nodes_per_p
            
            self.incident_energy = torch.zeros((ar_size, 1), dtype=torch.float32)
            self.showers = torch.zeros((ar_size, ar_input_size), dtype=torch.float32)
            self.showers_reduce = torch.zeros((ar_size, ar_input_size), dtype=torch.float32)
            self.showers_recon = torch.zeros((ar_size, ar_input_size), dtype=torch.float32)
            self.showers_reduce_recon = torch.zeros((ar_size, ar_input_size), dtype=torch.float32)
            self.post_samples = torch.zeros((ar_size, ar_latent_size * 4), dtype=torch.float32)
            self.post_logits = torch.zeros((ar_size, ar_latent_size * 3), dtype=torch.float32)

            self.showers_prior = torch.zeros((ar_size, ar_input_size), dtype=torch.float32)
            self.showers_reduce_prior = torch.zeros((ar_size, ar_input_size), dtype=torch.float32)
            self.prior_samples = torch.zeros((ar_size, ar_latent_size * 4), dtype=torch.float32)
            self.RBM_energy_prior = torch.zeros((ar_size, 1), dtype=torch.float32)
            self.RBM_energy_post = torch.zeros((ar_size, 1), dtype=torch.float32)

            for i, (x, x0) in enumerate(data_loader):
                x = x.to(self.device)
                x0 = x0.to(self.device)
                x_reduce = self._reduce(x, x0)
                # Forward pass
                output = self.model((x_reduce, x0))
                # Get prior samples
                if not self._config.engine.train_vae_separate:
                    prior_samples = self.model.prior.block_gibbs_sampling_cond(p0 = output[2][0])
                    _, shower_prior = self.model.decode(prior_samples, x_reduce, x0)
                # Compute loss
                loss_dict = self.model.loss(x_reduce, output)
                loss_dict["loss"] = torch.stack([loss_dict[key] * self._config.model.loss_coeff[key]  for key in loss_dict.keys() if "loss" != key]).sum()
                for key in list(loss_dict.keys()):
                    loss_dict['val_'+key] = loss_dict[key]
                    loss_dict.pop(key)
                
                # Aggregate loss
                for key in loss_dict.keys():
                    if key not in total_loss_dict:
                        total_loss_dict[key] = 0.0
                    total_loss_dict[key] += loss_dict[key].item()
                
                
                idx1, idx2 = int(np.sum(bs[:i])), int(np.sum(bs[:i+1]))
                self.incident_energy[idx1:idx2,:] = x0.cpu()
                self.showers[idx1:idx2,:] = x.cpu()
                self.showers_reduce[idx1:idx2,:] = x_reduce.cpu()
                self.showers_recon[idx1:idx2,:] = self._reduceinv(output[3], x0).cpu()
                self.showers_reduce_recon[idx1:idx2,:] = output[3].cpu()
                self.post_samples[idx1:idx2,:] = torch.cat(output[2],dim=1).cpu()
                self.post_logits[idx1:idx2,:] = torch.cat(output[1],dim=1).cpu()
                self.prior_samples[idx1:idx2,:] = torch.cat(prior_samples,dim=1).cpu()
                self.showers_prior[idx1:idx2,:] = self._reduceinv(shower_prior, x0).cpu()
                self.showers_reduce_prior[idx1:idx2,:] = shower_prior.cpu()
                self.RBM_energy_prior[idx1:idx2,:] = self.model.prior.energy_exp_cond(prior_samples[0], prior_samples[1], prior_samples[2], prior_samples[3]).cpu().unsqueeze(1)
                self.RBM_energy_post[idx1:idx2,:] = self.model.prior.energy_exp_cond(output[2][0], output[2][1], output[2][2], output[2][3]).cpu().unsqueeze(1)
            
            # Log average loss after loop
            num_batches = len(data_loader)
            avg_loss_dict = {key: value / num_batches for key, value in total_loss_dict.items()}
            logger.info("Epoch: {} - Average Validation Loss: {:.4f}".format(epoch, avg_loss_dict["val_loss"]))
            self.generate_plots(epoch)

    def evaluate_ae(self, data_loader, epoch):
        log_batch_idx = max(len(data_loader)//self._config.engine.n_batches_log_val, 1)
        self.model.eval()
        total_loss_dict = {}
        with torch.no_grad():
            bs = [batch[0].shape[0] for batch in data_loader]
            ar_size = np.sum(bs)
            ar_input_size = self._config.data.z * self._config.data.r * self._config.data.phi
            ar_latent_size = self._config.rbm.latent_nodes_per_p
            
            self.incident_energy = torch.zeros((ar_size, 1), dtype=torch.float32)
            self.showers = torch.zeros((ar_size, ar_input_size), dtype=torch.float32)
            self.showers_reduce = torch.zeros((ar_size, ar_input_size), dtype=torch.float32)
            self.showers_recon = torch.zeros((ar_size, ar_input_size), dtype=torch.float32)
            self.showers_reduce_recon = torch.zeros((ar_size, ar_input_size), dtype=torch.float32)
            self.post_samples = torch.zeros((ar_size, ar_latent_size * 4), dtype=torch.float32)
            self.post_logits = torch.zeros((ar_size, ar_latent_size * 3), dtype=torch.float32)

            self.showers_prior = torch.zeros((ar_size, ar_input_size), dtype=torch.float32)
            self.showers_reduce_prior = torch.zeros((ar_size, ar_input_size), dtype=torch.float32)
            self.prior_samples = torch.zeros((ar_size, ar_latent_size * 4), dtype=torch.float32)
            self.RBM_energy_prior = torch.zeros((ar_size, 1), dtype=torch.float32)
            self.RBM_energy_post = torch.zeros((ar_size, 1), dtype=torch.float32)

            for i, (x, x0) in enumerate(data_loader):
                x = x.to(self.device)
                x0 = x0.to(self.device)
                x_reduce = self._reduce(x, x0)
                # Forward pass
                output = self.model((x_reduce, x0))
                # Get prior samples
                if not self._config.engine.train_vae_separate:
                    prior_samples = self.model.prior.block_gibbs_sampling_cond(p0 = output[2][0])
                    _, shower_prior = self.model.decode(prior_samples, x_reduce, x0)
                # Compute loss
                loss_dict = self.model.loss(x_reduce, output)
                loss_dict["loss"] = torch.stack([loss_dict[key] * self._config.model.loss_coeff[key]  for key in loss_dict.keys() if "loss" != key]).sum()
                for key in list(loss_dict.keys()):
                    loss_dict['val_'+key] = loss_dict[key]
                    loss_dict.pop(key)
                
                # Aggregate loss
                for key in loss_dict.keys():
                    if key not in total_loss_dict:
                        total_loss_dict[key] = 0.0
                    total_loss_dict[key] += loss_dict[key].item()

                
                idx1, idx2 = int(np.sum(bs[:i])), int(np.sum(bs[:i+1]))
                self.incident_energy[idx1:idx2,:] = x0.cpu()
                self.showers[idx1:idx2,:] = x.cpu()
                self.showers_reduce[idx1:idx2,:] = x_reduce.cpu()
                self.showers_recon[idx1:idx2,:] = self._reduceinv(output[3], x0).cpu()
                self.showers_reduce_recon[idx1:idx2,:] = output[3].cpu()
                self.post_samples[idx1:idx2,:] = torch.cat(output[2],dim=1).cpu()
                self.post_logits[idx1:idx2,:] = torch.cat(output[1],dim=1).cpu()

    
                # Use recon as prior
                self.prior_samples[idx1:idx2,:] = torch.cat(output[2],dim=1).cpu()
                self.showers_prior[idx1:idx2,:] = self._reduceinv(output[3], x0).cpu()
                self.showers_reduce_prior[idx1:idx2,:] = output[3].cpu()
            
            # Log average loss after loop
            num_batches = len(data_loader)
            avg_loss_dict = {key: value / num_batches for key, value in total_loss_dict.items()}
            logger.info("Epoch: {} - Average Validation Loss: {:.4f}".format(epoch, avg_loss_dict["val_loss"]))
            wandb.log(avg_loss_dict)

            self.generate_plots(epoch)
    
    def generate_plots(self, epoch):
        if self._config.wandb.mode != "disabled": # Only log if wandb is enabled
            # Calorimeter layer plots

            calo_input, calo_recon, calo_sampled = plot_calorimeter_shower(
                cfg=self._config,
                showers=self.showers,
                showers_recon=self.showers_recon,
                showers_sampled=self.showers_prior,
                incident_energy=self.incident_energy,
                epoch=epoch,
                save_dir=None
            )
            
            # Log plots
            overall_fig, fig_energy_sum, fig_incidence_ratio, fig_target_recon_ratio, fig_sparsity = vae_plots(
                self.incident_energy, self.showers, self.showers_recon, self.showers_prior)
            
            if not self._config.engine.train_vae_separate:
                rbm_hist = plot_rbm_histogram(self.RBM_energy_post, self.RBM_energy_prior)
            
                wandb.log({
                    "overall_plots": wandb.Image(overall_fig),
                    "conditioned_energy_sum": wandb.Image(fig_energy_sum),
                    "conditioned_incidence_ratio": wandb.Image(fig_incidence_ratio),
                    "conditioned_target_recon_ratio": wandb.Image(fig_target_recon_ratio),
                    "conditioned_sparsity": wandb.Image(fig_sparsity),
                    "RBM histogram": wandb.Image(rbm_hist),
                    "calo_layer_input": wandb.Image(calo_input),
                    "calo_layer_recon": wandb.Image(calo_recon),
                    "calo_layer_sampled": wandb.Image(calo_sampled)       
                })
            else:
                wandb.log({
                    "overall_plots": wandb.Image(overall_fig),
                    "conditioned_energy_sum": wandb.Image(fig_energy_sum),
                    "conditioned_incidence_ratio": wandb.Image(fig_incidence_ratio),
                    "conditioned_target_recon_ratio": wandb.Image(fig_target_recon_ratio),
                    "conditioned_sparsity": wandb.Image(fig_sparsity),
                    "calo_layer_input": wandb.Image(calo_input),
                    "calo_layer_recon": wandb.Image(calo_recon)
                })

    
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
        # Add zero mask to avoid numerical instability when multiplying by large incidence energies
        zero_mask = (in_data == 0.0)
        
        x = (torch.sigmoid(in_data + torch.log(torch.tensor([R/(1-R)]).to(in_data.device)) ) - R)/(1-2*R) * true_energy 
        x[zero_mask] = 0.0
        x[torch.isclose(x, torch.tensor([0]).to(dtype=x.dtype, device=x.device)) ] = 0.0
        
        return x