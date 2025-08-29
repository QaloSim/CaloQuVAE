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
from utils.rbm_plots import plot_forward_output_v2, plot_rbm_histogram, plot_rbm_params, plot_forward_output_hidden
# from utils.correlation_plotting import correlation_plots


from collections import defaultdict
from omegaconf import OmegaConf

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
        self.best_val_loss = float("inf")

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
            # self.model.prior.gradient_rbm_stan(output[2])
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

    def fit_rbm(self, epoch):
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
            # self.model.prior.update_params()
            self.model.prior.update_params()

            if (i % log_batch_idx) == 0:
                    logger.info('Epoch: {} [{}/{} ({:.0f}%)]\t beta: {:.3f}, slope: {:.3f} \t Batch Loss: {:.4f}'.format(epoch,
                        i, len(self.data_mgr.train_loader),100.*i/len(self.data_mgr.train_loader),
                        self.beta, self.slope, loss_dict["loss"]))
                    wandb.log(loss_dict)

    def aggr_loss(self, data_loader, epoch, loss_dict=None):
        if loss_dict is not None:
            for key in loss_dict.keys():
                if key not in self.total_loss_dict:
                    self.total_loss_dict[key] = 0.0
                self.total_loss_dict[key] += loss_dict[key].item()
        else:
            for key in self.total_loss_dict.keys():
                self.total_loss_dict[key] /= len(data_loader)
            logger.info("Epoch: {} - Average Val Loss: {:.4f}".format(epoch, self.total_loss_dict["val_loss"]))
            wandb.log(self.total_loss_dict)
            return self.total_loss_dict
    
    def track_best_val_loss(self, loss_dict):
        if self.best_val_loss > loss_dict["val_ae_loss"]: #+ loss_dict["val_hit_loss"]:
            self.best_val_loss = loss_dict["val_ae_loss"] #+ loss_dict["val_hit_loss"]
            self.best_config_path = self._save_model(name="best")
            logger.info("Best Val loss: {:.4f}".format(self.best_val_loss))

    def evaluate_vae(self, data_loader, epoch):
        log_batch_idx = max(len(data_loader)//self._config.engine.n_batches_log_val, 1)
        self.model.eval()
        self.total_loss_dict = {}
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
                prior_samples = self.model.prior.block_gibbs_sampling_cond(p0 = output[2][0])
                _, shower_prior = self.model.decode(prior_samples, x_reduce, x0)
                # Compute loss
                loss_dict = self.model.loss(x_reduce, output)
                loss_dict["loss"] = torch.stack([loss_dict[key] * self._config.model.loss_coeff[key]  for key in self._config.model.loss_coeff if "loss" != key]).sum()
                # loss_dict["loss"] = torch.stack([loss_dict[key] * self._config.model.loss_coeff[key]  for key in loss_dict.keys() if "loss" != key]).sum()
                for key in list(loss_dict.keys()):
                    loss_dict['val_'+key] = loss_dict[key]
                    loss_dict.pop(key)
                
                # Aggregate loss
                self.aggr_loss(data_loader, epoch, loss_dict)
                
                
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
            return self.aggr_loss(data_loader, epoch)

    def evaluate_ae(self, data_loader, epoch):
        log_batch_idx = max(len(data_loader)//self._config.engine.n_batches_log_val, 1)
        self.model.eval()
        self.total_loss_dict = {}
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
                # Compute loss
                loss_dict = self.model.loss(x_reduce, output)
                loss_dict["loss"] = torch.stack([loss_dict[key] * self._config.model.loss_coeff[key]  for key in loss_dict.keys() if "loss" != key]).sum()
                for key in list(loss_dict.keys()):
                    loss_dict['val_'+key] = loss_dict[key]
                    loss_dict.pop(key)
                
                # Aggregate loss
                self.aggr_loss(data_loader, epoch, loss_dict)

                
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
            return self.aggr_loss(data_loader, epoch)
    
    def generate_plots(self, epoch, key):
        if self._config.wandb.mode != "disabled": # Only log if wandb is enabled
            
            # Correlation and Frobenius plots
            # fig_target_corr, fig_sampled_corr, fig_gt_grid, fig_prior_grid, fig_frob_layerwise, fig_gt, fig_prior, gt_spars_corr, 
            #   prior_spars_corr, fig_gt_sparsity, fig_prior_sparsity, fig_gt_sparsity_corr, fig_prior_sparsity_corr, fig_gt_patch, 
            #   fig_prior_patch =correlation_plots(
            #       cfg=self._config,
            #       incident_energy=self.incident_energy,
            #       showers=self.showers,
            #       showers_prior=self.showers_prior,
            #       epoch=epoch
            #   )
           
            # Calorimeter layer plots
            
            calo_input, calo_recon, calo_sampled, calo_input_avg, calo_recon_avg, calo_sampled_avg = plot_calorimeter_shower(
                cfg=self._config,
                showers=self.showers,
                showers_recon=self.showers_recon,
                showers_sampled=self.showers_prior,
                epoch=epoch,
                save_dir=None
            )
            
            # Log plots
            overall_fig, fig_energy_sum, fig_incidence_ratio, fig_target_recon_ratio, fig_sparsity, fig_sum_layers, fig_incidence_layers, fig_ratio_layers, fig_sparsity_layers = vae_plots(self._config,
                self.incident_energy, self.showers, self.showers_recon, self.showers_prior)
            
            if key != "ae":
                rbm_hist = plot_rbm_histogram(self.RBM_energy_post, self.RBM_energy_prior)
                rbm_params = plot_rbm_params(self)
                rbm_floppy = plot_forward_output_v2(self)
            
                wandb.log({
                    "overall_plots": wandb.Image(overall_fig),
                    "conditioned_energy_sum": wandb.Image(fig_energy_sum),
                    "conditioned_incidence_ratio": wandb.Image(fig_incidence_ratio),
                    "conditioned_target_recon_ratio": wandb.Image(fig_target_recon_ratio),
                    "conditioned_sparsity": wandb.Image(fig_sparsity),
                    "energy_sum_layers": wandb.Image(fig_sum_layers),
                    "incidence_ratio_layers": wandb.Image(fig_incidence_layers),
                    "target_recon_ratio_layers": wandb.Image(fig_ratio_layers),
                    "sparsity_layers": wandb.Image(fig_sparsity_layers),
                    "RBM histogram": wandb.Image(rbm_hist),
                    "RBM params": wandb.Image(rbm_params),
                    "RBM floppy": wandb.Image(rbm_floppy),
                    "calo_layer_input": wandb.Image(calo_input),
                    "calo_layer_recon": wandb.Image(calo_recon),
                    "calo_layer_sampled": wandb.Image(calo_sampled),
                    "calo_layer_input_avg": wandb.Image(calo_input_avg),
                    "calo_layer_recon_avg": wandb.Image(calo_recon_avg),
                    "calo_layer_sampled_avg": wandb.Image(calo_sampled_avg),
                    # "layer_energy_correlation_GT": wandb.Image(fig_target_corr),
                    # "layer_energy_correlation_sampled": wandb.Image(fig_sampled_corr),
                    # "frob_layerwise_GT_vs_sampled": wandb.Image(fig_frob_layerwise),
                    # "sparsity_GT": wandb.Image(fig_gt_sparsity),
                    # "sparsity_prior": wandb.Image(fig_prior_sparsity),
                    # "sparsity_correlation_GT": wandb.Image(fig_gt_sparsity_corr),
                    # "sparsity_correlation_prior": wandb.Image(fig_prior_sparsity_corr),
                    # "patch_corr_GT": wandb.Image(fig_gt_patch),
                    # "patch_corr_prior": wandb.Image(fig_prior_patch),
                    # "voxel_corr_GT": wandb.Image(fig_gt),
                    # "voxel_corr_prior": wandb.Image(fig_prior),
                })
            else:
                wandb.log({
                    "overall_plots": wandb.Image(overall_fig),
                    "conditioned_energy_sum": wandb.Image(fig_energy_sum),
                    "conditioned_incidence_ratio": wandb.Image(fig_incidence_ratio),
                    "conditioned_target_recon_ratio": wandb.Image(fig_target_recon_ratio),
                    "conditioned_sparsity": wandb.Image(fig_sparsity),
                    "energy_sum_layers": wandb.Image(fig_sum_layers),
                    "incidence_ratio_layers": wandb.Image(fig_incidence_layers),
                    "target_recon_ratio_layers": wandb.Image(fig_ratio_layers),
                    "sparsity_layers": wandb.Image(fig_sparsity_layers),
                    "calo_layer_input": wandb.Image(calo_input),
                    "calo_layer_recon": wandb.Image(calo_recon),
                    "calo_layer_input_avg": wandb.Image(calo_input_avg),
                    "calo_layer_recon_avg": wandb.Image(calo_recon_avg),
                    # "layer_energy_correlation_GT": wandb.Image(fig_target_corr),
                    # "sparsity_GT": wandb.Image(fig_gt_sparsity),
                    # "sparsity_correlation_GT": wandb.Image(fig_gt_sparsity_corr),
                    # "patch_corr_GT": wandb.Image(fig_gt_patch),
                    # "voxel_corr_GT": wandb.Image(fig_gt),
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
        config_path = self._model_creator.save_state(config_string, vae_opt=self.optimiser, rbm_opt=self.model.prior.opt)
        return config_path
    
    def load_best_model(self, epoch):
        best_config = OmegaConf.load(self.best_config_path)
        self._model_creator.load_state(best_config.run_path, self.device, vae_opt=self.optimiser, rbm_opt=self.model.prior.opt)
        self._config.epoch_start = epoch + 1
    
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
    
class EngineHidden(Engine):
    def __init__(self, cfg=None, **kwargs):
        # Use cooperative initialization so Engine.__init__ runs properly
        super().__init__(cfg, **kwargs)

        # child can override or extend any attributes set by Engine
        self._config = cfg
        self.beta = self._config.engine.beta_gumbel_start
        self.slope = self._config.engine.slope_act_fct_start 
        
        self._model = None
        self._optimiser = None
        self._optimiser_c = None
        self._data_mgr = None
        self._device = None
        self.best_val_loss = float("inf")

    def evaluate_vae(self, data_loader, epoch):
        log_batch_idx = max(len(data_loader)//self._config.engine.n_batches_log_val, 1)
        self.model.eval()
        self.total_loss_dict = {}
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
            self.post_samples = torch.zeros((ar_size, ar_latent_size * 3), dtype=torch.float32)
            self.post_logits = torch.zeros((ar_size, ar_latent_size * 2), dtype=torch.float32)

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
                post_sample_hidden = self.model.prior.sigmoid_C_k(self.model.prior.weight_dict['03'],   self.model.prior.weight_dict['13'],   
                                    self.model.prior.weight_dict['23'], output[2][0], output[2][1], output[2][2], self.model.prior.bias_dict['3'])
                # Get prior samples
                prior_samples = self.model.prior.block_gibbs_sampling_cond(p0 = output[2][0])
                _, shower_prior = self.model.decode(prior_samples[:-1], x_reduce, x0)
                # Compute loss
                loss_dict = self.model.loss(x_reduce, output)
                loss_dict["loss"] = torch.stack([loss_dict[key] * self._config.model.loss_coeff[key]  for key in loss_dict.keys() if "loss" != key]).sum()
                for key in list(loss_dict.keys()):
                    loss_dict['val_'+key] = loss_dict[key]
                    loss_dict.pop(key)
                
                # Aggregate loss
                self.aggr_loss(data_loader, epoch, loss_dict)
                
                
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
                self.RBM_energy_post[idx1:idx2,:] = self.model.prior.energy_exp_cond(output[2][0], output[2][1], output[2][2], post_sample_hidden).cpu().unsqueeze(1)
            
            # Log average loss after loop
            return self.aggr_loss(data_loader, epoch)
        
    def evaluate_ae(self, data_loader, epoch):
            log_batch_idx = max(len(data_loader) // self._config.engine.n_batches_log_val, 1)
            self.model.eval()
            self.total_loss_dict = {}
            with torch.no_grad():
                bs = [batch[0].shape[0] for batch in data_loader]
                ar_size = np.sum(bs)
                ar_input_size = self._config.data.z * self._config.data.r * self._config.data.phi
                ar_latent_size = self._config.rbm.latent_nodes_per_p

                # Initialize tensors for storing evaluation data
                self.incident_energy = torch.zeros((ar_size, 1), dtype=torch.float32)
                self.showers = torch.zeros((ar_size, ar_input_size), dtype=torch.float32)
                self.showers_reduce = torch.zeros((ar_size, ar_input_size), dtype=torch.float32)
                self.showers_recon = torch.zeros((ar_size, ar_input_size), dtype=torch.float32)
                self.showers_reduce_recon = torch.zeros((ar_size, ar_input_size), dtype=torch.float32)
                
                # **Difference**: Initialize tensors to match the 3-component latent output of EngineHidden's model
                self.post_samples = torch.zeros((ar_size, ar_latent_size * 3), dtype=torch.float32)
                self.post_logits = torch.zeros((ar_size, ar_latent_size * 2), dtype=torch.float32)

                # Initialize tensors for "prior" plots (using reconstructions in AE mode)
                self.showers_prior = torch.zeros((ar_size, ar_input_size), dtype=torch.float32)
                self.showers_reduce_prior = torch.zeros((ar_size, ar_input_size), dtype=torch.float32)
                # **Difference**: Match the dimensions of post_samples
                self.prior_samples = torch.zeros((ar_size, ar_latent_size * 3), dtype=torch.float32)
                
                # RBM energy tensors are not needed for AE evaluation

                for i, (x, x0) in enumerate(data_loader):
                    x = x.to(self.device)
                    x0 = x0.to(self.device)
                    x_reduce = self._reduce(x, x0)
                    
                    # Forward pass
                    output = self.model((x_reduce, x0))
                    
                    # Compute and format loss
                    loss_dict = self.model.loss(x_reduce, output)
                    loss_dict["loss"] = torch.stack([loss_dict[key] * self._config.model.loss_coeff[key] for key in loss_dict.keys() if "loss" != key]).sum()
                    for key in list(loss_dict.keys()):
                        loss_dict['val_' + key] = loss_dict[key]
                        loss_dict.pop(key)
                    
                    # Aggregate loss
                    self.aggr_loss(data_loader, epoch, loss_dict)

                    # Store results
                    idx1, idx2 = int(np.sum(bs[:i])), int(np.sum(bs[:i + 1]))
                    self.incident_energy[idx1:idx2, :] = x0.cpu()
                    self.showers[idx1:idx2, :] = x.cpu()
                    self.showers_reduce[idx1:idx2, :] = x_reduce.cpu()
                    self.showers_recon[idx1:idx2, :] = self._reduceinv(output[3], x0).cpu()
                    self.showers_reduce_recon[idx1:idx2, :] = output[3].cpu()
                    
                    # **Difference**: Concatenate the 3 latent sample tensors from output[2]
                    self.post_samples[idx1:idx2, :] = torch.cat(output[2], dim=1).cpu()
                    self.post_logits[idx1:idx2, :] = torch.cat(output[1], dim=1).cpu()

                    # Use reconstruction as "prior" for plotting purposes
                    self.prior_samples[idx1:idx2, :] = torch.cat(output[2], dim=1).cpu()
                    self.showers_prior[idx1:idx2, :] = self._reduceinv(output[3], x0).cpu()
                    self.showers_reduce_prior[idx1:idx2, :] = output[3].cpu()

                # Log average loss after iterating through all batches
                return self.aggr_loss(data_loader, epoch)        

    def generate_plots(self, epoch, key):
        if self._config.wandb.mode != "disabled": # Only log if wandb is enabled
            
            # Correlation and Frobenius plots
            # fig_target_corr, fig_sampled_corr, fig_gt_grid, fig_prior_grid, fig_frob_layerwise, fig_gt, fig_prior, gt_spars_corr, 
            #   prior_spars_corr, fig_gt_sparsity, fig_prior_sparsity, fig_gt_sparsity_corr, fig_prior_sparsity_corr, fig_gt_patch, 
            #   fig_prior_patch =correlation_plots(
            #       cfg=self._config,
            #       incident_energy=self.incident_energy,
            #       showers=self.showers,
            #       showers_prior=self.showers_prior,
            #       epoch=epoch
            #   )
           
            # Calorimeter layer plots
            
            calo_input, calo_recon, calo_sampled, calo_input_avg, calo_recon_avg, calo_sampled_avg = plot_calorimeter_shower(
                cfg=self._config,
                showers=self.showers,
                showers_recon=self.showers_recon,
                showers_sampled=self.showers_prior,
                epoch=epoch,
                save_dir=None
            )
            
            # Log plots
            overall_fig, fig_energy_sum, fig_incidence_ratio, fig_target_recon_ratio, fig_sparsity, fig_sum_layers, fig_incidence_layers, fig_ratio_layers, fig_sparsity_layers = vae_plots(self._config,
                self.incident_energy, self.showers, self.showers_recon, self.showers_prior)
            
            if key != "ae":
                rbm_hist = plot_rbm_histogram(self.RBM_energy_post, self.RBM_energy_prior)
                rbm_params = plot_rbm_params(self)
                rbm_floppy = plot_forward_output_hidden(self)
            
                wandb.log({
                    "overall_plots": wandb.Image(overall_fig),
                    "conditioned_energy_sum": wandb.Image(fig_energy_sum),
                    "conditioned_incidence_ratio": wandb.Image(fig_incidence_ratio),
                    "conditioned_target_recon_ratio": wandb.Image(fig_target_recon_ratio),
                    "conditioned_sparsity": wandb.Image(fig_sparsity),
                    "energy_sum_layers": wandb.Image(fig_sum_layers),
                    "incidence_ratio_layers": wandb.Image(fig_incidence_layers),
                    "target_recon_ratio_layers": wandb.Image(fig_ratio_layers),
                    "sparsity_layers": wandb.Image(fig_sparsity_layers),
                    "RBM histogram": wandb.Image(rbm_hist),
                    "RBM params": wandb.Image(rbm_params),
                    "RBM floppy": wandb.Image(rbm_floppy),
                    "calo_layer_input": wandb.Image(calo_input),
                    "calo_layer_recon": wandb.Image(calo_recon),
                    "calo_layer_sampled": wandb.Image(calo_sampled),
                    "calo_layer_input_avg": wandb.Image(calo_input_avg),
                    "calo_layer_recon_avg": wandb.Image(calo_recon_avg),
                    "calo_layer_sampled_avg": wandb.Image(calo_sampled_avg),
                    # "layer_energy_correlation_GT": wandb.Image(fig_target_corr),
                    # "layer_energy_correlation_sampled": wandb.Image(fig_sampled_corr),
                    # "frob_layerwise_GT_vs_sampled": wandb.Image(fig_frob_layerwise),
                    # "sparsity_GT": wandb.Image(fig_gt_sparsity),
                    # "sparsity_prior": wandb.Image(fig_prior_sparsity),
                    # "sparsity_correlation_GT": wandb.Image(fig_gt_sparsity_corr),
                    # "sparsity_correlation_prior": wandb.Image(fig_prior_sparsity_corr),
                    # "patch_corr_GT": wandb.Image(fig_gt_patch),
                    # "patch_corr_prior": wandb.Image(fig_prior_patch),
                    # "voxel_corr_GT": wandb.Image(fig_gt),
                    # "voxel_corr_prior": wandb.Image(fig_prior),
                })
            else:
                wandb.log({
                    "overall_plots": wandb.Image(overall_fig),
                    "conditioned_energy_sum": wandb.Image(fig_energy_sum),
                    "conditioned_incidence_ratio": wandb.Image(fig_incidence_ratio),
                    "conditioned_target_recon_ratio": wandb.Image(fig_target_recon_ratio),
                    "conditioned_sparsity": wandb.Image(fig_sparsity),
                    "energy_sum_layers": wandb.Image(fig_sum_layers),
                    "incidence_ratio_layers": wandb.Image(fig_incidence_layers),
                    "target_recon_ratio_layers": wandb.Image(fig_ratio_layers),
                    "sparsity_layers": wandb.Image(fig_sparsity_layers),
                    "calo_layer_input": wandb.Image(calo_input),
                    "calo_layer_recon": wandb.Image(calo_recon),
                    "calo_layer_input_avg": wandb.Image(calo_input_avg),
                    "calo_layer_recon_avg": wandb.Image(calo_recon_avg),
                    # "layer_energy_correlation_GT": wandb.Image(fig_target_corr),
                    # "sparsity_GT": wandb.Image(fig_gt_sparsity),
                    # "sparsity_correlation_GT": wandb.Image(fig_gt_sparsity_corr),
                    # "patch_corr_GT": wandb.Image(fig_gt_patch),
                    # "voxel_corr_GT": wandb.Image(fig_gt),
                })