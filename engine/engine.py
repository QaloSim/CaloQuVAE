"""
Base Class of Engines. Defines properties and methods.
"""

import torch
import numpy as np

# Weights and Biases
import wandb

# Plotting
from utils.plots import vae_plots, corr_plots
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

            if hasattr(self.model.prior, "step_on_batch") and callable(getattr(self.model.prior, "step_on_batch")):
                self.model.prior.step_on_batch([sample.detach() for sample in output[2]])
            else:
                self.model.prior.gradient_rbm_centered(output[2])
                self.model.prior.update_params()
            # self.model.prior.update_params_SGD()

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
            with torch.no_grad():
                # Forward pass
                output = self.model((x, x0), self.beta, self.slope)
                # Compute loss
                loss_dict = self.model.loss(x, output)
                loss_dict["loss"] = torch.stack([loss_dict[key] * self._config.model.loss_coeff[key]  for key in loss_dict.keys() if "loss" != key]).sum()
            if hasattr(self.model.prior, "step_on_batch") and callable(getattr(self.model.prior, "step_on_batch")):
                self.model.prior.step_on_batch([sample.detach() for sample in output[2]])
            else:
                if hasattr(self._config.rbm, "no_weights") and self._config.rbm.no_weights:
                    self.model.prior.gradient_rbm(output[2])
                else:
                    self.model.prior.gradient_rbm_centered(output[2])
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
    
    def track_best_val_loss(self, loss_dict, chi2, epoch=None):
        if self.best_val_loss > loss_dict["val_ae_loss"] + chi2*10: #+ loss_dict["val_hit_loss"]:
            self.best_val_loss = loss_dict["val_ae_loss"] + chi2*10 #+ loss_dict["val_hit_loss"]
            self.best_config_path = self._save_model(name="best"+(f"_epoch{epoch}" if epoch is not None else ""))
            logger.info("Best Val loss plus chi2: {:.4f}".format(self.best_val_loss+ chi2*10))

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

    def generate_showers_from_rbm(self, rbm_samples, incident_energies):
        """
        Generates showers by passing externally provided RBM samples through the decoder.
        The first partition of the latent space is replaced by the encoded incident energy.

        Args:
            rbm_samples (torch.Tensor): A tensor of latent samples from the RBM.
                Shape: (n_samples, latent_nodes_per_p * 4).
            incident_energies (torch.Tensor): A tensor of corresponding incident energies (x0).
                Shape: (n_samples, 1).
        """
        self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            # 1. Get sizes from input tensors and config
            n_samples = rbm_samples.shape[0]
            ar_input_size = self._config.data.z * self._config.data.r * self._config.data.phi
            ar_latent_size = self._config.rbm.latent_nodes_per_p * 3 + self._config.model.cond_p_size

            # 2. Perform sanity checks on input tensor dimensions
            if rbm_samples.shape[1] != ar_latent_size:
                raise ValueError("rbm_samples has an incorrect latent dimension.")
            if n_samples != incident_energies.shape[0]:
                raise ValueError("The number of rbm_samples and incident_energies must be the same.")

            # 3. Initialize tensors to store the results
            self.showers_prior_generated = torch.zeros((n_samples, ar_input_size), dtype=torch.float32)
            self.showers_reduce_prior_generated = torch.zeros((n_samples, ar_input_size), dtype=torch.float32)
            self.prior_samples_generated = rbm_samples.clone()
            self.incident_energy_generated = incident_energies.clone()
            
            # Note: If n_samples is very large, you might need to process in batches.
            # This implementation processes the entire tensor at once.

            # 4. Prepare tensors for the model
            rbm_samples_dev = rbm_samples.to(self.device)
            incident_energies_dev = incident_energies.to(self.device)
            
            # The input rbm_samples tensor is flat. It needs to be split into 4 parts
            # for the decoder, reversing the original torch.cat operation.
            samples_split = list(torch.split(rbm_samples_dev, ar_latent_size, dim=1))

            # Encode the incident energies to get the conditional part of the latent space
            incident_energies_encoded = self.model.encoder.binary_energy_refactored(incident_energies_dev)
            
            # *** NEW: Replace the first partition (p0) with the encoded incident energies ***
            # samples_split[0] = incident_energies_encoded

            # 5. Pass samples through the decoder to generate showers.
            # We pass `None` for the `x_reduce` argument, assuming the decoder's
            # architecture can handle generation from the prior without it.
            _, shower_prior_reduce = self.model.decode(samples_split, None, incident_energies_dev)
            
            # 6. Undo the reduction operation to get the full shower energy distribution
            shower_prior_full = self._reduceinv(shower_prior_reduce, incident_energies_dev)

            # 7. Store all results in the corresponding instance tensors on the CPU
            self.showers_reduce_prior_generated = shower_prior_reduce.cpu()
            self.showers_prior_generated = shower_prior_full.cpu()
            
            print(f"Successfully generated {n_samples} showers from RBM samples.")


    def evaluate_ae(self, data_loader, epoch):
        log_batch_idx = max(len(data_loader)//self._config.engine.n_batches_log_val, 1)
        self.model.eval()
        self.total_loss_dict = {}
        with torch.no_grad():
            bs = [batch[0].shape[0] for batch in data_loader]
            ar_size = np.sum(bs)
            ar_input_size = self._config.data.z * self._config.data.r * self._config.data.phi
            ar_latent_size = self._config.rbm.latent_nodes_per_p
            cond_size = self._config.model.cond_p_size
            
            self.incident_energy = torch.zeros((ar_size, 1), dtype=torch.float32)
            self.showers = torch.zeros((ar_size, ar_input_size), dtype=torch.float32)
            self.showers_reduce = torch.zeros((ar_size, ar_input_size), dtype=torch.float32)
            self.showers_recon = torch.zeros((ar_size, ar_input_size), dtype=torch.float32)
            self.showers_reduce_recon = torch.zeros((ar_size, ar_input_size), dtype=torch.float32)
            self.post_samples = torch.zeros((ar_size, ar_latent_size * 3+cond_size), dtype=torch.float32)
            self.post_logits = torch.zeros((ar_size, ar_latent_size * 3), dtype=torch.float32)

            self.showers_prior = torch.zeros((ar_size, ar_input_size), dtype=torch.float32)
            self.showers_reduce_prior = torch.zeros((ar_size, ar_input_size), dtype=torch.float32)
            self.prior_samples = torch.zeros((ar_size, ar_latent_size * 3+cond_size), dtype=torch.float32)
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
                loss_dict["loss"] = torch.stack([loss_dict[key] * self._config.model.loss_coeff[key]  for key in loss_dict.keys() if "loss" != key and key in self._config.model.loss_coeff]).sum()
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


    def evaluate_trivial(self):
        self.evaluate_ae(self.data_mgr.val_loader, 0)
        # create naive samples
        p_size = self._config.rbm.latent_nodes_per_p
        if hasattr(self._config.model, "cond_p_size"):
            cond_p_size = self._config.model.cond_p_size
        else:
            cond_p_size = p_size
        probs = torch.sigmoid(self.post_logits).mean(dim=0).cpu()
        num_samples = self.post_logits.shape[0]
        expanded_probs = probs.expand(num_samples, -1)
        naive_samples = torch.bernoulli(expanded_probs)
        p0_slice = self.post_samples[:,:cond_p_size]
        naive_samples = torch.cat((p0_slice, naive_samples), dim=1)

        # set up storage of naive showers
        batch_size = self.data_mgr.val_loader.batch_size

        bs = [batch[0].shape[0] for batch in self.data_mgr.val_loader]
        ar_input_size = self._config.data.z * self._config.data.r * self._config.data.phi


        # decode naive samples
        self.model.eval()
        for i, (x, x0) in enumerate(self.data_mgr.val_loader):
            idx1, idx2 = int(np.sum(bs[:i])), int(np.sum(bs[:i+1]))
            x_reduce = self._reduce(x, x0)
            naive_sample = naive_samples[i*batch_size:(i+1)*batch_size,:]
            split_samples = torch.split(naive_sample, [cond_p_size, p_size, p_size, p_size], dim=1)
            self.model.to(torch.device("cpu"))

            _, shower_naive = self.model.decode(split_samples, x_reduce, x0)
            self.showers_prior[i*batch_size:(i+1)*batch_size,:] = self._reduceinv(shower_naive, x0).detach().cpu()
            print(f"Processed batch {i+1}")

    
    def generate_plots(self, epoch, key):
        if self._config.wandb.mode != "disabled": # Only log if wandb is enabled
            
            # --- Calorimeter Plots ---
            calo_input, calo_recon, calo_sampled, calo_input_avg, calo_recon_avg, calo_sampled_avg = plot_calorimeter_shower(
                cfg=self._config,
                showers=self.showers,
                showers_recon=self.showers_recon,
                showers_sampled=self.showers_prior,
                epoch=epoch,
                save_dir=None
            )
            
            # --- VAE Plots & Chi2 Metrics (Updated Call) ---
            # Unpack all 10 items from the updated vae_plots function
            (overall_fig, fig_energy_sum, fig_incidence_ratio, fig_target_recon_ratio, 
             fig_sparsity, energy_sum_layer_fig, incidence_ratio_layer_fig, 
             target_recon_ratio_layer_fig, sparsity_layer_fig, 
             all_chi2_metrics) = vae_plots(self._config,
                                          self.incident_energy, self.showers, 
                                          self.showers_recon, self.showers_prior)
            
            # --- Corr Plots ---
            post_corr, prior_corr, post_partition, prior_partition = corr_plots(self._config, self.post_logits, self.post_samples, self.prior_samples)
            
            # --- Initialize wandb_log dictionary ---
            # Use your original keys, mapping to the new variable names
            wandb_log = {
                "overall_plots": wandb.Image(overall_fig),
                "conditioned_energy_sum": wandb.Image(fig_energy_sum),
                "conditioned_incidence_ratio": wandb.Image(fig_incidence_ratio),
                "conditioned_target_recon_ratio": wandb.Image(fig_target_recon_ratio),
                "conditioned_sparsity": wandb.Image(fig_sparsity),
                "energy_sum_layers": wandb.Image(energy_sum_layer_fig),         # Was fig_sum_layers
                "incidence_ratio_layers": wandb.Image(incidence_ratio_layer_fig), # Was fig_incidence_layers
                "target_recon_ratio_layers": wandb.Image(target_recon_ratio_layer_fig), # Was fig_ratio_layers
                "sparsity_layers": wandb.Image(sparsity_layer_fig),             # Was fig_sparsity_layers

                # Common Calo plots
                "calo_layer_input": wandb.Image(calo_input),
                "calo_layer_recon": wandb.Image(calo_recon),
                "calo_layer_input_avg": wandb.Image(calo_input_avg),
                "calo_layer_recon_avg": wandb.Image(calo_recon_avg),

                # Common Corr plots
                "post_corr": wandb.Image(post_corr),
                "prior_corr": wandb.Image(prior_corr),
                "post_partition": wandb.Image(post_partition),
                "prior_partition": wandb.Image(prior_partition),
                
            }

            
            # A. Log overall Chi2 metrics (simple key-value pairs)
            for metric, values in all_chi2_metrics.items():
                if metric.startswith('overall_'):
                    wandb_log[f"Chi2/{metric}_recon"] = values.get('recon', float('nan'))
                    wandb_log[f"Chi2/{metric}_sampled"] = values.get('sampled', float('nan'))

            # B. Log layer-wise Chi2 metrics
            layer_chi2 = all_chi2_metrics.get('layer', {}) # Use .get for safety
            for metric_name, pairs in layer_chi2.items(): # e.g., metric_name = 'energy_sum'
                if 'recon' in pairs:
                    for layer_num, chi2_val in pairs['recon']:
                        wandb_log[f"Chi2/layer_{layer_num}_{metric_name}_recon"] = chi2_val
                if 'sampled' in pairs:
                    for layer_num, chi2_val in pairs['sampled']:
                        wandb_log[f"Chi2/layer_{layer_num}_{metric_name}_sampled"] = chi2_val
                    
            # C. Log binned Chi2 metrics (as mean and as Table)
            for metric_name in ['binned_energy_sum', 'binned_incidence_ratio', 'binned_sparsity']:
                if metric_name not in all_chi2_metrics: continue
                
                recon_vals = [val for _, val in all_chi2_metrics[metric_name].get('recon', [])]
                sampled_vals = [val for _, val in all_chi2_metrics[metric_name].get('sampled', [])]
                
                if recon_vals:
                    wandb_log[f"Chi2/mean_{metric_name}_recon"] = np.mean(recon_vals)
                if sampled_vals:
                    wandb_log[f"Chi2/mean_{metric_name}_sampled"] = np.mean(sampled_vals)

                # Log W&B Table
                binned_data = []
                recon_dict = dict(all_chi2_metrics[metric_name].get('recon', []))
                sampled_dict = dict(all_chi2_metrics[metric_name].get('sampled', []))
                
                # Get all unique energy bins that have data
                all_bins = sorted(list(set(recon_dict.keys()) | set(sampled_dict.keys())))
                
                for energy_bin in all_bins:
                    binned_data.append([
                        energy_bin,
                        recon_dict.get(energy_bin),   # .get() returns None if key is missing
                        sampled_dict.get(energy_bin)
                    ])
                
                if binned_data: # Only log table if there is data
                    wandb_log[f"Chi2_Tables/{metric_name}"] = wandb.Table(
                        data=binned_data,
                        columns=["Energy Bin Center", "Chi2 Recon", "Chi2 Sampled"]
                    )
            # --- End of new Chi2 logic ---

            # --- Conditional Logging ---
            if key != "ae":
                # RBM plots
                rbm_hist = plot_rbm_histogram(self.RBM_energy_post, self.RBM_energy_prior)
                rbm_params = plot_rbm_params(self)
                rbm_floppy = plot_forward_output_v2(self)
            
                # Add RBM plots to log
                wandb_log.update({
                    "RBM histogram": wandb.Image(rbm_hist),
                    "RBM params": wandb.Image(rbm_params),
                    "RBM floppy": wandb.Image(rbm_floppy),
                })
                
                # Add sampled calo plots (which are omitted for "ae")
                wandb_log.update({
                    "calo_layer_sampled": wandb.Image(calo_sampled),
                    "calo_layer_sampled_avg": wandb.Image(calo_sampled_avg),
                })
            
            # The 'else' block for "ae" is now handled,
            # because the "sampled" and "RBM" plots are only added if key != "ae".
            
            # --- Final Log Call ---
            wandb.log(wandb_log)
            incidence_ratio_mean_chi2 = np.mean([val for _, val in all_chi2_metrics["binned_incidence_ratio"].get('recon', [])])
            return incidence_ratio_mean_chi2

    
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