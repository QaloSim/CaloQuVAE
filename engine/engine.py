"""
Base Class of Engines. Defines properties and methods.
"""

import torch
import numpy as np

# Weights and Biases
import wandb

# Plotting
from utils.plots import vae_plots, corr_plots
from utils.atlas_plots import plot_calorimeter_shower, create_grid_figure, to_np
import matplotlib.pyplot as plt
from IPython.display import display
from utils.rbm_plots import plot_forward_output_v2, plot_rbm_histogram, plot_rbm_params, plot_forward_output_hidden
# from utils.correlation_plotting import correlation_plots
from utils.HLF.atlasgeo import AtlasGeometry, DifferentiableFeatureExtractor, FeatureAdapter
from utils.plots import calculate_chi_squared_distance

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
        # Initialize Geometry and Extractor
        self.geo_handler = AtlasGeometry(self._config.data.binning_path) 
        self.feature_extractor = DifferentiableFeatureExtractor(self.geo_handler)
        
        self.feature_extractor.eval()

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
            loss_dict = self.model.loss(x, x0, output)
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
        # Calculate current score once
        current_score = loss_dict["val_ae_loss"] + chi2 * 10
        
        # Check for strict improvement
        if self.best_val_loss > current_score:
            self.best_val_loss = current_score
            self.best_config_path = self._save_model(name="best" + (f"_epoch{epoch}" if epoch is not None else ""))
            logger.info("New Best Val loss plus chi2: {:.4f}".format(self.best_val_loss))
            
        # Check if within 1% of the best score (but not better)
        elif current_score <= self.best_val_loss * 1.01:
            self._save_model(name="best" + (f"_epoch{epoch}" if epoch is not None else ""))
            logger.info("Near-best model saved (within 1%): {:.4f}".format(current_score))

        
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

    def generate_showers_from_rbm(self, rbm_samples, incident_energies, batch_size=None):
            """
            Generates showers by passing externally provided RBM samples through the decoder.
            Batched to prevent OOM errors on large sample sizes.
            
            Args:
                rbm_samples (torch.Tensor): Latent samples from RBM (n_samples, latent_dim).
                incident_energies (torch.Tensor): Incident energies (n_samples, 1).
                batch_size (int, optional): Batch size. Defaults to config val batch size or 1024.

            Returns:
                tuple: (showers_full, showers_reduce)
                    - showers_full (torch.Tensor): Generated showers in physical space (CPU).
                    - showers_reduce (torch.Tensor): Generated showers in reduced space (CPU).
            """
            self.model.eval()
            
            # 1. Configuration and Dimensions
            n_samples = rbm_samples.shape[0]
            # Calculate total latent size based on config
            p_size = self._config.rbm.latent_nodes_per_p
            cond_size = self._config.model.cond_p_size
            ar_latent_size = p_size * 3 + cond_size
            
            ar_input_size = self._config.data.z * self._config.data.r * self._config.data.phi

            # 2. Validation
            if rbm_samples.shape[1] != ar_latent_size:
                raise ValueError(f"RBM samples have incorrect dimension {rbm_samples.shape[1]}, expected {ar_latent_size}")
            if n_samples != incident_energies.shape[0]:
                raise ValueError("Size mismatch between RBM samples and incident energies")

            # 3. Pre-allocate CPU storage
            # Local variables instead of object attributes
            all_showers_full = torch.zeros((n_samples, ar_input_size), dtype=torch.float32)
            all_showers_reduce = torch.zeros((n_samples, ar_input_size), dtype=torch.float32)

            # 4. Determine Batch Size
            if batch_size is None:
                batch_size = getattr(self._config.data, 'batch_size_val', 1024)

            logger.info(f"Generating {n_samples} showers in batches of {batch_size}...")

            # 5. Batched Generation
            with torch.no_grad():
                for i in range(0, n_samples, batch_size):
                    # Slicing handles the last batch automatically
                    rbm_batch = rbm_samples[i : i + batch_size].to(self.device)
                    energy_batch = incident_energies[i : i + batch_size].to(self.device)
                    
                    # --- Decoding Logic ---
                    # Split flat latent vector into the 4 partitions expected by the decoder
                    samples_split = list(torch.split(rbm_batch, ar_latent_size, dim=1))
                    
                    # Pass None for x_reduce since we are generating from prior
                    _, shower_reduce = self.model.decode(samples_split, None, energy_batch)
                    
                    # Inverse reduction to get physical energy
                    shower_full = self._reduceinv(shower_reduce, energy_batch)

                    # --- Store Results ---
                    current_batch_len = rbm_batch.shape[0]
                    idx_start = i
                    idx_end = i + current_batch_len
                    
                    # Move to CPU and assign to local storage
                    all_showers_reduce[idx_start:idx_end] = shower_reduce.cpu()
                    all_showers_full[idx_start:idx_end] = shower_full.cpu()

            logger.info(f"Successfully generated {n_samples} showers.")
            
            return all_showers_full, all_showers_reduce


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
                loss_dict = self.model.loss(x_reduce, x0, output)
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

    def _compute_high_level_metrics(self, showers_gt, showers_recon, showers_sampled):
            """
            Computes Chi2 for high-level features (Centers, Widths, Energy).
            Uses full data range (no percentile clipping) and the user's custom Chi2 function.
            """
            metrics = {}
            
            # 1. Extract features (Ensure inputs and extractor are on CPU)
            self.feature_extractor.to('cpu')
            
            with torch.no_grad():
                feats_gt = self.feature_extractor(showers_gt.cpu())
                feats_recon = self.feature_extractor(showers_recon.cpu())
                feats_sampled = self.feature_extractor(showers_sampled.cpu())

            # 2. Iterate through features and calculate Chi2
            for key in feats_gt.keys():
                data_gt = feats_gt[key].numpy()
                data_recon = feats_recon[key].numpy()
                data_sampled = feats_sampled[key].numpy()

                # --- A. Layer-wise features (Shape: Batch, Layers) ---
                if data_gt.ndim == 2:
                    num_layers = data_gt.shape[1]
                    for l in range(num_layers):
                        # Slice specific layer
                        g_l = data_gt[:, l]
                        r_l = data_recon[:, l]
                        s_l = data_sampled[:, l]
                        
                        # Define bins using FULL Ground Truth range
                        min_val = np.min(g_l)
                        max_val = np.max(g_l)

                        # Safety: Handle constant features (e.g., empty layer = all 0s)
                        if max_val <= min_val:
                            max_val = min_val + 1e-6

                        bins = np.linspace(min_val, max_val, 100) 

                        # Calculate Chi2
                        metrics[f"Chi2_HighLevel/layer_{l}_{key}_recon"] = calculate_chi_squared_distance(g_l, r_l, bins)
                        metrics[f"Chi2_HighLevel/layer_{l}_{key}_sampled"] = calculate_chi_squared_distance(g_l, s_l, bins)
                
                # --- B. Global features (Shape: Batch) ---
                else:
                    # Define bins using FULL Ground Truth range
                    min_val = np.min(data_gt)
                    max_val = np.max(data_gt)
                    
                    if max_val <= min_val:
                        max_val = min_val + 1e-6
                        
                    bins = np.linspace(min_val, max_val, 100)

                    metrics[f"Chi2_HighLevel/global_{key}_recon"] = calculate_chi_squared_distance(data_gt, data_recon, bins)
                    metrics[f"Chi2_HighLevel/global_{key}_sampled"] = calculate_chi_squared_distance(data_gt, data_sampled, bins)

            return metrics


    def _compute_high_level_plots(self):
        """
        Uses the existing self.feature_extractor to get high-level features 
        and returns a dictionary of grid plots.
        """
        # Ensure extractor is in eval mode and on the correct device
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()

        # 1. Define Data to Process
        data_to_process = {
            "Data": (self.showers, self.incident_energy),
            "Recon": (self.showers_recon, self.incident_energy),
            "Sampled": (self.showers_prior, self.incident_energy)
        }
        
        # 2. Extract Features
        adapters = {}
        with torch.no_grad():
            for label, (sh, e_inc) in data_to_process.items():
                if not isinstance(sh, torch.Tensor):
                    sh = torch.tensor(sh)
                sh = sh.to(self.device)

                if not isinstance(e_inc, torch.Tensor):
                    e_inc = torch.tensor(e_inc)
                e_inc = e_inc.to(self.device)                
                # Use the existing class member
                feats = self.feature_extractor(sh)
                
                # Wrap in adapter for easy dictionary access
                adapters[label] = FeatureAdapter(feats, self.geo_handler.relevant_layers, e_inc)

        # 3. Sort into Reference vs Models
        ref_adapter = adapters["Data"]
        model_keys = ["Recon", "Sampled"]
        model_adapters = [adapters[k] for k in model_keys]
        
        # 4. Populate Data Structures for Plotting
        grids = {
            "energy": {}, "mean_eta": {}, "width_eta": {}, "mean_phi": {}, "width_phi": {}
        }

        for layer in self.geo_handler.relevant_layers:
            # Helper lambda for cleaner code
            get_data = lambda attr, lyr: [to_np(m.__dict__[attr][lyr]) for m in model_adapters]
            
            grids["energy"][layer] = {
                'ref': to_np(ref_adapter.E_layers[layer]),
                'models': get_data('E_layers', layer)
            }
            grids["mean_eta"][layer] = {
                'ref': to_np(ref_adapter.EC_etas[layer]),
                'models': get_data('EC_etas', layer)
            }
            grids["width_eta"][layer] = {
                'ref': to_np(ref_adapter.width_etas[layer]),
                'models': get_data('width_etas', layer)
            }
            grids["mean_phi"][layer] = {
                'ref': to_np(ref_adapter.EC_phis[layer]),
                'models': get_data('EC_phis', layer)
            }
            grids["width_phi"][layer] = {
                'ref': to_np(ref_adapter.width_phis[layer]),
                'models': get_data('width_phis', layer)
            }

        # 5. Create Figures
        figs = {}
        figs["HLF/Grid_Energy"] = create_grid_figure(grids["energy"], 'Layer Energy [MeV]', model_keys, yscale='log')
        figs["HLF/Grid_Mean_Eta"] = create_grid_figure(grids["mean_eta"], 'Mean Eta', model_keys, yscale='log')
        figs["HLF/Grid_Width_Eta"] = create_grid_figure(grids["width_eta"], 'Width Eta', model_keys, yscale='log')
        figs["HLF/Grid_Mean_Phi"] = create_grid_figure(grids["mean_phi"], 'Mean Phi', model_keys, yscale='log')
        figs["HLF/Grid_Width_Phi"] = create_grid_figure(grids["width_phi"], 'Width Phi', model_keys, yscale='log')

        return figs


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
            
            # --- VAE Plots & Chi2 Metrics ---
            (overall_fig, fig_energy_sum, fig_incidence_ratio, fig_target_recon_ratio, 
             fig_sparsity, energy_sum_layer_fig, incidence_ratio_layer_fig, 
             target_recon_ratio_layer_fig, sparsity_layer_fig, 
             all_chi2_metrics) = vae_plots(self._config,
                                          self.incident_energy, self.showers, 
                                          self.showers_recon, self.showers_prior)
            
            # --- Corr Plots ---
            post_corr, prior_corr, post_partition, prior_partition = corr_plots(
                self._config, self.post_logits, self.post_samples, self.prior_samples
            )

            # --- High Level Feature Grid Plots (NEW) ---
            # Call the internal method using the existing feature extractor
            hlf_grid_plots = self._compute_high_level_plots()
            
            # --- Initialize wandb_log dictionary ---
            wandb_log = {
                # VAE Plots
                "overall_plots": wandb.Image(overall_fig),
                "conditioned_energy_sum": wandb.Image(fig_energy_sum),
                "conditioned_incidence_ratio": wandb.Image(fig_incidence_ratio),
                "conditioned_target_recon_ratio": wandb.Image(fig_target_recon_ratio),
                "conditioned_sparsity": wandb.Image(fig_sparsity),
                "energy_sum_layers": wandb.Image(energy_sum_layer_fig),
                "incidence_ratio_layers": wandb.Image(incidence_ratio_layer_fig),
                "target_recon_ratio_layers": wandb.Image(target_recon_ratio_layer_fig),
                "sparsity_layers": wandb.Image(sparsity_layer_fig),

                # Calo Plots
                "calo_layer_input": wandb.Image(calo_input),
                "calo_layer_recon": wandb.Image(calo_recon),
                "calo_layer_input_avg": wandb.Image(calo_input_avg),
                "calo_layer_recon_avg": wandb.Image(calo_recon_avg),

                # Corr Plots
                "post_corr": wandb.Image(post_corr),
                "prior_corr": wandb.Image(prior_corr),
                "post_partition": wandb.Image(post_partition),
                "prior_partition": wandb.Image(prior_partition),
            }

            # Add HLF Grid Plots to log and close them
            for name, fig in hlf_grid_plots.items():
                if fig is not None:
                    wandb_log[name] = wandb.Image(fig)
                    display(fig)
                    plt.close(fig)

            # A. Log overall Chi2 metrics
            for metric, values in all_chi2_metrics.items():
                if metric.startswith('overall_'):
                    wandb_log[f"Chi2/{metric}_recon"] = values.get('recon', float('nan'))
                    wandb_log[f"Chi2/{metric}_sampled"] = values.get('sampled', float('nan'))

            # B. Log layer-wise Chi2 metrics
            layer_chi2 = all_chi2_metrics.get('layer', {}) 
            for metric_name, pairs in layer_chi2.items(): 
                if 'recon' in pairs:
                    for layer_num, chi2_val in pairs['recon']:
                        wandb_log[f"Chi2/layer_{layer_num}_{metric_name}_recon"] = chi2_val
                if 'sampled' in pairs:
                    for layer_num, chi2_val in pairs['sampled']:
                        wandb_log[f"Chi2/layer_{layer_num}_{metric_name}_sampled"] = chi2_val
                    
            # C. Log binned Chi2 metrics
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
                
                all_bins = sorted(list(set(recon_dict.keys()) | set(sampled_dict.keys())))
                
                for energy_bin in all_bins:
                    binned_data.append([
                        energy_bin,
                        recon_dict.get(energy_bin),   
                        sampled_dict.get(energy_bin)
                    ])
                
                if binned_data:
                    wandb_log[f"Chi2_Tables/{metric_name}"] = wandb.Table(
                        data=binned_data,
                        columns=["Energy Bin Center", "Chi2 Recon", "Chi2 Sampled"]
                    )
            
            # Compute scalar HL metrics (if you still need the scalar values for optimization/tracking)
            hl_metrics = self._compute_high_level_metrics(
                        self.showers, 
                        self.showers_recon, 
                        self.showers_prior
                    )
            wandb_log.update(hl_metrics)

            # --- Conditional Logging (RBM & Sampled Calo) ---
            if key != "ae":
                rbm_hist = plot_rbm_histogram(self.RBM_energy_post, self.RBM_energy_prior)
                rbm_params = plot_rbm_params(self)
                rbm_floppy = plot_forward_output_v2(self)
            
                wandb_log.update({
                    "RBM histogram": wandb.Image(rbm_hist),
                    "RBM params": wandb.Image(rbm_params),
                    "RBM floppy": wandb.Image(rbm_floppy),
                    "calo_layer_sampled": wandb.Image(calo_sampled),
                    "calo_layer_sampled_avg": wandb.Image(calo_sampled_avg),
                })
            
            # --- Final Log Call ---
            wandb.log(wandb_log)
            
            # Cleanup figures from vae_plots
            plt.close(overall_fig)
            plt.close(fig_energy_sum)
            # ... (close others if needed)

            incidence_ratio_mean_chi2 = np.mean([val for _, val in all_chi2_metrics["binned_incidence_ratio"].get('recon', [])])
            geo_mean_chi2 = np.mean([val for key, val in hl_metrics.items() if ("center" in key or "width" in key) and "recon" in key])
            
            return incidence_ratio_mean_chi2 + geo_mean_chi2
    
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