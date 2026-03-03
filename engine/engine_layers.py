"""
engine for layer conditioned AE
"""
import torch
import numpy as np
import wandb
import math

from engine.engine import Engine
from CaloQuVAE import logging
logger = logging.getLogger(__name__)


class EngineLayers(Engine):

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)

    def fit_ae(self, epoch):
        log_batch_idx = max(len(self.data_mgr.train_loader)//self._config.engine.n_batches_log_train, 1)
        self.model.train()
        for i, (x, x0, u, E) in enumerate(self.data_mgr.train_loader):
            # Anneal parameters
            self._anneal_params(len(self.data_mgr.train_loader), i, epoch)
            x = x.to(self.device).to(dtype=torch.float32)
            x0 = x0.to(self.device).to(dtype=torch.float32)
            u = u.to(self.device).to(dtype=torch.int32)
            x = self._reduce(x, x0)
            # Forward pass
            output = self.model((x, x0, u), beta_latent=self.beta_latent, beta_hits=self.beta_hits, act_fct_slope=self.slope)
            # Compute loss
            loss_dict = self.model.loss(x, output[2], output[3])
            total_loss = torch.stack([loss_dict[key] * self._config.model.loss_coeff[key]  for key in loss_dict.keys() if "loss" != key]).sum()
            loss_dict["loss"] = total_loss            
            if torch.isnan(total_loss):
                logger.error(f"NaN Loss detected at Epoch {epoch}, Batch {i}. Params: beta_latent={self.beta_latent}, slope={self.slope}")
                # Raising ValueError here triggers the failure handling in 'train_and_evaluate'
                raise ValueError("NaN detected in training loop")
            
            # Backward pass and optimization
            self.optimiser.zero_grad()
            loss_dict["loss"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimiser.step()
        
            if (i % log_batch_idx) == 0:
                    logger.info('Epoch: {} [{}/{} ({:.0f}%)]\t beta_latent: {:.3f}, beta_hits: {:.3f}, slope: {:.3f} \t Batch Loss: {:.4f}'.format(epoch,
                        i, len(self.data_mgr.train_loader),100.*i/len(self.data_mgr.train_loader),
                        self.beta_latent, self.beta_hits, self.slope, loss_dict["loss"]))
                    wandb.log(loss_dict)

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
            self.showers_recon = torch.zeros((ar_size, ar_input_size), dtype=torch.float32)
            self.post_samples = torch.zeros((ar_size, ar_latent_size * 3+cond_size), dtype=torch.float32)
            self.post_logits = torch.zeros((ar_size, ar_latent_size * 3), dtype=torch.float32)

            self.showers_prior = torch.zeros((ar_size, ar_input_size), dtype=torch.float32)
            self.prior_samples = torch.zeros((ar_size, ar_latent_size * 3+cond_size), dtype=torch.float32)

            for i, (x, x0, u, E) in enumerate(data_loader):
                x = x.to(self.device)
                x0 = x0.to(self.device)
                x_reduce = self._reduce(x, x0)
                u = u.to(self.device)
                E = E.to(self.device)
                # Forward pass
                output = self.model((x_reduce, x0, u))
                loss_dict = self.model.loss(x_reduce, output[2], output[3])
                loss_dict["loss"] = torch.stack([loss_dict[key] * self._config.model.loss_coeff[key]  for key in loss_dict.keys() if "loss" != key and key in self._config.model.loss_coeff]).sum()
                for key in list(loss_dict.keys()):
                    loss_dict['val_'+key] = loss_dict[key]
                    loss_dict.pop(key)
                
                # Aggregate loss
                self.aggr_loss(data_loader, epoch, loss_dict)

                
                idx1, idx2 = int(np.sum(bs[:i])), int(np.sum(bs[:i+1]))
                self.incident_energy[idx1:idx2,:] = x0.cpu()
                self.showers[idx1:idx2,:] = x.cpu()
                self.showers_recon[idx1:idx2,:] = self._reduceinv(output[3], x0, E).cpu()
                self.post_samples[idx1:idx2,:] = torch.cat(output[1],dim=1).cpu()
                self.post_logits[idx1:idx2,:] = torch.cat(output[0],dim=1).cpu()
                # Use recon as prior
                self.prior_samples[idx1:idx2,:] = torch.cat(output[1],dim=1).cpu()
                self.showers_prior[idx1:idx2,:] = self._reduceinv(output[3], x0, E).cpu()
            
            # Log average loss after loop
            return self.aggr_loss(data_loader, epoch)

    

    def _reduceinv(self, x, x0, E, R=1e-7):
        """
        Inverse of the preprocessing function used on voxels
        Inverts log scaling and normalizes by layer energies
        Args:
            x: preprocessed voxel data, shape (batch_size, n_voxels)
            x0: incident energy, shape (batch_size, 1)
            E: raw layer energies, shape (batch_size, n_layers)
        """
        zero_mask = (x == 0.0)
        logit_offset = math.log(R / (1 - R)) 
        x = (torch.sigmoid(x + logit_offset) - R) / (1 - 2 * R) * x0
        
        # Restore explicit zeros and squash floating point noise
        x[zero_mask] = 0.0
        x[x.abs() < 1e-7] = 0.0 

        # Layer normalization (optimized with broadcasting)
        B = x.shape[0]
        z = self._config.data.z
        
        # View x as 3D: (Batch, Z, Phi*R)
        x_3d = x.view(B, z, -1)
        x_layer_denom = x_3d.sum(dim=2, keepdim=True)
        
        E_3d = E.unsqueeze(2)
        
        # Broadcast handles the dimension matching automatically
        x_3d = x_3d * (E_3d / (x_layer_denom + R))
        
        return x_3d.view_as(x)

