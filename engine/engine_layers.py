"""
engine for layer conditioned AE
"""
import torch
import numpy as np
import wandb
import math
import os
import contextlib

from engine.engine import Engine
from CaloQuVAE import logging
logger = logging.getLogger(__name__)
from utils.evaluate_ae_layers import evaluate_layer_ae_distributions
from utils.atlas_plots import plot_calorimeter_shower
import matplotlib.pyplot as plt
from scripts.run import is_master, is_distributed
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from model.gumbel import GumbelNoNoise, GumbelTemperature


class EngineLayers(Engine):

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self.lr_scheduler = None
        self._ema_state = None
        self._ema_decay = float(getattr(cfg.engine, 'ema_decay', 0.0))

    def _update_ema(self):
        """Update (or lazily initialize) EMA shadow weights from the current model.

        The config value `ema_decay` is interpreted as a *per-step* decay (e.g. 0.999).
        Because _update_ema() is called once per epoch rather than once per step we
        correct for the frequency mismatch:

            d_epoch = d_step ^ steps_per_epoch

        so that the effective per-step behaviour is preserved.  The approximation
        treats end-of-epoch weights as representative of the intra-epoch trajectory,
        which is standard practice.
        """
        if self._ema_decay <= 0.0:
            return
        actual_model = self.model.module if hasattr(self.model, 'module') else self.model
        current_state = actual_model.state_dict()
        if self._ema_state is None:
            # First call: clone current weights as starting point
            self._ema_state = {k: v.detach().clone().cpu() for k, v in current_state.items()}
            steps_per_epoch = len(self.data_mgr.train_loader)
            epoch_decay = self._ema_decay ** steps_per_epoch
            logger.info(
                f"EMA shadow weights initialized "
                f"(per-step decay={self._ema_decay}, steps/epoch={steps_per_epoch}, "
                f"effective per-epoch decay={epoch_decay:.6f})"
            )
            return
        steps_per_epoch = len(self.data_mgr.train_loader)
        epoch_decay = self._ema_decay ** steps_per_epoch
        with torch.no_grad():
            for k, shadow in self._ema_state.items():
                current = current_state[k].detach().cpu()
                if shadow.is_floating_point():
                    shadow.mul_(epoch_decay).add_(current, alpha=1.0 - epoch_decay)
                else:
                    shadow.copy_(current)

    @contextlib.contextmanager
    def _ema_context(self):
        """Context manager: temporarily replace model weights with EMA shadow weights."""
        if self._ema_state is None or self._ema_decay <= 0.0:
            yield
            return
        actual_model = self.model.module if hasattr(self.model, 'module') else self.model
        backup = {k: v.clone() for k, v in actual_model.state_dict().items()}
        ema_on_device = {k: v.to(self.device) for k, v in self._ema_state.items()}
        actual_model.load_state_dict(ema_on_device)
        try:
            yield
        finally:
            actual_model.load_state_dict(backup)

    def _save_model(self, name="blank", override_path=None):
        """Save model, using EMA weights if available."""
        with self._ema_context():
            return super()._save_model(name=name, override_path=override_path)

    def fit_ae(self, epoch):
        log_batch_idx = max(len(self.data_mgr.train_loader)//self._config.engine.n_batches_log_train, 1)
        self.model.train()
        if isinstance(self.data_mgr.train_loader.sampler, DistributedSampler):
            self.data_mgr.train_loader.sampler.set_epoch(epoch)
        for i, (x, x0, u, E) in enumerate(self.data_mgr.train_loader):
            # Anneal parameters
            self._anneal_params(len(self.data_mgr.train_loader), i, epoch)
            x = x.to(self.device).to(dtype=torch.float32)
            x0 = x0.to(self.device).to(dtype=torch.float32)
            u = u.to(self.device).to(dtype=torch.int32)
            # x = self._reduce(x, x0)
            x = self._reduceBCE(x)
            # Forward pass
            output = self.model((x, x0, u), beta_latent=self.beta_latent, beta_hits=self.beta_hits, act_fct_slope=self.slope)
            # Compute loss
            if is_distributed():
                loss_dict = self.model.module.loss(x, output[2], output[3], post_logits=torch.cat(output[0], dim=1))
            else:
                loss_dict = self.model.loss(x, output[2], output[3], post_logits=torch.cat(output[0], dim=1))
            total_loss = torch.stack([loss_dict[key] * self._config.model.loss_coeff[key]  for key in loss_dict.keys() if "loss" != key and key in self._config.model.loss_coeff]).sum()
            loss_dict["loss"] = total_loss
            # Check for NaNs locally
            is_nan = torch.tensor(1 if torch.isnan(total_loss) else 0, device=self.device)

            # Synchronize the check across all GPUs. If ANY GPU has a 1, the max will be 1.
            if dist.is_initialized():
                dist.all_reduce(is_nan, op=dist.ReduceOp.MAX)

            if is_nan.item() > 0:
                logger.error(f"NaN Loss detected! Aborting across all processes.")
                raise ValueError("NaN detected in training loop")            
            
            # Backward pass and optimization
            self.optimiser.zero_grad()
            loss_dict["loss"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            # Check the gradient norm of the absolute bottleneck
            # last_enc_layer = self.model.encoder._networks[-1].seq2[-2].conv
            # grad_norm = last_enc_layer.weight.grad.norm().item() if last_enc_layer.weight.grad is not None else 0.0
            # print(f"Latent Bottleneck Gradient Norm: {grad_norm}")
            self.optimiser.step()
        
            if (i % log_batch_idx) == 0 and is_master():
                    current_lr = self.optimiser.param_groups[0]['lr']
                    logger.info('Epoch: {} [{}/{} ({:.0f}%)]\t beta_latent: {:.3f}, beta_hits: {:.3f}, slope: {:.3f}, lr: {:.2e} \t Batch Loss: {:.4f}'.format(epoch,
                        i, len(self.data_mgr.train_loader),100.*i/len(self.data_mgr.train_loader),
                        self.beta_latent, self.beta_hits, self.slope, current_lr, loss_dict["loss"]))
                    safe_wandb_dict = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}
                    safe_wandb_dict['lr'] = current_lr
                    wandb.log(safe_wandb_dict)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    def evaluate_ae(self, data_loader, epoch):
        self._update_ema()
        log_batch_idx = max(len(data_loader)//self._config.engine.n_batches_log_val, 1)
        self.model.eval()
        self.total_loss_dict = {}
        with self._ema_context(), torch.no_grad():
            bs = [data_loader.batch_size for _ in range(len(data_loader))]
            ar_size = len(data_loader.dataset)
            ar_input_size = self._config.data.z * self._config.data.r * self._config.data.phi
            ar_latent_size = self._config.rbm.latent_nodes_per_p
            cond_size = self._config.model.cond_p_size
            
            self.incident_energy = torch.zeros((ar_size, 1), dtype=torch.float32, device="cpu")
            self.showers = torch.zeros((ar_size, ar_input_size), dtype=torch.float32, device="cpu")
            self.showers_recon = torch.zeros((ar_size, ar_input_size), dtype=torch.float32, device="cpu")
            self.post_samples = torch.zeros((ar_size, ar_latent_size * 3+cond_size), dtype=torch.float32, device="cpu")
            self.post_logits = torch.zeros((ar_size, ar_latent_size * 3), dtype=torch.float32, device="cpu")

            for i, (x, x0, u, E) in enumerate(data_loader):
                x = x.to(self.device)
                x0 = x0.to(self.device)
                # x_reduce = self._reduce(x, x0)
                x_reduce = self._reduceBCE(x)
                u = u.to(self.device)
                E = E.to(self.device)
                # Forward pass
                if is_distributed():
                    output = self.model.module((x_reduce, x0, u))
                    loss_dict = self.model.module.loss(x_reduce, output[2], output[3], post_logits=torch.cat(output[0], dim=1))
                else:
                    output = self.model((x_reduce, x0, u))
                    loss_dict = self.model.loss(x_reduce, output[2], output[3], post_logits=torch.cat(output[0], dim=1))
                loss_dict["loss"] = torch.stack([loss_dict[key] * self._config.model.loss_coeff[key]  for key in loss_dict.keys() if "loss" != key and key in self._config.model.loss_coeff]).sum()
                for key in list(loss_dict.keys()):
                    loss_dict['val_'+key] = loss_dict[key]
                    loss_dict.pop(key)
                
                # Aggregate loss
                self.aggr_loss(data_loader, epoch, loss_dict)

                
                idx1, idx2 = int(np.sum(bs[:i])), int(np.sum(bs[:i+1]))
                self.incident_energy[idx1:idx2,:] = x0.cpu()
                self.showers[idx1:idx2,:] = x.cpu()
                # self.showers_recon[idx1:idx2,:] = self._reduceinv(output[3], x0, E).cpu()
                self.showers_recon[idx1:idx2,:] = self._reduceBCEinv(output[3], E).cpu()
                self.post_samples[idx1:idx2,:] = torch.cat(output[1],dim=1).cpu()
                self.post_logits[idx1:idx2,:] = torch.cat(output[0],dim=1).cpu()            
            # Log average loss after loop
            return self.aggr_loss(data_loader, epoch)


    def generate_plots(self, epoch, close_plots=True):
        if self._config.wandb.mode == "disabled":
            return 0.0

        # Narrow x-axis ranges for zoomed-in width plots on layers 1 and 2
        narrow_ranges = {
            "width_eta": {1: (0.0, 40.0), 2: (0.0, 40.0)},
            "width_phi": {1: (0.0, 40.0), 2: (0.0, 40.0)},
        }

        # Generate core analytical plots and WS metrics
        metrics, plots = evaluate_layer_ae_distributions(
            cfg=self._config,
            gt=self.showers,
            recon=self.showers_recon,
            incident_energies=self.incident_energy,
            post_logits=self.post_logits,
            post_samples=self.post_samples,
            feature_extractor=self.feature_extractor,
            geo_handler=self.geo_handler,
            close_plots=close_plots,
            device=self.device,
            narrow_ranges=narrow_ranges,
        )

        # Generate Calo visualisations
        calo_input, calo_recon, calo_sample, calo_input_avg, calo_recon_avg, calo_sample_avg = plot_calorimeter_shower(
            cfg=self._config,
            showers=self.showers,
            showers_recon=self.showers_recon,
            showers_sampled=self.showers_recon,  # Using recon as a proxy for sampled showers
            epoch=epoch,
            save_dir=None
        )

        plots.update({
            "calo_layer_input": wandb.Image(calo_input),
            "calo_layer_recon": wandb.Image(calo_recon),
            "calo_layer_input_avg": wandb.Image(calo_input_avg),
            "calo_layer_recon_avg": wandb.Image(calo_recon_avg),
        })

        if close_plots:
            for fig in [calo_input, calo_recon, calo_input_avg, calo_recon_avg, calo_sample, calo_sample_avg]:
                plt.close(fig)

        # Log to wandb
        if is_master():
            wandb_log = {**metrics, **plots}
            safe_wandb_log = {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in wandb_log.items()}
            wandb.log(safe_wandb_log)

        raw_weights = getattr(self._config.model, "feature_layer_weights", None) or getattr(self._config.model, "layer_weights", None)
        num_layers = self._config.data.z
        if raw_weights:
            w = np.array(raw_weights, dtype=np.float64)
            layer_weights_norm = w / w.sum()
        else:
            layer_weights_norm = np.ones(num_layers) / num_layers

        hlf_vals, hlf_wts = [], []
        for key, val in metrics.items():
            if "center" in key or "width" in key:
                # Key format: ws_HLF_layer_{l}_{feature} e.g. ws_HLF_layer_0_Eta_center
                layer_idx = int(key.split("_")[3])
                hlf_vals.append(val)
                hlf_wts.append(layer_weights_norm[layer_idx])
        mean_hlf_ws = float(np.average(hlf_vals))
        wandb.log({"overall_score": mean_hlf_ws})

        mean_hlf_ws_weighted  = float(np.average(hlf_vals, weights=hlf_wts))
        return mean_hlf_ws_weighted

    def track_best_val_loss(self, loss_dict, score, epoch=None):
        # Only the master process should track and save the best model
        if not is_master():
            return
        # Calculate current score once
        current_score = loss_dict["val_ae_loss"] + score * 20000
        wandb.log({"overall_score_weighted": current_score})
        
        # Check for strict improvement
        if self.best_val_loss > current_score:
            self.best_val_loss = current_score
            self.best_config_path = self._save_model(name="best" + (f"_epoch{epoch}" if epoch is not None else ""))
            logger.info("New Best Val loss plus normalized WD: {:.4f}".format(self.best_val_loss))
            
        # Check if within 10% of the best score (but not better)
        elif current_score <= self.best_val_loss * 1.1:
            self._save_model(name="best" + (f"_epoch{epoch}" if epoch is not None else ""))
            logger.info("Near-best model saved (within 10%): {:.4f}".format(current_score))


    def _reduceBCE(self, x):
        """
        Preprocesses voxels by normalizing as proportions of layer energies
        """
        B = x.shape[0]
        z = self._config.data.z
        
        # View as (Batch, Layers, Voxels_per_layer)
        x_3d = x.view(B, z, -1)
        layer_sums = x_3d.sum(dim=2, keepdim=True)
        
        # Divide by layer sum (add epsilon to prevent division by zero for empty layers)
        x_frac = x_3d / (layer_sums + 1e-12) 
        return x_frac.view_as(x)

    def _reduceBCEinv(self, x_frac, E):
        """
        Inverse of _reduceBCE: converts layer fractions back to voxel energies by multiplying by layer energies
        """
        B = x_frac.shape[0]
        z = self._config.data.z
        
        x_3d = x_frac.view(B, z, -1)
        E_3d = E.unsqueeze(2)
        
        # Multiply fractions by the true layer energies
        x_raw = x_3d * E_3d
        return x_raw.view_as(x_frac)


    

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

    def load_cond_encoding(self, x0, u):
        """
        Loads the conditional encoding for a given batch of incident energies and layer energies.
        x0 is raw incidence energy
        u is normalized to be in [0, 1]
        """
        x0 = x0.to(self.device)
        u = u.to(self.device)
        self.post_cond_samples = torch.cat((self.model.encoder.energy_encoding_fct(x0), self.model.encoder.gray_encoding_fct(u)), dim=1)

    def generate_showers_from_rbm(self, rbm_samples, x0, u, E, batch_size=1024):
        """
        Generates shower reconstructions by decoding samples from the RBM prior.
        """
        self.model.eval()
        rbm_samples = rbm_samples.to(self.device)
        x0 = x0.to(self.device)
        u = u.to(self.device)
        E = E.to(self.device)
        n_samples = rbm_samples.shape[0]
        p_size = self._config.rbm.latent_nodes_per_p
        cond_size = self._config.model.cond_p_size
        ar_latent_size = p_size * 3 + cond_size
        if rbm_samples.shape[1] != ar_latent_size:
            raise ValueError(f"RBM samples have incorrect dimension {rbm_samples.shape[1]}, expected {ar_latent_size}")
        if n_samples != x0.shape[0] or n_samples != u.shape[0] or n_samples != E.shape[0]:
            raise ValueError("Size mismatch between RBM samples and conditioning data")


        rbm_samples = [rbm_samples[:, :cond_size], rbm_samples[:, cond_size:cond_size+p_size], rbm_samples[:, cond_size+p_size:cond_size+2*p_size], rbm_samples[:, cond_size+2*p_size:cond_size+3*p_size]]

        ar_input_size = self._config.data.z * self._config.data.r * self._config.data.phi
        decoded_showers = torch.zeros((n_samples, ar_input_size), dtype=torch.float32, device=self.device)

        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                rbm_batch = [rbm_sample[i : i + batch_size] for rbm_sample in rbm_samples]
                x0_batch = x0[i : i + batch_size]
                u_batch = u[i : i + batch_size]
                E_batch = E[i : i + batch_size]

                outputs = self.model.decode(rbm_batch, torch.zeros(x0_batch.shape[0], 1).to(self.device), x0_batch, u_batch)
                decoded_shower_batch = self._reduceBCEinv(outputs[1], E_batch)
                decoded_showers[i : i + batch_size] = decoded_shower_batch
        return decoded_showers.cpu()

            
            


    def evaluate_ae_experimental(self, data_loader, epoch):
        # self.model._hit_smoothing = GumbelNoNoise()
        self.model.eval()
        all_diagnostics = {}
        with torch.no_grad():
            bs = [data_loader.batch_size for _ in range(len(data_loader))]
            ar_size = len(data_loader.dataset)
            ar_input_size = self._config.data.z * self._config.data.r * self._config.data.phi
            ar_latent_size = self._config.rbm.latent_nodes_per_p
            cond_size = self._config.model.cond_p_size
            
            self.incident_energy = torch.zeros((ar_size, 1), dtype=torch.float32, device="cpu")
            self.showers = torch.zeros((ar_size, ar_input_size), dtype=torch.float32, device="cpu")
            self.showers_recon = torch.zeros((ar_size, ar_input_size), dtype=torch.float32, device="cpu")
            self.post_samples = torch.zeros((ar_size, ar_latent_size * 3+cond_size), dtype=torch.float32, device="cpu")
            self.post_logits = torch.zeros((ar_size, ar_latent_size * 3), dtype=torch.float32, device="cpu")
            self.hits_recon = torch.zeros((ar_size, ar_input_size), dtype=torch.float32, device="cpu")
            self.u = torch.zeros((ar_size, self._config.data.z), dtype=torch.float32, device="cpu")


            for i, (x, x0, u, E) in enumerate(data_loader):
                x = x.to(self.device)
                x0 = x0.to(self.device)
                # x_reduce = self._reduce(x, x0)
                x_reduce = self._reduceBCE(x)
                u = u.to(self.device)
                E = E.to(self.device)
                # Forward pass
                if is_distributed():
                    output = self.model.module((x_reduce, x0, u))
                    loss_dict = self.model.module.loss(x_reduce, output[2], output[3])
                else:
                    # output = self.model((x_reduce, x0, u))
                    output = self.model((x_reduce, x0, u))
                    loss_dict = self.model.loss(x_reduce, output[2], output[3])
                if "diagnostic_data" in loss_dict:
                    batch_diag = loss_dict.pop("diagnostic_data")
                    for key, data in batch_diag.items():
                        if key not in all_diagnostics:
                            all_diagnostics[key] = {'error': [], 'val_gt': []}
                        all_diagnostics[key]['error'].append(data['error'])
                        all_diagnostics[key]['val_gt'].append(data['val_gt'])
                loss_dict["loss"] = torch.stack([loss_dict[key] * self._config.model.loss_coeff[key]  for key in loss_dict.keys() if "loss" != key and key in self._config.model.loss_coeff]).sum()
                for key in list(loss_dict.keys()):
                    loss_dict['val_'+key] = loss_dict[key]
                    loss_dict.pop(key)
                

                
                idx1, idx2 = int(np.sum(bs[:i])), int(np.sum(bs[:i+1]))
                self.incident_energy[idx1:idx2,:] = x0.cpu()
                self.showers[idx1:idx2,:] = x.cpu()
                # self.showers_recon[idx1:idx2,:] = Engine._reduceinv(self, output[3], x0).cpu()
                self.showers_recon[idx1:idx2,:] = self._reduceBCEinv(output[3], E).cpu()
                self.post_samples[idx1:idx2,:] = torch.cat(output[1],dim=1).cpu()
                self.post_logits[idx1:idx2,:] = torch.cat(output[0],dim=1).cpu()            
                self.hits_recon[idx1:idx2,:] = output[2].cpu()
                self.u[idx1:idx2,:] = u.cpu()

                self.diagnostics = {
                    key: {
                        'error': torch.cat(data['error'], dim=0),
                        'val_gt': torch.cat(data['val_gt'], dim=0)
                    }
                    for key, data in all_diagnostics.items()
                }