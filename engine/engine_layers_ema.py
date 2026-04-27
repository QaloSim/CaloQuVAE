"""
EngineLayers with batch-wise EMA and dual evaluation (raw + EMA weights).
"""
import torch
import numpy as np
import wandb

from engine.engine import Engine
from engine.engine_layers import EngineLayers
from CaloQuVAE import logging
logger = logging.getLogger(__name__)
from utils.evaluate_ae_layers import evaluate_layer_ae_distributions
from utils.atlas_plots import plot_calorimeter_shower
import matplotlib.pyplot as plt
from scripts.run import is_master, is_distributed
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist


class EngineLayersEMA(EngineLayers):
    """
    Extends EngineLayers with:
      - Batch-wise EMA updates (per optimiser step, not epoch-wise).
      - Dual evaluation: each validation epoch evaluates both raw and EMA weights
        and logs both sets of metrics/plots to wandb.
      - Separate best-model tracking and checkpointing for raw and EMA weights.
    """

    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        self._best_val_loss_raw = float("inf")
        self._best_val_loss_ema = float("inf")
        # EMA eval outputs populated by evaluate_ae and consumed by track_best_val_loss
        self._last_ema_score = None
        self.ema_val_loss = None
        self.showers_recon_ema = None
        self.post_samples_ema = None
        self.post_logits_ema = None

    # ------------------------------------------------------------------
    # Batch-wise EMA
    # ------------------------------------------------------------------

    def _update_ema_step(self):
        """Per-optimiser-step EMA using the raw per-step decay from config."""
        if self._ema_decay <= 0.0:
            return
        actual_model = self.model.module if hasattr(self.model, "module") else self.model
        current_state = actual_model.state_dict()
        if self._ema_state is None:
            self._ema_state = {k: v.detach().clone().cpu() for k, v in current_state.items()}
            logger.info(f"EMA shadow weights initialized (per-step decay={self._ema_decay})")
            return
        d = self._ema_decay
        with torch.no_grad():
            for k, shadow in self._ema_state.items():
                current = current_state[k].detach().cpu()
                if shadow.is_floating_point():
                    shadow.mul_(d).add_(current, alpha=1.0 - d)
                else:
                    shadow.copy_(current)

    def fit_ae(self, epoch):
        """Training loop identical to parent's but with per-batch EMA update."""
        log_batch_idx = max(len(self.data_mgr.train_loader) // self._config.engine.n_batches_log_train, 1)
        self.model.train()
        if isinstance(self.data_mgr.train_loader.sampler, DistributedSampler):
            self.data_mgr.train_loader.sampler.set_epoch(epoch)

        for i, (x, x0, u, E) in enumerate(self.data_mgr.train_loader):
            self._anneal_params(len(self.data_mgr.train_loader), i, epoch)
            x = x.to(self.device).to(dtype=torch.float32)
            x0 = x0.to(self.device).to(dtype=torch.float32)
            u = u.to(self.device).to(dtype=torch.int32)
            x = self._reduceBCE(x)

            output = self.model((x, x0, u), beta_latent=self.beta_latent, beta_hits=self.beta_hits, act_fct_slope=self.slope)
            if is_distributed():
                loss_dict = self.model.module.loss(x, output[2], output[3], post_logits=torch.cat(output[0], dim=1))
            else:
                loss_dict = self.model.loss(x, output[2], output[3], post_logits=torch.cat(output[0], dim=1))

            total_loss = torch.stack([
                loss_dict[key] * self._config.model.loss_coeff[key]
                for key in loss_dict.keys()
                if key != "loss" and key in self._config.model.loss_coeff
            ]).sum()
            loss_dict["loss"] = total_loss

            is_nan = torch.tensor(1 if torch.isnan(total_loss) else 0, device=self.device)
            if dist.is_initialized():
                dist.all_reduce(is_nan, op=dist.ReduceOp.MAX)
            if is_nan.item() > 0:
                logger.error("NaN Loss detected! Aborting across all processes.")
                raise ValueError("NaN detected in training loop")

            self.optimiser.zero_grad()
            loss_dict["loss"].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimiser.step()

            # Batch-wise EMA update — only master maintains shadow weights.
            # In DDP all ranks have identical parameters, so master's shadow is valid.
            if is_master():
                self._update_ema_step()

            if (i % log_batch_idx) == 0 and is_master():
                current_lr = self.optimiser.param_groups[0]["lr"]
                logger.info(
                    "Epoch: {} [{}/{} ({:.0f}%)]\t beta_latent: {:.3f}, beta_hits: {:.3f}, "
                    "slope: {:.3f}, lr: {:.2e} \t Batch Loss: {:.4f}".format(
                        epoch, i, len(self.data_mgr.train_loader),
                        100.0 * i / len(self.data_mgr.train_loader),
                        self.beta_latent, self.beta_hits, self.slope,
                        current_lr, loss_dict["loss"],
                    )
                )
                safe = {k: v.item() if isinstance(v, torch.Tensor) else v for k, v in loss_dict.items()}
                safe["lr"] = current_lr
                wandb.log(safe)

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def _run_eval_pass(self, data_loader, epoch):
        """
        Single evaluation pass with whatever weights are currently loaded.

        Returns:
            avg_loss: dict of per-key average losses (keys without "val_" prefix)
            tensors:  dict with incident_energy, showers, showers_recon,
                      post_samples, post_logits — all on CPU
        """
        self.model.eval()
        running_loss = {}
        bs = [data_loader.batch_size for _ in range(len(data_loader))]
        ar_size = len(data_loader.dataset)
        ar_input_size = self._config.data.z * self._config.data.r * self._config.data.phi
        ar_latent_size = self._config.rbm.latent_nodes_per_p
        cond_size = self._config.model.cond_p_size

        tensors = {
            "incident_energy": torch.zeros((ar_size, 1), dtype=torch.float32),
            "showers": torch.zeros((ar_size, ar_input_size), dtype=torch.float32),
            "showers_recon": torch.zeros((ar_size, ar_input_size), dtype=torch.float32),
            "post_samples": torch.zeros((ar_size, ar_latent_size * 3 + cond_size), dtype=torch.float32),
            "post_logits": torch.zeros((ar_size, ar_latent_size * 3), dtype=torch.float32),
        }

        with torch.no_grad():
            for i, (x, x0, u, E) in enumerate(data_loader):
                x = x.to(self.device)
                x0 = x0.to(self.device)
                x_reduce = self._reduceBCE(x)
                u = u.to(self.device)
                E = E.to(self.device)

                if is_distributed():
                    output = self.model.module((x_reduce, x0, u))
                    ld = self.model.module.loss(x_reduce, output[2], output[3], post_logits=torch.cat(output[0], dim=1))
                else:
                    output = self.model((x_reduce, x0, u))
                    ld = self.model.loss(x_reduce, output[2], output[3], post_logits=torch.cat(output[0], dim=1))

                ld["loss"] = torch.stack([
                    ld[k] * self._config.model.loss_coeff[k]
                    for k in ld.keys()
                    if k != "loss" and k in self._config.model.loss_coeff
                ]).sum()

                for k, v in ld.items():
                    running_loss[k] = running_loss.get(k, 0.0) + (v.item() if isinstance(v, torch.Tensor) else v)

                idx1 = int(np.sum(bs[:i]))
                idx2 = int(np.sum(bs[:i + 1]))
                tensors["incident_energy"][idx1:idx2] = x0.cpu()
                tensors["showers"][idx1:idx2] = x.cpu()
                tensors["showers_recon"][idx1:idx2] = self._reduceBCEinv(output[3], E).cpu()
                tensors["post_samples"][idx1:idx2] = torch.cat(output[1], dim=1).cpu()
                tensors["post_logits"][idx1:idx2] = torch.cat(output[0], dim=1).cpu()

        avg_loss = {k: v / len(data_loader) for k, v in running_loss.items()}
        return avg_loss, tensors

    def evaluate_ae(self, data_loader, epoch):
        """
        Evaluate with raw weights, and (if EMA is active) also with EMA weights.

        Raw results are stored in self.showers_recon / self.post_logits / self.post_samples.
        EMA results are stored in self.showers_recon_ema / self.post_logits_ema / self.post_samples_ema.
        self.showers and self.incident_energy are shared (same input data for both passes).

        Returns the val-prefixed raw loss dict for compatibility with run.py.
        """
        # --- Raw pass ---
        raw_loss, raw_tensors = self._run_eval_pass(data_loader, epoch)
        self.incident_energy = raw_tensors["incident_energy"]
        self.showers = raw_tensors["showers"]
        self.showers_recon = raw_tensors["showers_recon"]
        self.post_samples = raw_tensors["post_samples"]
        self.post_logits = raw_tensors["post_logits"]

        raw_val_loss = {"val_" + k: v for k, v in raw_loss.items()}
        self.total_loss_dict = raw_val_loss  # keeps aggr_loss-based callers happy

        if is_master():
            logger.info("Epoch {} - Raw Avg Val Loss: {:.4f}".format(
                epoch, raw_val_loss.get("val_loss", float("nan"))))
            wandb.log(raw_val_loss)

        # --- EMA pass ---
        self.showers_recon_ema = None
        self.post_samples_ema = None
        self.post_logits_ema = None
        self.ema_val_loss = None

        if self._ema_decay > 0.0 and self._ema_state is not None:
            with self._ema_context():
                ema_loss, ema_tensors = self._run_eval_pass(data_loader, epoch)

            self.showers_recon_ema = ema_tensors["showers_recon"]
            self.post_samples_ema = ema_tensors["post_samples"]
            self.post_logits_ema = ema_tensors["post_logits"]
            self.ema_val_loss = {"ema_val_" + k: v for k, v in ema_loss.items()}

            if is_master():
                logger.info("Epoch {} - EMA Avg Val Loss: {:.4f}".format(
                    epoch, self.ema_val_loss.get("ema_val_loss", float("nan"))))
                wandb.log(self.ema_val_loss)

        return raw_val_loss

    # ------------------------------------------------------------------
    # Plot generation
    # ------------------------------------------------------------------

    def _generate_plots_for(self, recon, post_logits, post_samples, epoch, close_plots, wandb_prefix=""):
        """
        Generate and log all diagnostic plots for one set of outputs.
        wandb_prefix: "" for raw weights, "ema/" for EMA weights.
        Returns mean_hlf_ws_weighted (the scalar optimisation score).
        """
        narrow_ranges = {
            "width_eta": {1: (0.0, 40.0), 2: (0.0, 40.0)},
            "width_phi": {1: (0.0, 40.0), 2: (0.0, 40.0)},
        }

        metrics, plots = evaluate_layer_ae_distributions(
            cfg=self._config,
            gt=self.showers,
            recon=recon,
            incident_energies=self.incident_energy,
            post_logits=post_logits,
            post_samples=post_samples,
            feature_extractor=self.feature_extractor,
            geo_handler=self.geo_handler,
            close_plots=close_plots,
            device=self.device,
            narrow_ranges=narrow_ranges,
        )

        calo_input, calo_recon, calo_sample, calo_input_avg, calo_recon_avg, calo_sample_avg = plot_calorimeter_shower(
            cfg=self._config,
            showers=self.showers,
            showers_recon=recon,
            showers_sampled=recon,
            epoch=epoch,
            save_dir=None,
        )

        # Compute weighted HLF score
        raw_weights = (
            getattr(self._config.model, "feature_layer_weights", None)
            or getattr(self._config.model, "layer_weights", None)
        )
        num_layers = self._config.data.z
        if raw_weights:
            w = np.array(raw_weights, dtype=np.float64)
            layer_weights_norm = w / w.sum()
        else:
            layer_weights_norm = np.ones(num_layers) / num_layers

        hlf_vals, hlf_wts = [], []
        for key, val in metrics.items():
            if "center" in key or "width" in key:
                layer_idx = int(key.split("_")[3])
                hlf_vals.append(val)
                hlf_wts.append(layer_weights_norm[layer_idx])
        mean_hlf_ws = float(np.average(hlf_vals))
        mean_hlf_ws_weighted = float(np.average(hlf_vals, weights=hlf_wts))

        if is_master():
            prefixed_metrics = {wandb_prefix + k: v for k, v in metrics.items()}
            prefixed_plots = {wandb_prefix + k: v for k, v in plots.items()}
            prefixed_plots.update({
                wandb_prefix + "calo_layer_input": wandb.Image(calo_input),
                wandb_prefix + "calo_layer_recon": wandb.Image(calo_recon),
                wandb_prefix + "calo_layer_input_avg": wandb.Image(calo_input_avg),
                wandb_prefix + "calo_layer_recon_avg": wandb.Image(calo_recon_avg),
            })
            wandb_log = {**prefixed_metrics, **prefixed_plots}
            safe = {k: (v.item() if isinstance(v, torch.Tensor) else v) for k, v in wandb_log.items()}
            safe[wandb_prefix + "overall_score"] = mean_hlf_ws
            wandb.log(safe)

        if close_plots:
            for fig in [calo_input, calo_recon, calo_input_avg, calo_recon_avg, calo_sample, calo_sample_avg]:
                plt.close(fig)

        return mean_hlf_ws_weighted

    def generate_plots(self, epoch, close_plots=True):
        if self._config.wandb.mode == "disabled":
            return 0.0

        raw_score = self._generate_plots_for(
            recon=self.showers_recon,
            post_logits=self.post_logits,
            post_samples=self.post_samples,
            epoch=epoch,
            close_plots=close_plots,
            wandb_prefix="",
        )

        self._last_ema_score = None
        if self._ema_decay > 0.0 and self.showers_recon_ema is not None:
            self._last_ema_score = self._generate_plots_for(
                recon=self.showers_recon_ema,
                post_logits=self.post_logits_ema,
                post_samples=self.post_samples_ema,
                epoch=epoch,
                close_plots=close_plots,
                wandb_prefix="ema/",
            )

        return raw_score

    # ------------------------------------------------------------------
    # Best-model tracking and checkpointing
    # ------------------------------------------------------------------

    def _save_model_raw(self, name="blank", override_path=None):
        """Save the live (non-EMA) model weights."""
        return Engine._save_model(self, name=name, override_path=override_path)

    def _save_model_ema(self, name="blank", override_path=None):
        """Save the EMA shadow weights."""
        with self._ema_context():
            return Engine._save_model(self, name=name, override_path=override_path)

    def track_best_val_loss(self, loss_dict, score, epoch=None):
        if not is_master():
            return

        suffix = f"_epoch{epoch}" if epoch is not None else ""

        # --- Raw model ---
        raw_score = loss_dict.get("val_ae_loss", float("inf")) + score * 20000
        wandb.log({"overall_score_weighted": raw_score})
        if self._best_val_loss_raw > raw_score:
            self._best_val_loss_raw = raw_score
            self.best_val_loss = raw_score  # keep parent attr in sync
            self.best_config_path = self._save_model_raw(name=f"best_raw{suffix}")
            logger.info("New Best Raw Val score: {:.4f}".format(raw_score))
        elif raw_score <= self._best_val_loss_raw * 1.1:
            self._save_model_raw(name=f"best_raw{suffix}")
            logger.info("Near-best Raw model saved (within 10%): {:.4f}".format(raw_score))

        # --- EMA model ---
        if self._ema_decay > 0.0 and self._last_ema_score is not None and self.ema_val_loss is not None:
            ema_ae_loss = self.ema_val_loss.get("ema_val_ae_loss", float("inf"))
            ema_score = ema_ae_loss + self._last_ema_score * 20000
            wandb.log({"ema/overall_score_weighted": ema_score})
            if self._best_val_loss_ema > ema_score:
                self._best_val_loss_ema = ema_score
                self._save_model_ema(name=f"best_ema{suffix}")
                logger.info("New Best EMA Val score: {:.4f}".format(ema_score))
            elif ema_score <= self._best_val_loss_ema * 1.1:
                self._save_model_ema(name=f"best_ema{suffix}")
                logger.info("Near-best EMA model saved (within 10%): {:.4f}".format(ema_score))
