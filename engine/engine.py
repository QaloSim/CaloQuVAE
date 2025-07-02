"""
Base Class of Engines. Defines properties and methods.
"""

import torch
import numpy as np

# Weights and Biases
import wandb

# Plotting libraries
import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict

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

            if (i % log_batch_idx) == 0 and self._config.wandb.watch:
                    logger.info('Epoch: {} [{}/{} ({:.0f}%)]\t Batch Loss: {:.4f}'.format(epoch,
                        i, len(self.data_mgr.train_loader),100.*i/len(self.data_mgr.train_loader),
                        loss_dict["loss"]))
                    wandb.log(loss_dict)
                    
            

    
    def evaluate(self):
        self.model.eval()

        incident_energies = []
        target_showers = []
        recon_showers = []

        # Initialize loss accumulators
        total_loss = defaultdict(float)
        total_samples = 0


        with torch.no_grad():
            for x, x0 in self.data_mgr.val_loader:
                x = x.to(self.device, dtype=torch.float)
                x0 = x0.to(self.device, dtype=torch.float)
                x_reduced = self._reduce(x, x0)

                output = self.model((x_reduced, x0))
                beta, post_logits, post_samples, output_activations, output_hits = output

                # Compute loss
                loss_dict = self.model.loss(x_reduced, output)
                batch_size = x.size(0)
                for key in loss_dict:
                    total_loss[key] += loss_dict[key].item() * batch_size
                total_samples += batch_size

                x_recon = self._reduceinv(output_activations, x0)
                incident_energies.append(x0)
                target_showers.append(x)
                recon_showers.append(x_recon)
        # Average the losses
        total_loss["loss"] = sum([total_loss[key] for key in total_loss.keys() if key != "loss"])
        for key in total_loss:
            total_loss[key] /= total_samples
        val_loss_dict = {f"val_{key}": value for key, value in total_loss.items()}
        wandb.log(val_loss_dict)
        logger.info(f"Validation Loss: {val_loss_dict["val_loss"]:.4f}")


        incident_energies = torch.cat(incident_energies, dim=0)
        target_showers = torch.cat(target_showers, dim=0)
        recon_showers = torch.cat(recon_showers, dim=0)
        self._log_vae_plots(incident_energies, target_showers, recon_showers)
    
    def _plot_histograms(self, ax, target, recon, xlabel, ylabel, title, bins=30, log_scale=True):
        max_value = max(target.max(), recon.max())
        min_value = min(target.min(), recon.min())
        binning = np.arange(min_value, max_value, (max_value - min_value) / bins)
        ax.hist(target, histtype="stepfilled", bins=binning, density=True, alpha=0.7, label='Target', color='b', linewidth=2.5)
        ax.hist(recon, histtype="step", bins=binning, density=True, label='Reconstructed', color='c', linewidth=2.5)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_yscale('log' if log_scale else 'linear')
        ax.grid(True)
        ax.legend()


        
    def _log_vae_plots(self, incident_energies, target_showers, recon_showers):
        """
        Plot the energy sums and ratios target showers and reconstructed showers (overall and conditioned on incident energy bins)
        """

        target_energy_sums = torch.sum(target_showers, dim=1)
        recon_energy_sums = torch.sum(recon_showers, dim=1)
        target_incidence_ratio = target_energy_sums / incident_energies
        recon_incidence_ratio = recon_energy_sums / incident_energies
        target_recon_ratio = target_energy_sums / recon_energy_sums
        target_sparsity = (target_showers == 0).sum(dim=1) / target_showers.shape[1]
        recon_sparsity = (recon_showers == 0).sum(dim=1) / recon_showers.shape[1]

        #move to cpu and numpy for plotting
        target_energy_sums_np = target_energy_sums.detach().cpu().numpy()
        recon_energy_sums_np = recon_energy_sums.detach().cpu().numpy()
        target_incidence_ratio_np = target_incidence_ratio.detach().cpu().numpy()
        recon_incidence_ratio_np = recon_incidence_ratio.detach().cpu().numpy()
        target_recon_ratio_np = target_recon_ratio.detach().cpu().numpy()
        incident_energies_np = incident_energies.detach().cpu().numpy().squeeze()
        target_sparsity_np = target_sparsity.detach().cpu().numpy()
        recon_sparsity_np = recon_sparsity.detach().cpu().numpy()



        overall_fig, overall_ax = plt.subplots(2, 2, figsize=(15, 15))
        self._plot_histograms(overall_ax[0, 0], target_energy_sums_np, recon_energy_sums_np, 
                                xlabel='Deposited Energy (GeV)', ylabel='Density', title="Overall Deposited Energy")
        self._plot_histograms(overall_ax[0, 1], target_incidence_ratio_np, recon_incidence_ratio_np, 
                                xlabel='Deposited Energy / Incident Energy', ylabel='Density', title="Overall Energy Ratio")

        max_ratio = target_recon_ratio.max()
        min_ratio = target_recon_ratio.min()
        binning_ratio = np.arange(min_ratio, max_ratio, (max_ratio - min_ratio) / 30)
        overall_ax[1, 0].hist(target_recon_ratio_np, histtype="stepfilled", bins=binning_ratio, density=True, alpha=0.7, label='Target / Reconstructed', color='c', linewidth=2.5)
        overall_ax[1, 0].set_xlabel('Target / Reconstructed Energy Ratio')
        overall_ax[1, 0].set_ylabel('Density')
        overall_ax[1, 0].set_yscale('log')
        overall_ax[1, 0].grid(True)
        overall_ax[1, 0].axvline(1, color='r', linestyle='--', label='Ideal Ratio (1.0)')
        overall_ax[1, 0].legend()

        self._plot_histograms(overall_ax[1, 1], target_sparsity_np, recon_sparsity_np,
                                xlabel='Sparsity', ylabel='Density', title="Overall Sparsity")

        overall_fig.tight_layout()


        energy_bin_centers = [2**i for i in range(8, 23)] #currently hardcoded, can be parameterized
        fig_energy_sum, ax_energy_sum = plt.subplots(3, 5, figsize=(16, 10))
        fig_incidence_ratio, ax_incidence_ratio = plt.subplots(3, 5, figsize=(16, 10))
        fig_target_recon_ratio, ax_target_recon_ratio = plt.subplots(3, 5, figsize=(16, 10))
        fig_sparsity, ax_sparsity = plt.subplots(3, 5, figsize=(16, 10))

        for i, energy_center in enumerate(energy_bin_centers):
            row = i // 5
            col = i % 5

            e_low = 2**(np.log2(energy_center) - 0.5)
            e_high = 2**(np.log2(energy_center) + 0.5)
            
            mask = (incident_energies_np >= e_low) & (incident_energies_np < e_high)
            target_energy_sums_e = target_energy_sums_np[mask]
            recon_energy_sums_e = recon_energy_sums_np[mask]
            target_incidence_ratio_e = target_incidence_ratio_np[mask]
            recon_incidence_ratio_e = recon_incidence_ratio_np[mask]
            target_recon_ratio_e = target_recon_ratio_np[mask]

            self._plot_histograms(ax_energy_sum[row, col], target_energy_sums_e, recon_energy_sums_e, 
                                  xlabel='Deposited Energy (GeV)', ylabel='Density',
                                  title=f'Energy ~ {e_low / 1000:.1f} - {e_high / 1000:.1f} GeV')
            self._plot_histograms(ax_incidence_ratio[row, col], target_incidence_ratio_e, recon_incidence_ratio_e,
                                    xlabel='Deposited Energy / Incident Energy', ylabel='Density',
                                    title=f'Energy Ratio ~ {e_low / 1000:.1f} - {e_high / 1000:.1f} GeV')
            max_ratio = target_recon_ratio_e.max()
            min_ratio = target_recon_ratio_e.min()
            binning_ratio = np.arange(min_ratio, max_ratio, (max_ratio - min_ratio) / 30)
            ax_target_recon_ratio[row, col].hist(target_recon_ratio_e, histtype="stepfilled", bins=binning_ratio, density=True, alpha=0.7, label='Target / Reconstructed', color='c', linewidth=2.5)
            ax_target_recon_ratio[row, col].set_xlabel('Target / Reconstructed Energy Ratio')
            ax_target_recon_ratio[row, col].set_ylabel('Density')
            ax_target_recon_ratio[row, col].set_yscale('log')
            ax_target_recon_ratio[row, col].grid(True)
            ax_target_recon_ratio[row, col].axvline(1, color='r', linestyle='--', label='Ideal Ratio (1.0)')
            ax_target_recon_ratio[row, col].legend()
            
            self._plot_histograms(ax_sparsity[row, col], target_sparsity_np[mask], recon_sparsity_np[mask],
                                  xlabel='Sparsity', ylabel='Density',
                                  title=f'Sparsity ~ {e_low / 1000:.1f} - {e_high / 1000:.1f} GeV')


        fig_energy_sum.tight_layout()
        fig_incidence_ratio.tight_layout()
        fig_target_recon_ratio.tight_layout()
        fig_sparsity.tight_layout()
        wandb.log({
            "overall_plots": wandb.Image(overall_fig),
            "conditioned_energy_sum": wandb.Image(fig_energy_sum),
            "conditioned_incidence_ratio": wandb.Image(fig_incidence_ratio),
            "conditioned_target_recon_ratio": wandb.Image(fig_target_recon_ratio),
            "conditioned_sparsity": wandb.Image(fig_sparsity)
        })
        plt.close("all")  # Close all figures to free memory

            
        



 

    
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