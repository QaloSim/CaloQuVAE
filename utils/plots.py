import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_histograms(ax, target, recon, sampled, xlabel, ylabel, title, bins=30, log_scale=True):
    max_value = max(target.max(), recon.max(), sampled.max())
    min_value = min(target.min(), recon.min(), sampled.min())
    if min_value == max_value:
        print("Warning: min and max values are the same, adjusting to avoid division by zero.")
        max_value += 0.1  # Avoid division by zero if all values are the same
    binning = np.arange(min_value, max_value, (max_value - min_value) / bins)
    
    ax.hist(target, histtype="stepfilled", bins=binning, density=True, alpha=0.7, label='Target', color='b', linewidth=2.5)
    ax.hist(recon, histtype="step", bins=binning, density=True, label='Recon', color='c', linewidth=2.5)
    ax.hist(sampled, histtype="step", bins=binning, density=True, label='Sampled', color='orange', linewidth=2.5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_yscale('log' if log_scale else 'linear')
    ax.grid(True)
    ax.legend()


def vae_plots(cfg, incident_energies, target_showers, recon_showers, sampled_showers):
    """
    Plot the energy sums and ratios for target, reconstructed, and sampled showers (overall and binned by incident energy).
    """
    dataset_name = cfg.data.dataset_name.lower()
    
    epsilon = 1e-7  # to avoid division by zero

    target_energy_sums = torch.sum(target_showers, dim=1)
    recon_energy_sums = torch.sum(recon_showers, dim=1)
    sampled_energy_sums = torch.sum(sampled_showers, dim=1)

    target_incidence_ratio = target_energy_sums / (incident_energies.view(-1) + epsilon)
    recon_incidence_ratio = recon_energy_sums / (incident_energies.view(-1) + epsilon)
    sampled_incidence_ratio = sampled_energy_sums / (incident_energies.view(-1) + epsilon)

    target_recon_ratio = recon_energy_sums / (target_energy_sums.view(-1) + epsilon)

    target_sparsity = (target_showers == 0).sum(dim=1) / target_showers.shape[1]
    recon_sparsity = (recon_showers == 0).sum(dim=1) / recon_showers.shape[1]
    sampled_sparsity = (sampled_showers == 0).sum(dim=1) / sampled_showers.shape[1]

    # move to numpy for plotting
    target_energy_sums_np = target_energy_sums.numpy()
    recon_energy_sums_np = recon_energy_sums.numpy()
    sampled_energy_sums_np = sampled_energy_sums.numpy()

    target_incidence_ratio_np = target_incidence_ratio.numpy()
    recon_incidence_ratio_np = recon_incidence_ratio.numpy()
    sampled_incidence_ratio_np = sampled_incidence_ratio.numpy()

    target_recon_ratio_np = target_recon_ratio.numpy()
    incident_energies_np = incident_energies.numpy().squeeze()

    target_sparsity_np = target_sparsity.numpy()
    recon_sparsity_np = recon_sparsity.numpy()
    sampled_sparsity_np = sampled_sparsity.numpy()

    # overall plots
    overall_fig, overall_ax = plt.subplots(2, 2, figsize=(15, 15))

    plot_histograms(overall_ax[0, 0], target_energy_sums_np, recon_energy_sums_np, sampled_energy_sums_np,
                    xlabel='Deposited Energy (GeV)', ylabel='Density', title="Overall Deposited Energy")

    plot_histograms(overall_ax[0, 1], target_incidence_ratio_np, recon_incidence_ratio_np, sampled_incidence_ratio_np,
                    xlabel='Deposited Energy / Incident Energy', ylabel='Density', title="Overall Energy Ratio")

    max_ratio = target_recon_ratio_np.max()
    min_ratio = target_recon_ratio_np.min()
    if min_ratio == max_ratio:
        print("Warning: min and max values are the same, adjusting to avoid division by zero.")
        max_ratio += 0.1
    binning_ratio = np.arange(min_ratio, max_ratio, (max_ratio - min_ratio) / 30)
    overall_ax[1, 0].hist(target_recon_ratio_np, histtype="stepfilled", bins=binning_ratio, density=True,
                          alpha=0.7, label='Reconstructed / Target', color='c', linewidth=2.5)
    overall_ax[1, 0].set_xlabel('Recon / Target Energy Ratio')
    overall_ax[1, 0].set_ylabel('Density')
    overall_ax[1, 0].set_yscale('log')
    overall_ax[1, 0].grid(True)
    overall_ax[1, 0].axvline(1, color='r', linestyle='--', label='Ideal Ratio (1.0)')
    overall_ax[1, 0].legend()

    plot_histograms(overall_ax[1, 1], target_sparsity_np, recon_sparsity_np, sampled_sparsity_np,
                    xlabel='Sparsity', ylabel='Density', title="Overall Sparsity")

    overall_fig.tight_layout()

    # set up the binning for conditioned plots:
    if 'atlas' in dataset_name:
        energy_bin_centers = [2 ** i for i in range(8, 23)] # for atlas
    else:
        energy_bin_centers = [10 ** i for i in np.linspace(3, 6, num=15, endpoint=True)] # for calo
        
    fig_energy_sum, ax_energy_sum = plt.subplots(3, 5, figsize=(16, 10))
    fig_incidence_ratio, ax_incidence_ratio = plt.subplots(3, 5, figsize=(16, 10))
    fig_target_recon_ratio, ax_target_recon_ratio = plt.subplots(3, 5, figsize=(16, 10))
    fig_sparsity, ax_sparsity = plt.subplots(3, 5, figsize=(16, 10))

    for i, energy_center in enumerate(energy_bin_centers):
        row = i // 5
        col = i % 5

        e_low = 2 ** (np.log2(energy_center) - 0.5)
        e_high = 2 ** (np.log2(energy_center) + 0.5)

        mask = (incident_energies_np >= e_low) & (incident_energies_np < e_high)
        if mask.sum() == 0:
            print(f"No data in energy range {e_low:.1f} - {e_high:.1f} GeV, skipping this bin.")
            continue

        target_energy_sums_e = target_energy_sums_np[mask]
        recon_energy_sums_e = recon_energy_sums_np[mask]
        sampled_energy_sums_e = sampled_energy_sums_np[mask]
        
        target_incidence_ratio_e = target_incidence_ratio_np[mask]
        recon_incidence_ratio_e = recon_incidence_ratio_np[mask]
        sampled_incidence_ratio_e = sampled_incidence_ratio_np[mask]
        
        target_recon_ratio_e = target_recon_ratio_np[mask]

        plot_histograms(ax_energy_sum[row, col], target_energy_sums_e, recon_energy_sums_e, sampled_energy_sums_e,
                        xlabel='Deposited Energy (GeV)', ylabel='Density',
                        title=f'Energy ~ {e_low / 1000:.1f} - {e_high / 1000:.1f} GeV')

        plot_histograms(ax_incidence_ratio[row, col], target_incidence_ratio_e, recon_incidence_ratio_e, sampled_incidence_ratio_e,
                        xlabel='Deposited Energy / Incident Energy', ylabel='Density',
                        title=f'Energy Ratio ~ {e_low / 1000:.1f} - {e_high / 1000:.1f} GeV')

        max_ratio = target_recon_ratio_e.max()
        min_ratio = target_recon_ratio_e.min()
        if max_ratio == min_ratio:
            print('Warning: min and max values are the same, adjusting to avoid division by zero.')
            max_ratio += 0.1 
        binning_ratio = np.arange(min_ratio, max_ratio, (max_ratio - min_ratio) / 30)

        ax_target_recon_ratio[row, col].hist(target_recon_ratio_e, histtype="stepfilled", bins=binning_ratio,
                                             density=True, alpha=0.7, label='Reconstructed / Target', color='c',
                                             linewidth=2.5)
        ax_target_recon_ratio[row, col].set_title(f'Recon/Target~{e_low / 1000:.1f} - {e_high / 1000:.1f} GeV')
        ax_target_recon_ratio[row, col].set_xlabel('Recon / Target Energy Ratio')
        ax_target_recon_ratio[row, col].set_ylabel('Density')
        ax_target_recon_ratio[row, col].set_yscale('log')
        ax_target_recon_ratio[row, col].grid(True)
        ax_target_recon_ratio[row, col].axvline(1, color='r', linestyle='--', label='Ideal Ratio (1.0)')
        ax_target_recon_ratio[row, col].legend()

        plot_histograms(ax_sparsity[row, col], target_sparsity_np[mask], recon_sparsity_np[mask], sampled_sparsity_np[mask],
                        xlabel='Sparsity', ylabel='Density',
                        title=f'Sparsity ~ {e_low / 1000:.1f} - {e_high / 1000:.1f} GeV')

    fig_energy_sum.tight_layout()
    fig_incidence_ratio.tight_layout()
    fig_target_recon_ratio.tight_layout()
    fig_sparsity.tight_layout()

    return overall_fig, fig_energy_sum, fig_incidence_ratio, fig_target_recon_ratio, fig_sparsity

