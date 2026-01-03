import numpy as np
import matplotlib.pyplot as plt
import torch
import itertools

def calculate_chi_squared_distance(data1, data2, bins):
    """
    Calculates the Chi-Squared distance between two 1D datasets using specified bins.
    Formula: sum( (P_i - Q_i)^2 / (P_i + Q_i) ) for bins where P_i + Q_i > 0
    P = counts from data1 (e.g., target)
    Q = counts from data2 (e.g., recon or sampled)
    
    Args:
        data1 (np.array): The first dataset (e.g., ground truth).
        data2 (np.array): The second dataset (e.g., reconstruction).
        bins (np.array): The bin edges to use for histogramming.
    
    Returns:
        float: The calculated Chi-Squared distance.
    """
    # Get counts (P) for data1
    counts1, _ = np.histogram(data1, bins=bins, density=False)
    # Get counts (Q) for data2
    counts2, _ = np.histogram(data2, bins=bins, density=False)

    # Calculate (P_i + Q_i)
    sum_counts = counts1 + counts2
    # Calculate (P_i - Q_i)^2
    diff_counts_sq = (counts1 - counts2) ** 2

    # Mask for bins where the sum is non-zero to avoid division by zero
    mask = sum_counts > 0

    # Calculate (P_i - Q_i)^2 / (P_i + Q_i) for each bin
    chi2_terms = np.zeros_like(sum_counts, dtype=float)
    chi2_terms[mask] = diff_counts_sq[mask] / sum_counts[mask]

    # The total Chi-Squared distance is the sum of these terms
    return np.sum(chi2_terms)


def plot_histograms(ax, target, recon, sampled, xlabel, ylabel, title, binning, log_scale=True):
    """
    Modified plot_histograms function that accepts pre-calculated binning.
    """
    # The binning calculation is now done *before* calling this function.
    
    ax.hist(target, histtype="stepfilled", bins=binning, density=True, alpha=0.7, label='Target', color='b', linewidth=2.5)
    ax.hist(recon, histtype="step", bins=binning, density=True, label='Recon', color='c', linewidth=2.5)
    ax.hist(sampled, histtype="step", bins=binning, density=True, label='Sampled', color='orange', linewidth=2.5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_yscale('log' if log_scale else 'linear')
    ax.grid(True)
    ax.legend()

def get_binning(data1, data2, data3, num_bins=30):
    """
    Helper function to calculate binning from three datasets.
    """
    all_data = np.concatenate([data1, data2, data3])
    if all_data.size == 0:
        print("Warning: No data to calculate binning.")
        return np.linspace(0, 1, num_bins + 1)
        
    max_value = all_data.max()
    min_value = all_data.min()
    if min_value == max_value:
        print(f"Warning: min and max values are the same ({min_value}), adjusting.")
        min_value = min_value - 0.1
        max_value = max_value + 0.1
        
    # Use linspace for num_bins + 1 edges (which gives num_bins)
    return np.linspace(min_value, max_value, num_bins + 1)

def layer_plots(cfg, incident_energies, target_showers, recon_showers, sampled_showers):
    """
    Plot the energy sums, ratios, and sparsity for each layer,
    and calculate Chi-Squared distances.
    """
    dataset_name = cfg.data.dataset_name.lower()
    layer_cell_count = cfg.data.r * cfg.data.phi
    epsilon = 1e-7
    num_rows = int(np.ceil(len(cfg.data.relevantLayers) / 4.0))

    fig_set = {
        'energy_sum': plt.subplots(num_rows, 4, figsize=(15, 3*num_rows), constrained_layout=True),
        'incidence_ratio': plt.subplots(num_rows, 4, figsize=(15, 3*num_rows), constrained_layout=True),
        'target_recon_ratio': plt.subplots(num_rows, 4, figsize=(15, 3*num_rows), constrained_layout=True),
        'sparsity': plt.subplots(num_rows, 4, figsize=(15, 3*num_rows), constrained_layout=True)
    }
    
    # This dictionary will store all the Chi-Squared metrics
    chi2_metrics = {
        'energy_sum': {'recon': [], 'sampled': []},
        'incidence_ratio': {'recon': [], 'sampled': []},
        'sparsity': {'recon': [], 'sampled': []}
    }

    for i, layer_num in enumerate(cfg.data.relevantLayers):
        row = i // 4
        col = i % 4
        idx_prev = i * layer_cell_count
        idx = (i + 1) * layer_cell_count

        target_layer = target_showers[:, idx_prev:idx]
        recon_layer = recon_showers[:, idx_prev:idx]
        sampled_layer = sampled_showers[:, idx_prev:idx]

        # --- Energy Sum ---
        target_data = torch.sum(target_layer, dim=1).numpy()
        recon_data = torch.sum(recon_layer, dim=1).numpy()
        sampled_data = torch.sum(sampled_layer, dim=1).numpy()

        binning = get_binning(target_data, recon_data, sampled_data)
        plot_histograms(fig_set['energy_sum'][1][row, col], target_data, recon_data, sampled_data,
                        xlabel='Deposited Energy (MeV)', ylabel='Density',
                        title=f'Layer {layer_num} - Energy Sum', binning=binning)
        
        chi2_metrics['energy_sum']['recon'].append((layer_num, calculate_chi_squared_distance(target_data, recon_data, binning)))
        chi2_metrics['energy_sum']['sampled'].append((layer_num, calculate_chi_squared_distance(target_data, sampled_data, binning)))

        # --- Incidence Ratio ---
        target_data = (torch.sum(target_layer, dim=1) / (incident_energies.view(-1) + epsilon)).numpy()
        recon_data = (torch.sum(recon_layer, dim=1) / (incident_energies.view(-1) + epsilon)).numpy()
        sampled_data = (torch.sum(sampled_layer, dim=1) / (incident_energies.view(-1) + epsilon)).numpy()

        binning = get_binning(target_data, recon_data, sampled_data)
        plot_histograms(fig_set['incidence_ratio'][1][row, col], target_data, recon_data, sampled_data,
                        xlabel='Deposited Energy / Incident Energy', ylabel='Density',
                        title=f'Layer {layer_num} - Incidence Ratio', binning=binning)

        chi2_metrics['incidence_ratio']['recon'].append((layer_num, calculate_chi_squared_distance(target_data, recon_data, binning)))
        chi2_metrics['incidence_ratio']['sampled'].append((layer_num, calculate_chi_squared_distance(target_data, sampled_data, binning)))

        # --- Sparsity ---
        target_data = ((target_layer == 0).sum(dim=1) / target_layer.shape[1]).numpy()
        recon_data = ((recon_layer == 0).sum(dim=1) / recon_layer.shape[1]).numpy()
        sampled_data = ((sampled_layer == 0).sum(dim=1) / sampled_layer.shape[1]).numpy()

        binning = get_binning(target_data, recon_data, sampled_data)
        plot_histograms(fig_set['sparsity'][1][row, col], target_data, recon_data, sampled_data,
                        xlabel='Sparsity', ylabel='Density',
                        title=f'Layer {layer_num} - Sparsity', binning=binning)

        chi2_metrics['sparsity']['recon'].append((layer_num, calculate_chi_squared_distance(target_data, recon_data, binning)))
        chi2_metrics['sparsity']['sampled'].append((layer_num, calculate_chi_squared_distance(target_data, sampled_data, binning)))

        # --- Target/Recon Ratio (No Chi-Squared as it has no target) ---
        target_recon_ratio_layer = (torch.sum(recon_layer, dim=1) / (torch.sum(target_layer, dim=1).view(-1) + epsilon)).numpy()
        
        max_ratio_layer = target_recon_ratio_layer.max()
        min_ratio_layer = target_recon_ratio_layer.min()
        if max_ratio_layer == min_ratio_layer:
            max_ratio_layer += 0.1
        binning_ratio_layer = np.arange(min_ratio_layer, max_ratio_layer, (max_ratio_layer - min_ratio_layer) / 30)
        
        ax = fig_set['target_recon_ratio'][1][row, col]
        ax.hist(target_recon_ratio_layer, histtype="stepfilled", bins=binning_ratio_layer,
                density=True, alpha=0.7, label='Reconstructed / Target', color='c', linewidth=2.5)
        ax.set_xlabel('Recon / Target Energy Ratio')
        ax.set_ylabel('Density')
        ax.set_yscale('log')
        ax.grid(True)
        ax.axvline(1, color='r', linestyle='--', label='Ideal Ratio (1.0)')
        ax.legend()
        ax.set_title(f'Layer {layer_num} - Recon/Target Ratio')


    for fig, _ in fig_set.values():
        fig.tight_layout()

    return (fig_set['energy_sum'][0], fig_set['incidence_ratio'][0],
            fig_set['target_recon_ratio'][0], fig_set['sparsity'][0],
            chi2_metrics)


def vae_plots(cfg, incident_energies, target_showers, recon_showers, sampled_showers, incidence_energy_sampled = None):
    """
    Plot overall and binned metrics, and calculate Chi-Squared distances.
    """
    dataset_name = cfg.data.dataset_name.lower()
    epsilon = 1e-7

    # --- Calculate all data first ---
    target_energy_sums = torch.sum(target_showers, dim=1)
    recon_energy_sums = torch.sum(recon_showers, dim=1)
    sampled_energy_sums = torch.sum(sampled_showers, dim=1)

    target_incidence_ratio = target_energy_sums / (incident_energies.view(-1) + epsilon)
    recon_incidence_ratio = recon_energy_sums / (incident_energies.view(-1) + epsilon)
    if incidence_energy_sampled is not None:
        sampled_incidence_ratio = sampled_energy_sums / (incidence_energy_sampled.view(-1) + epsilon)
    else:
        sampled_incidence_ratio = sampled_energy_sums / (incident_energies.view(-1) + epsilon)

    target_recon_ratio = recon_energy_sums / (target_energy_sums.view(-1) + epsilon)

    target_sparsity = (target_showers == 0).sum(dim=1) / target_showers.shape[1]
    recon_sparsity = (recon_showers == 0).sum(dim=1) / recon_showers.shape[1]
    sampled_sparsity = (sampled_showers == 0).sum(dim=1) / sampled_showers.shape[1]

    # --- Move to numpy ---
    target_energy_sums_np = target_energy_sums.numpy()
    recon_energy_sums_np = recon_energy_sums.numpy()
    sampled_energy_sums_np = sampled_energy_sums.numpy()

    target_incidence_ratio_np = target_incidence_ratio.numpy()
    recon_incidence_ratio_np = recon_incidence_ratio.numpy()
    sampled_incidence_ratio_np = sampled_incidence_ratio.numpy()

    target_recon_ratio_np = target_recon_ratio.numpy()
    incident_energies_np = incident_energies.numpy().squeeze()
    if incidence_energy_sampled is not None:
        incidence_energy_sampled_np = incidence_energy_sampled.numpy().squeeze()

    target_sparsity_np = target_sparsity.numpy()
    recon_sparsity_np = recon_sparsity.numpy()
    sampled_sparsity_np = sampled_sparsity.numpy()
    
    # --- Initialize Chi-Squared metric storage ---
    all_chi2_metrics = {
        'overall_energy_sum': {},
        'overall_incidence_ratio': {},
        'overall_sparsity': {},
        'binned_energy_sum': {'recon': [], 'sampled': []},
        'binned_incidence_ratio': {'recon': [], 'sampled': []},
        'binned_sparsity': {'recon': [], 'sampled': []},
        'layer': {} # Will be filled by layer_plots
    }

    # --- Overall Plots ---
    overall_fig, overall_ax = plt.subplots(2, 2, figsize=(15, 15))

    # Energy Sum
    binning = get_binning(target_energy_sums_np, recon_energy_sums_np, sampled_energy_sums_np)
    plot_histograms(overall_ax[0, 0], target_energy_sums_np, recon_energy_sums_np, sampled_energy_sums_np,
                    xlabel='Deposited Energy (MeV)', ylabel='Density', title="Overall Deposited Energy", binning=binning)
    all_chi2_metrics['overall_energy_sum']['recon'] = calculate_chi_squared_distance(target_energy_sums_np, recon_energy_sums_np, binning)
    all_chi2_metrics['overall_energy_sum']['sampled'] = calculate_chi_squared_distance(target_energy_sums_np, sampled_energy_sums_np, binning)

    # Incidence Ratio
    binning = get_binning(target_incidence_ratio_np, recon_incidence_ratio_np, sampled_incidence_ratio_np)
    plot_histograms(overall_ax[0, 1], target_incidence_ratio_np, recon_incidence_ratio_np, sampled_incidence_ratio_np,
                    xlabel='Deposited Energy / Incident Energy', ylabel='Density', title="Overall Energy Ratio", binning=binning)
    all_chi2_metrics['overall_incidence_ratio']['recon'] = calculate_chi_squared_distance(target_incidence_ratio_np, recon_incidence_ratio_np, binning)
    all_chi2_metrics['overall_incidence_ratio']['sampled'] = calculate_chi_squared_distance(target_incidence_ratio_np, sampled_incidence_ratio_np, binning)

    # Sparsity
    binning = get_binning(target_sparsity_np, recon_sparsity_np, sampled_sparsity_np)
    plot_histograms(overall_ax[1, 1], target_sparsity_np, recon_sparsity_np, sampled_sparsity_np,
                    xlabel='Sparsity', ylabel='Density', title="Overall Sparsity", binning=binning)
    all_chi2_metrics['overall_sparsity']['recon'] = calculate_chi_squared_distance(target_sparsity_np, recon_sparsity_np, binning)
    all_chi2_metrics['overall_sparsity']['sampled'] = calculate_chi_squared_distance(target_sparsity_np, sampled_sparsity_np, binning)

    # Target/Recon Ratio
    max_ratio = target_recon_ratio_np.max()
    min_ratio = target_recon_ratio_np.min()
    if min_ratio == max_ratio:
        print("Warning: min and max values are the same, adjusting to avoid division by zero.")
        max_ratio += 0.1
    binning_ratio = np.arange(min_ratio, max_ratio, (max_ratio - min_ratio) / 30)
    overall_ax[1, 0].hist(target_recon_ratio_np, histtype="stepfilled", bins=binning_ratio, density=True,
                          alpha=0.7, label='Recon / Target', color='c', linewidth=2.5)
    overall_ax[1, 0].set_xlabel('Recon / Target Energy Ratio')
    overall_ax[1, 0].set_ylabel('Density')
    overall_ax[1, 0].set_yscale('log')
    overall_ax[1, 0].grid(True)
    overall_ax[1, 0].axvline(1, color='r', linestyle='--', label='Ideal Ratio (1.0)')
    overall_ax[1, 0].legend()

    overall_fig.tight_layout()

    # --- Binned Plots ---
    if "custom" in dataset_name:
        max_energy = float(300100.0)
        bin_width = max_energy/ 15
        first_center = bin_width / 2.0       
        last_center = max_energy - (bin_width / 2.0) 
        energy_bin_centers = [first_center + i * bin_width for i in range(15)]
    elif 'atlas' in dataset_name:
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
        if "custom" in dataset_name:
            e_low = energy_center - (bin_width / 2.0)
            e_high = energy_center + (bin_width / 2.0)
        else:
            e_low = 2 ** (np.log2(energy_center) - 0.5)
            e_high = 2 ** (np.log2(energy_center) + 0.5)

        mask = (incident_energies_np >= e_low) & (incident_energies_np < e_high)
        if incidence_energy_sampled is not None:
            sampled_mask = (incidence_energy_sampled_np >= e_low) & (incidence_energy_sampled_np < e_high)
        else:
            sampled_mask = mask
            
        if mask.sum() == 0:
            print(f"No data in energy range {e_low:.1f} - {e_high:.1f} MeV, skipping this bin.")
            continue

        # Energy Sum (Binned)
        target_data = target_energy_sums_np[mask]
        recon_data = recon_energy_sums_np[mask]
        sampled_data = sampled_energy_sums_np[sampled_mask]
        
        binning = get_binning(target_data, recon_data, sampled_data)
        plot_histograms(ax_energy_sum[row, col], target_data, recon_data, sampled_data,
                        xlabel='Deposited Energy (MeV)', ylabel='Density',
                        title=f'Energy ~ {e_low / 1000:.1f} - {e_high / 1000:.1f} GeV', binning=binning)
        
        if target_data.size > 0 and recon_data.size > 0:
            chi2 = calculate_chi_squared_distance(target_data, recon_data, binning)
            all_chi2_metrics['binned_energy_sum']['recon'].append((energy_center, chi2))
        if target_data.size > 0 and sampled_data.size > 0:
            chi2 = calculate_chi_squared_distance(target_data, sampled_data, binning)
            all_chi2_metrics['binned_energy_sum']['sampled'].append((energy_center, chi2))

        # Incidence Ratio (Binned)
        target_data = target_incidence_ratio_np[mask]
        recon_data = recon_incidence_ratio_np[mask]
        sampled_data = sampled_incidence_ratio_np[sampled_mask]

        binning = get_binning(target_data, recon_data, sampled_data)
        plot_histograms(ax_incidence_ratio[row, col], target_data, recon_data, sampled_data,
                        xlabel='Deposited Energy / Incident Energy', ylabel='Density',
                        title=f'Energy Ratio ~ {e_low / 1000:.1f} - {e_high / 1000:.1f} GeV', binning=binning)

        if target_data.size > 0 and recon_data.size > 0:
            chi2 = calculate_chi_squared_distance(target_data, recon_data, binning)
            all_chi2_metrics['binned_incidence_ratio']['recon'].append((energy_center, chi2))
        if target_data.size > 0 and sampled_data.size > 0:
            chi2 = calculate_chi_squared_distance(target_data, sampled_data, binning)
            all_chi2_metrics['binned_incidence_ratio']['sampled'].append((energy_center, chi2))

        # Sparsity (Binned)
        target_data = target_sparsity_np[mask]
        recon_data = recon_sparsity_np[mask]
        sampled_data = sampled_sparsity_np[sampled_mask]

        binning = get_binning(target_data, recon_data, sampled_data)
        plot_histograms(ax_sparsity[row, col], target_data, recon_data, sampled_data,
                        xlabel='Sparsity', ylabel='Density',
                        title=f'Sparsity ~ {e_low / 1000:.1f} - {e_high / 1000:.1f} GeV', binning=binning)

        if target_data.size > 0 and recon_data.size > 0:
            chi2 = calculate_chi_squared_distance(target_data, recon_data, binning)
            all_chi2_metrics['binned_sparsity']['recon'].append((energy_center, chi2))
        if target_data.size > 0 and sampled_data.size > 0:
            chi2 = calculate_chi_squared_distance(target_data, sampled_data, binning)
            all_chi2_metrics['binned_sparsity']['sampled'].append((energy_center, chi2))


        # Target/Recon Ratio (Binned)
        target_recon_ratio_e = target_recon_ratio_np[mask]
        if target_recon_ratio_e.size > 0:
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
            ax_target_recon_ratio[row, col].set_title(f'Recon Ratio ~ {e_low / 1000:.1f} - {e_high / 1000:.1f} GeV')
        

    fig_energy_sum.tight_layout()
    fig_incidence_ratio.tight_layout()
    fig_target_recon_ratio.tight_layout()
    fig_sparsity.tight_layout()

    # --- Layer Plots ---
    (energy_sum_layer_fig, incidence_ratio_layer_fig, 
     target_recon_ratio_layer_fig, sparsity_layer_fig, 
     layer_chi2_metrics) = layer_plots(cfg, incident_energies, target_showers, recon_showers, sampled_showers)
    
    all_chi2_metrics['layer'] = layer_chi2_metrics

    return (overall_fig, fig_energy_sum, fig_incidence_ratio, fig_target_recon_ratio, fig_sparsity, 
            energy_sum_layer_fig, incidence_ratio_layer_fig, target_recon_ratio_layer_fig, sparsity_layer_fig,
            all_chi2_metrics)

def corr_plots(cfg, post_logits, post_samples, prior_samples):
    p_size = cfg.rbm.latent_nodes_per_p
    p_cond_size = cfg.model.cond_p_size
    p_0 = post_samples[:, :p_cond_size].cpu()
    post_probs = torch.sigmoid(post_logits)
    post_probs = torch.cat([p_0, post_probs], dim=1)
    post_correlations = torch.corrcoef(post_probs.cpu().T).cpu().numpy()
    # 0 out diagonal
    np.fill_diagonal(post_correlations, 0)
    # 0 out first p_cond_size x p_size block
    post_correlations[:p_cond_size, :p_cond_size] = 0
    post_correlations = np.nan_to_num(post_correlations, nan=0.0)

    post_fig = plt.figure(figsize=(8,8))
    plt.imshow(post_correlations, cmap='seismic', vmin=-1, vmax=1, interpolation="none")
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.title('Posterior Correlation Matrix')

    prior_correlations = torch.corrcoef(prior_samples.cpu().T).cpu().numpy()
    # 0 out diagonal
    np.fill_diagonal(prior_correlations, 0)
    prior_correlations = np.nan_to_num(prior_correlations, nan=0.0)
    # 0 out first p_cond_size x p_cond_size block
    prior_correlations[:p_cond_size, :p_cond_size] = 0

    prior_fig = plt.figure(figsize=(8,8))
    plt.imshow(prior_correlations, cmap='seismic', vmin=-1, vmax=1, interpolation="none")
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.title('Prior Correlation Matrix')

    # posterior partition to partition correlations
    post_partition_dict = {
        0: post_probs[:, :p_cond_size].cpu(),
        1: post_probs[:, p_cond_size:p_cond_size+p_size].cpu(),
        2: post_probs[:, p_cond_size+p_size:p_cond_size+2*p_size].cpu(),
        3: post_probs[:, p_cond_size+2*p_size:p_cond_size+3*p_size].cpu()
    }
    post_partition_fig, post_partition_ax = plt.subplots(2, 3, figsize=(12, 8))
    post_partition_ax = post_partition_ax.flatten()

    for idx, (i, j) in enumerate(itertools.combinations(range(4), 2)):
        part_i = post_partition_dict[i].numpy()
        part_j = post_partition_dict[j].numpy()
        size_i = part_i.shape[1]  # Get the size of the first partition

        corr_ij = np.corrcoef(
            part_i.T,
            part_j.T
        )[0:size_i, size_i:]  # Use dynamic size_i for slicing        corr_ij = np.nan_to_num(corr_ij, nan=0.0)

        ax = post_partition_ax[idx]
        im = ax.imshow(corr_ij, cmap='seismic', vmin=-1, vmax=1, interpolation="none")
        ax.set_title(f'Posterior Corr: Partition {i} vs {j}')
        plt.colorbar(im, ax=ax)
        ax.invert_yaxis()
    post_partition_fig.tight_layout()

    # prior partition to partition correlations
    prior_partition_dict = {
        0: prior_samples[:, :p_cond_size].cpu(),
        1: prior_samples[:, p_cond_size:p_cond_size+p_size].cpu(),
        2: prior_samples[:, p_cond_size+p_size:p_cond_size+2*p_size].cpu(),
        3: prior_samples[:, p_cond_size+2*p_size:p_cond_size+3*p_size].cpu()}
    prior_partition_fig, prior_partition_ax = plt.subplots(2, 3, figsize=(12, 8))
    prior_partition_ax = prior_partition_ax.flatten() 
    for idx, (i, j) in enumerate(itertools.combinations(range(4), 2)):
        part_i = prior_partition_dict[i].numpy()
        part_j = prior_partition_dict[j].numpy()
        size_i = part_i.shape[1]  # Get the size of the first partition

        corr_ij = np.corrcoef(
            part_i.T,
            part_j.T
        )[0:size_i, size_i:]  # Use dynamic size_i for slicing
        corr_ij = np.nan_to_num(corr_ij, nan=0.0)

        ax = prior_partition_ax[idx]
        im = ax.imshow(corr_ij, cmap='seismic', vmin=-1, vmax=1, interpolation="none")
        ax.set_title(f'Prior Corr: Partition {i} vs {j}')
        plt.colorbar(im, ax=ax)
        ax.invert_yaxis()
    prior_partition_fig.tight_layout()
    return post_fig, prior_fig, post_partition_fig, prior_partition_fig


def incidence_energy_plots(gt_incidence_energies, sampled_incidence_energies):
    """
    Plot the incidence energy distributions for ground truth and sampled data.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    max_value = max(gt_incidence_energies.max(), sampled_incidence_energies.max())
    min_value = min(gt_incidence_energies.min(), sampled_incidence_energies.min())
    if min_value == max_value:
        print("Warning: min and max values are the same, adjusting to avoid division by zero.")
        max_value += 0.1  # Avoid division by zero if all values are the same
    binning = np.arange(min_value, max_value, (max_value - min_value) / 30)

    ax.hist(gt_incidence_energies, histtype="stepfilled", bins=binning, density=True, alpha=0.7, label='Ground Truth', color='b', linewidth=2.5)
    ax.hist(sampled_incidence_energies, histtype="step", bins=binning, density=True, label='Sampled', color='orange', linewidth=2.5)

    ax.set_xlabel('Incidence Energy (MeV)')
    ax.set_ylabel('Density')
    ax.set_title('Incidence Energy Distribution')
    ax.set_yscale('log')
    ax.grid(True)
    ax.legend()

    return fig


