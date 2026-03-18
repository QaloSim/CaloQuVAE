import numpy as np
import matplotlib.pyplot as plt
import wandb
import torch
import itertools

from utils.optimization.scalar_metrics import calculate_wasserstein
from utils.atlas_plots import create_grid_figure, to_np
from utils.HLF.atlasgeo import FeatureAdapter
from utils.evaluate_transfusion import plot_hist_wrapper

def evaluate_sparsity(cfg, gt, recon, incident_energies):
    """Calculates WS and plots sparsity overall, per layer, and conditioned on incident energy."""
    metrics = {}
    layer_cell_count = cfg.data.r * cfg.data.phi
    dataset_name = cfg.data.dataset_name.lower()
    
    # Overall & Layer-wise Sparsity
    gt_overall = ((gt == 0).sum(dim=1).float() / gt.shape[1]).cpu().numpy()
    recon_overall = ((recon == 0).sum(dim=1).float() / recon.shape[1]).cpu().numpy()
    incident_energies_np = incident_energies.cpu().numpy().squeeze()
    
    metric_overall = calculate_wasserstein(gt_overall, recon_overall, name="ws_overall_sparsity", max_range=True)
    metrics[metric_overall.name] = metric_overall.score
    
    num_layers = len(cfg.data.relevantLayers)
    num_rows = int(np.ceil((num_layers + 1) / 4.0))
    fig_layer, axes_layer = plt.subplots(num_rows, 4, figsize=(15, 3 * num_rows), constrained_layout=True)
    axes_layer = axes_layer.flatten()
    
    plot_hist_wrapper(axes_layer[0], gt_overall, recon_overall, 'Sparsity', 'Density', 'Overall Sparsity', metric_overall)
    
    for i, layer_num in enumerate(cfg.data.relevantLayers):
        idx_prev = i * layer_cell_count
        idx = (i + 1) * layer_cell_count
        
        gt_layer = ((gt[:, idx_prev:idx] == 0).sum(dim=1).float() / layer_cell_count).cpu().numpy()
        recon_layer = ((recon[:, idx_prev:idx] == 0).sum(dim=1).float() / layer_cell_count).cpu().numpy()
        
        metric_layer = calculate_wasserstein(gt_layer, recon_layer, name=f"ws_sparsity_layer_{layer_num}", max_range=True)
        metrics[metric_layer.name] = metric_layer.score
        
        plot_hist_wrapper(axes_layer[i+1], gt_layer, recon_layer, 'Sparsity', 'Density', f'Layer {layer_num} Sparsity', metric_layer)
        
    # Hide any unused subplots in the layer grid
    for j in range(num_layers + 1, len(axes_layer)):
        axes_layer[j].set_visible(False)

    # Conditioned (Binned) Sparsity
    max_energy = float(300100.0)
    bin_width = max_energy / 15
    first_center = bin_width / 2.0       
    energy_bin_centers = [first_center + i * bin_width for i in range(15)]
        
    fig_cond, axes_cond = plt.subplots(3, 5, figsize=(16, 10), constrained_layout=True)
    
    for i, energy_center in enumerate(energy_bin_centers):
        row = i // 5
        col = i % 5
        
        e_low = energy_center - (bin_width / 2.0)
        e_high = energy_center + (bin_width / 2.0)

        mask = (incident_energies_np >= e_low) & (incident_energies_np < e_high)
        
        if mask.sum() == 0:
            axes_cond[row, col].set_title(f"Empty Bin\n{e_low/1000:.1f} - {e_high/1000:.1f} GeV")
            axes_cond[row, col].axis('off')
            continue
            
        gt_cond = gt_overall[mask]
        recon_cond = recon_overall[mask]
        
        metric_cond = calculate_wasserstein(gt_cond, recon_cond, name=f"ws_sparsity_binned_{energy_center}", max_range=True)
        metrics[metric_cond.name] = metric_cond.score
        
        plot_hist_wrapper(axes_cond[row, col], gt_cond, recon_cond, 
                          'Sparsity', 'Density', 
                          f'Sparsity ~ {e_low / 1000:.1f} - {e_high / 1000:.1f} GeV', 
                          metric_cond)

    return fig_layer, fig_cond, metrics


def evaluate_high_level_features(feature_extractor, geo_handler, gt, recon, incident_energies, device='cuda', batch_size=1024):
    """Extracts HLFs in chunks to prevent VRAM OOM, calculates WD, and plots grid figures."""
    feature_extractor.to(device)
    feature_extractor.eval()
    metrics = {}
    
    num_samples = gt.shape[0]
    feats_gt_accum = {}
    feats_recon_accum = {}

    with torch.no_grad():
        # Process the monolithic arrays in memory-safe chunks
        for i in range(0, num_samples, batch_size):
            # Slice the arrays and move only the chunk to the target device
            gt_batch = torch.as_tensor(gt[i:i+batch_size], dtype=torch.float32, device=device)
            recon_batch = torch.as_tensor(recon[i:i+batch_size], dtype=torch.float32, device=device)
            
            batch_feats_gt = feature_extractor(gt_batch)
            batch_feats_recon = feature_extractor(recon_batch)

            # Initialize the accumulation dictionaries on the first pass
            if not feats_gt_accum:
                for k in batch_feats_gt.keys():
                    feats_gt_accum[k] = []
                    feats_recon_accum[k] = []

            # Append features and instantly move them to CPU RAM to clear VRAM
            for k in batch_feats_gt.keys():
                feats_gt_accum[k].append(batch_feats_gt[k].cpu())
                feats_recon_accum[k].append(batch_feats_recon[k].cpu())

    # Concatenate the chunks back into monolithic tensors on the CPU
    feats_gt = {k: torch.cat(v, dim=0) for k, v in feats_gt_accum.items()}
    feats_recon = {k: torch.cat(v, dim=0) for k, v in feats_recon_accum.items()}
    
    # Ensure incident energies are also CPU tensors for the Adapter
    inc_e_tensor = torch.as_tensor(incident_energies, dtype=torch.float32, device='cpu')
    
    adapters = {
        "Data": FeatureAdapter(feats_gt, geo_handler.relevant_layers, inc_e_tensor),
        "Recon": FeatureAdapter(feats_recon, geo_handler.relevant_layers, inc_e_tensor)
    }

    # Calculate WS Metrics
    for key in feats_gt.keys():
        data_gt = feats_gt[key].numpy()
        data_recon = feats_recon[key].numpy()

        if data_gt.ndim == 2:
            for l in range(data_gt.shape[1]):
                metric = calculate_wasserstein(data_gt[:, l], data_recon[:, l], name=f"ws_HLF_layer_{l}_{key}", max_range=True)
                metrics[metric.name] = metric.score
        else:
            metric = calculate_wasserstein(data_gt, data_recon, name=f"ws_HLF_global_{key}", max_range=True)
            metrics[metric.name] = metric.score

    # Populate Grid Figures
    grids = {"mean_eta": {}, "width_eta": {}, "mean_phi": {}, "width_phi": {}}
    ref_adapter, model_adapter = adapters["Data"], adapters["Recon"]

    for layer in geo_handler.relevant_layers:
        grids["mean_eta"][layer] = {'ref': to_np(ref_adapter.EC_etas[layer]), 'models': [to_np(model_adapter.EC_etas[layer])]}
        grids["width_eta"][layer] = {'ref': to_np(ref_adapter.width_etas[layer]), 'models': [to_np(model_adapter.width_etas[layer])]}
        grids["mean_phi"][layer] = {'ref': to_np(ref_adapter.EC_phis[layer]), 'models': [to_np(model_adapter.EC_phis[layer])]}
        grids["width_phi"][layer] = {'ref': to_np(ref_adapter.width_phis[layer]), 'models': [to_np(model_adapter.width_phis[layer])]}

    figs = {
        "HLF/Grid_Mean_Eta": create_grid_figure(grids["mean_eta"], 'Mean Eta', ["Recon"], yscale='log'),
        "HLF/Grid_Width_Eta": create_grid_figure(grids["width_eta"], 'Width Eta', ["Recon"], yscale='log'),
        "HLF/Grid_Mean_Phi": create_grid_figure(grids["mean_phi"], 'Mean Phi', ["Recon"], yscale='log'),
        "HLF/Grid_Width_Phi": create_grid_figure(grids["width_phi"], 'Width Phi', ["Recon"], yscale='log')
    }

    return figs, metrics


def plot_posterior_correlations(cfg, post_logits, post_samples):
    """Plots correlation matrix for the posterior latent space (no prior)."""
    p_size = cfg.rbm.latent_nodes_per_p
    p_cond_size = cfg.model.cond_p_size
    p_0 = post_samples[:, :p_cond_size].cpu()

    post_probs = torch.sigmoid(post_logits).cpu()
    post_probs = torch.cat([p_0, post_probs], dim=1)
    post_correlations = torch.corrcoef(post_probs.T).numpy()
    np.fill_diagonal(post_correlations, 0)
    
    # 0 out conditional block if applicable
    if p_cond_size > 0:
        post_correlations[:p_cond_size, :p_cond_size] = 0
        
    post_correlations = np.nan_to_num(post_correlations, nan=0.0)

    fig = plt.figure(figsize=(8,8))
    plt.imshow(post_correlations, cmap='seismic', vmin=-1, vmax=1, interpolation="none")
    plt.colorbar()
    plt.gca().invert_yaxis()
    plt.title('Posterior Correlation Matrix')

    return fig

def plot_latent_node_activations(post_logits):
    """Plots the average activation probability of learned latent nodes to identify 'dead' nodes."""
    # Convert logits to probabilities (only learned nodes)
    post_probs = torch.sigmoid(post_logits).cpu()
    
    # Calculate mean activation
    mean_probs = post_probs.float().mean(dim=0).numpy()
    
    x = np.arange(len(mean_probs))
    shifted = mean_probs - 0.5

    fig, ax = plt.subplots(figsize=(14, 6))
    
    ax.bar(x, shifted, bottom=0.5, color='royalblue', alpha=0.8, edgecolor='black', linewidth=0.5, zorder=3)

    ax.set_title('Learned Latent Dimension Utilization (Deviation from 0.5 Baseline)', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Learned Latent Dimension Index', fontsize=12)
    ax.set_ylabel('Average Activation Probability', fontsize=12)
    
    ax.axhline(0.5, color='black', linewidth=1.5, linestyle='--', zorder=2)
    ax.set_ylim(0, 1)
    ax.set_xlim(-1, len(mean_probs))
    
    ax.grid(axis='y', linestyle='--', alpha=0.6, zorder=0)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    fig.tight_layout()
    return fig

def evaluate_layer_ae_distributions(cfg, gt, recon, incident_energies, post_logits, post_samples, feature_extractor, geo_handler, close_plots=True, device="cpu"):
    """Master orchestrator for Layer AE evaluation."""
    statistics = {}
    plots = {}

    # Sparsity
    fig_sparsity_layer, fig_sparsity_cond, ws_sparsity = evaluate_sparsity(cfg, gt, recon, incident_energies)
    statistics.update(ws_sparsity)
    plots["sparsity_distributions"] = wandb.Image(fig_sparsity_layer)
    plots["sparsity_conditioned"] = wandb.Image(fig_sparsity_cond)
    
    # High Level Features (WD + Grids)
    fig_hlf_grids, ws_hlf = evaluate_high_level_features(feature_extractor, geo_handler, gt, recon, incident_energies, device=device)
    statistics.update(ws_hlf)
    for name, fig in fig_hlf_grids.items():
        if fig is not None:
            plots[name] = wandb.Image(fig)
            if close_plots: plt.close(fig)

    # Latent Correlations
    fig_corr = plot_posterior_correlations(cfg, post_logits, post_samples)
    plots["posterior_correlation_matrix"] = wandb.Image(fig_corr)

    # Latent Activations ("Dead Nodes" - Learned Only)
    fig_activations = plot_latent_node_activations(post_logits)
    plots["latent_node_activations"] = wandb.Image(fig_activations)

    if close_plots:
        plt.close(fig_sparsity_layer)
        plt.close(fig_sparsity_cond)
        plt.close(fig_corr)
        plt.close(fig_activations)

    return statistics, plots