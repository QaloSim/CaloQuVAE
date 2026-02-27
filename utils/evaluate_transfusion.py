import numpy as np
import matplotlib.pyplot as plt
import wandb
import torch

# Adjust this import to point to wherever you keep your metric utilities
from utils.optimization.scalar_metrics import calculate_wasserstein

def plot_hist_wrapper(ax, target, sampled, xlabel, ylabel, title, metric_result, log_scale=True):
    """Plots GT vs Sampled histograms using cleanly computed adaptive bins."""
    bins = metric_result.bins
    
    ax.hist(target, histtype="stepfilled", bins=bins, density=True, alpha=0.7, label='Target', color='b', linewidth=2.5)
    ax.hist(sampled, histtype="step", bins=bins, density=True, label='Sampled', color='orange', linewidth=2.5)
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{title}\nNorm WS: {metric_result.score:.4f}")
    ax.set_yscale('log' if log_scale else 'linear')
    ax.grid(True)
    ax.legend()

def plot_layer_correlations(gt, samples, relevant_layers):
    """Computes and plots 5x5 correlation matrices and the Frobenius norm."""
    corr_gt = np.nan_to_num(np.corrcoef(gt.T), nan=0.0)
    corr_sampled = np.nan_to_num(np.corrcoef(samples.T), nan=0.0)
    
    corr_diff = corr_gt - corr_sampled
    frob_norm = np.linalg.norm(corr_diff, ord='fro')
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    im0 = axes[0].imshow(corr_gt, cmap='seismic', vmin=-1, vmax=1)
    axes[0].set_title('Ground Truth Correlation')
    fig.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].imshow(corr_sampled, cmap='seismic', vmin=-1, vmax=1)
    axes[1].set_title('Sampled Correlation')
    fig.colorbar(im1, ax=axes[1])
    
    im2 = axes[2].imshow(corr_diff, cmap='seismic', vmin=-2, vmax=2)
    axes[2].set_title(f'Difference (GT - Sampled)\nFrobenius Norm: {frob_norm:.4f}')
    fig.colorbar(im2, ax=axes[2])
    tick_pos = np.arange(len(relevant_layers))
    
    for ax in axes:
        ax.set_xlabel("Layer Index")
        ax.set_ylabel("Layer Index")
        ax.invert_yaxis()
        ax.set_xticks(tick_pos)
        ax.set_yticks(tick_pos)
        ax.set_xticklabels(relevant_layers)
        ax.set_yticklabels(relevant_layers)
        
    fig.tight_layout()
    return fig, frob_norm

def plot_sequence_and_individual_layers(gt, samples, relevant_layers):
    """Plots sequence sums and individual layer distributions."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    ws_scores = {}
    
    # 1. Total Sequence Sum
    gt_total = gt.sum(axis=1)
    sampled_total = samples.sum(axis=1)
    
    metric_total = calculate_wasserstein(gt_total, sampled_total, name="ws_total_energy", max_range=True)
    ws_scores["ws_total_energy"] = metric_total.score
    
    plot_hist_wrapper(axes[0], gt_total, sampled_total, 
                      xlabel='Total Energy (MeV)', ylabel='Density', 
                      title='Total Sequence Energy Sum', metric_result=metric_total)
    
    # 2-6. Individual Layers
    for i in range(5):
        gt_layer = gt[:, i]
        sampled_layer = samples[:, i]
        layer_name = relevant_layers[i] if i < len(relevant_layers) else i+1
        
        metric_layer = calculate_wasserstein(gt_layer, sampled_layer, name=f"ws_layer_{layer_name}_energy", max_range=True)
        ws_scores[metric_layer.name] = metric_layer.score
        
        plot_hist_wrapper(axes[i+1], gt_layer, sampled_layer,
                          xlabel=f'Layer {layer_name} Energy (MeV)', ylabel='Density',
                          title=f'Layer {layer_name} Energy Distribution', metric_result=metric_layer)
                          
    fig.tight_layout()
    return fig, ws_scores

def plot_sequence_and_individual_layers_ratio(gt, samples, incident_energies, relevant_layers):
    """Plots sequence sum and individual layer incidence ratios."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    ws_scores = {}
    
    inc_e = incident_energies.numpy().squeeze() if torch.is_tensor(incident_energies) else incident_energies.squeeze()
    inc_e_safe = inc_e + 1e-7 # Prevent division by zero
    
    # 1. Total Sequence Ratio
    gt_total = gt.sum(axis=1) / inc_e_safe
    sampled_total = samples.sum(axis=1) / inc_e_safe
    
    metric_total = calculate_wasserstein(gt_total, sampled_total, name="ws_total_energy_ratio", max_range=True)
    ws_scores["ws_total_energy_ratio"] = metric_total.score
    
    plot_hist_wrapper(axes[0], gt_total, sampled_total, 
                      xlabel='Deposited / Incident Energy', ylabel='Density', 
                      title='Total Energy Incidence Ratio', metric_result=metric_total)
    
    # 2-6. Individual Layers
    for i in range(5):
        gt_layer = gt[:, i] / inc_e_safe
        sampled_layer = samples[:, i] / inc_e_safe
        layer_name = relevant_layers[i] if i < len(relevant_layers) else i+1
        
        metric_layer = calculate_wasserstein(gt_layer, sampled_layer, name=f"ws_layer_{layer_name}_ratio", max_range=True)
        ws_scores[metric_layer.name] = metric_layer.score
        
        plot_hist_wrapper(axes[i+1], gt_layer, sampled_layer,
                          xlabel=f'Layer {layer_name} / Incident Energy', ylabel='Density',
                          title=f'Layer {layer_name} Incidence Ratio', metric_result=metric_layer)
                          
    fig.tight_layout()
    return fig, ws_scores

def plot_layers_conditioned(gt, samples, incident_energies, relevant_layers, num_bins=5):
    """Plots layer distributions conditioned on incident energy bins."""
    inc_e = incident_energies.numpy().squeeze() if torch.is_tensor(incident_energies) else incident_energies.squeeze()
    inc_bins = np.linspace(inc_e.min(), inc_e.max(), num_bins + 1)
    
    fig, axes = plt.subplots(num_bins, 5, figsize=(25, 4 * num_bins))
    ws_scores = {}
    
    for b in range(num_bins):
        b_min, b_max = inc_bins[b], inc_bins[b+1]
        mask = (inc_e >= b_min) & (inc_e < b_max)
        
        for l in range(5):
            ax = axes[b, l]
            if not np.any(mask):
                ax.set_title(f"Bin {b+1} Empty")
                continue
                
            gt_data = gt[mask, l]
            sampled_data = samples[mask, l]
            
            layer_name = relevant_layers[l] if l < len(relevant_layers) else l+1
            metric_cond = calculate_wasserstein(gt_data, sampled_data, name=f"ws_cond_bin{b+1}_layer_{layer_name}", max_range=True)
            ws_scores[metric_cond.name] = metric_cond.score
            
            plot_hist_wrapper(ax, gt_data, sampled_data,
                              xlabel=f'Layer {layer_name} Energy (MeV)', ylabel='Density',
                              title=f'Layer {layer_name} | E ~ [{b_min:.1f}, {b_max:.1f}) MeV', 
                              metric_result=metric_cond)
                            
    fig.tight_layout()
    return fig, ws_scores

def evaluate_transfusion_distributions(gt, samples, x0, close_plots=True, relevant_layers=[0, 1, 2, 3, 12]):
    """Master evaluation function."""
    gt = gt.numpy() if torch.is_tensor(gt) else gt
    samples = samples.numpy() if torch.is_tensor(samples) else samples
    # check for NaNs
    statistics = {}
    
    # 1. Correlations & Frob Norm
    fig_corr, frob_norm = plot_layer_correlations(gt, samples, relevant_layers)
    statistics["frob_norm_correlation_diff"] = frob_norm
    
    # 2. Sequence sum & 5 layers (Raw)
    fig_seq_layers, ws_raw = plot_sequence_and_individual_layers(gt, samples, relevant_layers)
    statistics.update(ws_raw)
    
    # 3. Sequence sum & 5 layers (Ratios)
    fig_seq_ratios, ws_ratios = plot_sequence_and_individual_layers_ratio(gt, samples, x0, relevant_layers)
    statistics.update(ws_ratios)
    
    # 4. Conditional layers
    fig_cond, ws_cond = plot_layers_conditioned(gt, samples, x0, relevant_layers)
    statistics.update(ws_cond)
    
    plots = {
        "correlation_matrices": wandb.Image(fig_corr),
        "sequence_and_layers_raw": wandb.Image(fig_seq_layers),
        "sequence_and_layers_ratio": wandb.Image(fig_seq_ratios),
        "conditioned_layers": wandb.Image(fig_cond)
    }
    if close_plots:
        plt.close(fig_corr)
        plt.close(fig_seq_layers)
        plt.close(fig_seq_ratios)
        plt.close(fig_cond)
        
    return statistics, plots