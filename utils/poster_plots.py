import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import os
from utils.atlas_plots import to_np
from utils.HLF.atlasgeo import DifferentiableFeatureExtractor, FeatureAdapter, AtlasGeometry
import torch
import mplhep as hep

# -----------------------------------------------------------------------------
# 1. Poster-Ready Plotting Core
# -----------------------------------------------------------------------------
def plot_poster_ratio(data_dict, xlabel, output_path, yscale='log', xscale='linear',
                      bins=50, color_cycle=None):
    """
    Generates a high-visibility histogram with fills and ratio subplot.
    """
    
    # --- Settings for Poster Visibility & ATLAS Style ---
    hep.style.use(hep.style.ATLAS) 
    plt.rcParams.update({'font.size': 14, 'axes.linewidth': 2})
    
    if color_cycle is None:
        color_cycle = ['black', '#DC267F', '#648FFF', '#E66100', '#5D3A9B']
    
    linestyle_cycle = ['-', '--', '-.', ':']
    
    # Extract and Validate Data
    labels = list(data_dict.keys())
    datasets = [np.array(d) for d in data_dict.values()]
    valid_datasets = [d[np.isfinite(d)] for d in datasets]
    
    if not valid_datasets or all(len(d) == 0 for d in valid_datasets):
        print(f"Skipping {output_path}: No valid data.")
        return

    # Determine Bins
    all_data_flat = np.concatenate(valid_datasets)
    if xscale == 'log':
        pos_data = all_data_flat[all_data_flat > 0]
        vmin = pos_data.min() if len(pos_data) > 0 else 1e-5
        vmax = pos_data.max() if len(pos_data) > 0 else 1.0
        bin_edges = np.logspace(np.log10(vmin), np.log10(vmax), bins)
    else:
        vmin, vmax = all_data_flat.min(), all_data_flat.max()
        bin_edges = np.linspace(vmin, vmax, bins)

    # Setup Canvas
    fig = plt.figure(figsize=(12, 12)) 
    gs = GridSpec(2, 1, height_ratios=[4, 1], hspace=0.08)
    ax_main = fig.add_subplot(gs[0])
    ax_ratio = fig.add_subplot(gs[1], sharex=ax_main)

    # --- Main Histogram Loop ---
    ref_hist = None
    
    for i, (label, data) in enumerate(zip(labels, valid_datasets)):
        color = color_cycle[i % len(color_cycle)]
        ls = linestyle_cycle[i % len(linestyle_cycle)]
        is_ref = (i == 0) 
        if is_ref: ls = '-' 
        lw = 2.5 if is_ref else 2.0

        counts, _ = np.histogram(data, bins=bin_edges)
        hist_density, _ = np.histogram(data, bins=bin_edges, density=True)
        y_step = np.append(hist_density, hist_density[-1])
        
        # Plot Main
        ax_main.step(bin_edges, y_step, color=color, linewidth=lw, linestyle=ls, where='post', label=label)
        
        if not is_ref:
            ax_main.fill_between(bin_edges, y_step, step='post', color=color, alpha=0.1)
        
        if is_ref:
            ref_hist = hist_density
            safe_counts = np.where(counts > 0, counts, 1)
            stat_err = hist_density / np.sqrt(safe_counts)
            err_low = np.append(hist_density - stat_err, (hist_density - stat_err)[-1])
            err_high = np.append(hist_density + stat_err, (hist_density + stat_err)[-1])
            ax_main.fill_between(bin_edges, err_low, err_high, step='post', color='black', alpha=0.2, label='Stat. Unc.')
        
        # Ratio Calculation
        if ref_hist is not None:
            safe_ref = np.where(ref_hist != 0, ref_hist, np.inf)
            ratio = hist_density / safe_ref
            ratio[ref_hist == 0] = np.nan
            y_ratio_step = np.append(ratio, ratio[-1])
            ax_ratio.step(bin_edges, y_ratio_step, color=color, linewidth=lw, linestyle=ls, where='post')

    # --- Styling ---
    ax_main.set_yscale(yscale)
    ax_main.set_ylabel("Normalized Density", fontsize=16, fontweight='bold')
    
    # ATLAS Label
    hep.atlas.label(
        text="Work in Progress", 
        data=False,              
        rlabel="",               
        ax=ax_main
    )

    # Clean Legend
    handles, plot_labels = ax_main.get_legend_handles_labels()
    by_label = dict(zip(plot_labels, handles))
    
    # [CHANGE 1] Legend moved to Top Left, but pushed down by bbox_to_anchor
    ax_main.legend(by_label.values(), by_label.keys(), 
                   fontsize=24, frameon=False, 
                   loc='upper left', bbox_to_anchor=(0, 0.90))
    
    ax_main.tick_params(axis='x', labelbottom=False)
    ax_main.tick_params(axis='both', which='major', labelsize=12)
    ax_main.set_ylim(top=2e-3)
    
    # --- Ratio Panel Styling ---
    ax_ratio.set_ylabel("Ratio", fontsize=14, fontweight='bold')
    ax_ratio.set_xlabel(xlabel, fontsize=24, fontweight='bold', labelpad=20)
    ax_ratio.axhline(1, color='gray', linestyle=':', linewidth=2, alpha=0.8)
    
    # [CHANGE 2] Clean up ticks: Hard limits on Y and remove minor ticks
    ax_ratio.set_ylim(0.5, 1.5)
    ax_ratio.set_yticks([0.5, 1.0, 1.5]) # Only show the bounds
    ax_ratio.tick_params(axis='y', which='major', pad=20)
    ax_ratio.tick_params(axis='y', which='minor', left=False, right=False) # Hide noisy minor ticks
    ax_ratio.tick_params(axis="x", left=False) # Hide vertical ticks on ratio plot
    
    ax_ratio.grid(True, which='both', linestyle=':', alpha=0.4)
    
    if xscale == 'log':
        ax_main.set_xscale('log')
        ax_ratio.set_xscale('log')

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=600, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close(fig)    
# -----------------------------------------------------------------------------
# 2. Metric Calculators (Physics Helpers)
# -----------------------------------------------------------------------------
def get_total_energy(shower_batch):
    """
    Sum of all energy in the shower.
    Input shape: [batch_size, num_voxels]
    Output shape: [batch_size]
    """
    data = to_np(shower_batch)
    # Sum across the voxel dimension (axis 1)
    return np.sum(data, axis=1)

def get_sparsity(shower_batch, threshold=1e-5):
    """
    Fraction of active cells (cells with E > threshold).
    Input shape: [batch_size, num_voxels]
    Output shape: [batch_size]
    """
    data = to_np(shower_batch)
    
    # Count how many voxels in each event are above threshold
    n_active = np.sum(data > threshold, axis=1)
    
    # Total voxels is just the size of the second dimension
    n_total = data.shape[1]
    
    return n_active / n_total




def plot_poster_layers(adapters_dict, layers_to_plot=[0, 1, 2], layer_names=None,
                       output_path='poster_layers.png', bins=50):
    """
    Plots selected layers side-by-side using data from FeatureAdapters.
    """
    
    # --- Poster Settings ---
    # Enforce ATLAS style context for fonts/ticks
    hep.style.use(hep.style.ATLAS)
    
    plt.rcParams.update({
        'font.size': 16,
        'axes.linewidth': 2.5,
        'xtick.major.width': 2, 'ytick.major.width': 2,
        'font.weight': 'bold', 'axes.labelweight': 'bold',
        'figure.facecolor': 'white'
    })
    
    # IBM Design Colorblind Safe Palette
    colors = {'Data': 'black', 'Recon': '#5D3A9B', 'GPU': '#785EF0', 'Geant': 'black'}
    fallback_colors = ['black', '#DC267F', '#648FFF', '#E66100', '#5D3A9B']
    linestyle_cycle = ['-', '--', '-.', ':']

    # Setup Canvas
    n_layers = len(layers_to_plot)
    fig, axes = plt.subplots(1, n_layers, figsize=(6 * n_layers, 6.5), 
                             sharey=True, constrained_layout=True)
    
    if n_layers == 1: axes = [axes] 

    # --- Plotting Loop ---
    for ax_idx, layer_id in enumerate(layers_to_plot):
        ax = axes[ax_idx]
        
        # 1. Collect all data for this layer to determine bins
        layer_all_data = []
        for adapter in adapters_dict.values():
            if layer_id in adapter.E_layers:
                layer_all_data.append(adapter.E_layers[layer_id])
        
        if not layer_all_data:
            print(f"Skipping Layer {layer_id}: No data found in adapters.")
            continue
            
        flat_data = np.concatenate(layer_all_data)
        flat_data = flat_data[flat_data > 0] # Filter for log scale
        
        # Dynamic Binning
        vmin = np.percentile(flat_data, 1) if len(flat_data) > 0 else 1e-4
        vmax = np.percentile(flat_data, 99) if len(flat_data) > 0 else 1.0
        if vmin <= 0: vmin = 1e-4
        
        bin_edges = np.linspace(vmin, vmax, bins)

        # 2. Plot Each Dataset
        for i, (label, adapter) in enumerate(adapters_dict.items()):
            if layer_id not in adapter.E_layers: continue
            
            data = adapter.E_layers[layer_id]
            
            # Styling
            color = colors.get(label, fallback_colors[i % len(fallback_colors)])
            is_ref = (label in ['Data', 'Geant']) or (i == 0)
            lw = 3.0 if is_ref else 2.5
            ls = '-' if is_ref else linestyle_cycle[i % len(linestyle_cycle)]
            alpha_fill = 0.0 if is_ref else 0.15

            # Histogram
            hist_density, _ = np.histogram(data, bins=bin_edges, density=True)
            y_step = np.append(hist_density, hist_density[-1])
            
            # Plot
            ax.step(bin_edges, y_step, where='post', label=label, 
                    color=color, linewidth=lw, linestyle=ls)
            
            if not is_ref:
                ax.fill_between(bin_edges, y_step, step='post', color=color, alpha=alpha_fill)

        # 3. Layer Formatting
        if layer_names and len(layer_names) == len(layers_to_plot):
            title_text = layer_names[ax_idx]
        else:
            title_text = f"Layer {layer_id}"        
        ax.set_yscale('log')
        ax.set_ylim(bottom=1e-6, top=1e-2) # Clean noise floor
        ax.set_title(title_text, fontsize=24, pad=10)
        ax.set_xlim(left=vmin, right=vmax)
        

    # 4. Global Legend
    hep.atlas.label(
        text="Work in Progress", 
        data=False,              
        rlabel="",               
        ax=axes[0],
        loc=4 # Places the label in the bottom left corner
    )
        # ------------------------------

    axes[0].set_ylabel("Normalized Density", fontsize=18)
    fig.supxlabel(r"$\mathbfit{E_{layer}}$ [MeV] $\mathbfit{E_{inc} =}$50 GeV", fontsize=24, fontweight='bold')

    handles, labels = axes[0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    
    # Note: Removed 'loc' from legend to avoid conflict if you want it purely external
    # or you can keep it 'upper center' below the plot as you had it.
    fig.legend(by_label.values(), by_label.keys(), 
               loc='upper center', bbox_to_anchor=(0.5, 0.0), 
               ncol=len(by_label), fontsize=18, frameon=False)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved Poster Plot: {output_path}")
    plt.close(fig)
# -----------------------------------------------------------------------------
# 2. The Execution Function
# -----------------------------------------------------------------------------
def evaluate_and_plot_poster(data_dict, binning_path, 
                             layers_to_plot=[1, 2, 12], 
                             layer_names = None,
                             output_dir="plots/", device="cpu"):
    """
    Orchestrates: Raw Data -> Extractor -> Adapter -> Poster Plotter
    """
    
    # 1. Setup Geometry & Extractor
    # Assuming AtlasGeometry is imported or defined
    geo = AtlasGeometry(filename=binning_path)
    extractor = DifferentiableFeatureExtractor(geo).to(device)
    extractor.eval()

    # Store populated adapters here
    adapters = {}

    # 2. Process Datasets
    with torch.no_grad():
        for label, (showers, e_inc) in data_dict.items():
            print(f"Extracting features for: {label}...")
            
            if not isinstance(showers, torch.Tensor):
                showers = torch.tensor(showers, dtype=torch.float32)
            showers = showers.to(device)

            # --- THE FAST PART ---
            features = extractor(showers)
            
            # --- THE ADAPTER ---
            # Using your FeatureAdapter class to wrap the tensors
            adapter = FeatureAdapter(features, geo.relevant_layers, e_inc)
            
            adapters[label] = adapter

    # 3. Call the Poster Plotter
    output_filename = os.path.join(output_dir, "poster_layer_comparison.svg")
    
    plot_poster_layers(
        adapters, 
        layers_to_plot=layers_to_plot,
        layer_names=layer_names,
        output_path=output_filename
    )