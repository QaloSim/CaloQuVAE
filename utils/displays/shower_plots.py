import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
from matplotlib.collections import PatchCollection
from matplotlib.colors import LogNorm
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

import numpy as np
import os
# local imports
from utils.HighLevelFeatsAtlasReg import HighLevelFeatures_ATLAS_regular
from utils.HLF.atlasgeo import AtlasGeometry, DifferentiableFeatureExtractor
import mplhep as hep




def plot_poster_comparison(shower_atlas, shower_qpu, layer_names, layer_ids, cfg):
    """
    Plots a direct comparison between a specific ATLAS shower and a QPU shower 
    for a poster, including ATLAS labelling and a shared colorbar.

    Args:
        shower_atlas: Tensor (1D) of the ATLAS shower event.
        shower_qpu: Tensor (1D) of the QPU shower event.
        layer_names: List of strings or ints naming the layers (e.g. ["EMB0", "TileBar 0"]).
        layer_ids: List of corresponding layer IDs for geometry retrieval (e.g. [1, 2, 3, 4, 13]).
        cfg: Configuration object for HLF initialization.
    
    Returns:
        fig: The matplotlib figure object.
    """
    
    # 1. Initialize HighLevelFeatures (HLF)
    # We use this ONLY for geometry/binning calculations
    dataset_name = cfg.data.dataset_name.lower()
    HLF = HighLevelFeatures_ATLAS_regular(
        particle=cfg.data.particle,
        filename=cfg.data.binning_path,
        relevantLayers=cfg.data.relevantLayers
    )
    # 2. Setup Figure Grid
    # Rows = 2 (ATLAS vs QPU), Cols = Number of Layers
    num_layers = len(layer_names)
    fig, axes = plt.subplots(2, num_layers, figsize=(4 * num_layers, 8), dpi=300,
    gridspec_kw={'hspace': 0.02, 'wspace': 0.2}, # Add this line to control spacing,
    # constrained_layout=True
    )
    
    # Global Plot Settings
    cmap = 'rainbow'
    vmin, vmax = 2, 1e3 # Adjust based on your typical energy range
    norm = LogNorm(vmin=vmin, vmax=vmax)
    vox_per_layer = 14 * 24 # Standard voxel size per layer based on your HLF
    
    # Helper function to extract geometry and draw patches on a specific axis
    def _draw_on_ax(ax, energy_tensor, layer_id):
        # Inject state into HLF to retrieve geometry for this layer
        # Note: We need to ensure tensor is on CPU/Numpy for plotting
        if isinstance(energy_tensor, torch.Tensor):
            e_vals = energy_tensor.detach().cpu().numpy()
        else:
            e_vals = energy_tensor

        HLF.single_event_energy = e_vals
        HLF.current_layer = str(layer_id)
        
        # Get Geometry arrays
        r0, r1, a0, a1, e = HLF.get_sector_arrays(HLF.current_layer)
        
        # Transform for equal area visualization
        transform = HLF._make_equal_bin_transform(r0, r1)
        r0p, r1p = transform(r0), transform(r1)
        
        # Create Wedges
        patches = []
        for inner, outer, start, end in zip(r0p, r1p, a0, a1):
            width = outer - inner
            patches.append(Wedge((0, 0), outer, start, end, width=width))
            
        # Add Collection
        pc = PatchCollection(patches, cmap=cmap, norm=norm, edgecolor="grey", linewidths=0.1)
        pc.set_array(e)
        ax.add_collection(pc)
        
        # Styling
        Rmax = r1p.max()
        ax.set_xlim(-Rmax - 0.1, Rmax + 0.1)
        ax.set_ylim(-Rmax - 0.1, Rmax + 0.1)
        ax.set_aspect('equal')
        ax.axis('off')
        return pc

    # 3. Plot Loops
    
    # --- Row 0: ATLAS ---
    for i, layer_id in enumerate(layer_ids):
        ax = axes[0, i]
        # Slice the flat tensor for this layer
        start_idx = i * vox_per_layer
        end_idx = (i + 1) * vox_per_layer
        layer_data = shower_atlas[start_idx:end_idx]
        
        _draw_on_ax(ax, layer_data, layer_id)
                
        # Add row label to the first column
        if i == 0:
            ax.text(-0.2, 0.5, "ATLAS\nSimulation", transform=ax.transAxes, 
                    fontsize=24, va='center', ha='right', fontweight='bold', rotation=90)

    # --- Row 1: QPU ---
    for i, layer_id in enumerate(layer_ids):
        ax = axes[1, i]
        # Slice the flat tensor for this layer
        start_idx = i * vox_per_layer
        end_idx = (i + 1) * vox_per_layer
        layer_data = shower_qpu[start_idx:end_idx]
        
        _draw_on_ax(ax, layer_data, layer_id)
        ax.set_title(f"{layer_names[i]}", transform=ax.transAxes, y=-0.1,
                ha='center', va='top', fontsize=16, fontweight='medium')

        # Add row label to the first column
        if i == 0:
            ax.text(-0.2, 0.5, "QPU\nGeneration", transform=ax.transAxes, 
                    fontsize=24, va='center', ha='right', fontweight='bold', rotation=90)

    # 4. Add Colorbar
    # We add one colorbar at the bottom for the whole figure
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), orientation='vertical', 
                        fraction=0.05, pad=0.05, aspect=15, location="right")
    cbar.set_label('Energy (MeV)', fontsize=18)

    # 5. Add ATLAS Label (mplhep)
    # Usually placed on the top left axis or the figure super-title area
    # We'll attach it to the first axis of the ATLAS row
    hep.atlas.label(ax=axes[0,0], text="Work in Progress", loc=0, data=False, rlabel="") 
    fig.suptitle(r"Displays of ATLAS Simulation and QPU Generated Showers, $E_{inc} = 50$ GeV", fontsize=28, fontweight='bold')
    return fig


def plot_layer_on_ax(ax, hlf_instance, layer_id, energy_data, title=None, 
                     norm=None, cmap='rainbow', title_fontsize=10):
    """
    Draws a single calorimeter layer directly onto a provided Matplotlib axis.
    """
    # 1. Inject state into HLF
    hlf_instance.single_event_energy = energy_data
    hlf_instance.current_layer = str(layer_id)
    
    # 2. Get Geometry
    r0, r1, a0, a1, e = hlf_instance.get_sector_arrays(hlf_instance.current_layer)

    # 3. Precompute Transform (Equal Bin Area)
    transform = hlf_instance._make_equal_bin_transform(r0, r1)
    r0p, r1p = transform(r0), transform(r1)

    # 4. Build Wedges
    patches = []
    for inner, outer, start, end in zip(r0p, r1p, a0, a1):
        width = outer - inner
        patches.append(Wedge((0, 0), outer, start, end, width=width))

    # 5. Create Collection
    # Use the passed norm and cmap
    pc = PatchCollection(patches, cmap=cmap, norm=norm, edgecolor="grey", linewidths=0.1)
    pc.set_array(e)
    ax.add_collection(pc)
    
    # 6. Styling
    Rmax = r1p.max()
    ax.set_xlim(-Rmax - 0.1, Rmax + 0.1)
    ax.set_ylim(-Rmax - 0.1, Rmax + 0.1)
    ax.set_aspect('equal')
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=title_fontsize, pad=8)
    
    return pc

def visualize_sliced_distribution(data_dict, binning_path, layer=2, 
                                  feature_name='Phi_center', 
                                  bounds=(-float('inf'), float('inf')), 
                                  mode='ratio',  # <--- NEW: 'ratio' or 'energy'
                                  device="cpu", output_dir="plots/slices"):
    
    # --- 1. Setup ---
    geo = AtlasGeometry(filename=binning_path)
    if layer not in geo.relevant_layers:
        print(f"Layer {layer} not found.")
        return
    layer_idx = geo.relevant_layers.index(layer)
    
    HLF = HighLevelFeatures_ATLAS_regular(
        particle='electron', filename=binning_path, relevantLayers=geo.relevant_layers
    )
    
    extractor = DifferentiableFeatureExtractor(geo).to(device)
    extractor.eval()
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n--- Slicing ({mode}): Layer {layer} | {feature_name} in {bounds} ---")

    # --- 2. Pre-Calculate Global Averages (for Ratio Mode) ---
    global_avgs = {}
    if mode == 'ratio':
        print("Pre-calculating global averages...")
        for label, (showers, _) in data_dict.items():
            if isinstance(showers, torch.Tensor):
                avg = showers.mean(dim=0).cpu().numpy()
            else:
                avg = np.mean(showers, axis=0)
            global_avgs[label] = avg

    # --- 3. Collect Valid Data ---
    valid_plots = [] 
    
    # Determine the Voxels for this specific layer
    vox_per_layer = 14 * 24 
    start_idx = layer_idx * vox_per_layer
    end_idx = (layer_idx + 1) * vox_per_layer

    key_map = {'mean_phi': 'Phi_center', 'Phi_center': 'Phi_center',
               'mean_eta': 'Eta_center', 'Eta_center': 'Eta_center',
               'energy': 'E_layer', 'E_layer': 'E_layer'}
    target_key = key_map.get(feature_name, feature_name)

    with torch.no_grad():
        for label, (showers, e_inc) in data_dict.items():
            # Prep Data
            if isinstance(showers, np.ndarray):
                showers_t = torch.tensor(showers, dtype=torch.float32).to(device)
                showers_np = showers
            else:
                showers_t = showers.to(device)
                showers_np = showers.cpu().numpy()

            # Extract Features for Masking
            feats = extractor(showers_t)
            values = feats[target_key][:, layer_idx].cpu().numpy()

            # Mask
            mask = (values >= bounds[0]) & (values <= bounds[1])
            count = np.sum(mask)
            
            if count > 0:
                selected = showers_np[mask]
                slice_avg_full = np.mean(selected, axis=0)
                
                # --- APPLY NORMALIZATION LOGIC ---
                if mode == 'ratio':
                    # Ratio: (Slice + eps) / (Global + eps)
                    # We use a small epsilon to avoid divide-by-zero or massive spikes
                    epsilon = 1e-6 
                    g_avg = global_avgs[label]
                    
                    # Compute ratio on full vector first
                    ratio_full = (slice_avg_full + epsilon) / (g_avg + epsilon)
                    
                    # Extract layer
                    plot_data = ratio_full[start_idx:end_idx]
                    
                    # Plot Styling for Ratio
                    # RdBu_r: Red (high), White (1.0), Blue (low)
                    cmap = 'RdBu_r' 
                    # LogNorm centered at 1.0. 
                    # vmin=0.2, vmax=5.0 means 1/5th to 5x relative activation
                    norm = LogNorm(vmin=0.2, vmax=5.0) 
                    cbar_label = "Relative Activation (Slice / Global)"
                    
                else:
                    # Standard Energy
                    plot_data = slice_avg_full[start_idx:end_idx]
                    cmap = 'viridis' # or rainbow
                    norm = LogNorm(vmin=1e-4, vmax=1e1) # Adjust based on your energy scale
                    cbar_label = "Mean Energy [MeV]"

                pct = count / len(showers_np) * 100
                valid_plots.append({
                    "label": label,
                    "data": plot_data,
                    "count": count,
                    "pct": pct,
                    "cmap": cmap,
                    "norm": norm,
                    "cbar_label": cbar_label
                })
            else:
                print(f"Skipping {label} (0 events)")

    # --- 4. Plotting ---
    n_plots = len(valid_plots)
    if n_plots == 0: return

    # Vertical Stack
    fig, axes = plt.subplots(n_plots, 1, figsize=(6, 5 * n_plots), dpi=150)
    if n_plots == 1: axes = [axes]

    print(f"Plotting {n_plots} panels...")
    
    for i, (ax, item) in enumerate(zip(axes, valid_plots)):
        label = item['label']
        count = item['count']
        pct = item['pct']
        
        plot_title = f"{label} | N={count} ({pct:.2f}%)"
        
        pc = plot_layer_on_ax(
            ax, HLF, layer, item['data'], 
            title=plot_title,
            norm=item['norm'],
            cmap=item['cmap']
        )
        
        # Add colorbar only to the last plot, or individual ones? 
        # Individual is safer if scales differ, but here they are fixed.
        # Let's add a small one per plot for clarity.
        cbar = plt.colorbar(pc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(item['cbar_label'], fontsize=8)

    # Global Title
    fig.suptitle(f"Layer {layer} [{mode.upper()}]: {bounds[0]} < {feature_name} < {bounds[1]}", 
                 y=1.0, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    slice_name = f"L{layer}_{feature_name}_{mode}_{bounds[0]}_to_{bounds[1]}"
    save_path = os.path.join(output_dir, f"Vert_{slice_name}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")




def plot_correlated_slice(data_dict, binning_path, 
                          target_spec, slice_spec,
                          reference_label='GEANT4',
                          device="cpu", output_dir="plots/correlations",
                          xscale='linear', yscale='log'):
    """
    Plots the distribution of `target_spec` for events that satisfy `slice_spec`.
    Bins are calculated using the full min/max of ALL data to preserve outliers.
    """
    
    # --- 1. Setup Feature Extractor ---
    # (Assuming imports are handled externally or at top of script)
    geo = AtlasGeometry(filename=binning_path)
    extractor = DifferentiableFeatureExtractor(geo).to(device)
    extractor.eval()
    
    tgt_layer, tgt_feat = target_spec
    slice_layer, slice_feat, slice_min, slice_max = slice_spec
    
    # Map friendly names to extractor keys
    key_map = {'mean_phi': 'Phi_center', 'Phi_center': 'Phi_center',
               'mean_eta': 'Eta_center', 'Eta_center': 'Eta_center',
               'energy': 'E_layer', 'E_layer': 'E_layer', 
               'width_eta': 'Eta_width', 'Eta_width': 'Eta_width',
               'width_phi': 'Phi_width', 'Phi_width': 'Phi_width'}
    
    tgt_key = key_map.get(tgt_feat, tgt_feat)
    slice_key = key_map.get(slice_feat, slice_feat)
    
    tgt_idx = geo.relevant_layers.index(tgt_layer)
    slice_idx = geo.relevant_layers.index(slice_layer)

    os.makedirs(output_dir, exist_ok=True)
    print(f"\n--- Correlation Plot ---")
    print(f"Target: L{tgt_layer} {tgt_feat}")
    print(f"Slice:  L{slice_layer} {slice_feat} in [{slice_min}, {slice_max}]")

    # --- 2. Extract Data & Apply Masks ---
    processed_data = {} 
    all_target_values = [] # Collection of all values to determine bin range
    
    def dup_last(arr):
        return np.append(arr, arr[-1])

    with torch.no_grad():
        for label, content in data_dict.items():
            if isinstance(content, tuple): showers = content[0]
            else: showers = content
            
            if isinstance(showers, np.ndarray):
                showers_t = torch.tensor(showers, dtype=torch.float32).to(device)
            else:
                showers_t = showers.to(device)
            
            # Extract Features
            feats = extractor(showers_t)
            tgt_vals = feats[tgt_key][:, tgt_idx].cpu().numpy()
            slice_vals = feats[slice_key][:, slice_idx].cpu().numpy()
            
            # Create Mask
            mask = (slice_vals >= slice_min) & (slice_vals <= slice_max)
            sliced_vals = tgt_vals[mask]
            
            # Store data
            processed_data[label] = {
                'full': tgt_vals,
                'sliced': sliced_vals
            }
            
            # Add to global list for binning (filter NaNs)
            valid_vals = tgt_vals[np.isfinite(tgt_vals)]
            if len(valid_vals) > 0:
                all_target_values.append(valid_vals)

            eff = len(sliced_vals) / len(tgt_vals) if len(tgt_vals) > 0 else 0
            print(f"  {label}: {len(sliced_vals)} events in slice ({eff*100:.1f}%)")

    if reference_label not in processed_data:
        print(f"Error: Reference label '{reference_label}' not found.")
        return
    
    if not all_target_values:
        print("No valid data found.")
        return

    # --- 3. Determine Global Bins (Min/Max Method) ---
    # Concatenate all data to find the absolute global min and max
    global_data = np.concatenate(all_target_values)
    
    if xscale == 'log':
        # Filter for positive values only for log binning
        pos_data = global_data[global_data > 0]
        if len(pos_data) == 0:
            print("Warning: Log scale requested but no positive data found.")
            return
        low = pos_data.min()
        high = pos_data.max()
        # Add a tiny buffer so the max value isn't on the edge
        bins = np.logspace(np.log10(low), np.log10(high * 1.01), 50)
    else:
        low = global_data.min()
        high = global_data.max()
        # Add a tiny buffer
        bins = np.linspace(low, high * 1.01, 50)

    # --- 4. Plotting ---
    fig = plt.figure(figsize=(8, 7))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)

    colors = ['black', 'red', 'blue', 'green', 'purple', 'orange']
    linestyles = ['-', '--', '-.', ':', '--', '-.']
    
    # --- Plot Reference ---
    ref_full = processed_data[reference_label]['full']
    ref_sliced = processed_data[reference_label]['sliced']
    
    # Full distribution (Outline)
    n_ref_full, _ = np.histogram(ref_full, bins=bins, density=True)
    ax0.step(bins, dup_last(n_ref_full), color='gray', alpha=0.4, linestyle=':', 
             linewidth=1, where='post', label=f"{reference_label} (Full)")

    # Sliced distribution (Filled)
    n_ref_slice, _ = np.histogram(ref_sliced, bins=bins, density=True)
    counts_ref_slice, _ = np.histogram(ref_sliced, bins=bins)
    
    # Error bars on Reference Slice
    mask = counts_ref_slice > 0
    ref_err = np.zeros_like(n_ref_slice)
    ref_err[mask] = n_ref_slice[mask] / np.sqrt(counts_ref_slice[mask])

    ax0.step(bins, dup_last(n_ref_slice), color='black', linewidth=1.5, 
             where='post', label=f"{reference_label} (Slice)")
    ax0.fill_between(bins, dup_last(n_ref_slice), alpha=0.15, color='black', step='post')
    
    # --- Plot Models ---
    color_idx = 1
    for label, data in processed_data.items():
        if label == reference_label:
            # Calculate ratio of Slice / Full to see how the shape changes
            ratio = np.divide(n_ref_slice, n_ref_full, out=np.zeros_like(n_ref_full), 
                            where=n_ref_full > 1e-9)
            # Plot this special self-ratio in Black
            ax1.step(bins, dup_last(ratio), color='black', linewidth=1.5, where='post')
            continue        
        sliced_mod = data['sliced']
        n_mod_slice, _ = np.histogram(sliced_mod, bins=bins, density=True)
        
        col = colors[color_idx % len(colors)]
        ls = linestyles[color_idx % len(linestyles)]
        
        ax0.step(bins, dup_last(n_mod_slice), color=col, linestyle=ls, 
                 linewidth=1.5, where='post', label=f"{label} (Slice)")
        
        # Ratio
        ratio = np.divide(n_mod_slice, n_ref_slice, out=np.zeros_like(n_ref_slice), 
                          where=n_ref_slice > 1e-9)
        ax1.step(bins, dup_last(ratio), color=col, linestyle=ls, 
                 linewidth=1.5, where='post')
        
        color_idx += 1

    # --- Styling ---
    slice_text = (f"Slice Condition:\n"
                  f"Layer {slice_layer} {slice_feat}\n"
                  f"$\\in [{slice_min}, {slice_max}]$")
    ax0.text(0.03, 0.96, slice_text, transform=ax0.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    ax0.set_yscale(yscale)
    ax0.set_xscale(xscale)
    ax0.set_ylabel("Normalized Entries", fontsize=12)
    ax0.legend(loc='best', fontsize=9, frameon=False) # 'best' logic helps avoid covering outliers
    ax0.tick_params(labelbottom=False)
    
    ax1.set_ylabel("Ratio to Ref", fontsize=10)
    ax1.set_xlabel(f"Layer {tgt_layer} {tgt_feat}", fontsize=12)
    ax1.set_xscale(xscale)
    ax1.axhline(1, color='black', linestyle='-', alpha=0.3, linewidth=1)
    
    # Expanded Y-limit for ratio to see severe disagreements
    ax1.set_ylim(0.0, 2.5) 
    ax1.grid(True, which='both', linestyle=':', alpha=0.5)

    fname = f"Corr_L{tgt_layer}{tgt_feat}_vs_L{slice_layer}{slice_feat}.png"
    save_path = os.path.join(output_dir, fname)
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")




def visualize_tail_events_at_energy(data_dict, binning_path, 
                                    target_energy, 
                                    selection_layer=2, 
                                    display_layers=[1, 2, 12], 
                                    feature_name='Phi_center', 
                                    bounds=(-float('inf'), float('inf')), 
                                    num_events=4, 
                                    energy_tol=0.05,
                                    mode='absolute', 
                                    seed=42,
                                    device="cpu"):
    """
    Visualizes specific layers of individual events with per-subplot metadata titles.
    Includes E_inc, E_sum (total deposited), and the feature value in the title.
    """
    
    # --- 1. Setup ---
    rng = np.random.default_rng(seed)
    geo = AtlasGeometry(filename=binning_path)
    
    # Validate Layers
    if selection_layer not in geo.relevant_layers:
        print(f"Selection Layer {selection_layer} not found.")
        return
    for l in display_layers:
        if l not in geo.relevant_layers:
            print(f"Display Layer {l} not found.")
            return

    sel_layer_idx = geo.relevant_layers.index(selection_layer)
    
    HLF = HighLevelFeatures_ATLAS_regular(
        particle='electron', filename=binning_path, relevantLayers=geo.relevant_layers
    )
    
    extractor = DifferentiableFeatureExtractor(geo).to(device)
    extractor.eval()
    
    e_min = target_energy * (1 - energy_tol)
    e_max = target_energy * (1 + energy_tol)
    
    model_label = list(data_dict.keys())[0]
    showers, e_inc = data_dict[model_label]
    
    print(f"\n--- Visualizing Tails ({mode}) @ {target_energy}GeV for {model_label} ---")

    # --- 2. Feature Extraction & Filtering ---
    key_map = {'mean_phi': 'Phi_center', 'Phi_center': 'Phi_center',
               'mean_eta': 'Eta_center', 'Eta_center': 'Eta_center',
               'energy': 'E_layer', 'E_layer': 'E_layer'}
    target_key = key_map.get(feature_name, feature_name)

    with torch.no_grad():
        if isinstance(showers, np.ndarray):
            showers_t = torch.tensor(showers, dtype=torch.float32).to(device)
            showers_np = showers
            e_inc_np = e_inc if isinstance(e_inc, np.ndarray) else e_inc.cpu().numpy()
        else:
            showers_t = showers.to(device)
            showers_np = showers.cpu().numpy()
            e_inc_np = e_inc.cpu().numpy()

        if e_inc_np.ndim > 1: e_inc_np = e_inc_np.flatten()

        feats = extractor(showers_t)
        values = feats[target_key][:, sel_layer_idx].cpu().numpy()

        mask_energy = (e_inc_np >= e_min) & (e_inc_np <= e_max)
        mask_feat = (values >= bounds[0]) & (values <= bounds[1])
        final_mask = mask_feat & mask_energy
        
        # Stats for Title
        total_in_window = np.sum(mask_energy)
        total_selected = np.sum(final_mask)
        pct = (total_selected / total_in_window * 100) if total_in_window > 0 else 0.0
            
        print(f"Events Selected: {total_selected}/{total_in_window} ({pct:.2f}%)")

        candidate_indices = np.where(final_mask)[0]
        if len(candidate_indices) == 0:
            print("No events found.")
            return

        if len(candidate_indices) > num_events:
            selected_indices = rng.choice(candidate_indices, num_events, replace=False)
        else:
            selected_indices = candidate_indices
            num_events = len(selected_indices)

    # --- 3. Pre-Calculate Means (Relative Mode) ---
    mean_full_shower = None
    if mode == 'relative':
        mean_full_shower = np.mean(showers_np, axis=0)

    # --- 4. Plotting Configuration ---
    if mode == 'relative':
        cmap = plt.cm.magma_r.copy() 
        cbar_label =r"Relative Activation ($E_{\text{vox}} / \langle E \rangle$)"
        norm = LogNorm(vmin=0.1, vmax=500.0)
    else:
        cmap = plt.cm.viridis.copy()
        cbar_label = "Energy Deposition [MeV]"
        norm = LogNorm(vmin=1e-1, vmax=1e2)

    cmap.set_bad(color='white')   
    cmap.set_under(color='white') 

    n_cols = len(display_layers)
    # Increased vertical size slightly to accommodate titles
    fig, axes = plt.subplots(num_events, n_cols, 
                             figsize=(3.0 * n_cols, 3.0 * num_events), 
                             dpi=150, squeeze=False)

    vox_per_layer = 14 * 24 
    last_pc = None 

    # --- 5. Render Loop ---
    for row_idx, event_idx in enumerate(selected_indices):
        
        inc_e = e_inc_np[event_idx]
        feat_val = values[event_idx]
        full_shower_data = showers_np[event_idx]
        
        # --- NEW: Calculate total deposited energy in the shower ---
        total_dep = np.sum(full_shower_data)

        # Row Label (Just Event ID now, since data is in title)
        axes[row_idx, 0].set_ylabel(f"Event {event_idx}", fontsize=11, fontweight='bold', labelpad=10)

        # --- UPDATED: Common subtitle for this event (row) ---
        # Added E_sum to the string
        meta_title = f"val={feat_val:.2f} \n$E_{{inc}}$={inc_e:.0f} | $E_{{sum}}$={total_dep:.0f}"

        for col_idx, layer_id in enumerate(display_layers):
            ax = axes[row_idx, col_idx]
            
            l_idx_geo = geo.relevant_layers.index(layer_id)
            start_idx = l_idx_geo * vox_per_layer
            end_idx = (l_idx_geo + 1) * vox_per_layer
            
            layer_data = full_shower_data[start_idx:end_idx]

            if mode == 'relative':
                mean_layer = mean_full_shower[start_idx:end_idx]
                safe_mean = mean_layer.copy()
                safe_mean[safe_mean == 0] = 1e-9 
                plot_payload = layer_data / safe_mean
            else:
                plot_payload = layer_data

            # --- Title Logic ---
            # Top row gets "Layer X" + Metadata
            # Subsequent rows get just Metadata
            if row_idx == 0:
                final_title = f"Layer {layer_id}\n{meta_title}"
            else:
                final_title = meta_title

            last_pc = plot_layer_on_ax(
                ax, HLF, layer_id, plot_payload, 
                title=final_title,
                norm=norm,
                cmap=cmap,
                title_fontsize=6
            )
            
            # Ensure title doesn't overlap with plot content
            ax.title.set_fontsize(10)

    # --- 6. Global Elements ---
    title_str = (f"Tail Events ({mode}) @ {target_energy/1000:.0f} GeV | "
                 f"Sel: Layer {selection_layer}, {bounds[0]}<{feature_name}<{bounds[1]}\n"
                 f"Subset: {total_selected}/{total_in_window} events ({pct:.2f}%)")
    
    fig.suptitle(title_str, y=0.99, fontsize=13, fontweight='bold')
    
    fig.subplots_adjust(bottom=0.15, hspace=0.4, wspace=0.2, top=0.92)
    
    cbar_ax = fig.add_axes([0.15, 0.05, 0.7, 0.025])
    cbar = fig.colorbar(last_pc, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(cbar_label, fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    plt.show()


def visualize_tail_events_comparison(data_dict, binning_path, 
                                     target_energy, 
                                     selection_layer=2, 
                                     display_layers=[1, 2, 12], 
                                     feature_name='Phi_center', 
                                     bounds=(-float('inf'), float('inf')), 
                                     num_events=4, 
                                     energy_tol=0.05,
                                     mode='absolute', 
                                     seed=42,
                                     device="cpu"):
    """
    Visualizes specific layers of individual events: GT, Model, and Difference.
    Calculates and displays feature values for BOTH GT and Model.
    """
    
    # --- 1. Setup ---
    rng = np.random.default_rng(seed)
    geo = AtlasGeometry(filename=binning_path)
    
    # Validate Layers
    if selection_layer not in geo.relevant_layers:
        print(f"Selection Layer {selection_layer} not found.")
        return
    for l in display_layers:
        if l not in geo.relevant_layers:
            print(f"Display Layer {l} not found.")
            return

    sel_layer_idx = geo.relevant_layers.index(selection_layer)
    
    HLF = HighLevelFeatures_ATLAS_regular(
        particle='electron', filename=binning_path, relevantLayers=geo.relevant_layers
    )
    
    extractor = DifferentiableFeatureExtractor(geo).to(device)
    extractor.eval()
    
    e_min = target_energy * (1 - energy_tol)
    e_max = target_energy * (1 + energy_tol)
    
    # Extract Datasets (Assume Index 0 is GT, Index 1 is Model)
    keys = list(data_dict.keys())
    label_gt = keys[0]
    label_model = keys[1]
    
    showers_gt, e_inc_gt = data_dict[label_gt]
    showers_model, _ = data_dict[label_model] 
    
    print(f"\n--- Visualizing Tails ({mode}) @ {target_energy}GeV ---")
    print(f"Comparison: {label_gt} (GT) vs {label_model} (Model)")

    # --- 2. Feature Extraction & Filtering ---
    key_map = {'mean_phi': 'Phi_center', 'Phi_center': 'Phi_center',
               'mean_eta': 'Eta_center', 'Eta_center': 'Eta_center',
               'energy': 'E_layer', 'E_layer': 'E_layer'}
    target_key = key_map.get(feature_name, feature_name)

    with torch.no_grad():
        # --- A. Prepare GT Data ---
        if isinstance(showers_gt, np.ndarray):
            showers_t_gt = torch.tensor(showers_gt, dtype=torch.float32).to(device)
            showers_np_gt = showers_gt
            e_inc_np = e_inc_gt if isinstance(e_inc_gt, np.ndarray) else e_inc_gt.cpu().numpy()
        else:
            showers_t_gt = showers_gt.to(device)
            showers_np_gt = showers_gt.cpu().numpy()
            e_inc_np = e_inc_gt.cpu().numpy()

        if e_inc_np.ndim > 1: e_inc_np = e_inc_np.flatten()

        # --- B. Prepare Model Data ---
        if isinstance(showers_model, np.ndarray):
            showers_t_model = torch.tensor(showers_model, dtype=torch.float32).to(device)
            showers_np_model = showers_model
        else:
            showers_t_model = showers_model.to(device)
            showers_np_model = showers_model.cpu().numpy()

        # --- C. Extract Features for BOTH ---
        # 1. Ground Truth Features
        feats_gt = extractor(showers_t_gt)
        values_gt = feats_gt[target_key][:, sel_layer_idx].cpu().numpy()

        # 2. Model Features (New addition)
        feats_model = extractor(showers_t_model)
        values_model = feats_model[target_key][:, sel_layer_idx].cpu().numpy()

        # --- D. Filter (Based on GT) ---
        mask_energy = (e_inc_np >= e_min) & (e_inc_np <= e_max)
        mask_feat = (values_gt >= bounds[0]) & (values_gt <= bounds[1])
        final_mask = mask_feat & mask_energy
        
        # Stats
        total_in_window = np.sum(mask_energy)
        total_selected = np.sum(final_mask)
        pct = (total_selected / total_in_window * 100) if total_in_window > 0 else 0.0
            
        print(f"Events Selected (based on GT): {total_selected}/{total_in_window} ({pct:.2f}%)")

        candidate_indices = np.where(final_mask)[0]
        if len(candidate_indices) == 0:
            print("No events found.")
            return

        if len(candidate_indices) > num_events:
            selected_indices = rng.choice(candidate_indices, num_events, replace=False)
        else:
            selected_indices = candidate_indices
            num_events = len(selected_indices)

    # --- 3. Pre-Calculate Means (Relative Mode) ---
    mean_full_shower = None
    if mode == 'relative':
        mean_full_shower = np.mean(showers_np_gt, axis=0) 

    # --- 4. Plotting Configuration ---
    if mode == 'relative':
        cmap_main = plt.cm.magma_r.copy() 
        cbar_label_main = r"Relative ($E / \langle E_{GT} \rangle$)"
        norm_main = LogNorm(vmin=0.1, vmax=500.0)
    else:
        cmap_main = plt.cm.viridis.copy()
        cbar_label_main = "Energy [MeV]"
        norm_main = LogNorm(vmin=1e-1, vmax=1e2)

    cmap_main.set_bad(color='white')   
    cmap_main.set_under(color='white') 

    cmap_diff = plt.cm.coolwarm.copy()
    cmap_diff.set_bad(color='white')
    cbar_label_diff = "Difference (Model - GT)"
    
    n_layers = len(display_layers)
    n_cols = n_layers * 3 
    
    fig, axes = plt.subplots(num_events, n_cols, 
                             figsize=(3.0 * n_cols, 3.4 * num_events), 
                             dpi=150, squeeze=False)

    vox_per_layer = 14 * 24 
    pc_main = None
    pc_diff = None

    # --- 5. Render Loop ---
    for row_idx, event_idx in enumerate(selected_indices):
        
        # Metadata
        inc_e = e_inc_np[event_idx]
        
        # Retrieve Pre-calculated Feature Values
        val_gt = values_gt[event_idx]
        val_model = values_model[event_idx]
        
        # Full Shower Data
        full_gt = showers_np_gt[event_idx]
        full_model = showers_np_model[event_idx]

        total_dep_gt = np.sum(full_gt)
        total_dep_model = np.sum(full_model)

        # Row Label
        axes[row_idx, 0].set_ylabel(f"Event {event_idx}", fontsize=11, fontweight='bold', labelpad=10)

        # --- UPDATED: Meta string with both GT and Model feature values ---
        meta_str = (f"v_GT={val_gt:.2f} | v_M={val_model:.2f}\n"
                    f"$\Sigma GT$={total_dep_gt:.0f} | $\Sigma M$={total_dep_model:.0f}\n"
                    f"$E_{{inc}}$={inc_e:.0f}")

        # Loop through requested layers
        for l_idx, layer_id in enumerate(display_layers):
            
            l_idx_geo = geo.relevant_layers.index(layer_id)
            start_idx = l_idx_geo * vox_per_layer
            end_idx = (l_idx_geo + 1) * vox_per_layer
            
            data_gt = full_gt[start_idx:end_idx]
            data_model = full_model[start_idx:end_idx]
            
            if mode == 'relative':
                mean_layer = mean_full_shower[start_idx:end_idx]
                safe_mean = mean_layer.copy()
                safe_mean[safe_mean == 0] = 1e-9 
                
                pl_gt = data_gt / safe_mean
                pl_model = data_model / safe_mean
            else:
                pl_gt = data_gt
                pl_model = data_model
            
            pl_diff = pl_model - pl_gt
            
            col_gt = l_idx * 3
            col_model = l_idx * 3 + 1
            col_diff = l_idx * 3 + 2
            
            if row_idx == 0:
                title_gt = f"Layer {layer_id} (GT)\n{meta_str}"
                title_model = f"Layer {layer_id} (Model)\n{meta_str}"
                title_diff = f"Layer {layer_id} (Diff)\n(M - GT)"
            else:
                title_gt = meta_str
                title_model = meta_str
                title_diff = "(M - GT)"

            pc_main = plot_layer_on_ax(
                axes[row_idx, col_gt], HLF, layer_id, pl_gt, 
                title=title_gt, norm=norm_main, cmap=cmap_main, title_fontsize=7
            )

            plot_layer_on_ax(
                axes[row_idx, col_model], HLF, layer_id, pl_model, 
                title=title_model, norm=norm_main, cmap=cmap_main, title_fontsize=7
            )

            max_diff = np.max(np.abs(pl_diff))
            if max_diff == 0: max_diff = 1.0 
            norm_diff = Normalize(vmin=-max_diff, vmax=max_diff)
            
            pc_diff = plot_layer_on_ax(
                axes[row_idx, col_diff], HLF, layer_id, pl_diff, 
                title=title_diff, norm=norm_diff, cmap=cmap_diff, title_fontsize=7
            )
            
            axes[row_idx, col_gt].title.set_fontsize(8)
            axes[row_idx, col_model].title.set_fontsize(8)
            axes[row_idx, col_diff].title.set_fontsize(8)

    # --- 6. Global Elements ---
    title_str = (f"Tail Events Comparison ({mode}) @ {target_energy/1000:.0f} GeV | "
                 f"Sel: L{selection_layer}, {bounds[0]}<{feature_name}<{bounds[1]}\n"
                 f"GT: {label_gt} | Model: {label_model}")
    
    fig.suptitle(title_str, y=0.99, fontsize=14, fontweight='bold')
    
    fig.subplots_adjust(bottom=0.15, hspace=0.45, wspace=0.25, top=0.91)
    
    cbar_ax_main = fig.add_axes([0.15, 0.05, 0.35, 0.025])
    cbar_main = fig.colorbar(pc_main, cax=cbar_ax_main, orientation='horizontal')
    cbar_main.set_label(cbar_label_main, fontsize=11)
    
    cbar_ax_diff = fig.add_axes([0.55, 0.05, 0.35, 0.025])
    cbar_diff = fig.colorbar(pc_diff, cax=cbar_ax_diff, orientation='horizontal')
    cbar_diff.set_label(cbar_label_diff, fontsize=11)

    plt.show()





def visualize_low_ratio_events(showers, e_inc, binning_path, 
                               cutoff=0.9, 
                               num_events=5, 
                               display_layers=[0, 1, 2, 3, 12], 
                               seed=42,
                               device="cpu"):
    """
    Plots individual shower events where the ratio (Deposited Energy / Incidence Energy)
    is BELOW a specified cutoff.
    
    Args:
        showers (torch.Tensor): Shape (batch, shower_size)
        e_inc (torch.Tensor): Shape (batch, 1) or (batch,)
        binning_path (str): Path to binning file for Geometry initialization.
        cutoff (float): The threshold ratio. Events with (sum(E) / E_inc) < cutoff are selected.
        num_events (int): Number of events to plot.
        display_layers (list): List of layer IDs to visualize.
    """
    
    # --- 1. Setup ---
    rng = np.random.default_rng(seed)
    geo = AtlasGeometry(filename=binning_path)
    
    # Validate Layers
    for l in display_layers:
        if l not in geo.relevant_layers:
            print(f"Display Layer {l} not found in geometry.")
            return

    # Initialize HLF (needed for geometry/wedge plotting)
    HLF = HighLevelFeatures_ATLAS_regular(
        particle='electron', filename=binning_path, relevantLayers=geo.relevant_layers
    )

    print(f"\n--- Visualizing Low Ratio Events (Ratio < {cutoff}) ---")

    # --- 2. Data Preparation & Filtering ---
    with torch.no_grad():
        if isinstance(showers, np.ndarray):
            showers_t = torch.tensor(showers, dtype=torch.float32).to(device)
            e_inc_np = e_inc if isinstance(e_inc, np.ndarray) else e_inc.cpu().numpy()
        else:
            showers_t = showers.to(device)
            e_inc_np = e_inc.cpu().numpy()

        # Handle e_inc shape
        if e_inc_np.ndim > 1: 
            e_inc_np = e_inc_np.flatten()

        # Calculate Total Deposited Energy per event
        # Sum across all voxels (dim 1)
        e_dep = torch.sum(showers_t, dim=1).cpu().numpy()
        
        # Calculate Ratios
        # Avoid division by zero with a small epsilon if needed, though E_inc should be > 0
        ratios = e_dep / (e_inc_np + 1e-9)

        # Create Selection Mask
        mask_low_ratio = ratios < cutoff
        
        # Statistics
        total_events = len(ratios)
        selected_count = np.sum(mask_low_ratio)
        pct = (selected_count / total_events * 100) if total_events > 0 else 0.0
        
        print(f"Total Events: {total_events}")
        print(f"Events below cutoff: {selected_count} ({pct:.2f}%)")

        if selected_count == 0:
            print("No events found below this cutoff.")
            return

        # Select Indices
        candidate_indices = np.where(mask_low_ratio)[0]
        
        if len(candidate_indices) > num_events:
            selected_indices = rng.choice(candidate_indices, num_events, replace=False)
            # Sort them so they appear in consistent order if re-run with same seed
            selected_indices.sort()
        else:
            selected_indices = candidate_indices
            num_events = len(selected_indices)

    # --- 3. Plotting Configuration ---
    # Standard Energy styling
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color='white')
    cmap.set_under(color='white') 
    
    # Adjust norm based on expected energy range (MeV)
    norm = LogNorm(vmin=1e-1, vmax=1e4) 
    
    n_cols = len(display_layers)
    fig, axes = plt.subplots(num_events, n_cols, 
                             figsize=(3.0 * n_cols, 3.2 * num_events), 
                             dpi=150, squeeze=False)

    vox_per_layer = 14 * 24 
    last_pc = None

    # --- 4. Render Loop ---
    # Convert showers to numpy for plotting access
    showers_np = showers_t.cpu().numpy()

    for row_idx, event_idx in enumerate(selected_indices):
        
        # Event Data
        inc_e = e_inc_np[event_idx]
        dep_e = e_dep[event_idx]
        ratio = ratios[event_idx]
        full_shower_data = showers_np[event_idx]

        # Row Label (Event ID)
        axes[row_idx, 0].set_ylabel(f"Evt {event_idx}", fontsize=10, fontweight='bold', labelpad=8)

        # Meta string for titles
        meta_str = (f"Ratio = {ratio:.3f}\n"
                    f"$E_{{inc}}$={inc_e:.0f} | $E_{{dep}}$={dep_e:.0f}")

        for col_idx, layer_id in enumerate(display_layers):
            ax = axes[row_idx, col_idx]
            
            # Slice out the specific layer data
            l_idx_geo = geo.relevant_layers.index(layer_id)
            start_idx = l_idx_geo * vox_per_layer
            end_idx = (l_idx_geo + 1) * vox_per_layer
            
            layer_data = full_shower_data[start_idx:end_idx]

            # Title Logic
            if row_idx == 0:
                final_title = f"Layer {layer_id}\n{meta_str}"
            else:
                final_title = meta_str

            # Plot using the helper
            last_pc = plot_layer_on_ax(
                ax, HLF, layer_id, layer_data, 
                title=final_title,
                norm=norm,
                cmap=cmap,
                title_fontsize=7
            )
            
            # Adjust title size
            ax.title.set_fontsize(8)

    # --- 5. Global Elements ---
    title_str = (f"Low Ratio Events (Ratio < {cutoff}) | N={num_events}\n"
                 f"Subset found: {selected_count}/{total_events} ({pct:.2f}%)")
    
    fig.suptitle(title_str, y=0.99, fontsize=13, fontweight='bold')
    
    # Adjust layout to make room for titles and colorbar
    fig.subplots_adjust(bottom=0.1, hspace=0.5, wspace=0.2, top=0.90)
    
    # Add Colorbar
    cbar_ax = fig.add_axes([0.2, 0.04, 0.6, 0.02])
    cbar = fig.colorbar(last_pc, cax=cbar_ax, orientation='horizontal')
    cbar.set_label("Energy Deposition [MeV]", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    plt.show()