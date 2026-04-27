import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr
import itertools
from typing import Tuple, List, Optional
import h5py
import os
from utils.pca_utils import compute_U, plot_PCA

def plot_weight_distribution(rbm, abs_tol: float, bins: int = 100):
    """
    Plots the distribution of the weight matrix from an RBM object 
    and overlays the absolute clipping tolerance.
    """
    if "weight_matrix" not in rbm.params:
        raise ValueError("The provided RBM object lacks a 'weight_matrix' in its params dictionary.")

    # Detach from graph, move to CPU, and flatten to 1D array
    weights = rbm.params["weight_matrix"].detach().cpu().numpy().flatten()
    
    plt.figure(figsize=(8, 5))
    
    # Plot the histogram of the weights
    plt.hist(weights, bins=bins, color='steelblue', edgecolor='black', alpha=0.7, density=True, label='Weights')
    
    # Add vertical lines for the absolute tolerance boundaries
    plt.axvline(x=abs_tol, color='crimson', linestyle='dashed', linewidth=2, label=f'+Tol ({abs_tol})')
    plt.axvline(x=-abs_tol, color='crimson', linestyle='dashed', linewidth=2, label=f'-Tol ({-abs_tol})')
    
    plt.title("RBM Weight Distribution under Clipping")
    plt.xlabel("Weight Value")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.yscale("log")
    plt.tight_layout()

    
    plt.show()

def plot_weight_distribution_checkpoints(
    checkpoint_paths: List[str],
    labels: Optional[List[str]] = None,
    epochs: Optional[List[Optional[int]]] = None,
    bins: int = 100,
    abs_tol: Optional[float] = None,
    title: str = "RBM Weight Distribution across Checkpoints",
) -> plt.Figure:
    """
    Plots overlaid weight distributions for a list of RBM checkpoint (.h5) files.

    Reads ``weight_matrix`` directly from each HDF5 file without requiring a
    full RBM instantiation, so no config or data objects are needed.

    Args:
        checkpoint_paths: List of paths to HDF5 checkpoint files.
        labels: Display label for each checkpoint. Defaults to the filename stem.
        epochs: Epoch to load from each file. ``None`` entries (or omitting the
                argument entirely) load the latest epoch recorded in that file.
        bins: Number of histogram bins.
        abs_tol: If given, draw symmetric dashed vertical lines at ±abs_tol.
        title: Figure title.

    Returns:
        matplotlib Figure.
    """
    if not checkpoint_paths:
        raise ValueError("checkpoint_paths must be a non-empty list.")

    n = len(checkpoint_paths)

    if labels is None:
        labels = [os.path.splitext(os.path.basename(p))[0] for p in checkpoint_paths]
    if len(labels) != n:
        raise ValueError("len(labels) must match len(checkpoint_paths).")

    if epochs is None:
        epochs = [None] * n
    if len(epochs) != n:
        raise ValueError("len(epochs) must match len(checkpoint_paths).")

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (path, label, epoch) in enumerate(zip(checkpoint_paths, labels, epochs)):
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")

        with h5py.File(path, "r") as f:
            # Resolve epoch
            if epoch is None:
                if "last_epoch" not in f.attrs:
                    raise KeyError(
                        f"'last_epoch' attribute missing in {path}. "
                        "Specify an explicit epoch number."
                    )
                loaded_epoch = int(f.attrs["last_epoch"])
            else:
                loaded_epoch = int(epoch)

            group_name = f"epoch_{loaded_epoch}"
            if group_name not in f:
                raise KeyError(f"Group '{group_name}' not found in {path}.")

            if "weight_matrix" not in f[group_name]:
                raise KeyError(f"'weight_matrix' not found in {group_name} of {path}.")

            weights = f[group_name]["weight_matrix"][()].flatten()

        rms = np.sqrt(np.mean(weights ** 2))
        color = colors[i % len(colors)]
        ax.hist(
            weights,
            bins=bins,
            density=True,
            alpha=0.6,
            color=color,
            edgecolor="none",
            label=f"{label} (RMS={rms:.4f})",
        )

    if abs_tol is not None:
        ax.axvline(x=abs_tol, color="crimson", linestyle="--", linewidth=1.8)
        ax.axvline(x=-abs_tol, color="crimson", linestyle="--", linewidth=1.8)

    ax.set_yscale("log")
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Weight Value", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig


def prepare_dataframe(proj_real, proj_gen, n_components):
    pc_cols = [f"PC{i}" for i in range(n_components)]
    df_real = pd.DataFrame(proj_real, columns=pc_cols)
    df_real['Source'] = 'Real Data'
    df_gen = pd.DataFrame(proj_gen, columns=pc_cols)
    df_gen['Source'] = 'RBM Samples'
    return pd.concat([df_real, df_gen], ignore_index=True), pc_cols

def _calculate_relative_limit(df, pc_columns, bins, sigma, epsilon, count_threshold, quantile=0.99):
    """
    Calculates limits, ignoring bins where raw counts are below the threshold.
    """
    all_diffs = []
    
    df_real = df[df['Source'] == 'Real Data']
    df_gen = df[df['Source'] == 'RBM Samples']
    
    for col_x, col_y in itertools.combinations(pc_columns, 2):
        x_real, y_real = df_real[col_x], df_real[col_y]
        x_gen, y_gen = df_gen[col_x], df_gen[col_y]
        
        global_min = min(x_real.min(), x_gen.min(), y_real.min(), y_gen.min())
        global_max = max(x_real.max(), x_gen.max(), y_real.max(), y_gen.max())
        r_lims = [[global_min, global_max], [global_min, global_max]]
        
        # 1. Compute RAW COUNTS for thresholding
        counts_r, _, _ = np.histogram2d(x_real, y_real, bins=bins, range=r_lims)
        counts_g, _, _ = np.histogram2d(x_gen, y_gen, bins=bins, range=r_lims)
        
        # 2. Compute DENSITIES for plotting
        dens_r, _, _ = np.histogram2d(x_real, y_real, bins=bins, range=r_lims, density=True)
        dens_g, _, _ = np.histogram2d(x_gen, y_gen, bins=bins, range=r_lims, density=True)
        
        # 3. Smooth Densities
        s_r = gaussian_filter(dens_r, sigma)
        s_g = gaussian_filter(dens_g, sigma)
        
        # 4. Apply Relative Diff Logic
        denom = s_r.copy()
        denom[denom < epsilon] = epsilon
        rel_diff = (s_g - s_r) / denom
        
        # 5. Apply Count Thresholding (The "Noise Floor")
        # If BOTH real and gen have low counts, this bin is noise -> set diff to 0
        mask_noise = (counts_r < count_threshold) & (counts_g < count_threshold)
        rel_diff[mask_noise] = 0.0
        
        all_diffs.append(np.abs(rel_diff).flatten())

    concat_diffs = np.concatenate(all_diffs)
    # Remove zeros so they don't drag down the percentile calculation
    concat_diffs = concat_diffs[concat_diffs > 1e-6] 
    
    if len(concat_diffs) == 0:
        return 1.0 # Fallback
        
    return np.percentile(concat_diffs, quantile * 100)

def plot_pca_analysis(
    df: pd.DataFrame, 
    pc_columns: list[str], 
    bins: int = 30, 
    sigma: float = 1.0,
    epsilon: float = 1e-4, 
    vmax_cap: float = 2.0,
    count_threshold: int = 5  # New Parameter
):
    """
    Generates a PairGrid with Relative Density Difference.
    Ignores relative errors in bins where raw counts < count_threshold.
    """
    
    print("Calculating relative density limits with thresholding...")
    robust_max = _calculate_relative_limit(df, pc_columns, bins, sigma, epsilon, count_threshold)
    limit = min(robust_max, vmax_cap)
    print(f"Color scale set to: +/- {limit:.2f}")

    def plot_density_diff_closure(x, y, **kwargs):
        d_subset = df.loc[x.index]
        x_real = x[d_subset['Source'] == 'Real Data']
        y_real = y[d_subset['Source'] == 'Real Data']
        x_gen = x[d_subset['Source'] == 'RBM Samples']
        y_gen = y[d_subset['Source'] == 'RBM Samples']

        global_min = min(x.min(), y.min())
        global_max = max(x.max(), y.max())
        r_lims = [[global_min, global_max], [global_min, global_max]]
        
        # 1. Raw Counts (for Mask)
        counts_r, xed, yed = np.histogram2d(x_real, y_real, bins=bins, range=r_lims)
        counts_g, _, _ = np.histogram2d(x_gen, y_gen, bins=bins, range=r_lims)
        
        # 2. Densities (for Plot)
        dens_r, _, _ = np.histogram2d(x_real, y_real, bins=bins, range=r_lims, density=True)
        dens_g, _, _ = np.histogram2d(x_gen, y_gen, bins=bins, range=r_lims, density=True)
        
        # 3. Smooth
        s_r = gaussian_filter(dens_r, sigma)
        s_g = gaussian_filter(dens_g, sigma)
        
        # 4. Relative Diff
        denom = s_r.copy()
        denom[denom < epsilon] = epsilon
        rel_diff = (s_g - s_r) / denom
        
        # 5. Apply Threshold Mask
        # If raw count is low in BOTH, we treat it as background/noise
        mask_noise = (counts_r < count_threshold) & (counts_g < count_threshold)
        
        # Set noise to NaN so it renders transparent (white), rather than 0 (color)
        rel_diff[mask_noise] = np.nan 

        ax = plt.gca()
        ax.imshow(
            rel_diff.T, origin='lower', extent=[xed[0], xed[-1], yed[0], yed[-1]],
            cmap='seismic', aspect='auto',
            vmin=-limit, vmax=limit 
        )

    # --- Standard Scatter/Diag Logic ---
    def plot_scatter_closure(x, y, **kwargs):
        labels = ['Real Data', 'RBM Samples']
        colors = {'Real Data': 'black', 'RBM Samples': 'red'}
        alphas = {'Real Data': 0.15, 'RBM Samples': 0.15}
        d_subset = df.loc[x.index]
        for s in labels:
            mask = d_subset['Source'] == s
            if mask.any():
                plt.scatter(x[mask], y[mask], c=colors[s], s=3, alpha=alphas[s], edgecolors='none', rasterized=True)

    def plot_diag_closure(x, **kwargs):
        d_subset = df.loc[x.index]
        colors = {'Real Data': 'black', 'RBM Samples': 'red'}
        for s in ['Real Data', 'RBM Samples']:
            mask = d_subset['Source'] == s
            if mask.any():
                sns.histplot(x[mask], element='step', fill=False, stat='density', color=colors[s], lw=1.5)

    # --- Build Plot ---
    g = sns.PairGrid(df, vars=pc_columns, corner=False)
    g.map_lower(plot_scatter_closure)
    g.map_diag(plot_diag_closure)
    g.map_upper(plot_density_diff_closure)

    legend_elements = [
        Line2D([0], [0], color='black', lw=2, label='Real Data'),
        Line2D([0], [0], color='red', lw=2, label='RBM Samples'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='red', label='Excess (>0)', markersize=10),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', label='Deficit (<0)', markersize=10)
    ]
    g.fig.legend(handles=legend_elements, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.05))
    
    sm = plt.cm.ScalarMappable(cmap='seismic', norm=plt.Normalize(vmin=-limit, vmax=limit))
    sm.set_array([])
    cbar_ax = g.fig.add_axes([1.0, 0.3, 0.01, 0.2])
    g.fig.colorbar(sm, cax=cbar_ax, label=f'Relative Density Diff\n(Filtered count < {count_threshold})')
    
    return g



def prepare_dataframe_incident_energy(proj_real, proj_gen, energies, n_components):
    """
    Prepares dataframe including incidence energies.
    Assumes 'energies' aligns 1:1 with both proj_real and proj_gen 
    (i.e., real[i] and gen[i] both correspond to energies[i]).
    """
    pc_cols = [f"PC{i}" for i in range(n_components)]
    
    # Ensure energies is a flat numpy array
    if torch.is_tensor(energies):
        energies_np = energies.cpu().detach().numpy().flatten()
    else:
        energies_np = np.array(energies).flatten()

    # Create Real Dataframe
    df_real = pd.DataFrame(proj_real, columns=pc_cols)
    df_real['Source'] = 'Real Data'
    # Check if lengths match to avoid pandas assignment errors
    if len(df_real) == len(energies_np):
        df_real['Energy'] = energies_np
    else:
        raise ValueError(f"Shape mismatch: Real data has {len(df_real)} samples but energies has {len(energies_np)}")

    # Create Gen Dataframe
    df_gen = pd.DataFrame(proj_gen, columns=pc_cols)
    df_gen['Source'] = 'RBM Samples'
    if len(df_gen) == len(energies_np):
        df_gen['Energy'] = energies_np
    else:
        raise ValueError(f"Shape mismatch: Gen data has {len(df_gen)} samples but energies has {len(energies_np)}")

    return pd.concat([df_real, df_gen], ignore_index=True), pc_cols

def _calculate_relative_limit(df, pc_columns, bins, sigma, epsilon, count_threshold, quantile=0.99):
    """
    (Unchanged) Calculates limits, ignoring bins where raw counts are below the threshold.
    """
    all_diffs = []
    
    df_real = df[df['Source'] == 'Real Data']
    df_gen = df[df['Source'] == 'RBM Samples']
    
    for col_x, col_y in itertools.combinations(pc_columns, 2):
        x_real, y_real = df_real[col_x], df_real[col_y]
        x_gen, y_gen = df_gen[col_x], df_gen[col_y]
        
        global_min = min(x_real.min(), x_gen.min(), y_real.min(), y_gen.min())
        global_max = max(x_real.max(), x_gen.max(), y_real.max(), y_gen.max())
        r_lims = [[global_min, global_max], [global_min, global_max]]
        
        counts_r, _, _ = np.histogram2d(x_real, y_real, bins=bins, range=r_lims)
        counts_g, _, _ = np.histogram2d(x_gen, y_gen, bins=bins, range=r_lims)
        
        dens_r, _, _ = np.histogram2d(x_real, y_real, bins=bins, range=r_lims, density=True)
        dens_g, _, _ = np.histogram2d(x_gen, y_gen, bins=bins, range=r_lims, density=True)
        
        s_r = gaussian_filter(dens_r, sigma)
        s_g = gaussian_filter(dens_g, sigma)
        
        denom = s_r.copy()
        denom[denom < epsilon] = epsilon
        rel_diff = (s_g - s_r) / denom
        
        mask_noise = (counts_r < count_threshold) & (counts_g < count_threshold)
        rel_diff[mask_noise] = 0.0
        
        all_diffs.append(np.abs(rel_diff).flatten())

    concat_diffs = np.concatenate(all_diffs)
    concat_diffs = concat_diffs[concat_diffs > 1e-6] 
    
    if len(concat_diffs) == 0:
        return 1.0
        
    return np.percentile(concat_diffs, quantile * 100)

def plot_pca_analysis_with_energy(
    df: pd.DataFrame, 
    pc_columns: list[str], 
    bins: int = 30, 
    sigma: float = 1.0,
    epsilon: float = 1e-4, 
    vmax_cap: float = 2.0,
    count_threshold: int = 5
):
    """
    Generates a PairGrid.
    Upper: Relative Density Difference (Real vs Gen).
    Diag: Histograms (Real vs Gen).
    Lower: Scatter plots COLORED by Energy.
    """
    
    print("Calculating relative density limits with thresholding...")
    robust_max = _calculate_relative_limit(df, pc_columns, bins, sigma, epsilon, count_threshold)
    limit = min(robust_max, vmax_cap)
    print(f"Density diff scale set to: +/- {limit:.2f}")

    e_min = df['Energy'].min()
    e_max = df['Energy'].max()
    print(f"Energy color scale: {e_min:.2f} to {e_max:.2f}")

    # --- Upper: Density Difference ---
    def plot_density_diff_closure(x, y, **kwargs):
        d_subset = df.loc[x.index]
        x_real = x[d_subset['Source'] == 'Real Data']
        y_real = y[d_subset['Source'] == 'Real Data']
        x_gen = x[d_subset['Source'] == 'RBM Samples']
        y_gen = y[d_subset['Source'] == 'RBM Samples']

        global_min = min(x.min(), y.min())
        global_max = max(x.max(), y.max())
        r_lims = [[global_min, global_max], [global_min, global_max]]
        
        counts_r, xed, yed = np.histogram2d(x_real, y_real, bins=bins, range=r_lims)
        counts_g, _, _ = np.histogram2d(x_gen, y_gen, bins=bins, range=r_lims)
        
        dens_r, _, _ = np.histogram2d(x_real, y_real, bins=bins, range=r_lims, density=True)
        dens_g, _, _ = np.histogram2d(x_gen, y_gen, bins=bins, range=r_lims, density=True)
        
        s_r = gaussian_filter(dens_r, sigma)
        s_g = gaussian_filter(dens_g, sigma)
        
        denom = s_r.copy()
        denom[denom < epsilon] = epsilon
        rel_diff = (s_g - s_r) / denom
        
        mask_noise = (counts_r < count_threshold) & (counts_g < count_threshold)
        rel_diff[mask_noise] = np.nan 

        ax = plt.gca()
        ax.imshow(
            rel_diff.T, origin='lower', extent=[xed[0], xed[-1], yed[0], yed[-1]],
            cmap='seismic', aspect='auto',
            vmin=-limit, vmax=limit 
        )

    # --- Lower: Scatter Colored by Energy ---
    def plot_scatter_energy_closure(x, y, **kwargs):
        d_subset = df.loc[x.index]
        
        # Configuration for sources
        # 'edgecolors' is ONLY present for Real Data. 
        # It is omitted for RBM Samples to prevent the warning.
        source_config = {
            'Real Data':   {'marker': 'o', 'zorder': 1, 'alpha': 0.3, 'edgecolors': 'none'},
            'RBM Samples': {'marker': 'x', 'zorder': 2, 'alpha': 0.6}
        }

        for s, config in source_config.items():
            mask = d_subset['Source'] == s
            if mask.any():
                energy_vals = d_subset.loc[mask, 'Energy']
                
                # Base arguments common to both
                scatter_kwargs = {
                    'c': energy_vals,
                    'cmap': 'magma',
                    'vmin': e_min,
                    'vmax': e_max,
                    's': 10,
                    'rasterized': True
                }
                
                # Merge specific config (overwrites/adds keys)
                scatter_kwargs.update(config)
                
                # Pass unpacked dictionary
                plt.scatter(x[mask], y[mask], **scatter_kwargs)

    # --- Diag: Standard Histogram ---
    def plot_diag_closure(x, **kwargs):
        d_subset = df.loc[x.index]
        colors = {'Real Data': 'black', 'RBM Samples': 'red'}
        for s in ['Real Data', 'RBM Samples']:
            mask = d_subset['Source'] == s
            if mask.any():
                sns.histplot(x[mask], element='step', fill=False, stat='density', color=colors[s], lw=1.5)

    # --- Build Plot ---
    g = sns.PairGrid(df, vars=pc_columns, corner=False)
    g.map_lower(plot_scatter_energy_closure)
    g.map_diag(plot_diag_closure)
    g.map_upper(plot_density_diff_closure)

    # --- Legends & Colorbars ---
    legend_elements = [
        Line2D([0], [0], color='black', lw=2, label='Real (Hist)'),
        Line2D([0], [0], color='red', lw=2, label='RBM (Hist)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', label='Real (Scatter)', markersize=8),
        Line2D([0], [0], marker='x', color='black', lw=0, label='RBM (Scatter)', markersize=8, markeredgewidth=2),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='red', label='Excess (>0)', markersize=10),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', label='Deficit (<0)', markersize=10)
    ]
    g.fig.legend(handles=legend_elements, loc='upper center', ncol=6, bbox_to_anchor=(0.5, 1.05), fontsize='small')
    
    sm_diff = plt.cm.ScalarMappable(cmap='seismic', norm=plt.Normalize(vmin=-limit, vmax=limit))
    sm_diff.set_array([])
    cbar_ax_diff = g.fig.add_axes([0.98, 0.3, 0.015, 0.25]) 
    g.fig.colorbar(sm_diff, cax=cbar_ax_diff, label=f'Rel. Density Diff\n(Count > {count_threshold})')

    sm_eng = plt.cm.ScalarMappable(cmap='magma', norm=plt.Normalize(vmin=e_min, vmax=e_max))
    sm_eng.set_array([])
    cbar_ax_eng = g.fig.add_axes([0.98, 0.65, 0.015, 0.25])
    g.fig.colorbar(sm_eng, cax=cbar_ax_eng, label='Incidence Energy')
    
    return g



def prepare_dataframe_features(
    proj_real, 
    proj_gen, 
    recon_features, 
    sample_features, 
    feature_name, 
    n_components
):
    """
    Prepares dataframe including a specific physics feature.
    
    Args:
        proj_real: Tensor/Array of PCA projections for real data.
        proj_gen: Tensor/Array of PCA projections for generated data.
        recon_features: Tensor/Array of features corresponding to proj_real (aligned with real samples).
        sample_features: Tensor/Array of features corresponding to proj_gen (aligned with gen samples).
        feature_name: String name for the feature (e.g., 'Layer 1 Phi').
        n_components: Number of PC components.
    """
    pc_cols = [f"PC{i}" for i in range(n_components)]
    
    # Helper to flatten the features (handles (N,) and (N,1) shapes automatically)
    def to_flat_numpy(tensor):
        if torch.is_tensor(tensor):
            data = tensor.cpu().detach().numpy()
        else:
            data = np.array(tensor)
        return data.flatten()

    feat_real_np = to_flat_numpy(recon_features)
    feat_gen_np = to_flat_numpy(sample_features)

    # Create Real Dataframe
    df_real = pd.DataFrame(proj_real, columns=pc_cols)
    df_real['Source'] = 'Real Data'
    
    if len(df_real) == len(feat_real_np):
        df_real[feature_name] = feat_real_np
    else:
        raise ValueError(f"Shape mismatch: Real proj has {len(df_real)} samples but feature tensor has {len(feat_real_np)}")

    # Create Gen Dataframe
    df_gen = pd.DataFrame(proj_gen, columns=pc_cols)
    df_gen['Source'] = 'RBM Samples'
    
    if len(df_gen) == len(feat_gen_np):
        df_gen[feature_name] = feat_gen_np
    else:
        raise ValueError(f"Shape mismatch: Gen proj has {len(df_gen)} samples but feature tensor has {len(feat_gen_np)}")

    return pd.concat([df_real, df_gen], ignore_index=True), pc_cols

def plot_pca_analysis_with_features(
    df: pd.DataFrame, 
    pc_columns: list[str], 
    feature_name: str, 
    bins: int = 30, 
    sigma: float = 1.0,
    epsilon: float = 1e-4, 
    vmax_cap: float = 2.0,
    count_threshold: int = 5
):
    """
    Generates a PairGrid colored by the specified physics feature.
    
    Args:
        df: DataFrame output from prepare_dataframe_features.
        pc_columns: List of PC column names.
        feature_name: The specific column name to use for coloring the scatter plots.
    """
    
    # --- 1. Calculate Limits ---
    print("Calculating relative density limits with thresholding...")
    # Assumes _calculate_relative_limit is defined in your scope (same as previous code)
    robust_max = _calculate_relative_limit(df, pc_columns, bins, sigma, epsilon, count_threshold)
    limit = min(robust_max, vmax_cap)
    print(f"Density diff scale set to: +/- {limit:.2f}")

    # Determine Range for the Custom Feature for consistent coloring
    f_min = df[feature_name].min()
    f_max = df[feature_name].max()
    print(f"Feature '{feature_name}' color scale: {f_min:.2f} to {f_max:.2f}")

    # --- 2. Define Plotting Closures ---

    # Upper: Density Difference
    def plot_density_diff_closure(x, y, **kwargs):
        d_subset = df.loc[x.index]
        x_real = x[d_subset['Source'] == 'Real Data']
        y_real = y[d_subset['Source'] == 'Real Data']
        x_gen = x[d_subset['Source'] == 'RBM Samples']
        y_gen = y[d_subset['Source'] == 'RBM Samples']

        global_min = min(x.min(), y.min())
        global_max = max(x.max(), y.max())
        r_lims = [[global_min, global_max], [global_min, global_max]]
        
        counts_r, xed, yed = np.histogram2d(x_real, y_real, bins=bins, range=r_lims)
        counts_g, _, _ = np.histogram2d(x_gen, y_gen, bins=bins, range=r_lims)
        
        dens_r, _, _ = np.histogram2d(x_real, y_real, bins=bins, range=r_lims, density=True)
        dens_g, _, _ = np.histogram2d(x_gen, y_gen, bins=bins, range=r_lims, density=True)
        
        s_r = gaussian_filter(dens_r, sigma)
        s_g = gaussian_filter(dens_g, sigma)
        
        denom = s_r.copy()
        denom[denom < epsilon] = epsilon
        rel_diff = (s_g - s_r) / denom
        
        mask_noise = (counts_r < count_threshold) & (counts_g < count_threshold)
        rel_diff[mask_noise] = np.nan 

        ax = plt.gca()
        ax.imshow(
            rel_diff.T, origin='lower', extent=[xed[0], xed[-1], yed[0], yed[-1]],
            cmap='seismic', aspect='auto',
            vmin=-limit, vmax=limit 
        )

    # Lower: Scatter Colored by the specific Feature
    def plot_scatter_feature_closure(x, y, **kwargs):
        d_subset = df.loc[x.index]
        
        source_config = {
            'Real Data':   {'marker': 'o', 'zorder': 1, 'alpha': 0.3, 'edgecolors': 'none'},
            'RBM Samples': {'marker': 'x', 'zorder': 2, 'alpha': 0.6}
        }

        for s, config in source_config.items():
            mask = d_subset['Source'] == s
            if mask.any():
                # Dynamically select the feature column based on the argument
                feat_vals = d_subset.loc[mask, feature_name]
                
                scatter_kwargs = {
                    'c': feat_vals,
                    'cmap': 'magma',
                    'vmin': f_min,
                    'vmax': f_max,
                    's': 10,
                    'rasterized': True
                }
                scatter_kwargs.update(config)
                plt.scatter(x[mask], y[mask], **scatter_kwargs)

    # Diag: Standard Histogram
    def plot_diag_closure(x, **kwargs):
        d_subset = df.loc[x.index]
        colors = {'Real Data': 'black', 'RBM Samples': 'red'}
        for s in ['Real Data', 'RBM Samples']:
            mask = d_subset['Source'] == s
            if mask.any():
                sns.histplot(x[mask], element='step', fill=False, stat='density', color=colors[s], lw=1.5)

    # --- 3. Build Plot ---
    g = sns.PairGrid(df, vars=pc_columns, corner=False)
    g.map_lower(plot_scatter_feature_closure)
    g.map_diag(plot_diag_closure)
    g.map_upper(plot_density_diff_closure)

    # --- 4. Legends & Colorbars ---
    legend_elements = [
        Line2D([0], [0], color='black', lw=2, label='Real (Hist)'),
        Line2D([0], [0], color='red', lw=2, label='RBM (Hist)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='black', label='Real (Scatter)', markersize=8),
        Line2D([0], [0], marker='x', color='black', lw=0, label='RBM (Scatter)', markersize=8, markeredgewidth=2),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='red', label='Excess (>0)', markersize=10),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='blue', label='Deficit (<0)', markersize=10)
    ]
    g.fig.legend(handles=legend_elements, loc='upper center', ncol=6, bbox_to_anchor=(0.5, 1.05), fontsize='small')
    
    # Relative Density Colorbar
    sm_diff = plt.cm.ScalarMappable(cmap='seismic', norm=plt.Normalize(vmin=-limit, vmax=limit))
    sm_diff.set_array([])
    cbar_ax_diff = g.fig.add_axes([0.98, 0.3, 0.015, 0.25]) 
    g.fig.colorbar(sm_diff, cax=cbar_ax_diff, label=f'Rel. Density Diff\n(Count > {count_threshold})')

    # Physics Feature Colorbar
    sm_feat = plt.cm.ScalarMappable(cmap='magma', norm=plt.Normalize(vmin=f_min, vmax=f_max))
    sm_feat.set_array([])
    cbar_ax_feat = g.fig.add_axes([0.98, 0.65, 0.015, 0.25])
    g.fig.colorbar(sm_feat, cax=cbar_ax_feat, label=feature_name)
    
    return g



def prepare_dataframe_pls(
    proj_real: np.ndarray,
    proj_gen: np.ndarray,
    n_components: int,
) -> Tuple[pd.DataFrame, List[str]]:
    """Mirrors prepare_dataframe but labels columns LV0, LV1, … ."""
    lv_cols = [f"LV{i}" for i in range(n_components)]
    df_real = pd.DataFrame(proj_real, columns=lv_cols)
    df_real["Source"] = "Real Data"
    df_gen = pd.DataFrame(proj_gen, columns=lv_cols)
    df_gen["Source"] = "RBM Samples"
    return pd.concat([df_real, df_gen], ignore_index=True), lv_cols


def _calculate_relative_limit_pls(df, lv_columns, bins, sigma, epsilon, count_threshold, quantile=0.99):
    """Same logic as _calculate_relative_limit but works with any column label set."""
    all_diffs = []
    df_real = df[df["Source"] == "Real Data"]
    df_gen  = df[df["Source"] == "RBM Samples"]
    for col_x, col_y in itertools.combinations(lv_columns, 2):
        x_real, y_real = df_real[col_x], df_real[col_y]
        x_gen,  y_gen  = df_gen[col_x],  df_gen[col_y]
        global_min = min(x_real.min(), x_gen.min(), y_real.min(), y_gen.min())
        global_max = max(x_real.max(), x_gen.max(), y_real.max(), y_gen.max())
        r_lims = [[global_min, global_max], [global_min, global_max]]
        counts_r, _, _ = np.histogram2d(x_real, y_real, bins=bins, range=r_lims)
        counts_g, _, _ = np.histogram2d(x_gen,  y_gen,  bins=bins, range=r_lims)
        dens_r,   _, _ = np.histogram2d(x_real, y_real, bins=bins, range=r_lims, density=True)
        dens_g,   _, _ = np.histogram2d(x_gen,  y_gen,  bins=bins, range=r_lims, density=True)
        s_r = gaussian_filter(dens_r, sigma)
        s_g = gaussian_filter(dens_g, sigma)
        denom = s_r.copy()
        denom[denom < epsilon] = epsilon
        rel_diff = (s_g - s_r) / denom
        mask_noise = (counts_r < count_threshold) & (counts_g < count_threshold)
        rel_diff[mask_noise] = 0.0
        all_diffs.append(np.abs(rel_diff).flatten())
    concat_diffs = np.concatenate(all_diffs)
    concat_diffs = concat_diffs[concat_diffs > 1e-6]
    return np.percentile(concat_diffs, quantile * 100) if len(concat_diffs) else 1.0


def plot_pls_analysis(
    df: pd.DataFrame,
    lv_columns: List[str],
    bins: int = 30,
    sigma: float = 1.0,
    epsilon: float = 1e-4,
    vmax_cap: float = 2.0,
    count_threshold: int = 5,
) -> sns.PairGrid:
    """
    PairGrid for PLS latent-variable scores.

    Lower : scatter (Real = black, Gen = red)
    Diag  : overlaid step histograms
    Upper : relative density difference (seismic colour map)
    """
    print("Calculating relative density limits...")
    robust_max = _calculate_relative_limit_pls(df, lv_columns, bins, sigma, epsilon, count_threshold)
    limit = min(robust_max, vmax_cap)
    print(f"Colour scale: +/- {limit:.2f}")

    def plot_density_diff(x, y, **kwargs):
        d = df.loc[x.index]
        x_r = x[d["Source"] == "Real Data"]; y_r = y[d["Source"] == "Real Data"]
        x_g = x[d["Source"] == "RBM Samples"]; y_g = y[d["Source"] == "RBM Samples"]
        gmin = min(x.min(), y.min()); gmax = max(x.max(), y.max())
        r_lims = [[gmin, gmax], [gmin, gmax]]
        counts_r, xed, yed = np.histogram2d(x_r, y_r, bins=bins, range=r_lims)
        counts_g, _,   _   = np.histogram2d(x_g, y_g, bins=bins, range=r_lims)
        dens_r,   _,   _   = np.histogram2d(x_r, y_r, bins=bins, range=r_lims, density=True)
        dens_g,   _,   _   = np.histogram2d(x_g, y_g, bins=bins, range=r_lims, density=True)
        s_r = gaussian_filter(dens_r, sigma); s_g = gaussian_filter(dens_g, sigma)
        denom = s_r.copy(); denom[denom < epsilon] = epsilon
        rel_diff = (s_g - s_r) / denom
        rel_diff[(counts_r < count_threshold) & (counts_g < count_threshold)] = np.nan
        plt.gca().imshow(
            rel_diff.T, origin="lower", extent=[xed[0], xed[-1], yed[0], yed[-1]],
            cmap="seismic", aspect="auto", vmin=-limit, vmax=limit,
        )

    def plot_scatter(x, y, **kwargs):
        d = df.loc[x.index]
        for s, c, a in [("Real Data", "black", 0.15), ("RBM Samples", "red", 0.15)]:
            m = d["Source"] == s
            if m.any():
                plt.scatter(x[m], y[m], c=c, s=3, alpha=a, edgecolors="none", rasterized=True)

    def plot_diag(x, **kwargs):
        d = df.loc[x.index]
        for s, c in [("Real Data", "black"), ("RBM Samples", "red")]:
            m = d["Source"] == s
            if m.any():
                sns.histplot(x[m], element="step", fill=False, stat="density", color=c, lw=1.5)

    g = sns.PairGrid(df, vars=lv_columns, corner=False)
    g.map_lower(plot_scatter)
    g.map_diag(plot_diag)
    g.map_upper(plot_density_diff)

    legend_elements = [
        Line2D([0], [0], color="black", lw=2, label="Real Data"),
        Line2D([0], [0], color="red",   lw=2, label="RBM Samples"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="red",  label="Excess (>0)", markersize=10),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="blue", label="Deficit (<0)", markersize=10),
    ]
    g.fig.legend(handles=legend_elements, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.05))
    sm = plt.cm.ScalarMappable(cmap="seismic", norm=plt.Normalize(vmin=-limit, vmax=limit))
    sm.set_array([])
    cbar_ax = g.fig.add_axes([1.0, 0.3, 0.01, 0.2])
    g.fig.colorbar(sm, cax=cbar_ax, label=f"Relative Density Diff\n(Count > {count_threshold})")
    return g


def plot_pls_loadings(
    y_weights: np.ndarray,
    feature_names: List[str],
    n_components: Optional[int] = None,
    x_weights: Optional[np.ndarray] = None,
    top_k_visible: int = 20,
) -> plt.Figure:
    """
    Visualise PLS weight vectors.

    Left panel  : Y-weights — how strongly each classical feature loads onto each LV.
    Right panel : Top-k X-weights by absolute magnitude for each LV (optional).

    Parameters
    ----------
    y_weights : np.ndarray  (n_features, n_components)
    feature_names : list of str  length n_features
    n_components : int or None — defaults to y_weights.shape[1]
    x_weights : np.ndarray  (n_visible, n_components), optional
    top_k_visible : int — how many visible nodes to show in the X panel
    """
    if n_components is None:
        n_components = y_weights.shape[1]

    n_panels = 2 if x_weights is not None else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, max(4, 0.4 * len(feature_names) + 2)))

    # --- Y-weights heatmap ---
    ax_y = axes[0] if n_panels == 2 else axes
    lv_labels = [f"LV{i}" for i in range(n_components)]
    vmax = np.abs(y_weights[:, :n_components]).max()
    sns.heatmap(
        y_weights[:, :n_components],
        annot=True, fmt=".2f",
        cmap="coolwarm", center=0, vmin=-vmax, vmax=vmax,
        xticklabels=lv_labels, yticklabels=feature_names,
        ax=ax_y, linewidths=0.3,
    )
    ax_y.set_title("Y-Weights: Classical Features → PLS Latent Variables")
    ax_y.set_xlabel("Latent Variable")
    ax_y.set_ylabel("Classical Feature")

    # --- X-weights bar chart (top-k by |w|) ---
    if x_weights is not None:
        ax_x = axes[1]
        # Average absolute weight across all LVs to rank nodes
        mean_abs = np.abs(x_weights[:, :n_components]).mean(axis=1)
        top_idx = np.argsort(mean_abs)[-top_k_visible:][::-1]

        x = np.arange(len(top_idx))
        width = 0.8 / n_components
        colors = plt.cm.tab10(np.linspace(0, 1, n_components))
        for k in range(n_components):
            ax_x.bar(
                x + k * width,
                x_weights[top_idx, k],
                width=width, color=colors[k], alpha=0.8, label=f"LV{k}",
            )
        ax_x.set_xticks(x + width * (n_components - 1) / 2)
        ax_x.set_xticklabels([str(i) for i in top_idx], rotation=90, fontsize=7)
        ax_x.axhline(0, color="black", lw=0.7)
        ax_x.set_title(f"X-Weights: Top {top_k_visible} Visible Nodes by Mean |w|")
        ax_x.set_xlabel("Visible Node Index")
        ax_x.set_ylabel("Weight")
        ax_x.legend(fontsize=8)

    plt.tight_layout()
    return fig


def plot_pls_r2(
    r2: np.ndarray,
    feature_names: Optional[List[str]] = None,
    proj_real: Optional[np.ndarray] = None,
    classical_features=None,
    lv_idx: int = 0,
) -> plt.Figure:
    """
    Left panel : cumulative R² vs number of LVs.
    Right panel: scatter of LV{lv_idx} scores vs the feature (optional).
                 Shown when both proj_real and classical_features are provided.

    Parameters
    ----------
    r2 : np.ndarray  (n_components, n_features)
        Output of calculate_pls_r2.
    feature_names : list of str, optional
    proj_real : np.ndarray  (n_real, n_components), optional
        LV scores for real data (from run_pls_pipeline).
    classical_features : Tensor or ndarray  (n_real,) or (n_real, n_features), optional
        Feature values aligned with proj_real.
    lv_idx : int
        Which latent variable to show in the scatter panel (default 0).
    """
    show_scatter = (proj_real is not None) and (classical_features is not None)
    n_panels = 2 if show_scatter else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    # --- Left: cumulative R² ---
    ax = axes[0]
    n_components, n_features = r2.shape
    x = np.arange(1, n_components + 1)
    colors = plt.cm.tab10(np.linspace(0, 1, max(n_features, 2)))
    for i in range(n_features):
        label = feature_names[i] if feature_names is not None else f"Feature {i}"
        ax.plot(x, r2[:, i], marker="o", color=colors[i], label=label, lw=2)
    ax.axhline(1.0, color="grey", linestyle="--", lw=0.8, alpha=0.6)
    ax.set_xlabel("Number of PLS Latent Variables")
    ax.set_ylabel("Cumulative $R^2$ (Y-variance explained)")
    ax.set_title("PLS: Cumulative Feature Variance Explained")
    ax.set_xticks(x)
    ax.set_xticklabels([f"LV{k}" for k in range(n_components)])
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    if n_features > 1:
        ax.legend(fontsize=8)

    # --- Right: LV vs feature scatter ---
    if show_scatter:
        plot_lv_feature_scatter(
            proj_real, classical_features,
            lv_idx=lv_idx, feature_idx=0,
            feature_names=feature_names,
            ax=axes[1],
        )

    plt.tight_layout()
    return fig


def plot_lv_feature_scatter(
    proj_real: np.ndarray,
    classical_features,
    lv_idx: int = 0,
    feature_idx: int = 0,
    feature_names: Optional[List[str]] = None,
    proj_gen: Optional[np.ndarray] = None,
    ax: Optional[plt.Axes] = None,
) -> plt.Figure:
    """
    Scatter plot of a single LV score vs a single classical feature value.
    Overlays a linear regression line so the strength of the relationship
    is immediately visible.

    Parameters
    ----------
    proj_real : np.ndarray  (n_real, n_components)
    classical_features : Tensor or ndarray  (n_real,) or (n_real, n_features)
    lv_idx : int   — which latent variable column to plot on the x-axis
    feature_idx : int — which feature column to plot on the y-axis
    feature_names : list of str, optional
    proj_gen : np.ndarray  (n_gen, n_components), optional
        If provided, gen samples are overlaid in red.
    ax : plt.Axes, optional — draw into an existing axis; otherwise a new figure.
    """
    if torch.is_tensor(classical_features):
        Y = classical_features.detach().cpu().numpy().astype(np.float64)
    else:
        Y = np.array(classical_features, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]

    t = proj_real[:, lv_idx]
    y = Y[:, feature_idx]

    feat_label = (
        feature_names[feature_idx] if feature_names is not None else f"Feature {feature_idx}"
    )
    lv_label = f"LV{lv_idx}"

    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.get_figure()

    # Real data scatter
    ax.scatter(t, y, c="black", s=4, alpha=0.25, edgecolors="none",
               rasterized=True, label="Real Data", zorder=1)

    # Gen data scatter (optional)
    if proj_gen is not None:
        ax.scatter(proj_gen[:, lv_idx], np.full(len(proj_gen), np.nan),
                   c="red", s=4, alpha=0.25, edgecolors="none",
                   rasterized=True, label="RBM Samples", zorder=2)
        # We don't have gen features, so gen points are plotted at y=nan (invisible)
        # but the legend entry is there for context. If you have gen features, pass them.

    # Linear regression line
    coeffs = np.polyfit(t, y, 1)
    t_line = np.linspace(t.min(), t.max(), 200)
    ax.plot(t_line, np.polyval(coeffs, t_line), color="crimson", lw=1.5,
            label=f"OLS fit (slope={coeffs[0]:.3f})", zorder=3)

    # Pearson r and Spearman ρ annotation
    r = np.corrcoef(t, y)[0, 1]
    rho, _ = spearmanr(t, y)
    ax.annotate(
        f"$r$ = {r:.3f}    $\\rho$ = {rho:.3f}",
        xy=(0.05, 0.93), xycoords="axes fraction", fontsize=10,
    )

    ax.set_xlabel(lv_label)
    ax.set_ylabel(feat_label)
    ax.set_title(f"{lv_label} vs {feat_label}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    if standalone:
        plt.tight_layout()
    return fig


def plot_lv_quantile_tracking(
    proj_real: np.ndarray,
    classical_features,
    lv_idx: int = 0,
    feature_idx: int = 0,
    n_quantiles: int = 10,
    tail_pct: float = 5.0,
    feature_names: Optional[List[str]] = None,
) -> plt.Figure:
    """
    Answers "are the tails of Y consistently landing at the extremes of LV{lv_idx}?"

    Left panel — Quantile-tracking plot:
        Samples are binned by their LV score percentile (x-axis).
        For each bin the median Y and 10th–90th percentile band of Y are shown.
        A flat line = no relationship; a monotone staircase = perfect rank separation.

    Right panel — Tail concentration:
        For a range of tail thresholds (top/bottom p% of Y), shows what fraction
        of those tail samples fall in the corresponding top/bottom p% of LV scores.
        Random chance = diagonal; perfect = 1.0 everywhere.
        The Spearman ρ is annotated as the headline scalar.

    Parameters
    ----------
    proj_real : np.ndarray  (n_real, n_components)
    classical_features : Tensor or ndarray  (n_real,) or (n_real, n_features)
    lv_idx : int
    feature_idx : int
    n_quantiles : int — number of equal-frequency LV bins for the left panel
    tail_pct : float — not used directly; tail panel sweeps 1% to 30%
    feature_names : list of str, optional
    """
    if torch.is_tensor(classical_features):
        Y = classical_features.detach().cpu().numpy().astype(np.float64)
    else:
        Y = np.array(classical_features, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]

    t = proj_real[:, lv_idx]
    y = Y[:, feature_idx]
    feat_label = feature_names[feature_idx] if feature_names is not None else f"Feature {feature_idx}"
    lv_label = f"LV{lv_idx}"

    rho, _ = spearmanr(t, y)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # --- Left: quantile-tracking ---
    ax = axes[0]
    bin_edges = np.percentile(t, np.linspace(0, 100, n_quantiles + 1))
    bin_edges[-1] += 1e-10  # include right edge
    bin_ids = np.digitize(t, bin_edges) - 1
    bin_ids = np.clip(bin_ids, 0, n_quantiles - 1)

    bin_centers, medians, p10s, p90s = [], [], [], []
    for b in range(n_quantiles):
        mask = bin_ids == b
        if mask.sum() < 2:
            continue
        y_bin = y[mask]
        bin_centers.append(np.percentile(t, (b + 0.5) * 100 / n_quantiles))
        medians.append(np.median(y_bin))
        p10s.append(np.percentile(y_bin, 10))
        p90s.append(np.percentile(y_bin, 90))

    bin_centers = np.array(bin_centers)
    medians = np.array(medians)
    p10s = np.array(p10s)
    p90s = np.array(p90s)

    ax.fill_between(bin_centers, p10s, p90s, alpha=0.25, color="steelblue", label="10th–90th pct")
    ax.plot(bin_centers, medians, marker="o", color="steelblue", lw=2, label="Median Y")
    ax.axhline(np.median(y), color="grey", linestyle="--", lw=0.8, alpha=0.7, label="Overall median")
    ax.set_xlabel(f"{lv_label} score (binned by percentile)")
    ax.set_ylabel(feat_label)
    ax.set_title(f"Quantile Tracking: {lv_label} → {feat_label}\n$\\rho$ = {rho:.3f}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Right: tail concentration curve ---
    ax2 = axes[1]
    thresholds = np.linspace(1, 30, 30)  # 1% to 30% tail
    top_capture, bot_capture = [], []

    for p in thresholds:
        t_hi = np.percentile(t, 100 - p)
        t_lo = np.percentile(t, p)
        y_hi = np.percentile(y, 100 - p)
        y_lo = np.percentile(y, p)

        top_mask = y > y_hi
        bot_mask = y < y_lo
        n_top = top_mask.sum()
        n_bot = bot_mask.sum()

        top_capture.append(((t > t_hi) & top_mask).sum() / n_top if n_top > 0 else 0.0)
        bot_capture.append(((t < t_lo) & bot_mask).sum() / n_bot if n_bot > 0 else 0.0)

    ax2.plot(thresholds, top_capture, color="crimson",  lw=2, marker="o", markersize=3, label="Top tail (high Y)")
    ax2.plot(thresholds, bot_capture, color="royalblue", lw=2, marker="o", markersize=3, label="Bottom tail (low Y)")
    ax2.plot(thresholds, thresholds / 100, color="grey", linestyle="--", lw=1.2, label="Random chance")
    ax2.set_xlabel("Tail threshold (top/bottom p%)")
    ax2.set_ylabel(f"Fraction of tail captured in LV{lv_idx} extreme p%")
    ax2.set_title(f"Tail Concentration: {lv_label} → {feat_label}")
    ax2.set_xlim(0, 31)
    ax2.set_ylim(0, 1.05)
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def prepare_dataframe_sir(
    proj_real: np.ndarray,
    proj_gen: np.ndarray,
    n_components: int,
) -> Tuple[pd.DataFrame, List[str]]:
    """Mirrors prepare_dataframe_pls but labels columns SIR0, SIR1, …"""
    sir_cols = [f"SIR{i}" for i in range(n_components)]
    df_real = pd.DataFrame(proj_real, columns=sir_cols)
    df_real["Source"] = "Real Data"
    df_gen = pd.DataFrame(proj_gen, columns=sir_cols)
    df_gen["Source"] = "RBM Samples"
    return pd.concat([df_real, df_gen], ignore_index=True), sir_cols


def plot_sir_analysis(
    df: pd.DataFrame,
    sir_columns: List[str],
    bins: int = 30,
    sigma: float = 1.0,
    epsilon: float = 1e-4,
    vmax_cap: float = 2.0,
    count_threshold: int = 5,
) -> sns.PairGrid:
    """PairGrid for SIR directions — identical layout to plot_pls_analysis."""
    # Reuse the PLS limit calculation (works for any column label set)
    robust_max = _calculate_relative_limit_pls(df, sir_columns, bins, sigma, epsilon, count_threshold)
    limit = min(robust_max, vmax_cap)
    print(f"Colour scale: +/- {limit:.2f}")

    def plot_density_diff(x, y, **kwargs):
        d = df.loc[x.index]
        x_r = x[d["Source"] == "Real Data"]; y_r = y[d["Source"] == "Real Data"]
        x_g = x[d["Source"] == "RBM Samples"]; y_g = y[d["Source"] == "RBM Samples"]
        gmin = min(x.min(), y.min()); gmax = max(x.max(), y.max())
        r_lims = [[gmin, gmax], [gmin, gmax]]
        counts_r, xed, yed = np.histogram2d(x_r, y_r, bins=bins, range=r_lims)
        counts_g, _,   _   = np.histogram2d(x_g, y_g, bins=bins, range=r_lims)
        dens_r,   _,   _   = np.histogram2d(x_r, y_r, bins=bins, range=r_lims, density=True)
        dens_g,   _,   _   = np.histogram2d(x_g, y_g, bins=bins, range=r_lims, density=True)
        s_r = gaussian_filter(dens_r, sigma); s_g = gaussian_filter(dens_g, sigma)
        denom = s_r.copy(); denom[denom < epsilon] = epsilon
        rel_diff = (s_g - s_r) / denom
        rel_diff[(counts_r < count_threshold) & (counts_g < count_threshold)] = np.nan
        plt.gca().imshow(
            rel_diff.T, origin="lower", extent=[xed[0], xed[-1], yed[0], yed[-1]],
            cmap="seismic", aspect="auto", vmin=-limit, vmax=limit,
        )

    def plot_scatter(x, y, **kwargs):
        d = df.loc[x.index]
        for s, c, a in [("Real Data", "black", 0.15), ("RBM Samples", "red", 0.15)]:
            m = d["Source"] == s
            if m.any():
                plt.scatter(x[m], y[m], c=c, s=3, alpha=a, edgecolors="none", rasterized=True)

    def plot_diag(x, **kwargs):
        d = df.loc[x.index]
        for s, c in [("Real Data", "black"), ("RBM Samples", "red")]:
            m = d["Source"] == s
            if m.any():
                sns.histplot(x[m], element="step", fill=False, stat="density", color=c, lw=1.5)

    g = sns.PairGrid(df, vars=sir_columns, corner=False)
    g.map_lower(plot_scatter)
    g.map_diag(plot_diag)
    g.map_upper(plot_density_diff)

    legend_elements = [
        Line2D([0], [0], color="black", lw=2, label="Real Data"),
        Line2D([0], [0], color="red",   lw=2, label="RBM Samples"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="red",  label="Excess (>0)", markersize=10),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="blue", label="Deficit (<0)", markersize=10),
    ]
    g.fig.legend(handles=legend_elements, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.05))
    sm = plt.cm.ScalarMappable(cmap="seismic", norm=plt.Normalize(vmin=-limit, vmax=limit))
    sm.set_array([])
    cbar_ax = g.fig.add_axes([1.0, 0.3, 0.01, 0.2])
    g.fig.colorbar(sm, cax=cbar_ax, label=f"Relative Density Diff\n(Count > {count_threshold})")
    return g


def plot_sir_spectrum(
    eigenvalues: np.ndarray,
    feature_name: str = "Y",
) -> plt.Figure:
    """
    Bar chart of SIR eigenvalues (the between-slice variance captured by each direction).
    Analogous to a scree plot — shows how many directions matter.
    A sharp drop-off after k means only k directions are worth using.
    """
    n = len(eigenvalues)
    x = np.arange(n)
    total = eigenvalues.sum()
    frac = eigenvalues / total if total > 0 else eigenvalues

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Raw eigenvalues
    axes[0].bar(x, eigenvalues, color="steelblue", edgecolor="black", alpha=0.8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([f"SIR{i}" for i in x])
    axes[0].set_xlabel("SIR Direction")
    axes[0].set_ylabel("Eigenvalue of M")
    axes[0].set_title(f"SIR Spectrum\n(response: {feature_name})")
    axes[0].grid(axis="y", alpha=0.3)

    # Cumulative fraction
    axes[1].plot(x, np.cumsum(frac), marker="o", color="steelblue", lw=2)
    axes[1].axhline(1.0, color="grey", linestyle="--", lw=0.8, alpha=0.6)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"SIR{i}" for i in x])
    axes[1].set_xlabel("SIR Direction")
    axes[1].set_ylabel("Cumulative Fraction of Σ(eigenvalues)")
    axes[1].set_title("Cumulative Eigenvalue Fraction")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_basis_evaluation(metrics: dict):
    """
    Generates a two-panel plot showing Basis Alignment and Cumulative Variance Explained.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    n_components = metrics["n_components"]
    pc_labels = [f"PC{i}" for i in range(n_components)]
    
    # --- Panel 1: Absolute Cosine Similarity Heatmap ---
    sns.heatmap(
        metrics["cosine_similarity"], 
        annot=True, 
        cmap="Blues", 
        vmin=0, vmax=1,
        xticklabels=[f"Cond {pc}" for pc in pc_labels],
        yticklabels=[f"Marg {pc}" for pc in pc_labels],
        ax=axes[0]
    )
    axes[0].set_title("Absolute Cosine Similarity: $U_m^T U_c$")
    axes[0].set_xlabel("Conditional Basis ($U_c$)")
    axes[0].set_ylabel("Marginalized Basis ($U_m$)")
    
    # --- Panel 2: Cumulative Variance Explained ---
    # Convert individual PC variances to cumulative sum
    cum_var_native = np.cumsum(metrics["var_native"])
    cum_var_marginal = np.cumsum(metrics["var_marginal"])
    
    x_axis = np.arange(n_components)
    
    axes[1].plot(x_axis, cum_var_native, marker='o', linestyle='-', color='black', label="Native Basis ($U_c$)")
    axes[1].plot(x_axis, cum_var_marginal, marker='s', linestyle='--', color='red', label="Marginal Basis ($U_m$)")
    
    # Formatting
    axes[1].set_title("Cumulative Variance Explained on Conditional Data")
    axes[1].set_xlabel("Principal Component Index")
    axes[1].set_ylabel("Fraction of Total Variance")
    axes[1].set_xticks(x_axis)
    axes[1].set_xticklabels(pc_labels)
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    plt.show()


def plot_tail_weighting(
    features_list: List[torch.Tensor],
    incidence_energies: torch.Tensor,
    weights: torch.Tensor,
    feature_names: Optional[List[str]] = None,
    tail_threshold: float = 1e-4,
    z_tail_scatter: float = 3.0,
):
    """
    Visualise how the importance-weighting scheme identifies and upweights tails
    for each feature.

    For every feature this produces a two-panel figure:
      Left  — Z-score CCDF (survival function), matching the notebook style.
              A vertical dashed line marks the z-score that corresponds to the
              survival-probability tail_threshold.  A horizontal dashed line marks
              the threshold itself.
      Right — Energy vs Z-score scatter / hexbin overlay (core + tail) showing
              which samples are considered tails.

    Additionally a final figure overlays the unweighted vs importance-weighted
    marginal histogram for each feature so you can see how the distribution shifts.

    Args:
        features_list: list of (N,) or (N,1) tensors, one per feature.
        incidence_energies: (N,) or (N,1) tensor of incidence energies.
        weights: (N,) tensor of final combined importance weights.
        feature_names: optional display names for each feature.
        tail_threshold: the survival-probability threshold used during training.
        z_tail_scatter: z-score above which points are drawn as a scatter overlay
                        in the right panel (matches the notebook's 3-sigma split).
    """
    n_features = len(features_list)
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    E = incidence_energies.flatten().numpy().astype(np.float64)
    w = weights.numpy()

    # ------------------------------------------------------------------ #
    # Per-feature figures (CCDF + Energy vs Z-Score)                      #
    # ------------------------------------------------------------------ #
    for feat_idx, (feat_tensor, fname) in enumerate(zip(features_list, feature_names)):
        y = feat_tensor.squeeze().numpy().astype(np.float64)

        # Reproduce the same z-score computation used during training
        coeffs_mu = np.polyfit(E, y, 1)
        mu_local = np.polyval(coeffs_mu, E)
        residuals_sq = (y - mu_local) ** 2
        coeffs_var = np.polyfit(E, residuals_sq, 1)
        var_local = np.clip(np.polyval(coeffs_var, E), 1e-6, None)
        sig_local = np.sqrt(var_local)
        z_scores = np.abs(y - mu_local) / sig_local

        sorted_z = np.sort(z_scores)
        p_greater = 1.0 - np.arange(1, len(sorted_z) + 1) / len(sorted_z)

        lookup_idx = np.clip(np.searchsorted(sorted_z, z_scores), 0, len(sorted_z) - 1)
        sample_probs = p_greater[lookup_idx]
        is_tail = sample_probs <= tail_threshold

        # z value at the threshold boundary (for the CCDF vertical line)
        threshold_z_idx = np.searchsorted(p_greater[::-1], tail_threshold)
        threshold_z = sorted_z[max(0, len(sorted_z) - 1 - threshold_z_idx)]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f"Tail Weighting — {fname}", fontsize=14)

        # ── Left: CCDF ─────────────────────────────────────────────────
        ax1.plot(sorted_z, p_greater, color="indigo", lw=2.5)
        ax1.set_yscale("log")
        ax1.set_xlim(0, sorted_z[-1] * 1.05)
        ax1.set_ylim(1.0 / len(z_scores), 1.0)
        ax1.set_xlabel("Absolute Z-Score threshold ($z$)")
        ax1.set_ylabel("Fraction of Data > $z$")
        ax1.set_title("Z-Score Survival Function (CCDF)")
        ax1.grid(True, which="both", ls="--", alpha=0.4)

        # Sigma reference lines
        for sig in [1, 3, 5, 10]:
            if sig < sorted_z[-1]:
                ax1.axvline(sig, color="gray", ls=":", alpha=0.7)
                ax1.text(sig + 0.1, 0.5, f"{sig}$\\sigma$",
                         rotation=90, color="gray", va="center", fontsize=9)

        # Threshold annotation
        ax1.axhline(tail_threshold, color="crimson", ls="--", lw=1.5,
                    label=f"tail_threshold = {tail_threshold:.0e}")
        ax1.axvline(threshold_z, color="crimson", ls="--", lw=1.5)
        n_tail = int(is_tail.sum())
        ax1.text(0.98, 0.98,
                 f"{n_tail}/{len(y)} samples in tail\n({100*n_tail/len(y):.3f}%)",
                 transform=ax1.transAxes, ha="right", va="top",
                 fontsize=10, color="crimson",
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
        ax1.legend(fontsize=10)

        # ── Right: Energy vs Z-Score (hexbin core + scatter tail) ──────
        mask_core = z_scores < z_tail_scatter
        mask_scatter = z_scores >= z_tail_scatter

        hb = ax2.hexbin(E[mask_core], z_scores[mask_core],
                        gridsize=60, cmap="Blues", bins="log", mincnt=1)
        ax2.scatter(E[mask_scatter], z_scores[mask_scatter],
                    color="crimson", s=4, alpha=0.4,
                    label=f"Tail ($Z \\geq {z_tail_scatter}$)")

        # Mark the threshold boundary on the y-axis
        ax2.axhline(threshold_z, color="crimson", ls="--", lw=1.5,
                    label=f"z at tail_threshold")
        ax2.set_xlabel("Incident energy $E$")
        ax2.set_ylabel("Absolute Z-Score")
        ax2.set_title("Energy vs Z-Score (Core + Scatter Overlay)")
        ax2.legend(loc="upper right", fontsize=9)
        fig.colorbar(hb, ax=ax2, label="Log Counts (Core only)")

        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------ #
    # Summary figure: unweighted vs weighted marginal histograms           #
    # ------------------------------------------------------------------ #
    ncols = min(3, n_features)
    nrows = int(np.ceil(n_features / ncols))
    fig2, axes = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4 * nrows), squeeze=False)
    fig2.suptitle("Feature Distributions: Unweighted vs Importance-Weighted", fontsize=14)

    for feat_idx, (feat_tensor, fname) in enumerate(zip(features_list, feature_names)):
        row, col = divmod(feat_idx, ncols)
        ax = axes[row][col]
        y = feat_tensor.squeeze().numpy().astype(np.float64)

        bins = np.histogram_bin_edges(y, bins=80)
        ax.hist(y, bins=bins, density=True, alpha=0.55, color="steelblue",
                label="Unweighted", histtype="stepfilled")
        ax.hist(y, bins=bins, weights=w, density=True, alpha=0.55,
                color="crimson", label="Weighted", histtype="stepfilled")
        ax.set_title(fname, fontsize=11)
        ax.set_xlabel("Feature value")
        ax.set_ylabel("Density")
        ax.set_yscale("log")
        ax.legend(fontsize=9)
        ax.grid(True, ls="--", alpha=0.4)


    # Hide unused subplot cells
    for extra in range(n_features, nrows * ncols):
        row, col = divmod(extra, ncols)
        axes[row][col].axis("off")

    plt.tight_layout()
    plt.show()


def plot_energy_dependent_sigma(
    features_list: List[torch.Tensor],
    incidence_energies: torch.Tensor,
    feature_names: Optional[List[str]] = None,
    n_energy_bins: int = 40,
) -> List[plt.Figure]:
    """
    Visualise the energy-dependent mu(E) and sigma(E) implied by the z-score
    computation used during importance-weight training.

    For each feature two panels are shown:
      Left  — Raw scatter of feature vs energy with the fitted mu(E) line and
              ±1σ, ±2σ, ±3σ bands.
      Right — sigma(E) itself vs energy, with per-bin empirical std overlaid
              so you can see how well the linear-variance model fits.

    Fits mirror exactly what LatentDataset / plot_tail_weighting compute:
        mu(E)    = polyfit(E, y, deg=1)
        var(E)   = clip(polyfit(E, (y - mu(E))^2, deg=1), 1e-6)
        sigma(E) = sqrt(var(E))

    Args:
        features_list: list of (N,) or (N,1) tensors, one per feature.
        incidence_energies: (N,) or (N,1) tensor of incidence energies.
        feature_names: optional display names.
        n_energy_bins: number of equal-width energy bins used for the empirical
                       std overlay in the right panel.
    """
    n_features = len(features_list)
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]

    E = incidence_energies.flatten().numpy().astype(np.float64)
    E_line = np.linspace(E.min(), E.max(), 300)

    figs = []
    for feat_tensor, fname in zip(features_list, feature_names):
        y = feat_tensor.squeeze().numpy().astype(np.float64)

        # Reproduce the training z-score fits
        coeffs_mu = np.polyfit(E, y, 1)
        mu_E = np.polyval(coeffs_mu, E)
        mu_line = np.polyval(coeffs_mu, E_line)

        residuals_sq = (y - mu_E) ** 2
        coeffs_var = np.polyfit(E, residuals_sq, 1)
        var_line = np.clip(np.polyval(coeffs_var, E_line), 1e-6, None)
        sig_line = np.sqrt(var_line)

        # Empirical per-bin std for comparison
        bin_edges = np.linspace(E.min(), E.max(), n_energy_bins + 1)
        bin_centers, emp_std = [], []
        for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
            mask = (E >= lo) & (E < hi)
            if mask.sum() >= 2:
                bin_centers.append(0.5 * (lo + hi))
                emp_std.append((y[mask] - np.polyval(coeffs_mu, 0.5 * (lo + hi))).std())
        bin_centers = np.array(bin_centers)
        emp_std = np.array(emp_std)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
        fig.suptitle(f"Energy-Dependent σ — {fname}", fontsize=13)

        # ── Left: raw data + mu ± k*sigma bands ────────────────────────
        ax1.scatter(E, y, s=1, alpha=0.15, color="steelblue",
                    rasterized=True, label="Data", zorder=1)
        ax1.plot(E_line, mu_line, color="black", lw=2, label=r"$\mu(E)$", zorder=3)
        for k, alpha in [(1, 0.25), (2, 0.15), (3, 0.08)]:
            ax1.fill_between(
                E_line,
                mu_line - k * sig_line,
                mu_line + k * sig_line,
                alpha=alpha, color="crimson",
                label=rf"$\pm {k}\sigma(E)$",
                zorder=2,
            )
        ax1.set_xlabel("Incidence Energy $E$")
        ax1.set_ylabel(fname)
        ax1.set_title(r"Feature vs Energy with $\mu \pm k\sigma$ bands")
        ax1.legend(fontsize=8, loc="best")
        ax1.grid(True, ls="--", alpha=0.3)

        # ── Right: sigma(E) model vs empirical binned std ───────────────
        ax2.plot(E_line, sig_line, color="crimson", lw=2.5,
                 label=r"$\sigma(E) = \sqrt{\hat{\mathrm{var}}(E)}$ (linear fit)")
        if len(bin_centers):
            ax2.scatter(bin_centers, emp_std, s=30, color="black", zorder=5,
                        label="Empirical std (binned, residuals)", edgecolors="none")
        ax2.set_xlabel("Incidence Energy $E$")
        ax2.set_ylabel(r"$\sigma(E)$")
        ax2.set_title(r"Energy-Dependent $\sigma(E)$: model vs empirical")
        ax2.legend(fontsize=9)
        ax2.grid(True, ls="--", alpha=0.3)
        ax2.set_ylim(bottom=0)

        slope_sig = 1000 * (sig_line[-1] - sig_line[0]) / (E_line[-1] - E_line[0])
        ax2.text(0.05, 0.93,
                 rf"$d\sigma/dE \approx {slope_sig:.4f}\ \mathrm{{GeV}}^{{-1}}$",
                 transform=ax2.transAxes, fontsize=10,
                 bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))

        plt.tight_layout()
        figs.append(fig)
        plt.show()

    return figs
