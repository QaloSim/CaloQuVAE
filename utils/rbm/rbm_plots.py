import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter
import itertools
from typing import Tuple, List, Optional
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

