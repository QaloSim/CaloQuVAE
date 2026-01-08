from __future__ import annotations  # 1. Must be the very first line!

import matplotlib.pyplot as plt
import dwave_networkx as dnx
import numpy as np
import torch
from utils.HighLevelFeatsAtlasReg import HighLevelFeatures_ATLAS_regular
from utils.HighLevelFeatures import HighLevelFeatures
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import seaborn as sns
import dwave.embedding
from typing import TYPE_CHECKING
from utils.dwave.physics import rbm_to_expanded_ising
from utils.dwave.graphs import build_expanded_embedding

if TYPE_CHECKING:
    from utils.dwave.sampling_backend import ChainAnalysisResult


def visualize_embedding(sampler, graph, left_chains_dict, right_chains_dict, conditioning_sets, colors):
    """
    Draws the QPU graph with all embedded nodes colored using draw_zephyr_embedding.
    """
    print("\n--- 4. Generating Visualization ---")
    
    # 1. Build the 'emb' (embedding) dictionary
    #    This maps a unique logical node ID to its physical chain (or set)
    emb = {}
    
    # Add left chains: {'L_0': [q1, q2], 'L_1': [q3, q4], ...}
    for logical_node, chain in left_chains_dict.items():
        emb[f'L_{logical_node}'] = chain
        
    # Add right chains: {'R_0': [q5, q6], 'R_1': [q7, q8], ...}
    for logical_node, chain in right_chains_dict.items():
        emb[f'R_{logical_node}'] = chain
        
    # Add conditioning nodes: {'C_0': {q9, q10}, 'C_1': {q11, q12}, ...}
    for i, q_set in enumerate(conditioning_sets):
        emb[f'C_{i}'] = q_set # The function accepts iterables (sets are fine)

    # 2. Build the 'chain_color' dictionary
    #    This maps the same unique logical IDs to their colors
    chain_color = {}
    for logical_node in emb:
        if logical_node.startswith('L_'):
            chain_color[logical_node] = colors['left']
        elif logical_node.startswith('R_'):
            chain_color[logical_node] = colors['right']
        elif logical_node.startswith('C_'):
            chain_color[logical_node] = colors['cond']

    print("Embedding and color map created. Drawing plot (this may take a moment)...")

    # 3. Draw the graph using the correct function
    plt.figure(figsize=(18, 18))
    
    dnx.draw_zephyr_embedding(
        graph, 
        # sampler=sampler,
        emb=emb,
        chain_color=chain_color,
        unused_color=colors['unused_tuple'], # Must be an RGBA tuple
        node_size=10,
        show_labels=False, # This is the correct param instead of with_labels
        width=0.5 # Edge line width
        # REMOVED: edge_color=colors['edges'] -- This caused the TypeError
    )
    
    # 4. Create a legend
    # We still create the legend manually, but we get the counts
    # for a more informative label.
    left_qubits = set()
    for chain in left_chains_dict.values():
        left_qubits.update(chain)
        
    right_qubits = set()
    for chain in right_chains_dict.values():
        right_qubits.update(chain)
        
    conditioning_qubits = set()
    for q_set in conditioning_sets:
        conditioning_qubits.update(q_set)
        
    # We create dummy plots for the legend handles
    l_patch = plt.Line2D([0], [0], marker='o', color='w', label=f'Left Chains ({len(left_qubits)} qubits)',
                          markerfacecolor=colors['left'], markersize=10)
    r_patch = plt.Line2D([0], [0], marker='o', color='w', label=f'Right Chains ({len(right_qubits)} qubits)',
                          markerfacecolor=colors['right'], markersize=10)
    c_patch = plt.Line2D([0], [0], marker='o', color='w', label=f'Conditioning Nodes ({len(conditioning_qubits)} qubits)',
                          markerfacecolor=colors['cond'], markersize=10)
    u_patch = plt.Line2D([0], [0], marker='o', color='w', label='Unused Qubits',
                          markerfacecolor=colors['unused'], markersize=10) # Use hex for legend
    
    plt.legend(handles=[l_patch, r_patch, c_patch, u_patch], loc='upper right', fontsize=18)
    plt.title(f"Embedding on {sampler.solver.name}", fontsize=20)
    plt.show()

def plot_beta_optimization(
    beta_hist: list | np.ndarray, 
    rbm_e_hist: list | np.ndarray, 
    qpu_e_hist: list | np.ndarray, 
    figsize: tuple = (10, 10)
):
    """
    Plots the beta schedule optimization and energy convergence comparison.
    """
    
    # 1. Setup
    epochs = np.arange(len(beta_hist))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)
    
    # --- Plot 1: Beta Progression ---
    ax1.plot(epochs, beta_hist, marker='o', linestyle='-', color='purple', label=r'$\beta$ Value')
    ax1.set_ylabel(r'Inverse Temperature ($\beta$)', fontsize=12)
    ax1.set_title(r'Optimization of $\beta$ (RBM vs QPU)', fontsize=14)
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)
    ax1.legend()

    # Annotate the final value
    # Note: xytext is offset relative to the data point. You might need to adjust 
    # the offset if your beta values change scale drastically.
    final_beta = beta_hist[-1]
    ax1.annotate(f'Final $\\beta$: {final_beta:.4f}', 
                 xy=(epochs[-1], final_beta),           # The point to look at (data coords)
                 xytext=(0, 40),                        # The text position (0pts x, 40pts y from point)
                 textcoords='offset points',            # <--- CRITICAL FIX
                 ha='center',
                 arrowprops=dict(facecolor='black', shrink=0.05))

    # --- Plot 2: Energy Comparison ---
    ax2.plot(epochs, rbm_e_hist, label='RBM (Target) Energy', color='blue', linestyle='--', linewidth=2)
    ax2.plot(epochs, qpu_e_hist, label='QPU (Sampled) Energy', color='red', marker='x', linestyle='-')

    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Mean Joint Energy', fontsize=12)
    ax2.grid(True, which='both', linestyle='--', alpha=0.5)
    ax2.legend()

    # Highlight the convergence gap
    final_diff = abs(qpu_e_hist[-1] - rbm_e_hist[-1])
    ax2.set_title(f'Energy Matching (Final Diff: {final_diff:.4f})', fontsize=14)

    plt.tight_layout()
    plt.show()



def plot_energy_comparison(rbm_energies, qpu_energies, beta):
    """
    Plots aligned histograms of RBM and QPU energy distributions and 
    calculates the Chi-Squared statistic.
    """
    # Convert tensors to numpy arrays
    rbm_e = rbm_energies.detach().cpu().numpy().flatten()
    qpu_e = qpu_energies.detach().cpu().numpy().flatten()
    num_samples = min(len(rbm_e), len(qpu_e))
    
    plt.figure(figsize=(10, 6))
    
    # Determine aligned bins for both distributions
    min_val = min(rbm_e.min(), qpu_e.min())
    max_val = max(rbm_e.max(), qpu_e.max())
    bins = np.linspace(min_val, max_val, 30)
    
    # Calculate histograms manually first to get density for Chi-Squared
    # We use density=True to account for potentially different batch sizes
    hist_rbm, _ = np.histogram(rbm_e, bins=bins, density=True)
    hist_qpu, _ = np.histogram(qpu_e, bins=bins, density=True)
    
    # Calculate Chi-Squared
    # Adding epsilon to denominator to prevent division by zero
    epsilon = 1e-10
    chi_sq = np.sum(((hist_rbm - hist_qpu) ** 2) / (hist_rbm + epsilon))
    
    # Plotting
    plt.hist(rbm_e, bins=bins, alpha=0.6, label='RBM (Target)', color='blue', density=True)
    plt.hist(qpu_e, bins=bins, alpha=0.6, label=f'QPU (Beta={beta:.4f})', color='orange', density=True)
    
    plt.title(f"Energy Distribution Comparison\nBeta: {beta:.4f} | $\chi^2$: {chi_sq:.4f} | Samples: {num_samples}", fontsize=14)
    plt.xlabel("Energy")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Display the plot
    plt.show()


def plot_single_shower(
    shower_data, 
    target_energy, 
    epoch, 
    cfg, 
    save_dir=None
):
    """
    Visualizes a single pre-generated shower.
    """
    # 1. Setup HLF
    dataset_name = cfg.data.dataset_name.lower()
    if "atlas" in dataset_name:
        HLF = HighLevelFeatures_ATLAS_regular(
            particle=cfg.data.particle, 
            filename=cfg.data.binning_path, 
            relevantLayers=cfg.data.relevantLayers
        )
    else:
        HLF = HighLevelFeatures(
            particle=cfg.data.particle, 
            filename=cfg.data.binning_path, 
            relevantLayers=cfg.data.relevantLayers
        )

    # 2. Plot
    print(f"--- Generating Plot for {target_energy:.2f} MeV ---")
    qpu_path = None
    if save_dir:
        qpu_path = os.path.join(save_dir, f"val_qpu_epoch{epoch}.png")
    
    image_qpu = HLF.DrawSingleShower(
        shower_data, 
        title=f"QPU Generated (Epoch {epoch}) (Energy: {target_energy:.1f} MeV)", 
        filename=qpu_path, 
        cmap='rainbow'
    )
    
    return image_qpu

def plot_chain_break_distribution(result: ChainAnalysisResult, ax=None):
    """
    Plots a histogram/bar chart of how many chain breaks occur per sample.
    Useful for answering: "Is my solution mostly clean?"
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
        
    counts = result.breaks_per_sample
    max_breaks = int(np.max(counts)) if len(counts) > 0 else 0
    
    # Create bins for discrete integer values
    bins = np.arange(-0.5, max_breaks + 1.5, 1)
    
    # Plot
    n, _, patches = ax.hist(counts, bins=bins, color='#4C72B0', edgecolor='white', alpha=0.8)
    
    # Aesthetics
    ax.set_title("Distribution of Chain Breaks per Sample", fontsize=14)
    ax.set_xlabel("Number of Broken Chains", fontsize=12)
    ax.set_ylabel("Count of Samples", fontsize=12)
    ax.set_xticks(np.arange(0, max_breaks + 1, max(1, max_breaks//10)))
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Annotation for clean samples
    clean_count = np.sum(counts == 0)
    total_count = len(counts)
    clean_pct = (clean_count / total_count) * 100
    
    stats_text = (f"Total Samples: {total_count}\n"
                  f"Perfectly Clean: {clean_pct:.1f}%")
    
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    return ax

def plot_top_broken_chains(result: ChainAnalysisResult, top_n=10, ax=None):
    """
    Plots the Top N variables that suffer from chain breaks.
    Useful for answering: "Which specific qubits/chains are failing?"
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))
    
    # 1. Get data
    break_freqs = result.breaks_per_variable
    labels = np.array(result.variable_labels)
    
    # 2. Sort indices by frequency (descending)
    sorted_indices = np.argsort(break_freqs)[::-1]
    
    # 3. Slice Top N
    top_indices = sorted_indices[:top_n]
    top_freqs = break_freqs[top_indices]
    top_labels = labels[top_indices]
    
    # Filter out zero-break variables to keep plot clean
    mask = top_freqs > 0
    top_freqs = top_freqs[mask]
    top_labels = top_labels[mask]
    
    if len(top_freqs) == 0:
        ax.text(0.5, 0.5, "No Chain Breaks Found!", 
                ha='center', va='center', fontsize=14)
        return ax

    # 4. Plot Horizontal Bar Chart (easier to read labels)
    y_pos = np.arange(len(top_labels))
    ax.barh(y_pos, top_freqs, color='#C44E52', align='center', height=0.7)
    
    # Aesthetics
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_labels)
    ax.invert_yaxis()  # Highest value at top
    ax.set_xlabel("Total Breaks observed", fontsize=12)
    ax.set_title(f"Top {top_n} Most Fragile Chains", fontsize=14)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    
    # Add percentage labels to bars
    total_samples = len(result.breaks_per_sample)
    for i, v in enumerate(top_freqs):
        pct = (v / total_samples) * 100
        ax.text(v + (max(top_freqs)*0.01), i, f" {pct:.1f}%", va='center', fontsize=10, color='black')

    return ax


def plot_effective_fields(rbm, n_cond: int = 0, top_k: int = 20, weight_k: int = 5):
    """
    Plots three metrics for the top nodes in the RBM:
    1. Effective Field: |bias| + sum(|W|)
    2. Relative Field: sum(|W|) / |bias|
    3. Sparse Influence: sum(Top-K largest |W|) / |bias|
    
    Args:
        rbm: The RBM_TwoPartite object.
        n_cond: The number of conditioning visible nodes to exclude.
        top_k: The number of top nodes to display on the x-axis.
        weight_k: The 'k' used for the 3rd subplot (sum of top k weights).
    """
    # Create 3 subplots vertically
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15))
    fig.subplots_adjust(hspace=0.5) 

    # 1. Extract parameters
    w = rbm.params["weight_matrix"].detach().cpu().abs()
    v_bias = rbm.params["vbias"].detach().cpu().abs()
    h_bias = rbm.params["hbias"].detach().cpu().abs()
    
    epsilon = 1e-6 
    n_vis, n_hid = w.shape

    # 2. Calculate Metrics

    # --- Metric A: Effective Field (|b| + sum|W|) ---
    v_sum_w = torch.sum(w, dim=1)
    h_sum_w = torch.sum(w, dim=0)
    
    v_eff = v_bias + v_sum_w
    h_eff = h_bias + h_sum_w

    # --- Metric B: Relative Field (sum|W| / |b|) ---
    v_rel = v_sum_w / (v_bias + epsilon)
    h_rel = h_sum_w / (h_bias + epsilon)

    # --- Metric C: Top-K Weight Influence (sum(top_k|W|) / |b|) ---
    # Safe k: cannot take top 5 if dimension size is 3
    k_v = min(weight_k, n_hid) # For visible nodes, we look across hiddens (cols)
    k_h = min(weight_k, n_vis) # For hidden nodes, we look across visibles (rows)

    # Values for Visibles: Top k weights in each row
    v_topk_sum = torch.topk(w, k=k_v, dim=1).values.sum(dim=1)
    v_sparse = v_topk_sum / (v_bias + epsilon)

    # Values for Hiddens: Top k weights in each col
    h_topk_sum = torch.topk(w, k=k_h, dim=0).values.sum(dim=0)
    h_sparse = h_topk_sum / (h_bias + epsilon)

    # 3. Helper to build and sort data lists
    def get_sorted_data(v_metric, h_metric):
        data = []
        # Visible Nodes (skip conditioning)
        for i in range(n_cond, len(v_metric)):
            data.append({
                'label': f"v{i}", 
                'value': v_metric[i].item(), 
                'type': 'visible'
            })
        # Hidden Nodes
        for i in range(len(h_metric)):
            data.append({
                'label': f"h{i}", 
                'value': h_metric[i].item(), 
                'type': 'hidden'
            })
        
        # Sort descending
        data.sort(key=lambda x: x['value'], reverse=True)
        return data[:top_k]

    # Get data for all plots
    data_eff = get_sorted_data(v_eff, h_eff)
    data_rel = get_sorted_data(v_rel, h_rel)
    data_sparse = get_sorted_data(v_sparse, h_sparse)

    # 4. Plotting Helper
    def draw_subplot(ax, data, title, ylabel):
        if not data:
            ax.text(0.5, 0.5, "No nodes to plot", ha='center')
            return
            
        labels = [d['label'] for d in data]
        values = [d['value'] for d in data]
        colors = ['#C44E52' if d['type'] == 'hidden' else '#4C72B0' for d in data]

        x_pos = np.arange(len(labels))
        ax.bar(x_pos, values, color=colors)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Draw Plot 1: Effective Field
    draw_subplot(ax1, data_eff, 
                 f"Total Effective Field (|bias| + $\sum|W|$)", 
                 "Magnitude")

    # Draw Plot 2: Total Relative Influence
    draw_subplot(ax2, data_rel, 
                 f"Relative Influence ($\sum|W| / |bias|$)", 
                 "Ratio")

    # Draw Plot 3: Sparse Influence
    draw_subplot(ax3, data_sparse, 
                 f"Sparse Influence (Sum of Top {weight_k} Weights / |bias|)", 
                 f"Ratio (Top {weight_k} Weights / Bias)")
    
    # Shared Legend
    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='#4C72B0', lw=4),
                    Line2D([0], [0], color='#C44E52', lw=4)]
    fig.legend(custom_lines, ['Visible', 'Hidden'], loc='upper right', bbox_to_anchor=(0.95, 0.95))
    fig.suptitle(f"Top {top_k} Nodes by Different Field Strength Measures \n(Conditioning nodes 0-{n_cond} excluded)")

    return fig
    
    
def plot_physical_chain_integrity(node_label, result: ChainAnalysisResult, n_vis=None, ax=None):
    """
    Visualizes which SPECIFIC physical qubit in a chain is flipping against the group.
    
    Args:
        node_label: The label to plot (e.g., "h29", "v10", or integer 157).
        result: The analysis result object.
        n_vis: (Required if using string labels like 'h29') The number of visible units 
               used to decode the integer ID.
    """
    if ax is None: fig, ax = plt.subplots(figsize=(10, 5))
    
    # --- 1. Resolve the Key ---
    lookup_key = node_label
    
    # If the user asks for "h29" but the dict has integers, we must translate.
    if node_label not in result.embedding and isinstance(node_label, str) and n_vis is not None:
        if node_label.startswith('h'):
            # h29 -> 29 + n_vis
            idx = int(node_label[1:])
            lookup_key = idx + n_vis
        elif node_label.startswith('v'):
            # v10 -> 10 + n_cond (usually just 10, but depends on your indexing)
            # Assuming strictly v_i -> i here based on previous code
            idx = int(node_label[1:])
            lookup_key = idx
            
    # Fallback check
    if lookup_key not in result.embedding:
        print(f"Error: Could not find key '{node_label}' (or ID {lookup_key}) in embedding.")
        print(f"Available keys (first 5): {list(result.embedding.keys())[:5]}...")
        return

    # --- 2. Get the Chain ---
    chain_indices = result.embedding[lookup_key] 
    
    if len(chain_indices) < 2:
        print(f"Node {node_label} (ID {lookup_key}) is not a chain (len={len(chain_indices)})")
        return

    # --- 3. Extract Physical Data ---
    # Map physical label -> column index in result.physical_matrix
    # (Assuming physical_labels are strictly the column headers)
    p_map = {lbl: i for i, lbl in enumerate(result.physical_labels)}
    
    try:
        cols = [p_map[q] for q in chain_indices]
    except KeyError as e:
        print(f"Error: Physical qubit {e} from embedding not found in readout.")
        return

    # shape: (n_samples, chain_len)
    chain_data = result.physical_matrix[:, cols]
    
    # --- 4. Calculate "Disagreement Rate" ---
    # The majority vote for each sample
    majority_vote = np.sign(np.sum(chain_data, axis=1))
    majority_vote[majority_vote == 0] = 1 # Handle ties
    
    # Check disagreement: Does qubit i != majority?
    disagreements = (chain_data != majority_vote[:, None])
    
    # Rate of failure per qubit
    fail_rates = np.mean(disagreements, axis=0) * 100
    
    # --- 5. Plot ---
    x_pos = np.arange(len(chain_indices))
    bars = ax.bar(x_pos, fail_rates, color='#C44E52')
    
    ax.set_xticks(x_pos)
    ax.set_xticklabels(chain_indices, rotation=45)
    ax.set_xlabel("Physical Qubit Index")
    ax.set_ylabel("Disagreement Rate (%)")
    ax.set_title(f"Physical Failure Analysis for {node_label} (ID: {lookup_key})")
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Highlight the worst offender
    if len(fail_rates) > 0:
        worst_idx = np.argmax(fail_rates)
        bars[worst_idx].set_color('#8B0000') 
        ax.text(worst_idx, fail_rates[worst_idx] + 0.5, f"{fail_rates[worst_idx]:.1f}%", 
                ha='center', fontsize=10, color='black', fontweight='bold')
    
    return ax



def plot_experiment_energies(experiment_data: dict):
    """
    Plots Classical vs Clean QPU vs Dirty QPU energy distributions.
    
    Args:
        experiment_data: The dictionary returned by 'run_chain_break_experiment'
    """
    energy_val = experiment_data['incidence_energy']
    e_classical = experiment_data['classical']
    e_clean = experiment_data['clean']
    e_dirty = experiment_data['dirty']
    stats = experiment_data['stats']

    plt.figure(figsize=(12, 7))
    
    # --- 1. Plotting ---
    
    # Classical RBM: Black, strong outline, no fill (Reference)
    sns.histplot(
        e_classical, color="black", stat="density", kde=True,
        element="step", fill=False, linewidth=2.5, label="_nolegend_"
    )
    
    # Clean QPU: Green, filled, step
    if len(e_clean) > 0:
        sns.histplot(
            e_clean, color="green", stat="density", kde=True,
            element="step", alpha=0.25, linewidth=1.5, label="_nolegend_"
        )

    # Dirty QPU: Red, filled, step
    if len(e_dirty) > 0:
        sns.histplot(
            e_dirty, color="red", stat="density", kde=True,
            element="step", alpha=0.25, linewidth=1.5, label="_nolegend_"
        )

    # --- 2. Legend & Labels ---
    
    legend_handles = [
        mlines.Line2D([], [], color='black', linewidth=2.5, label='Classical RBM'),
        mpatches.Patch(color='green', alpha=0.25, label='Clean Samples (No Breaks)'),
        mpatches.Patch(color='red', alpha=0.25, label='Dirty Samples (Has Breaks)'),
    ]
    
    # Statistics Box
    stats_text = (
        f"Total QPU Samples: {stats['n_total']}\n"
        f"Clean: {stats['n_clean']} ({stats['pct_clean']:.1f}%)\n"
        f"Dirty: {stats['n_dirty']} ({100 - stats['pct_clean']:.1f}%)"
    )
    
    # Add text box to top-left or top-right depending on preference
    plt.gca().text(
        0.02, 0.95, stats_text, transform=plt.gca().transAxes,
        fontsize=11, verticalalignment='top', 
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    plt.title(f"Impact of Chain Breaks on Energy (Incidence Energy = {energy_val} MeV)", fontsize=14)
    plt.xlabel("Joint Energy", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.legend(handles=legend_handles, fontsize=12, loc='upper right')
    
    plt.tight_layout()
    plt.show()



def plot_chain_break_correlations(experiment_data: dict, n_clamped: int = 53, min_samples: int = 5, use_srt=True):
    """
    Plots 3 correlation matrices side-by-side with correct Latent Node Index labels.
    """
    # Unpack metadata
    energy = experiment_data['incidence_energy']
    stats = experiment_data['stats']
    
    # Create Figure with sharey=True to reduce clutter on y-axis labels
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
    
    # Title
    title = f"Visible Unit Correlation Matrices (E = {energy} MeV)"
    if use_srt:
        title += " | SRT Applied"

    fig.suptitle(title, fontsize=16)
    
    # Track the last valid image for the shared colorbar
    last_im = None

    def _plot_single_matrix(ax, samples, title, subtext):
        nonlocal last_im
        if len(samples) < min_samples:
            ax.text(0.5, 0.5, f"Insufficient Data\n(n={len(samples)})", 
                    ha='center', va='center', fontsize=12)
            ax.axis('off')
            return

        # 1. Calculate Correlation
        if isinstance(samples, torch.Tensor):
            samples = samples.float().cpu()
        
        # Check for NaNs
        if torch.isnan(samples).any():
             samples = torch.nan_to_num(samples)
             
        corr = torch.corrcoef(samples.T).numpy()
        
        # 2. Crop Clamped Bits
        corr = corr[n_clamped:, n_clamped:]
        
        # 3. Zero Diagonal & Handle NaNs
        np.fill_diagonal(corr, 0)
        corr = np.nan_to_num(corr, nan=0.0)
        
        # 4. Determine Axis Extents (for labels)
        n_gen = corr.shape[0]
        n_total = n_clamped + n_gen
        # extent = [left, right, bottom, top]
        # We use origin='lower' so index n_clamped starts at the bottom-left corner
        extent = [n_clamped, n_total, n_clamped, n_total]

        # 5. Plot
        # origin='lower' places the [0,0] index at the bottom-left, matching your 
        # previous invert_yaxis preference.
        last_im = ax.imshow(corr, cmap='seismic', vmin=-1, vmax=1, 
                            interpolation="none", origin='lower', extent=extent)
        
        ax.set_title(f"{title}\n{subtext}", fontsize=12)
        ax.set_xlabel("Latent Node Index", fontsize=10)
        
        # Only set Y label for the first plot since we share Y axes
        if ax.get_subplotspec().is_first_col():
            ax.set_ylabel("Latent Node Index", fontsize=10)

    # --- Plot 1: Classical Baseline ---
    _plot_single_matrix(
        axes[0], 
        experiment_data['classical_samples'], 
        "Classical RBM Baseline",
        f"(n={len(experiment_data['classical_samples'])})"
    )

    # --- Plot 2: QPU Clean ---
    _plot_single_matrix(
        axes[1], 
        experiment_data['clean_samples'], 
        "QPU Clean Chains",
        f"(n={stats['n_clean']} | {stats['pct_clean']:.1f}%)"
    )

    # --- Plot 3: QPU Dirty ---
    _plot_single_matrix(
        axes[2], 
        experiment_data['dirty_samples'], 
        "QPU Broken Chains",
        f"(n={stats['n_dirty']} | {100 - stats['pct_clean']:.1f}%)"
    )

    # --- Shared Colorbar ---
    if last_im:
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7]) 
        fig.colorbar(last_im, cax=cbar_ax)

    plt.show()

def plot_magnetization_diagnostics(experiment_data, n_clamped=53, use_srt=True):
    """
    Plots the Average Magnetization <sigma_z> for each latent node.
    Compares Classical Baseline vs QPU Clean vs QPU Broken.
    """
    energy = experiment_data['incidence_energy']
    
    # 1. Extract Samples
    # Ensure everything is on CPU and float for calculation
    samples_cl = experiment_data['classical_samples'].float().cpu()
    samples_clean = experiment_data['clean_samples'].float().cpu()
    samples_dirty = experiment_data['dirty_samples'].float().cpu()

    # 2. Compute Magnetization (Mean across batch dim=0)
    # We slice [n_clamped:] immediately to ignore the clamped visible units
    mag_cl = samples_cl.mean(dim=0)[n_clamped:].numpy()
    mag_clean = samples_clean.mean(dim=0)[n_clamped:].numpy()
    mag_dirty = samples_dirty.mean(dim=0)[n_clamped:].numpy()
    
    # Create X-axis indices (shifted to match your plots)
    indices = np.arange(n_clamped, n_clamped + len(mag_cl))
    
    # 3. Plotting
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    suffix = " | SRT Applied" if use_srt else ""
    fig.suptitle(f"Node Diagnostic: Magnetization Profiles (E = {energy} MeV){suffix}", fontsize=16)

    # --- Subplot 1: The Raw Profiles ---
    ax = axes[0]
    ax.plot(indices, mag_cl, label='Classical Baseline', color='black', linestyle='--', alpha=0.7)
    ax.plot(indices, mag_clean, label='QPU Clean Chains', color='#1f77b4', linewidth=2)
    ax.plot(indices, mag_dirty, label='QPU Broken Chains', color='#d62728', alpha=0.6)
    
    
    ax.set_ylabel(r"Average Magnetization $\langle \sigma_z \rangle$", fontsize=12)
    ax.set_ylim(0.0, 1.1)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # --- Subplot 2: Deviation from Baseline (The "Error Signal") ---
    # This shows purely the *error* introduced by the QPU
    ax = axes[1]
    
    error_clean = mag_clean - mag_cl
    error_dirty = mag_dirty - mag_cl
    
    ax.bar(indices, error_clean, color='#1f77b4', alpha=0.6, label='Error (Clean Chains)')
    # We plot dirty errors as a line/scatter to not clutter the bars
    ax.plot(indices, error_dirty, color='#d62728', linestyle=':', alpha=0.8, label='Error (Broken Chains)')

    ax.set_ylabel(r"Bias Error $\langle \sigma_{QPU} \rangle - \langle \sigma_{RBM} \rangle$", fontsize=12)
    ax.set_xlabel("Latent Node Index", fontsize=12)
    ax.set_ylim(-0.4, 0.6)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title("Magnitude of Bias Shift")

    plt.tight_layout()
    plt.show()

def plot_expanded_j_distribution(
    rbm, 
    raw_sampler, 
    conditioning_sets, 
    left_chains, 
    right_chains, 
    hidden_side='right',
    beta=1.0, 
    chain_strength=None, 
    rho=28.0,
    save_path=None
):
    """
    Plots the distribution of Physical J values for the expanded embedding.
    Matches the logic of 'sample_expanded_flux_conditioned_rigorous' to ensure
    the plot reflects the actual hardware submission.
    """
    
    # --- 1. Setup Dimensions & Sides (Identical to Sampling Function) ---
    n_vis = rbm.params["vbias"].shape[0]
    
    if hidden_side == 'right':
        visible_side = 'left' 
    elif hidden_side == 'left':
        visible_side = 'right'
    else:
        raise ValueError("hidden_side must be 'left' or 'right'")

    # --- 2. Build Embedding Internally ---
    exp_embedding, fragment_map = build_expanded_embedding(
        conditioning_sets, 
        left_chains, 
        right_chains, 
        num_visible=n_vis, 
        hidden_side=hidden_side
    )

    # --- 3. Get Logical Ising ---
    h_exp, J_exp = rbm_to_expanded_ising(
        rbm, fragment_map, exp_embedding, raw_sampler.adjacency, beta
    )

    # --- 4. Handle Dynamic Chain Strength ---
    # This duplicates the logic in your sampler to ensure the plot is accurate
    if chain_strength is None:
        calc_strength = calculate_rms_chain_strength(J_exp, rho=rho)
        max_j = raw_sampler.properties.get('extended_j_range', [None, 2.0])[1]
        chain_strength = min(calc_strength, max_j)
        print(f"[Plot Debug] Calculated Dynamic Chain Strength: {chain_strength:.4f} (rho={rho})")
    else:
        print(f"[Plot Debug] Using Manual Chain Strength: {chain_strength}")

    # --- 5. Embed to Physical (What the QPU sees) ---
    target_adj = raw_sampler.adjacency
    # We pass h_exp even though we only care about J, because embed_ising requires it
    _, J_phys = dwave.embedding.embed_ising(
        h_exp, J_exp, exp_embedding, target_adj, chain_strength=chain_strength
    )

    # --- 6. Clamp to Hardware Limits ---
    # You specified limits of +/- 1.0. 
    # The sampler might support extended ranges, but we clamp to standard Ising limits here.
    j_min, j_max = -1.0, 1.0
    
    j_values_raw = np.array(list(J_phys.values()))
    j_values_clamped = np.clip(j_values_raw, j_min, j_max)

    # Separate Chains vs Logical for visualization
    # Heuristic: Chains are usually set exactly to -chain_strength
    is_chain = np.isclose(j_values_raw, -chain_strength, atol=1e-4)
    logical_couplings = j_values_clamped[~is_chain]
    chain_couplings = j_values_clamped[is_chain]

    # --- 7. Plotting ---
    plt.figure(figsize=(12, 6))
    
    # Histogram for Logical Couplings
    plt.hist(logical_couplings, bins=60, alpha=0.7, color='#4c72b0', 
             label=f'Logical Interactions (Beta={beta})', edgecolor='k', linewidth=0.5)
    
    # Histogram for Chain Couplings
    if len(chain_couplings) > 0:
        plt.hist(chain_couplings, bins=10, alpha=0.6, color='#dd8452', 
                 label=f'Chain Couplings (Str={chain_strength:.2f})', edgecolor='k', linewidth=0.5)

    # Hardware Limits
    plt.axvline(x=j_min, color='r', linestyle='--', linewidth=2, label='Hardware Limit (-1.0)')
    plt.axvline(x=j_max, color='r', linestyle='--', linewidth=2, label='Hardware Limit (+1.0)')
    
    # Zero line
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)

    # Stats Titles
    n_clipped = np.sum(j_values_raw < j_min) + np.sum(j_values_raw > j_max)
    mean_abs_j = np.mean(np.abs(logical_couplings))
    
    plt.title(f"Physical J Distribution | Beta: {beta} | Chain Str: {chain_strength:.2f}\n"
              f"Mean Abs Logical J: {mean_abs_j:.3f} | Total Clipped: {n_clipped}")
    plt.xlabel("Physical J Strength")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Plot saved to {save_path}")
    
    plt.show()