from __future__ import annotations  # 1. Must be the very first line!

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
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
from utils.dwave.graphs import build_expanded_embedding, EmbeddingPoint

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

import matplotlib.pyplot as plt
import dwave_networkx as dnx
import networkx as nx

def visualize_embedding_poster(sampler, graph, left_chains_dict, right_chains_dict, conditioning_sets):
    print("\n--- Generating Poster Visualization (Cropped) ---")
    
    # --- 0. DETERMINE TOPOLOGY ---
    # Cast sampler to string to safely check its name whether it's an object or string
    is_zephyr = "Advantage2" in str(sampler)

    # --- 1. CONFIGURATION ---
    c_palette = {
        'left': '#D55E00',      # Vermilion
        'right': '#009E73',     # Bluish Green
        'cond': '#56B4E9',      # Sky Blue
        'unused_rgba': (0.8, 0.8, 0.8, 0.1) # Ghost the background
    }

    # --- 2. DATA PREP ---
    emb = {}
    l_qubits, r_qubits, c_qubits = set(), set(), set()

    for node, chain in left_chains_dict.items():
        emb[f'L_{node}'] = chain
        l_qubits.update(chain)
        
    for node, chain in right_chains_dict.items():
        emb[f'R_{node}'] = chain
        r_qubits.update(chain)
        
    for i, q_set in enumerate(conditioning_sets):
        emb[f'C_{i}'] = q_set
        c_qubits.update(q_set)

    chain_color = {}
    for logical_node in emb:
        if logical_node.startswith('L_'):
            chain_color[logical_node] = c_palette['left']
        elif logical_node.startswith('R_'):
            chain_color[logical_node] = c_palette['right']
        elif logical_node.startswith('C_'):
            chain_color[logical_node] = c_palette['cond']

    # --- 3. PLOTTING & CROPPING ---
    fig, ax = plt.subplots(figsize=(12, 12), dpi=300) 
    
    # Dynamically select layout and draw function based on the sampler
    if is_zephyr:
        pos = dnx.zephyr_layout(graph)
        dnx.draw_zephyr_embedding(
            graph, 
            emb=emb,
            chain_color=chain_color,
            unused_color=c_palette['unused_rgba'], 
            node_size=20,       
            show_labels=False,
            width=0.2,
            ax=ax
        )
    else:
        pos = dnx.pegasus_layout(graph)
        dnx.draw_pegasus_embedding(
            graph, 
            emb=emb,
            chain_color=chain_color,
            unused_color=c_palette['unused_rgba'], 
            node_size=20,       
            show_labels=False,
            width=0.2,
            ax=ax
        )

    # --- CALCULATE BOUNDING BOX ---
    # Gather all active physical qubits
    all_active_qubits = l_qubits | r_qubits | c_qubits
    
    # Extract their x and y coordinates
    xs = [pos[q][0] for q in all_active_qubits]
    ys = [pos[q][1] for q in all_active_qubits]
    
    # Add a small margin so nodes aren't cut off at the edge
    margin = 0.1
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)

    # --- 4. LEGEND ---
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', 
                   label=f'Left Chains ({len(l_qubits)} q)',
                   markerfacecolor=c_palette['left'], markersize=18),
        plt.Line2D([0], [0], marker='o', color='w', 
                   label=f'Right Chains ({len(r_qubits)} q)',
                   markerfacecolor=c_palette['right'], markersize=18),
        plt.Line2D([0], [0], marker='o', color='w', 
                   label=f'Conditioning ({len(c_qubits)} q)',
                   markerfacecolor=c_palette['cond'], markersize=18),
    ]
    
    ax.legend(handles=legend_elements, 
              loc='upper right', 
              fontsize=24,           
              frameon=True, 
              framealpha=1.0, 
              edgecolor='black')

    plt.axis('off')
    plt.tight_layout()
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

def plot_hamming_energies(results: dict):
    energies = list(results.keys())
    e1, e2 = energies[0], energies[1]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    fig.suptitle("Impact of Hamming Cliff on Energy Distributions (Raw QPU)", fontsize=16)

    def _plot_hist(ax, energy_key):
        data = results[energy_key]
        
        # Classical (Black line)
        sns.histplot(
            data['classical_energies'], color="black", stat="density", kde=True,
            element="step", fill=False, linewidth=2.5, ax=ax, label="_nolegend_"
        )
        
        # QPU (Red fill)
        sns.histplot(
            data['qpu_energies'], color="firebrick", stat="density", kde=True,
            element="step", alpha=0.3, linewidth=1.5, ax=ax, label="_nolegend_"
        )
        
        ax.set_title(f"Incidence Energy = {energy_key} MeV", fontsize=14)
        ax.set_xlabel("Joint Energy", fontsize=12)

    # Plot both sides
    _plot_hist(axes[0], e1)
    _plot_hist(axes[1], e2)
    
    axes[0].set_ylabel("Density", fontsize=12)

    # Legend
    legend_handles = [
        mlines.Line2D([], [], color='black', linewidth=2.5, label='Classical RBM'),
        mpatches.Patch(color='firebrick', alpha=0.3, label='QPU (Raw)'),
    ]
    fig.legend(handles=legend_handles, loc='upper right', bbox_to_anchor=(0.95, 0.95), fontsize=12)
    
    plt.tight_layout()
    plt.show()


def plot_hamming_correlations(results: dict, n_cond: int = 53):
    energies = list(results.keys())
    e1, e2 = energies[0], energies[1]
    
    # 3 Rows (E1, E2, Difference), 2 Cols (Solvers)
    # Increased height to accommodate the 3rd row
    fig, axes = plt.subplots(3, 2, figsize=(14, 16), sharex=True, sharey=True)
    
    # --- Headers & Labels ---
    
    # Col Labels (Solver)
    axes[0, 0].set_title("Classical RBM", fontsize=14, fontweight='bold')
    axes[0, 1].set_title("QPU (Raw)", fontsize=14, fontweight='bold')
    
    # Row Labels (Energy / Diff)
    axes[0, 0].set_ylabel(f"E = {e1}\nLatent Index", fontsize=14, fontweight='bold')
    axes[1, 0].set_ylabel(f"E = {e2}\nLatent Index", fontsize=14, fontweight='bold')
    axes[2, 0].set_ylabel(f"Diff (E{e2} - E{e1})\nLatent Index", fontsize=14, fontweight='bold')

    # --- Helper Functions ---

    def _get_corr_matrix(samples):
        """ Computes and processes correlation matrix without plotting. """
        if isinstance(samples, torch.Tensor):
            samples = samples.float().cpu()
            
        # Compute Correlation
        corr = torch.corrcoef(samples.T).numpy()
        
        # Slice off clamped bits
        corr = corr[n_cond:, n_cond:]
        
        # Zero diagonal & fix NaNs
        np.fill_diagonal(corr, 0)
        return np.nan_to_num(corr, nan=0.0)

    def _plot_matrix(ax, data):
        """ Plots the pre-computed matrix. """
        # Using vmin=-1, vmax=1 ensures the difference plot uses the same scale
        # (i.e. if the difference is small, the map will look faint/white, which is correct)
        im = ax.imshow(data, cmap='seismic', vmin=-1, vmax=1, origin='lower')
        return im

    # --- 1. Compute Matrices ---
    
    # Classical
    c_e1 = _get_corr_matrix(results[e1]['classical_samples'])
    c_e2 = _get_corr_matrix(results[e2]['classical_samples'])
    c_diff = c_e2 - c_e1  # Difference
    
    # QPU
    q_e1 = _get_corr_matrix(results[e1]['qpu_samples'])
    q_e2 = _get_corr_matrix(results[e2]['qpu_samples'])
    q_diff = q_e2 - q_e1  # Difference

    # --- 2. Plotting ---

    # Row 1: Energy 1
    im1 = _plot_matrix(axes[0, 0], c_e1)
    im2 = _plot_matrix(axes[0, 1], q_e1)
    
    # Row 2: Energy 2
    im3 = _plot_matrix(axes[1, 0], c_e2)
    im4 = _plot_matrix(axes[1, 1], q_e2)
    
    # Row 3: Difference
    im5 = _plot_matrix(axes[2, 0], c_diff)
    im6 = _plot_matrix(axes[2, 1], q_diff)

    # --- 3. Formatting ---

    # Axis Labels
    for ax in axes.flat:
        ax.set_xlabel("Latent Node Index")

    # Shared Colorbar
    fig.subplots_adjust(right=0.9)
    # Adjusted position for the taller figure
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7]) 
    fig.colorbar(im1, cax=cbar_ax, label="Pearson Correlation")

    fig.suptitle(f"Correlations & Drift: Hamming Cliff ({e1} vs {e2})", fontsize=16)
    plt.show()


def plot_hamming_magnetization_diagnostics(results: dict, n_clamped=53):
    """
    Plots the Magnetization <sigma_z> diagnostics.
    
    Top Plot: 
      - Compares RBM vs QPU profiles using a "Dashed (E1) vs Solid (E2)" logic.
      - Uses High-Contrast colors (Greyscale for RBM, Blue/Orange for QPU).
      
    Bottom Plot:
      - RBM Diff (Bar): The "Required Jump" (RBM E2 - RBM E1).
      - QPU Diff (Line): The "Actual Jump" (QPU E2 - QPU E1).
    """
    energies = list(results.keys())
    e1, e2 = energies[0], energies[1]
    
    # --- 1. Extract & Compute Means ---
    def get_mag(samples):
        # Convert to float cpu, take mean across batch (dim 0), slice off clamped
        return samples.float().cpu().mean(dim=0)[n_clamped:].numpy()

    # Classical
    mag_rbm1 = get_mag(results[e1]['classical_samples'])
    mag_rbm2 = get_mag(results[e2]['classical_samples'])
    
    # QPU
    mag_qpu1 = get_mag(results[e1]['qpu_samples'])
    mag_qpu2 = get_mag(results[e2]['qpu_samples'])

    # X-axis indices
    indices = np.arange(n_clamped, n_clamped + len(mag_rbm1))

    # --- 2. Setup Plot ---
    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
    fig.suptitle(f"Hamming Cliff Diagnostics: {e1} vs {e2}", fontsize=16)

    # --- Subplot 1: Absolute Magnetization Profiles ---
    # Visual Logic: 
    #   - Greyscale = RBM (Ground Truth)
    #   - Color = QPU (Experiment)
    #   - Dashed = Energy 1 (Pre-Cliff)
    #   - Solid = Energy 2 (Post-Cliff)
    
    ax = axes[0]
    
    # RBM Baselines
    ax.plot(indices, mag_rbm1, color='gray', linestyle='--', alpha=0.6, linewidth=1.5, label=f'RBM E={e1} (Anchor)')
    ax.plot(indices, mag_rbm2, color='black', linestyle='-', linewidth=2, label=f'RBM E={e2} (Target)')
    
    # QPU Experiment (Blue for E1, Orange for E2 - Colorblind friendly contrast)
    ax.plot(indices, mag_qpu1, color='#648FFF', linestyle='--', linewidth=2, label=f'QPU E={e1}') # Soft Blue
    ax.plot(indices, mag_qpu2, color='#DC267F', linestyle='-', linewidth=2, alpha=0.9, label=f'QPU E={e2}') # Magenta/Pink
    
    ax.set_ylabel(r"Magnetization $\langle \sigma_z \rangle$", fontsize=12)
    ax.set_title("Absolute Profiles (Dashed=Pre-Cliff, Solid=Post-Cliff)", fontsize=12)
    ax.legend(loc='upper right', ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)


    # --- Subplot 2: The "Jump" Comparison ---
    # Question: Did the QPU jump (Red Line) match the RBM jump (Grey Bars)?
    
    ax = axes[1]
    
    # 1. The Ideal Jump (RBM E2 - RBM E1)
    # We plot this as bars to serve as the "background truth"
    diff_rbm = mag_rbm2 - mag_rbm1
    ax.bar(indices, diff_rbm, color='black', alpha=0.2, label=f'Required Jump (RBM Diff)')
    
    # 2. The Actual Jump (QPU E2 - QPU E1)
    # We plot this as a bright line
    diff_qpu = mag_qpu2 - mag_qpu1
    ax.plot(indices, diff_qpu, color='#DC267F', linewidth=2.5, marker='o', markersize=3, label=f'Actual Jump (QPU Diff)')

    ax.set_ylabel(r"$\Delta \langle \sigma_z \rangle$ (Post - Pre)", fontsize=12)
    ax.set_xlabel("Latent Node Index", fontsize=12)
    ax.set_title("Tunneling Success: Does the Line (QPU) follow the Bars (RBM)?", fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Zero line
    ax.axhline(0, color='black', linewidth=1, alpha=0.5)

    plt.tight_layout()
    plt.show()

def plot_correlation_comparison(experiment_result: dict, n_cond: int = 53):
    """
    Plots a 1x3 grid comparing Classical vs QPU correlations and their difference.
    
    Args:
        experiment_result: The dictionary output from 'run_spin_gauge_experiment_discretized'
        n_cond: Number of conditioning bits to slice off (to focus on latent correlations)
    """
    
    # --- 1. Data Preparation ---
    incidence_energy = experiment_result.get("incidence_energy", "Unknown")
    n_steps = experiment_result.get("stats", {}).get("n_quantization_steps", "Unknown")

    # Classical Samples
    c_samples = experiment_result['classical_samples']

    # QPU Samples (Combine Clean + Dirty)
    # We check for existence just in case one set is empty
    q_parts = []
    if experiment_result['clean_samples'] is not None:
        q_parts.append(experiment_result['clean_samples'])
    if experiment_result['dirty_samples'] is not None:
        q_parts.append(experiment_result['dirty_samples'])
    
    if q_parts:
        q_samples = torch.cat(q_parts, dim=0)
    else:
        # Fallback if no samples exist
        q_samples = torch.zeros((1, c_samples.shape[1]))

    # --- 2. Helper Function ---
    def _compute_corr_matrix(samples):
        """ Computes Pearson correlation, slices n_cond, zeroes diagonal. """
        if isinstance(samples, torch.Tensor):
            samples = samples.float().cpu()
        
        # Edge case: If batch size is 0 or 1, correlation is undefined/NaN
        if samples.shape[0] < 2:
            return np.zeros((samples.shape[1] - n_cond, samples.shape[1] - n_cond))

        # Compute Correlation (Rows=Variables, so we transpose)
        corr = torch.corrcoef(samples.T).numpy()
        
        # Slice off conditioning bits (focus on latent)
        corr = corr[n_cond:, n_cond:]
        
        # Zero diagonal (auto-correlation is always 1, distracting in plots)
        np.fill_diagonal(corr, 0)
        
        # Handle NaNs (e.g., constant columns have 0 std dev -> NaN correlation)
        return np.nan_to_num(corr, nan=0.0)

    # --- 3. Compute Matrices ---
    mat_c = _compute_corr_matrix(c_samples)
    mat_q = _compute_corr_matrix(q_samples)
    mat_diff = mat_q - mat_c  # (QPU - Classical)

    # --- 4. Plotting ---
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    
    # Shared plotting helper
    def _plot_heatmap(ax, data, title):
        im = ax.imshow(data, cmap='seismic', vmin=-1, vmax=1, origin='lower')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Latent Node Index")
        ax.set_ylabel("Latent Node Index")
        return im

    # Plot 1: Classical
    im1 = _plot_heatmap(axes[0], mat_c, "Classical RBM (Baseline)")
    
    # Plot 2: QPU (Aggregate)
    im2 = _plot_heatmap(axes[1], mat_q, "QPU (Clean + Dirty)")
    
    # Plot 3: Difference
    im3 = _plot_heatmap(axes[2], mat_diff, "Difference (QPU - Classical)")

    # --- 5. Formatting ---
    
    # Main Title
    fig.suptitle(f"Correlation Analysis: E={incidence_energy} | Quantization Steps={n_steps}", fontsize=16, y=0.98)

    # Shared Colorbar
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7]) 
    cbar = fig.colorbar(im1, cax=cbar_ax)
    cbar.set_label("Pearson Correlation", fontsize=12)

    plt.show()




def plot_calibration_history(history, save_path=None):
    """
    Generates diagnostic plots matching Figure 6 of the D-Wave Shimming Tutorial.
    """
    shims = np.array(history['shims'])           # [iter, n_qubits]
    mags = np.array(history['magnetizations'])   # [iter, n_qubits]
    stds = np.array(history['std_dev'])          # [iter]
    iterations = len(history['rmse'])
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=False)
    
    # --- Plot 1: Flux Bias Offsets Evolution ---
    # Shows the "random walk" or convergence of the shims
    axes[0].plot(shims)
    axes[0].set_ylabel(r"Flux-bias offsets ($\Phi_i / \Phi_0$)")
    axes[0].set_xlabel("Iteration")
    axes[0].set_title("Evolution of Flux-Bias Offsets")
    # Add safe limit lines for reference
    axes[0].axhline(y=0, color='black', linestyle='--', linewidth=0.5)

    # --- Plot 2: Magnetization Distributions (First 5 vs Last 5) ---
    # Compares the spread of biases before and after calibration
    first_5_mags = mags[:5].flatten()
    last_5_mags = mags[-5:].flatten()
    
    axes[1].hist(first_5_mags, bins=50, alpha=0.5, label='First 5 Iterations', density=True, color='skyblue')
    axes[1].hist(last_5_mags, bins=50, alpha=0.5, label='Last 5 Iterations', density=True, color='orange')
    axes[1].set_xlabel(r"Magnetization $\langle s_i \rangle$")
    axes[1].set_ylabel("Prob. Density")
    axes[1].set_title("Magnetization Distribution (Before vs After)")
    axes[1].legend()
    axes[1].set_xlim(-0.75, 0.75)

    # --- Plot 3: Standard Deviation of Magnetizations ---
    # Measures the "tightness" of the distribution over time.
    axes[2].plot(stds, color='steelblue')
    axes[2].set_ylabel(r"$\sigma$ of qubit magnetizations")
    axes[2].set_xlabel("Iteration")
    axes[2].set_title("Standard Deviation of Magnetizations over Time")
    axes[2].grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
    
    plt.show()




def plot_shim_verification(experiment_data, n_cond=53):
    """
    Plots a 2x3 matrix comparison:
    Row 1: Classical, No Shims, Shimmed (Correlations)
    Row 2: Empty, Error (No Shims), Error (Shimmed)
    """
    
    # 1. Unpack Samples
    s_classical = experiment_data["classical_samples"]
    s_no_shim = experiment_data["no_shim_samples"]
    s_shimmed = experiment_data["shimmed_samples"]
    energy = experiment_data["incidence_energy"]

    # 2. Helper to get Latent Correlations
    def get_latent_corr(samples):
        if isinstance(samples, torch.Tensor):
            samples = samples.float().cpu()
        
        # Calculate full correlation
        corr = torch.corrcoef(samples.T).numpy()
        
        # Slice off conditioning units (rows and cols 0 to n_cond)
        latent_corr = corr[n_cond:, n_cond:]
        
        # Zero diagonal for better contrast
        np.fill_diagonal(latent_corr, 0)
        return np.nan_to_num(latent_corr, nan=0.0)

    # 3. Compute Matrices
    mat_classical = get_latent_corr(s_classical)
    mat_no_shim = get_latent_corr(s_no_shim)
    mat_shimmed = get_latent_corr(s_shimmed)
    
    # Compute Differences (Error Maps)
    mat_diff_shim = mat_shimmed - mat_classical
    mat_diff_no_shim = mat_no_shim - mat_classical

    # 4. Plotting Setup (2 rows, 3 columns)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Helper to plot individual matrices
    def plot_mat(ax, data, title, is_diff=False):
        cmap = 'seismic' if not is_diff else 'bwr'
        # Differences often have smaller ranges, but keeping fixed scales helps comparison
        vmin, vmax = (-1, 1) if not is_diff else (-0.5, 0.5)
        
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
        return im

    # --- ROW 1: Correlations ---
    # 1. Classical (Top Left)
    plot_mat(axes[0, 0], mat_classical, "Classical RBM (Target)")

    # 2. No Shim (Top Center)
    plot_mat(axes[0, 1], mat_no_shim, "QPU Raw (No Shims)")

    # 3. Shimmed (Top Right)
    im_main = plot_mat(axes[0, 2], mat_shimmed, "QPU Calibrated (With Shims)")

    # --- ROW 2: Errors ---
    # 4. Empty Slot (Bottom Left) -> Hide this axis
    axes[1, 0].axis('off')

    # 5. Error No Shim (Bottom Center)
    # We use is_diff=True here
    plot_mat(axes[1, 1], mat_diff_no_shim, "Error: No Shim - Classical", is_diff=True)

    # 6. Error Shimmed (Bottom Right)
    im_diff = plot_mat(axes[1, 2], mat_diff_shim, "Error: Shimmed - Classical", is_diff=True)

    # Titles and Layout
    fig.suptitle(f"Flux Shim Verification: {energy} MeV", fontsize=18, y=0.96)
    
    # --- Colorbars ---
    # Since we have 3 columns, we position colorbars on the far right
    
    # Colorbar for Row 1 (Correlations)
    cbar_ax = fig.add_axes([0.92, 0.53, 0.015, 0.35]) # [left, bottom, width, height]
    fig.colorbar(im_main, cax=cbar_ax, label="Pearson Correlation")
    
    # Colorbar for Row 2 (Errors)
    cbar_diff_ax = fig.add_axes([0.92, 0.11, 0.015, 0.35])
    fig.colorbar(im_diff, cax=cbar_diff_ax, label="Correlation Error (Delta)")

    # Adjust spacing to prevent overlap with titles
    plt.subplots_adjust(wspace=0.1, hspace=0.2, right=0.9)
    plt.show()

    # Quantitative Metric
    err_no_shim = np.linalg.norm(mat_diff_no_shim)
    err_shimmed = np.linalg.norm(mat_diff_shim)
    
    print("\n--- Quantitative Improvement (Matrix Norm Distance from Classical) ---")
    print(f"Error (No Shim): {err_no_shim:.4f}")
    print(f"Error (Shimmed): {err_shimmed:.4f}")
    print(f"Improvement:     {((err_no_shim - err_shimmed)/err_no_shim)*100:.2f}%")




def plot_orbit_verification(experiment_data, n_cond=53):
    """
    Plots a 2x3 matrix comparison for Orbit Rotation:
    Row 1: Classical, No Orbits, With Orbits (Correlations)
    Row 2: Empty, Error (No Orbits), Error (With Orbits)
    """
    
    # 1. Unpack Samples
    s_classical = experiment_data["classical_samples"]
    s_no_orbit = experiment_data["no_orbit_samples"]
    s_with_orbit = experiment_data["with_orbit_samples"]
    energy = experiment_data["incidence_energy"]
    n_orbits = experiment_data.get("num_orbits_used", "?")

    # 2. Helper to get Latent Correlations
    def get_latent_corr(samples):
        if isinstance(samples, torch.Tensor):
            samples = samples.float().cpu()
        
        # Calculate full correlation
        corr = torch.corrcoef(samples.T).numpy()
        
        # Slice off conditioning units (rows and cols 0 to n_cond)
        latent_corr = corr[n_cond:, n_cond:]
        
        # Zero diagonal for better contrast
        np.fill_diagonal(latent_corr, 0)
        return np.nan_to_num(latent_corr, nan=0.0)

    # 3. Compute Matrices
    mat_classical = get_latent_corr(s_classical)
    mat_no_orbit = get_latent_corr(s_no_orbit)
    mat_with_orbit = get_latent_corr(s_with_orbit)
    
    # Compute Differences (Error Maps)
    mat_diff_with_orbit = mat_with_orbit - mat_classical
    mat_diff_no_orbit = mat_no_orbit - mat_classical

    # 4. Plotting Setup (2 rows, 3 columns)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Helper to plot individual matrices
    def plot_mat(ax, data, title, is_diff=False):
        cmap = 'seismic' if not is_diff else 'bwr'
        # Differences often have smaller ranges, but keeping fixed scales helps comparison
        vmin, vmax = (-1, 1) if not is_diff else (-0.5, 0.5)
        
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')
        return im

    # --- ROW 1: Correlations ---
    # 1. Classical (Top Left)
    plot_mat(axes[0, 0], mat_classical, "Classical RBM (Target)")

    # 2. No Orbit (Top Center)
    plot_mat(axes[0, 1], mat_no_orbit, "QPU Static (No Orbits)")

    # 3. With Orbit (Top Right)
    im_main = plot_mat(axes[0, 2], mat_with_orbit, f"QPU Rotated ({n_orbits} Orbits)")

    # --- ROW 2: Errors ---
    # 4. Empty Slot (Bottom Left) -> Hide this axis
    axes[1, 0].axis('off')

    # 5. Error No Orbit (Bottom Center)
    plot_mat(axes[1, 1], mat_diff_no_orbit, "Error: No Orbits - Classical", is_diff=True)

    # 6. Error With Orbit (Bottom Right)
    im_diff = plot_mat(axes[1, 2], mat_diff_with_orbit, "Error: With Orbits - Classical", is_diff=True)

    # Titles and Layout
    fig.suptitle(f"Orbit Rotation Verification: {energy} MeV", fontsize=18, y=0.96)
    
    # --- Colorbars ---
    # Position colorbars on the far right
    
    # Colorbar for Row 1 (Correlations)
    cbar_ax = fig.add_axes([0.92, 0.53, 0.015, 0.35]) # [left, bottom, width, height]
    fig.colorbar(im_main, cax=cbar_ax, label="Pearson Correlation")
    
    # Colorbar for Row 2 (Errors)
    cbar_diff_ax = fig.add_axes([0.92, 0.11, 0.015, 0.35])
    fig.colorbar(im_diff, cax=cbar_diff_ax, label="Correlation Error (Delta)")

    # Adjust spacing to prevent overlap with titles
    plt.subplots_adjust(wspace=0.1, hspace=0.2, right=0.9)
    plt.show()

    # Quantitative Metric
    err_no_orbit = np.linalg.norm(mat_diff_no_orbit)
    err_with_orbit = np.linalg.norm(mat_diff_with_orbit)
    
    print("\n--- Quantitative Improvement (Matrix Norm Distance from Classical) ---")
    print(f"Error (No Orbit):   {err_no_orbit:.4f}")
    print(f"Error (With Orbit): {err_with_orbit:.4f}")
    improvement = ((err_no_orbit - err_with_orbit)/err_no_orbit)*100
    print(f"Improvement:        {improvement:.2f}%")



def plot_orbit_sweep_analysis(sweep_results):
    """
    Visualizes the results of the Orbit Sweep.
    1. Bar chart of Error vs Shift.
    2. Matrix comparison: Classical vs Best Orbit vs Worst Orbit.
    """
    metrics = sweep_results["orbit_metrics"]
    mat_classical = sweep_results["classical_matrix"]
    best = sweep_results["best_orbit"]
    worst = sweep_results["worst_orbit"]
    
    # Extract data for bar chart
    shifts = [m['shift'] for m in metrics]
    errors = [m['error_norm'] for m in metrics]
    
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 4) # 2 rows, 4 cols

    # --- 1. Error Landscape (Top Row, spanning 2 cols) ---
    ax_bar = fig.add_subplot(gs[0, 1:3])
    bars = ax_bar.bar(shifts, errors, color='skyblue', width=3)
    
    # Highlight Best and Worst
    best_idx = shifts.index(best['shift'])
    worst_idx = shifts.index(worst['shift'])
    bars[best_idx].set_color('forestgreen')
    bars[worst_idx].set_color('firebrick')
    
    ax_bar.set_xlabel("Orbit Shift Amount")
    ax_bar.set_ylabel("Matrix Error Norm (Lower is Better)")
    ax_bar.set_title("Orbit Quality Spectrum")
    ax_bar.grid(axis='y', alpha=0.3)
    
    # --- 2. Matrix Visualization (Bottom Row) ---
    # Helper
    def plot_mat(ax, data, title, diff=False):
        cmap = 'seismic' if not diff else 'bwr'
        vmin, vmax = (-1, 1) if not diff else (-0.5, 0.5)
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
        ax.set_title(title, fontweight='bold')
        ax.axis('off')
        return im

    # Classical
    ax_clas = fig.add_subplot(gs[1, 0])
    plot_mat(ax_clas, mat_classical, "Target (Classical)")
    
    # Best Orbit
    ax_best = fig.add_subplot(gs[1, 1])
    plot_mat(ax_best, best['matrix'], f"Best Orbit (Shift {best['shift']})\nError: {best['error_norm']:.2f}")

    # Worst Orbit
    ax_worst = fig.add_subplot(gs[1, 2])
    plot_mat(ax_worst, worst['matrix'], f"Worst Orbit (Shift {worst['shift']})\nError: {worst['error_norm']:.2f}")
    
    # Difference (Best - Worst) -> Shows what artifacts the bad orbit introduces
    ax_diff = fig.add_subplot(gs[1, 3])
    diff_mat = worst['matrix'] - best['matrix']
    im_diff = plot_mat(ax_diff, diff_mat, "Diff: Worst - Best", diff=True)
    
    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.01, 0.3])
    fig.colorbar(im_diff, cax=cbar_ax, label="Correlation Delta")

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()


def plot_permutation_sweep_analysis(sweep_results):
    """
    Visualizes Monte Carlo Permutation Sweep.

    When ``sweep_results`` contains an ``"aggregated_orbit"`` key (produced by
    the upgraded run_monte_carlo_permutation_sweep / run_mc_permutation_sweep_single),
    a 4-column layout is used that adds the orbit-aggregation strategy as a
    direct competitor alongside Default and Best:

    Row 0: [Error Hist] | [Chain Break Hist] | [Scatter: Error vs Breaks] | [Bar: Strategy Comparison]
    Row 1: Classical    | Default            | Best                        | Aggregated (all orbits)
    Row 2: [Empty]      | Diff (Default-Cl)  | Diff (Best-Cl)              | Diff (Aggregated-Cl)

    If ``"aggregated_orbit"`` is absent (legacy results), the original 3-column
    layout is used with Default and Best only.
    """
    metrics = sweep_results["perm_metrics"]
    mat_classical = sweep_results["classical_matrix"]

    default_run = sweep_results["default_orbit"]
    best_run = sweep_results["best_orbit"]
    agg_run = sweep_results.get("aggregated_orbit")  # None for legacy results

    all_errors = [m['error_norm'] for m in metrics]
    all_breaks = [m['chain_break_frac'] for m in metrics]

    has_agg = agg_run is not None
    n_cols = 4 if has_agg else 3
    fig_width = 24 if has_agg else 18

    fig = plt.figure(figsize=(fig_width, 15))
    gs = fig.add_gridspec(3, n_cols, height_ratios=[0.25, 0.4, 0.4])

    # --- TOP ROW: STATS ---

    # 1. Error Histogram
    ax_hist_err = fig.add_subplot(gs[0, 0])
    ax_hist_err.hist(all_errors, bins=15, color='lightgray', edgecolor='white', alpha=0.8)
    ax_hist_err.axvline(default_run['error_norm'], color='firebrick', ls='--', lw=2, label='Default')
    ax_hist_err.axvline(best_run['error_norm'], color='forestgreen', ls='--', lw=2, label='Best')
    if has_agg:
        ax_hist_err.axvline(agg_run['error_norm'], color='mediumpurple', ls='--', lw=2, label='Aggregated')
    ax_hist_err.set_title("Distribution of Error Norms", fontweight='bold')
    ax_hist_err.set_xlabel("Euclidean Error")
    ax_hist_err.legend()

    # 2. Chain Break Histogram
    ax_hist_brk = fig.add_subplot(gs[0, 1])
    ax_hist_brk.hist(all_breaks, bins=15, color='peachpuff', edgecolor='white', alpha=0.8)
    ax_hist_brk.axvline(default_run['chain_break_frac'], color='firebrick', ls='--', lw=2, label='Default')
    ax_hist_brk.axvline(best_run['chain_break_frac'], color='forestgreen', ls='--', lw=2, label='Best')
    if has_agg:
        ax_hist_brk.axvline(agg_run['break_frac'], color='mediumpurple', ls='--', lw=2, label='Aggregated')
    ax_hist_brk.set_title("Distribution of Chain Breaks", fontweight='bold')
    ax_hist_brk.set_xlabel("Fraction of Broken Chains")

    # 3. Scatter: Error vs Breaks
    ax_scatter = fig.add_subplot(gs[0, 2])
    ax_scatter.scatter(all_errors, all_breaks, alpha=0.6, c='gray')
    ax_scatter.scatter(default_run['error_norm'], default_run['chain_break_frac'],
                       c='firebrick', s=100, marker='*', label='Default')
    ax_scatter.scatter(best_run['error_norm'], best_run['chain_break_frac'],
                       c='forestgreen', s=100, marker='*', label='Best')
    if has_agg:
        ax_scatter.scatter(agg_run['error_norm'], agg_run['break_frac'],
                           c='mediumpurple', s=100, marker='*', label='Aggregated')
    ax_scatter.set_title("Correlation: Error vs Stability", fontweight='bold')
    ax_scatter.set_xlabel("Error Norm")
    ax_scatter.set_ylabel("Chain Break Fraction")
    ax_scatter.legend()

    # 4. Bar: Strategy Comparison (only when aggregated is available)
    if has_agg:
        ax_bar = fig.add_subplot(gs[0, 3])
        labels = ['Default', 'Best', 'Aggregated']
        values = [default_run['error_norm'], best_run['error_norm'], agg_run['error_norm']]
        colors = ['firebrick', 'forestgreen', 'mediumpurple']
        bars = ax_bar.bar(labels, values, color=colors, alpha=0.8, width=0.5)
        ax_bar.set_title("Strategy Comparison", fontweight='bold')
        ax_bar.set_ylabel("Error Norm vs Classical")
        ax_bar.set_ylim(0, max(values) * 1.25)
        for bar, v in zip(bars, values):
            ax_bar.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                        f"{v:.3f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax_bar.spines['top'].set_visible(False)
        ax_bar.spines['right'].set_visible(False)

    # --- MATRIX HELPER ---
    def plot_mat(ax, data, title, diff=False):
        cmap = 'seismic' if not diff else 'bwr'
        vmin, vmax = (-1, 1) if not diff else (-0.5, 0.5)
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axis('off')
        return im

    # -- ROW 1: Absolute Matrices --
    ax_cl = fig.add_subplot(gs[1, 0])
    plot_mat(ax_cl, mat_classical, "Target (Classical RBM)")

    ax_def = fig.add_subplot(gs[1, 1])
    plot_mat(ax_def, default_run['matrix'], f"Default Embedding\nErr: {default_run['error_norm']:.3f}")

    ax_best = fig.add_subplot(gs[1, 2])
    im_main = plot_mat(ax_best, best_run['matrix'], f"Best Permutation\nErr: {best_run['error_norm']:.3f}")

    if has_agg:
        ax_agg = fig.add_subplot(gs[1, 3])
        plot_mat(ax_agg, agg_run['matrix'], f"Aggregated (All Orbits)\nErr: {agg_run['error_norm']:.3f}")

    # -- ROW 2: Differences --
    ax_empty = fig.add_subplot(gs[2, 0])
    ax_empty.axis('off')

    diff_default = default_run['matrix'] - mat_classical
    ax_diff_def = fig.add_subplot(gs[2, 1])
    plot_mat(ax_diff_def, diff_default, "Error: Default - Classical", diff=True)

    diff_best = best_run['matrix'] - mat_classical
    ax_diff_best = fig.add_subplot(gs[2, 2])
    im_diff = plot_mat(ax_diff_best, diff_best, "Error: Best - Classical", diff=True)

    if has_agg:
        diff_agg = agg_run['matrix'] - mat_classical
        ax_diff_agg = fig.add_subplot(gs[2, 3])
        plot_mat(ax_diff_agg, diff_agg, "Error: Aggregated - Classical", diff=True)

    # --- Colorbars ---
    cbar_ax_main = fig.add_axes([0.92, 0.45, 0.015, 0.25])
    fig.colorbar(im_main, cax=cbar_ax_main, label="Pearson Correlation")

    cbar_ax_diff = fig.add_axes([0.92, 0.12, 0.015, 0.25])
    fig.colorbar(im_diff, cax=cbar_ax_diff, label="Correlation Delta")

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()


def plot_ga_sweep_analysis(ga_results):
    """
    Summarises the output of run_ga_permutation_sweep_multi_energy.

    Layout (4 rows × 3 cols)
    ------------------------
    Row 0 [stats]:
        Left   – GA convergence curve (best / mean / median per generation)
        Centre – Scatter of all evaluated individuals (error vs chain-break-frac),
                 coloured by generation
        Right  – Per-energy error bars: identity orbit vs best orbit

    Row 1 [absolute matrices, averaged over energies]:
        Classical target | Default (identity) orbit | Best orbit

    Row 2 [difference matrices, averaged over energies]:
        (empty)          | Default − Classical      | Best − Classical
    """
    perm_metrics      = ga_results["perm_metrics"]
    ga_history        = ga_results["ga_history"]
    classical_matrices = ga_results["classical_matrices"]
    default_orbit     = ga_results["default_orbit"]
    best_orbit        = ga_results["best_orbit"]
    config            = ga_results.get("config", {})

    energies = list(classical_matrices.keys())

    # ---- averaged matrices across energies ----
    mat_cl_avg = np.mean([classical_matrices[e] for e in energies], axis=0)

    def _avg_qpu_mat(orbit_record):
        mats = orbit_record.get("matrices", {})
        if mats:
            return np.mean([mats[e] for e in energies if e in mats], axis=0)
        return np.zeros_like(mat_cl_avg)

    mat_def_avg  = _avg_qpu_mat(default_orbit)
    mat_best_avg = _avg_qpu_mat(best_orbit)

    # ---- per-generation stats ----
    gens    = [h["generation"] for h in ga_history]
    g_best  = [h["best"]       for h in ga_history]
    g_mean  = [h["mean"]       for h in ga_history]
    g_med   = [h["median"]     for h in ga_history]

    # ---- all-individual arrays ----
    all_errors  = [m["error_norm"]        for m in perm_metrics]
    all_breaks  = [m["chain_break_frac"]  for m in perm_metrics]
    all_gens    = [m["generation"]        for m in perm_metrics]

    n_gen = max(all_gens) + 1 if all_gens else 1

    # ------------------------------------------------------------------ figure
    fig = plt.figure(figsize=(20, 17))
    gs  = fig.add_gridspec(3, 3, height_ratios=[0.22, 0.39, 0.39],
                           hspace=0.28, wspace=0.32)

    # ---- colour palette for generation scatter ----
    cmap_gen = plt.cm.viridis
    gen_norm = plt.Normalize(vmin=0, vmax=max(n_gen - 1, 1))

    # ================================================================ ROW 0
    # --- (0,0) convergence curve ---
    ax_conv = fig.add_subplot(gs[0, 0])
    ax_conv.plot(gens, g_best,  color='forestgreen', lw=2,   label='Best',   marker='o', ms=4)
    ax_conv.plot(gens, g_mean,  color='steelblue',   lw=1.5, label='Mean',   marker='s', ms=3, ls='--')
    ax_conv.plot(gens, g_med,   color='darkorange',  lw=1.5, label='Median', marker='^', ms=3, ls=':')
    if default_orbit is not None:
        ax_conv.axhline(default_orbit["error_norm"], color='firebrick', ls='--',
                        lw=1.2, label='Identity baseline', alpha=0.7)
    ax_conv.set_title("GA Convergence", fontweight='bold')
    ax_conv.set_xlabel("Generation")
    ax_conv.set_ylabel("Frobenius Error")
    ax_conv.legend(fontsize=7)
    pop  = config.get("population_size", "?")
    xop  = config.get("crossover_op", "?")
    mop  = config.get("mutation_op",  "?")
    ax_conv.text(0.98, 0.97,
                 f"pop={pop}, xover={xop}, mut={mop}",
                 transform=ax_conv.transAxes, fontsize=7,
                 ha='right', va='top', color='gray')

    # --- (0,1) scatter: all individuals coloured by generation ---
    ax_sc = fig.add_subplot(gs[0, 1])
    sc = ax_sc.scatter(all_errors, all_breaks,
                       c=all_gens, cmap=cmap_gen, norm=gen_norm,
                       alpha=0.6, s=20, zorder=2)
    if default_orbit is not None:
        ax_sc.scatter(default_orbit["error_norm"], default_orbit["chain_break_frac"],
                      c='firebrick', s=120, marker='*', zorder=5, label='Identity')
    ax_sc.scatter(best_orbit["error_norm"], best_orbit["chain_break_frac"],
                  c='gold',     s=120, marker='*', zorder=5, label='Best',
                  edgecolors='k', linewidths=0.5)
    cb_sc = fig.colorbar(sc, ax=ax_sc, pad=0.02)
    cb_sc.set_label("Generation", fontsize=8)
    ax_sc.set_title("Population Evolution", fontweight='bold')
    ax_sc.set_xlabel("Error Norm")
    ax_sc.set_ylabel("Chain Break Fraction")
    ax_sc.legend(fontsize=7)

    # --- (0,2) per-energy bar chart: identity vs best ---
    ax_bar = fig.add_subplot(gs[0, 2])
    x      = np.arange(len(energies))
    w      = 0.35

    def_per_e  = [default_orbit["per_energy"].get(e, np.nan) for e in energies] \
                 if default_orbit and default_orbit.get("per_energy") else [np.nan] * len(energies)
    best_per_e = [best_orbit["per_energy"].get(e, np.nan) for e in energies] \
                 if best_orbit.get("per_energy") else [np.nan] * len(energies)

    bars_def  = ax_bar.bar(x - w / 2, def_per_e,  w, label='Identity',
                           color='firebrick', alpha=0.7)
    bars_best = ax_bar.bar(x + w / 2, best_per_e, w, label='Best orbit',
                           color='forestgreen', alpha=0.7)

    for bar, v in zip(list(bars_def) + list(bars_best),
                      def_per_e + best_per_e):
        if not np.isnan(v):
            ax_bar.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() * 1.02,
                        f"{v:.3f}", ha='center', va='bottom', fontsize=6)

    ax_bar.set_title("Per-Energy Error: Identity vs Best", fontweight='bold')
    ax_bar.set_ylabel("Frobenius Error")
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([f"{e}" for e in energies], fontsize=8)
    ax_bar.set_xlabel("Incidence Energy")
    ax_bar.legend(fontsize=7)

    # ================================================================ ROW 1 — absolute matrices
    def _plot_mat(ax, data, title, diff=False):
        cmap = 'seismic' if not diff else 'bwr'
        vmin, vmax = (-1, 1) if not diff else (-0.5, 0.5)
        im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')
        return im

    n_e_label = f" (avg {len(energies)} E)" if len(energies) > 1 else f" (E={energies[0]})"

    ax_cl   = fig.add_subplot(gs[1, 0])
    _plot_mat(ax_cl,   mat_cl_avg,   "Target (Classical RBM)" + n_e_label)

    ax_def  = fig.add_subplot(gs[1, 1])
    _plot_mat(ax_def,  mat_def_avg,
              f"Identity Orbit{n_e_label}\nErr: {default_orbit['error_norm']:.3f}" if default_orbit
              else f"Identity Orbit{n_e_label}")

    ax_best = fig.add_subplot(gs[1, 2])
    im_main = _plot_mat(ax_best, mat_best_avg,
                        f"Best Orbit{n_e_label}\nErr: {best_orbit['error_norm']:.3f}")

    # ================================================================ ROW 2 — diff matrices
    ax_empty = fig.add_subplot(gs[2, 0])
    ax_empty.axis('off')
    # summary text in the empty panel
    if default_orbit is not None:
        delta_abs = default_orbit["error_norm"] - best_orbit["error_norm"]
        delta_pct = 100.0 * delta_abs / (default_orbit["error_norm"] + 1e-12)
        n_unique  = ga_history[-1]["cache_size"] if ga_history else "?"
        summary = (
            f"GA Summary\n"
            f"{'─'*22}\n"
            f"Generations : {len(ga_history)}\n"
            f"Population  : {config.get('population_size', '?')}\n"
            f"Unique orbits: {n_unique}\n\n"
            f"Identity err : {default_orbit['error_norm']:.4f}\n"
            f"Best err     : {best_orbit['error_norm']:.4f}\n"
            f"Improvement  : {delta_abs:.4f} ({delta_pct:.1f}%)\n\n"
            f"Crossover : {config.get('crossover_op','?')}\n"
            f"Mutation  : {config.get('mutation_op','?')}\n"
        )
        ax_empty.text(0.05, 0.95, summary, transform=ax_empty.transAxes,
                      fontsize=9, va='top', family='monospace',
                      bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    diff_def  = mat_def_avg  - mat_cl_avg
    ax_dd = fig.add_subplot(gs[2, 1])
    _plot_mat(ax_dd,  diff_def,  "Identity − Classical" + n_e_label, diff=True)

    diff_best = mat_best_avg - mat_cl_avg
    ax_db = fig.add_subplot(gs[2, 2])
    im_diff = _plot_mat(ax_db, diff_best, "Best − Classical" + n_e_label, diff=True)

    # --- colorbars ---
    cbar_ax_main = fig.add_axes([0.93, 0.37, 0.014, 0.25])
    fig.colorbar(im_main, cax=cbar_ax_main, label="Pearson Correlation")

    cbar_ax_diff = fig.add_axes([0.93, 0.07, 0.014, 0.25])
    fig.colorbar(im_diff, cax=cbar_ax_diff, label="Correlation Delta")

    plt.suptitle("Genetic Algorithm Orbit Search — Results", fontsize=14, fontweight='bold', y=1.01)
    plt.tight_layout(rect=[0, 0, 0.92, 1])
    plt.show()


def plot_srt_comparison(comparison_results):
    """
    Visualises Physical (inter-chain) vs Logical (intra-chain) SRT comparison.

    Row 0: Error norm bars | Chain break bars | Summary text
    Row 1: Classical matrix | Physical SRT matrix | Logical SRT matrix
    Row 2: [empty]          | Diff: Phys - Cl     | Diff: Logical - Cl
    Row 3: Classical mag    | ΔMag: Phys - Cl     | ΔMag: Logical - Cl
    """
    mat_cl = comparison_results["classical_matrix"]
    phys   = comparison_results["physical_srt"]
    log    = comparison_results["logical_srt"]
    mag_cl   = comparison_results.get("classical_magnetization")

    fig = plt.figure(figsize=(16, 19))
    gs  = fig.add_gridspec(4, 3, height_ratios=[0.15, 0.3, 0.3, 0.25], hspace=0.45, wspace=0.3)

    bar_labels = ["Physical SRT\n(Intra-Chain)", "Logical SRT\n(Inter-Chain)"]
    bar_colors = ["steelblue", "darkorange"]

    # --- Row 0: Summary Stats ---
    # Error norm
    ax_err   = fig.add_subplot(gs[0, 0])
    vals_err = [phys["error_norm"], log["error_norm"]]
    bars     = ax_err.bar(bar_labels, vals_err, color=bar_colors, alpha=0.8, width=0.4)
    ax_err.set_title("Error Norm vs Classical", fontweight='bold')
    ax_err.set_ylabel("||QPU – Classical||")
    ax_err.set_ylim(0, max(vals_err) * 1.2)
    for bar, v in zip(bars, vals_err):
        ax_err.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                    f"{v:.3f}", ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Chain break fraction
    ax_brk   = fig.add_subplot(gs[0, 1])
    vals_brk = [phys["break_frac"], log["break_frac"]]
    bars2    = ax_brk.bar(bar_labels, vals_brk, color=bar_colors, alpha=0.8, width=0.4)
    ax_brk.set_title("Chain Break Fraction", fontweight='bold')
    ax_brk.set_ylabel("Fraction of Broken Chains")
    ax_brk.set_ylim(0, max(vals_brk) * 1.35 + 1e-4)
    for bar, v in zip(bars2, vals_brk):
        ax_brk.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                    f"{v:.1%}", ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Summary text
    ax_txt    = fig.add_subplot(gs[0, 2])
    ax_txt.axis('off')
    delta_err = phys["error_norm"] - log["error_norm"]
    winner    = "Logical" if delta_err > 0 else "Physical"
    summary   = (
        f"Physical SRT:\n"
        f"  Error : {phys['error_norm']:.4f}\n"
        f"  Breaks: {phys['break_frac']:.2%}\n\n"
        f"Logical SRT:\n"
        f"  Error : {log['error_norm']:.4f}\n"
        f"  Breaks: {log['break_frac']:.2%}\n\n"
        f"Δ Error: {abs(delta_err):.4f}\n"
        f"Winner: {winner} SRT"
    )
    ax_txt.text(0.5, 0.5, summary, transform=ax_txt.transAxes, fontsize=11,
                verticalalignment='center', horizontalalignment='center', fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow', alpha=0.8))

    # --- Matrix helper ---
    def plot_mat(ax, data, title, diff=False):
        vmin, vmax = (-0.5, 0.5) if diff else (-1, 1)
        im = ax.imshow(data, cmap='seismic', vmin=vmin, vmax=vmax, origin='lower')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axis('off')
        return im

    # --- Row 1: Absolute correlation matrices ---
    ax_cl = fig.add_subplot(gs[1, 0])
    plot_mat(ax_cl, mat_cl, "Target (Classical RBM)")

    ax_phys = fig.add_subplot(gs[1, 1])
    plot_mat(ax_phys, phys["matrix"],
             f"Physical (Inter-Chain) SRT\nErr: {phys['error_norm']:.3f}")

    ax_log  = fig.add_subplot(gs[1, 2])
    im_main = plot_mat(ax_log, log["matrix"],
                       f"Logical (Intra-Chain) SRT\nErr: {log['error_norm']:.3f}")

    # --- Row 2: Difference matrices ---
    ax_empty = fig.add_subplot(gs[2, 0])
    ax_empty.axis('off')

    ax_diff_phys = fig.add_subplot(gs[2, 1])
    plot_mat(ax_diff_phys, phys["matrix"] - mat_cl, "Error: Physical – Classical", diff=True)

    ax_diff_log = fig.add_subplot(gs[2, 2])
    im_diff = plot_mat(ax_diff_log, log["matrix"] - mat_cl, "Error: Logical – Classical", diff=True)

    # --- Row 3: Magnetization bar charts ---
    def _style_mag_ax(ax, n_nodes, ylabel):
        ax.set_xlabel("Latent Node Index", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlim(-1, n_nodes)
        ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    if mag_cl is not None and "magnetization" in phys and "magnetization" in log:
        n_nodes = len(mag_cl)
        x = np.arange(n_nodes)

        # col 0: Classical baseline (absolute magnetization)
        ax_mag_cl = fig.add_subplot(gs[3, 0])
        ax_mag_cl.bar(x, mag_cl, color='dimgray', alpha=0.75, linewidth=0)
        ax_mag_cl.set_title("Classical Magnetization\n(Target)", fontsize=11, fontweight='bold')
        ax_mag_cl.set_ylim(0, 1)
        _style_mag_ax(ax_mag_cl, n_nodes, r"$\langle\sigma\rangle$")

        # cols 1–2: QPU deltas (diverging, colour-coded by sign)
        def plot_mag_delta(ax, mag_qpu, title):
            delta = mag_qpu - mag_cl
            colors = ['steelblue' if d >= 0 else 'firebrick' for d in delta]
            ax.bar(x, delta, color=colors, alpha=0.85, linewidth=0)
            ax.axhline(0, color='black', linewidth=1.0, linestyle='--', zorder=3)
            ax.set_title(title, fontsize=11, fontweight='bold')
            lim = max(abs(delta).max() * 1.15, 0.05)
            ax.set_ylim(-lim, lim)
            _style_mag_ax(ax, n_nodes, r"$\Delta\langle\sigma\rangle$ (QPU – Classical)")

        ax_mag_phys = fig.add_subplot(gs[3, 1])
        plot_mag_delta(ax_mag_phys, phys["magnetization"], "ΔMag: Physical – Classical")

        ax_mag_log = fig.add_subplot(gs[3, 2])
        plot_mag_delta(ax_mag_log, log["magnetization"], "ΔMag: Logical – Classical")
    else:
        for col in [0, 1, 2]:
            ax = fig.add_subplot(gs[3, col])
            ax.text(0.5, 0.5, "Magnetization data\nnot available",
                    ha='center', va='center', transform=ax.transAxes, fontsize=10, color='gray')
            ax.axis('off')

    # Colorbars
    cbar_main = fig.add_axes([0.92, 0.50, 0.015, 0.20])
    fig.colorbar(im_main, cax=cbar_main, label="Pearson Correlation")

    cbar_diff = fig.add_axes([0.92, 0.28, 0.015, 0.20])
    fig.colorbar(im_diff, cax=cbar_diff, label="Correlation Delta")

    fig.suptitle("SRT Comparison: Physical (Inter-Chain) vs Logical (Intra-Chain)",
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 0.9, 0.97])
    plt.show()


def plot_orbit_sensitivity_analysis(data):
    """
    Produces the final 4x3 Figure + Summary Stats described.
    
    Layout:
    Row 0: Histograms/Scatter Stats
    Row 1: Spacer
    Row 2: Classical Baselines
    Row 3: QPU Default Orbit
    Row 4: QPU Best Orbit
    Row 5: Error (Best Orbit - Classical)
    """
    
    orbits = data["orbits"]
    baselines = data["baselines"]
    best_idx = data["best_orbit_index"]
    
    default_orb = orbits[0]
    best_orb = orbits[best_idx]
    
    # Extract lists for plotting stats
    modes = ['normal', 'ferro', 'anti']
    colors = {'normal': 'black', 'ferro': 'firebrick', 'anti': 'royalblue'}
    labels = {'normal': 'Normal', 'ferro': 'Ferro (+)', 'anti': 'Anti-Ferro (-)'}
    
    stats = {m: {'errors': [], 'breaks': []} for m in modes}
    
    for o in orbits:
        for m in modes:
            stats[m]['errors'].append(o["modes"][m]["error"])
            stats[m]['breaks'].append(o["modes"][m]["breaks"])

    # --- Setup Figure ---
    fig = plt.figure(figsize=(20, 25)) # Increased height slightly for the extra row
    
    # FIX: Increase nrows to 6 and add a height ratio for the 4th matrix row
    # Ratios: [Stats, Spacer, Mat1, Mat2, Mat3, Mat4]
    gs = GridSpec(6, 3, figure=fig, height_ratios=[0.2, 0.02, 0.19, 0.19, 0.19, 0.19])
    
    # === ROW 0: SUMMARY STATISTICS ===
    
    # 1. Histogram of Errors
    ax_hist_err = fig.add_subplot(gs[0, 0])
    for m in modes:
        ax_hist_err.hist(stats[m]['errors'], bins=15, alpha=0.5, color=colors[m], label=labels[m])
    ax_hist_err.set_title("Distribution of Error Norms", fontweight='bold')
    ax_hist_err.set_xlabel("Euclidean Error vs Classical Baseline")
    ax_hist_err.legend()
    
    # 2. Histogram of Chain Breaks
    ax_hist_brk = fig.add_subplot(gs[0, 1])
    for m in modes:
        ax_hist_brk.hist(stats[m]['breaks'], bins=15, alpha=0.5, color=colors[m], label=labels[m])
    ax_hist_brk.set_title("Distribution of Chain Breaks", fontweight='bold')
    ax_hist_brk.set_xlabel("Chain Break Fraction")
    
    # 3. Scatter Plot
    ax_scat = fig.add_subplot(gs[0, 2])
    for m in modes:
        ax_scat.scatter(stats[m]['errors'], stats[m]['breaks'], c=colors[m], alpha=0.6, label=labels[m])
        
        # Highlight Best Orbit
        ax_scat.scatter(best_orb["modes"][m]["error"], best_orb["modes"][m]["breaks"], 
                        facecolors='none', edgecolors=colors[m], s=150, linewidth=2, marker='s')

    ax_scat.set_title("Error vs Stability (Square = Best Orbit)", fontweight='bold')
    ax_scat.set_xlabel("Error Norm")
    ax_scat.set_ylabel("Break Fraction")
    
    # === MATRIX GRID ===
    
    def plot_mat(ax, mat, title, is_diff=False):
        cmap = 'seismic'
        if is_diff:
            vmin, vmax = -0.5, 0.5 
        else:
            vmin, vmax = -1.0, 1.0
            
        im = ax.imshow(mat, cmap=cmap, vmin=vmin, vmax=vmax, origin='lower')
        ax.set_title(title, fontsize=10)
        ax.axis('off')
        return im

    # Iterate over Columns (Modes)
    for col_idx, m in enumerate(modes):
        
        # Row 1 (GS Row 2): Classical RBM
        ax_cl = fig.add_subplot(gs[2, col_idx])
        plot_mat(ax_cl, baselines[m], f"Classical ({labels[m]})")
        if col_idx == 0: ax_cl.set_ylabel("Classical RBM", fontsize=12, fontweight='bold')

        # Row 2 (GS Row 3): QPU Default Orbit
        ax_def = fig.add_subplot(gs[3, col_idx])
        err_def = default_orb["modes"][m]["error"]
        plot_mat(ax_def, default_orb["modes"][m]["matrix"], f"QPU Default (Err: {err_def:.3f})")
        if col_idx == 0: ax_def.set_ylabel("QPU Default", fontsize=12, fontweight='bold')

        # Row 3 (GS Row 4): QPU Best Orbit
        ax_best = fig.add_subplot(gs[4, col_idx])
        err_best = best_orb["modes"][m]["error"]
        im = plot_mat(ax_best, best_orb["modes"][m]["matrix"], f"QPU Best (Err: {err_best:.3f})")
        if col_idx == 0: ax_best.set_ylabel("QPU Best", fontsize=12, fontweight='bold')

        # Row 4 (GS Row 5): Difference (Best - Classical)
        ax_diff = fig.add_subplot(gs[5, col_idx])
        diff_mat = best_orb["modes"][m]["matrix"] - baselines[m]
        plot_mat(ax_diff, diff_mat, "Diff: Best - Classical", is_diff=True)
        if col_idx == 0: ax_diff.set_ylabel("Difference", fontsize=12, fontweight='bold')

    # Add Colorbar for Matrices
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.4])
    fig.colorbar(im, cax=cbar_ax, label="Correlation / Difference")
    
    plt.suptitle("Orbit Sensitivity Analysis: Weight Polarity Impact", fontsize=16, fontweight='bold', y=0.99)
    plt.show()




def plot_hamming_correlations_classical(results: dict, n_cond: int = 52):
    energies = list(results.keys())
    e1, e2 = energies[0], energies[1]
    use_gray = results[e1].get('use_gray', False)
    
    # 1 Row, 3 Columns (E1, E2, Difference)
    # Changed figsize to be wide (20, 6)
    fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True)
    
    # --- Helper Functions ---
    def _get_corr_matrix(samples):
        if isinstance(samples, torch.Tensor):
            samples = samples.float().cpu()
        
        # Compute Correlation
        corr = torch.corrcoef(samples.T).numpy()
        
        # Slice off clamped bits (visible conditioning units)
        corr = corr[n_cond:, n_cond:]
        
        # Zero diagonal & fix NaNs
        np.fill_diagonal(corr, 0)
        return np.nan_to_num(corr, nan=0.0)

    # --- 1. Compute Matrices ---
    c_e1 = _get_corr_matrix(results[e1]['samples'])
    c_e2 = _get_corr_matrix(results[e2]['samples'])
    c_diff = c_e2 - c_e1

    # --- 2. Plotting ---
    
    # Plot E1 (Left)
    im1 = axes[0].imshow(c_e1, cmap='seismic', vmin=-1, vmax=1, origin='lower')
    axes[0].set_title(f"Correlation: Energy {e1}", fontsize=14, fontweight='bold')
    axes[0].set_ylabel("Latent Index", fontsize=12)
    axes[0].set_xlabel("Latent Index", fontsize=12)

    # Plot E2 (Middle)
    im2 = axes[1].imshow(c_e2, cmap='seismic', vmin=-1, vmax=1, origin='lower')
    axes[1].set_title(f"Correlation: Energy {e2}", fontsize=14, fontweight='bold')
    # No y-label needed due to sharey=True
    axes[1].set_xlabel("Latent Index", fontsize=12)

    # Plot Difference (Right)
    im3 = axes[2].imshow(c_diff, cmap='seismic', vmin=-1, vmax=1, origin='lower')
    axes[2].set_title(f"Difference (E{e2} - E{e1})", fontsize=14, fontweight='bold')
    axes[2].set_xlabel("Latent Index", fontsize=12)

    # --- 3. Formatting ---
    
    # Adjust subplots to make room for colorbar on the right
    plt.subplots_adjust(right=0.9)
    
    # Add a vertical colorbar on the far right
    # [left, bottom, width, height] relative to figure size
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7]) 
    fig.colorbar(im1, cax=cbar_ax, label="Pearson Correlation")
    if use_gray:
        rbm_type = "Gray Code RBM"
    else:
        rbm_type = "Binary Code RBM"

    fig.suptitle(f"{rbm_type} Correlation Drift: Hamming Cliff ({e1} vs {e2})", fontsize=16)
    plt.show()


def plot_hamming_magnetization_classical(results: dict, n_cond: int = 52):
    energies = list(results.keys())
    e1, e2 = energies[0], energies[1]
    use_gray = results[e1].get('use_gray', False)

    
    # --- 1. Extract & Compute Means ---
    def get_mag(samples):
        # Mean across batch, slice off clamped nodes
        return samples.float().cpu().mean(dim=0)[n_cond:].numpy()

    mag_e1 = get_mag(results[e1]['samples'])
    mag_e2 = get_mag(results[e2]['samples'])
    
    indices = np.arange(n_cond, n_cond + len(mag_e1))

    # --- 2. Setup Plot ---
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    if use_gray:
        rbm_type = "Gray Code RBM"
    else:
        rbm_type = "Binary Code RBM"
    
     # Overall Title
    fig.suptitle(f"{rbm_type} Transition: {e1} $\\to$ {e2}", fontsize=16)

    # --- Subplot 1: Absolute Profiles ---
    ax = axes[0]
    
    # Plot E1 (Pre-Cliff)
    ax.plot(indices, mag_e1, color='black', linestyle='--', alpha=0.6, linewidth=2, label=f'Energy {e1} (Pre-Cliff)')
    # Plot E2 (Post-Cliff)
    ax.fill_between(indices, mag_e1, mag_e2, color='red', alpha=0.1, label='Drift Region')
    ax.plot(indices, mag_e2, color='#DC267F', linestyle='-', linewidth=2.5, label=f'Energy {e2} (Post-Cliff)')
    
    ax.set_ylabel(r"Magnetization $\langle \sigma_z \rangle$", fontsize=12)
    ax.set_title("Magnetization Profile Shift", fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    # --- Subplot 2: The "Jump" (Difference) ---
    ax = axes[1]
    
    diff = mag_e2 - mag_e1
    
    # Use stem plot to highlight specific bits that changed
    markerline, stemlines, baseline = ax.stem(indices, diff, basefmt=" ")
    plt.setp(stemlines, 'color', 'black', 'linewidth', 1, 'alpha', 0.5)
    plt.setp(markerline, 'color', '#DC267F', 'markersize', 6)
    
    # Add a fill for better visibility of magnitude
    ax.fill_between(indices, 0, diff, color='#DC267F', alpha=0.2)

    ax.set_ylabel(r"$\Delta \langle \sigma_z \rangle$ (E2 - E1)", fontsize=12)
    ax.set_xlabel("Latent Node Index", fontsize=12)
    ax.set_title("Bit Difference Magnitude", fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Zero line
    ax.axhline(0, color='black', linewidth=1)

    plt.tight_layout()
    plt.show()


def plot_hamming_cliff_comparison(results_binary: dict, results_gray: dict, n_cond: int = 52):
    """
    Plots the difference in magnetization (Drift) for both Binary and Gray code 
    experiments on the same axes for direct poster comparison.
    """
    
    # --- 1. Helper to compute drift ---
    def compute_diff(results):
        energies = sorted(list(results.keys())) # Ensure E1 < E2
        e1, e2 = energies[0], energies[1]
        
        # Get raw samples (B, latent_dim)
        s1 = results[e1]['samples'].float()
        s2 = results[e2]['samples'].float()
        
        # Compute mean magnetization, slice off conditional nodes
        mag1 = s1.mean(dim=0)[n_cond:].cpu().numpy()
        mag2 = s2.mean(dim=0)[n_cond:].cpu().numpy()
        
        # Return difference
        return mag2 - mag1, (e1, e2)

    diff_binary, (be1, be2) = compute_diff(results_binary)
    diff_gray, (ge1, ge2) = compute_diff(results_gray)
    
    # Create Indices (Latent Nodes)
    indices = np.arange(len(diff_binary))
    
    # --- 2. Poster Style Settings ---
    plt.rcParams.update({
    # 1. Use the generic 'serif' family
    "font.family": "serif",
    
    # 2. Specify a list of serif fonts to try (in order of priority)
    # This fixes the "findfont" error by giving it options that definitely exist
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],

    # 3. Match the math font (equations) to Times New Roman
    # 'stix' is a math font that looks very similar to Times
    "mathtext.fontset": "stix", 
    
    # 4. Your sizing settings
    'font.size': 14,
    'axes.titlesize': 32,
    'axes.labelsize': 18,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 16,
    'legend.title_fontsize': 18,
    'lines.linewidth': 3
})
    # --- 3. Plot Setup ---
    fig, ax = plt.subplots(figsize=(20, 7)) # Wide aspect ratio for posters
    
    # Colors (IBM Color Blind Safe Palette)
    color_gray = '#DC267F'  # Magenta (Your original preference)
    color_binary = '#648FFF'    # Indigo/Blue (High contrast to Magenta)

    # --- 4. Plot Binary Code (The "Bad" Cliff) ---
    # Using 'step' plots is cleaner than stem for overlapping data
    ax.step(indices, diff_binary, where='mid', color=color_binary, label='Binary Code', alpha=0.8)
    ax.fill_between(indices, diff_binary, step='mid', color=color_binary, alpha=0.1)

    # --- 5. Plot Gray Code (The "Good" Cliff) ---
    ax.step(indices, diff_gray, where='mid', color=color_gray, label='Gray Code', alpha=1.0, linestyle='--')
    # Hatching helps distinguish the "good" overlap on a printed poster
    ax.fill_between(indices, diff_gray, step='mid', color=color_gray, alpha=0.4, hatch='//')

    # --- 6. Formatting & Annotations ---
    # Zero line (Reference)
    ax.axhline(0, color='black', linewidth=1.5, alpha=0.5)
    
    # Dynamic Title based on Energy Transition
    ax.set_title(f"Hamming Cliff Sensitivity: ${be1} \\to {be2}$ MeV", pad=20)
    ax.set_xlabel("Latent Node Index")
    ax.set_ylabel(r"Magnetization Drift $\Delta \langle \sigma_z \rangle$ (Lower is Better)")
    
    # Force Integer Ticks for x-axis if not too dense
    if len(indices) < 20:
        ax.set_xticks(indices)
    
    # Large Legend
    ax.legend(loc='upper right', frameon=True, framealpha=0.95, shadow=True, borderpad=1)
    
    # Grid
    # ax.grid(True, linestyle=':', alpha=0.6)
    
    # Clean spines for professional look
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()
    return fig


def plot_j_phys(j_phys, label="RBM"):
    """
    Plots a histogram of physical couplers from a single RBM embedding.
    """
    def extract_couplers(j_phys_dict):
        return [val for val in j_phys_dict.values()]

    couplers = extract_couplers(j_phys)

    # 2. Setup the plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6), dpi=120)

    # Define color
    c_color = '#4c72b0' # Blue

    # 3. Plot Histogram
    sns.histplot(
        couplers, 
        bins=30,  
        color=c_color, 
        edgecolor='white',
        linewidth=0.5,
        alpha=0.8,  # Increased alpha since there is no overlapping plot
        label=label,
        ax=ax
    )

    # 4. Add Mean
    mean_val = np.mean([coupler for coupler in couplers if coupler != -1])  # Exclude chains from mean calculation
    
    # Changed mean line to a dark gray for better contrast against the single color
    ax.axvline(mean_val, color='#333333', linestyle='--', linewidth=2, label=f'{label} Mean: {mean_val:.2f}')

    # 5. Aesthetics
    ax.set_title("Distribution of Physical Coupler Values ($J_{phys}$)", fontsize=16, fontweight='bold', pad=15)
    ax.set_xlabel("Coupler Strength", fontsize=12)
    ax.set_ylabel("Count (Log Scale)", fontsize=12)
    
    # Log scale as requested
    ax.set_yscale("log")
    
    # Legend location
    ax.legend(frameon=True, fontsize=10, loc='upper right')

    # Clean borders
    sns.despine()
    plt.tight_layout()


def plot_pareto_frontier(
    results: list,           # List[EmbeddingPoint] — all sweep results
    pareto: list = None,     # List[EmbeddingPoint] — Pareto-optimal subset (optional)
    size_metric: str = "total",
    color_by: str = "density_floor",
    figsize: tuple = (12, 5),
):
    """
    Visualises the embedding Pareto frontier from a pareto_sweep_expansion run.

    Left panel  — scatter of all sweep points (size vs density), with the Pareto
                  frontier drawn as a step function.  Points are coloured by the
                  density_floor parameter used for that run.
    Right panel — degree quality metrics (min_degree and mean_degree) across the
                  same sweep, so you can see where dead units appear.

    Args:
        results:     All EmbeddingPoint objects returned by pareto_sweep_expansion.
        pareto:      Pareto-optimal subset returned by extract_pareto_frontier.
                     If None, all points are plotted without a frontier curve.
        size_metric: 'total' (n_vis + n_hid) or 'min' (min(n_vis, n_hid)).
        color_by:    Which EmbeddingPoint field drives the colormap ('density_floor'
                     or 'min_hits_used').
        figsize:     Overall figure size.
    """
    from typing import List

    def size_of(p: EmbeddingPoint) -> int:
        return p.total_nodes if size_metric == "total" else min(p.n_vis, p.n_hid)

    fig, (ax_pareto, ax_degree) = plt.subplots(1, 2, figsize=figsize)

    # ── colour map ──────────────────────────────────────────────────────────
    cvals = np.array([getattr(p, color_by) for p in results], dtype=float)
    norm  = plt.Normalize(cvals.min(), cvals.max())
    cmap  = plt.cm.viridis_r          # high density_floor → bright (dense, fewer nodes)

    sizes_all   = np.array([size_of(p)   for p in results])
    density_all = np.array([p.density    for p in results])
    min_deg_all = np.array([p.min_degree for p in results])
    mean_deg_all= np.array([p.mean_degree for p in results])

    # ── Left panel: Pareto scatter ───────────────────────────────────────────
    sc = ax_pareto.scatter(
        sizes_all, density_all,
        c=cvals, cmap=cmap, norm=norm,
        s=80, zorder=3, edgecolors='k', linewidths=0.5,
    )
    plt.colorbar(sc, ax=ax_pareto, label=color_by.replace("_", " "))

    # Annotate each point with its density_floor value
    for p in results:
        ax_pareto.annotate(
            f"{p.density_floor:.2f}",
            (size_of(p), p.density),
            textcoords="offset points", xytext=(4, 4),
            fontsize=7, color="0.3",
        )

    # Draw the Pareto frontier as a step function if provided
    if pareto:
        px = np.array([size_of(p) for p in pareto])
        py = np.array([p.density  for p in pareto])
        # step-after so the frontier reads as "achievable region"
        ax_pareto.step(px, py, where="post", color="#D55E00", linewidth=2,
                       label="Pareto frontier", zorder=4)
        ax_pareto.scatter(px, py, color="#D55E00", s=100, zorder=5,
                          edgecolors='k', linewidths=0.8)

    x_label = "n_vis + n_hid" if size_metric == "total" else "min(n_vis, n_hid)"
    ax_pareto.set_xlabel(f"RBM size  ({x_label})", fontsize=12)
    ax_pareto.set_ylabel("Edge density  (actual / n_vis × n_hid)", fontsize=12)
    ax_pareto.set_title("Size vs Connectivity", fontsize=13)
    if pareto:
        ax_pareto.legend(fontsize=10)
    ax_pareto.grid(True, linestyle="--", alpha=0.4)
    sns.despine(ax=ax_pareto)

    # ── Right panel: degree quality ──────────────────────────────────────────
    floors = np.array([p.density_floor for p in results])
    ax_degree.plot(floors, mean_deg_all, marker='o', color='#009E73',
                   linewidth=1.8, label="mean degree")
    ax_degree.plot(floors, min_deg_all, marker='s', color='#D55E00',
                   linewidth=1.8, linestyle='--', label="min degree")

    # Shade the dead-unit danger zone (min_degree == 0)
    ax_degree.axhline(0, color='red', linewidth=0.8, linestyle=':')
    ax_degree.fill_between(floors, 0, min_deg_all,
                           where=(min_deg_all == 0), alpha=0.15, color='red',
                           label="dead units")

    ax_degree.set_xlabel("density_floor  (sparsity parameter)", fontsize=12)
    ax_degree.set_ylabel("Node degree", fontsize=12)
    ax_degree.set_title("Degree Quality vs Sparsity", fontsize=13)
    ax_degree.legend(fontsize=10)
    ax_degree.grid(True, linestyle="--", alpha=0.4)
    sns.despine(ax=ax_degree)

    plt.suptitle("RBM Embedding Pareto Sweep", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.show()
    plt.show()


def plot_susceptibility_comparison(comparison_results):
    """
    Visualises Uniform Spreading vs Susceptibility-Compensated logical-J sampling.

    Layout:
        Row 0: Error norm bars | Chain break bars | Summary text
        Row 1: Classical matrix | Uniform matrix  | Compensated matrix
        Row 2: J distribution   | Diff: Uni - Cl  | Diff: Comp - Cl
        Row 3: Classical mag    | ΔMag: Uni - Cl  | ΔMag: Comp - Cl
    """
    mat_cl   = comparison_results["classical_matrix"]
    uni      = comparison_results["uniform"]
    comp     = comparison_results["compensated"]
    mag_cl   = comparison_results.get("classical_magnetization")
    j_uni    = comparison_results.get("j_uniform")
    j_comp   = comparison_results.get("j_compensated")

    fig = plt.figure(figsize=(16, 21))
    gs  = fig.add_gridspec(4, 3, height_ratios=[0.15, 0.3, 0.3, 0.25],
                           hspace=0.45, wspace=0.3)

    bar_labels = ["Uniform\nSpreading", "Susceptibility\nCompensated"]
    bar_colors = ["steelblue", "darkorange"]

    # ── Row 0: Summary Stats ──
    # Error norm
    ax_err   = fig.add_subplot(gs[0, 0])
    vals_err = [uni["error_norm"], comp["error_norm"]]
    bars     = ax_err.bar(bar_labels, vals_err, color=bar_colors, alpha=0.8, width=0.4)
    ax_err.set_title("Error Norm vs Classical", fontweight='bold')
    ax_err.set_ylabel("||QPU – Classical||")
    ax_err.set_ylim(0, max(vals_err) * 1.2)
    for bar, v in zip(bars, vals_err):
        ax_err.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                    f"{v:.3f}", ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Chain break fraction
    ax_brk   = fig.add_subplot(gs[0, 1])
    vals_brk = [uni["break_frac"], comp["break_frac"]]
    bars2    = ax_brk.bar(bar_labels, vals_brk, color=bar_colors, alpha=0.8, width=0.4)
    ax_brk.set_title("Chain Break Fraction", fontweight='bold')
    ax_brk.set_ylabel("Fraction of Broken Chains")
    ax_brk.set_ylim(0, max(vals_brk) * 1.35 + 1e-4)
    for bar, v in zip(bars2, vals_brk):
        ax_brk.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                    f"{v:.1%}", ha='center', va='bottom', fontsize=8, fontweight='bold')

    # Summary text
    ax_txt = fig.add_subplot(gs[0, 2])
    ax_txt.axis('off')
    delta_err = uni["error_norm"] - comp["error_norm"]
    winner    = "Compensated" if delta_err > 0 else "Uniform"
    summary   = (
        f"Uniform Spreading:\n"
        f"  Error : {uni['error_norm']:.4f}\n"
        f"  Breaks: {uni['break_frac']:.2%}\n\n"
        f"Susceptibility Compensated:\n"
        f"  Error : {comp['error_norm']:.4f}\n"
        f"  Breaks: {comp['break_frac']:.2%}\n\n"
        f"Δ Error: {abs(delta_err):.4f}\n"
        f"Winner: {winner}"
    )
    ax_txt.text(0.5, 0.5, summary, transform=ax_txt.transAxes, fontsize=11,
                verticalalignment='center', horizontalalignment='center',
                fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow', alpha=0.8))

    # ── Matrix helper ──
    def plot_mat(ax, data, title, diff=False):
        vmin, vmax = (-0.5, 0.5) if diff else (-1, 1)
        im = ax.imshow(data, cmap='seismic', vmin=vmin, vmax=vmax, origin='lower')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axis('off')
        return im

    # ── Row 1: Absolute correlation matrices ──
    ax_cl = fig.add_subplot(gs[1, 0])
    plot_mat(ax_cl, mat_cl, "Target (Classical RBM)")

    ax_uni = fig.add_subplot(gs[1, 1])
    plot_mat(ax_uni, uni["matrix"],
             f"Uniform Spreading\nErr: {uni['error_norm']:.3f}")

    ax_comp = fig.add_subplot(gs[1, 2])
    im_main = plot_mat(ax_comp, comp["matrix"],
                       f"Susceptibility Compensated\nErr: {comp['error_norm']:.3f}")

    # ── Row 2: J-distribution (col 0) + difference matrices (cols 1-2) ──
    ax_j = fig.add_subplot(gs[2, 0])
    if j_uni is not None and j_comp is not None:
        bins = np.linspace(
            min(j_uni.min(), j_comp.min()),
            max(j_uni.max(), j_comp.max()),
            50
        )
        ax_j.hist(j_uni, bins=bins, alpha=0.6, color='steelblue',
                  label='Uniform', edgecolor='k', linewidth=0.3)
        ax_j.hist(j_comp, bins=bins, alpha=0.6, color='darkorange',
                  label='Compensated', edgecolor='k', linewidth=0.3)
        ax_j.set_title("Logical-J Distribution", fontsize=11, fontweight='bold')
        ax_j.set_xlabel("J value")
        ax_j.set_ylabel("Count")
        ax_j.legend(fontsize=8)
        ax_j.grid(True, alpha=0.3)
    else:
        ax_j.axis('off')

    ax_diff_uni = fig.add_subplot(gs[2, 1])
    plot_mat(ax_diff_uni, uni["matrix"] - mat_cl,
             "Error: Uniform – Classical", diff=True)

    ax_diff_comp = fig.add_subplot(gs[2, 2])
    im_diff = plot_mat(ax_diff_comp, comp["matrix"] - mat_cl,
                       "Error: Compensated – Classical", diff=True)

    # ── Row 3: Magnetization ──
    def _style_mag_ax(ax, n_nodes, ylabel):
        ax.set_xlabel("Latent Node Index", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlim(-1, n_nodes)
        ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    if mag_cl is not None and "magnetization" in uni and "magnetization" in comp:
        n_nodes = len(mag_cl)
        x = np.arange(n_nodes)

        ax_mag_cl = fig.add_subplot(gs[3, 0])
        ax_mag_cl.bar(x, mag_cl, color='dimgray', alpha=0.75, linewidth=0)
        ax_mag_cl.set_title("Classical Magnetization\n(Target)",
                            fontsize=11, fontweight='bold')
        ax_mag_cl.set_ylim(0, 1)
        _style_mag_ax(ax_mag_cl, n_nodes, r"$\langle\sigma\rangle$")

        def plot_mag_delta(ax, mag_qpu, title):
            delta  = mag_qpu - mag_cl
            colors = ['steelblue' if d >= 0 else 'firebrick' for d in delta]
            ax.bar(x, delta, color=colors, alpha=0.85, linewidth=0)
            ax.axhline(0, color='black', linewidth=1.0, linestyle='--', zorder=3)
            ax.set_title(title, fontsize=11, fontweight='bold')
            lim = max(abs(delta).max() * 1.15, 0.05)
            ax.set_ylim(-lim, lim)
            _style_mag_ax(ax, n_nodes,
                          r"$\Delta\langle\sigma\rangle$ (QPU – Classical)")

        ax_mag_uni = fig.add_subplot(gs[3, 1])
        plot_mag_delta(ax_mag_uni, uni["magnetization"],
                       "ΔMag: Uniform – Classical")

        ax_mag_comp = fig.add_subplot(gs[3, 2])
        plot_mag_delta(ax_mag_comp, comp["magnetization"],
                       "ΔMag: Compensated – Classical")
    else:
        for col in [0, 1, 2]:
            ax = fig.add_subplot(gs[3, col])
            ax.text(0.5, 0.5, "Magnetization data\nnot available",
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=10, color='gray')
            ax.axis('off')

    # Colorbars
    cbar_main = fig.add_axes([0.92, 0.50, 0.015, 0.20])
    fig.colorbar(im_main, cax=cbar_main, label="Pearson Correlation")

    cbar_diff = fig.add_axes([0.92, 0.28, 0.015, 0.20])
    fig.colorbar(im_diff, cax=cbar_diff, label="Correlation Delta")

    fig.suptitle(
        "Susceptibility Compensation: Uniform Spreading vs Logical-J Compensated",
        fontsize=14, fontweight='bold', y=0.98
    )
    plt.tight_layout(rect=[0, 0, 0.9, 0.97])
    plt.show()


def plot_flux_drift_comparison(comparison_results):
    """
    Visualises QPU sampling with Flux Drift Compensation off vs on.

    Layout:
        Row 0: Error norm bars | Chain break bars | Summary text
        Row 1: Classical matrix | FDC Off matrix  | FDC On matrix
        Row 2: Diff: FDC Off – Classical           | Diff: FDC On – Classical
        Row 3: Classical mag    | ΔMag: FDC Off    | ΔMag: FDC On
    """
    mat_cl  = comparison_results["classical_matrix"]
    no_fdc  = comparison_results["no_fdc"]
    fdc     = comparison_results["fdc"]
    mag_cl  = comparison_results.get("classical_magnetization")

    fig = plt.figure(figsize=(16, 21))
    gs  = fig.add_gridspec(4, 3, height_ratios=[0.15, 0.3, 0.3, 0.25],
                           hspace=0.45, wspace=0.3)

    bar_labels = ["FDC Off\n(baseline)", "FDC On"]
    bar_colors = ["steelblue", "darkorange"]

    # ── Row 0: Summary Stats ──
    ax_err   = fig.add_subplot(gs[0, 0])
    vals_err = [no_fdc["error_norm"], fdc["error_norm"]]
    bars     = ax_err.bar(bar_labels, vals_err, color=bar_colors, alpha=0.8, width=0.4)
    ax_err.set_title("Error Norm vs Classical", fontweight='bold')
    ax_err.set_ylabel("||QPU – Classical||")
    ax_err.set_ylim(0, max(vals_err) * 1.2)
    for bar, v in zip(bars, vals_err):
        ax_err.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                    f"{v:.3f}", ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax_brk   = fig.add_subplot(gs[0, 1])
    vals_brk = [no_fdc["break_frac"], fdc["break_frac"]]
    bars2    = ax_brk.bar(bar_labels, vals_brk, color=bar_colors, alpha=0.8, width=0.4)
    ax_brk.set_title("Chain Break Fraction", fontweight='bold')
    ax_brk.set_ylabel("Fraction of Broken Chains")
    ax_brk.set_ylim(0, max(vals_brk) * 1.35 + 1e-4)
    for bar, v in zip(bars2, vals_brk):
        ax_brk.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                    f"{v:.1%}", ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax_txt = fig.add_subplot(gs[0, 2])
    ax_txt.axis('off')
    delta_err = no_fdc["error_norm"] - fdc["error_norm"]
    winner    = "FDC On" if delta_err > 0 else "FDC Off"
    summary   = (
        f"FDC Off (baseline):\n"
        f"  Error : {no_fdc['error_norm']:.4f}\n"
        f"  Breaks: {no_fdc['break_frac']:.2%}\n\n"
        f"FDC On:\n"
        f"  Error : {fdc['error_norm']:.4f}\n"
        f"  Breaks: {fdc['break_frac']:.2%}\n\n"
        f"Δ Error: {abs(delta_err):.4f}\n"
        f"Winner: {winner}"
    )
    ax_txt.text(0.5, 0.5, summary, transform=ax_txt.transAxes, fontsize=11,
                verticalalignment='center', horizontalalignment='center',
                fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow', alpha=0.8))

    # ── Matrix helper ──
    def plot_mat(ax, data, title, diff=False):
        vmin, vmax = (-0.5, 0.5) if diff else (-1, 1)
        im = ax.imshow(data, cmap='seismic', vmin=vmin, vmax=vmax, origin='lower')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axis('off')
        return im

    # ── Row 1: Absolute correlation matrices ──
    ax_cl = fig.add_subplot(gs[1, 0])
    plot_mat(ax_cl, mat_cl, "Target (Classical RBM)")

    ax_nofdc = fig.add_subplot(gs[1, 1])
    plot_mat(ax_nofdc, no_fdc["matrix"],
             f"FDC Off\nErr: {no_fdc['error_norm']:.3f}")

    ax_fdc = fig.add_subplot(gs[1, 2])
    im_main = plot_mat(ax_fdc, fdc["matrix"],
                       f"FDC On\nErr: {fdc['error_norm']:.3f}")

    # ── Row 2: Difference matrices ──
    ax_j = fig.add_subplot(gs[2, 0])
    ax_j.axis('off')

    ax_diff_nofdc = fig.add_subplot(gs[2, 1])
    plot_mat(ax_diff_nofdc, no_fdc["matrix"] - mat_cl,
             "Error: FDC Off – Classical", diff=True)

    ax_diff_fdc = fig.add_subplot(gs[2, 2])
    im_diff = plot_mat(ax_diff_fdc, fdc["matrix"] - mat_cl,
                       "Error: FDC On – Classical", diff=True)

    # ── Row 3: Magnetization ──
    def _style_mag_ax(ax, n_nodes, ylabel):
        ax.set_xlabel("Latent Node Index", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlim(-1, n_nodes)
        ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    if mag_cl is not None and "magnetization" in no_fdc and "magnetization" in fdc:
        n_nodes = len(mag_cl)
        x = np.arange(n_nodes)

        ax_mag_cl = fig.add_subplot(gs[3, 0])
        ax_mag_cl.bar(x, mag_cl, color='dimgray', alpha=0.75, linewidth=0)
        ax_mag_cl.set_title("Classical Magnetization\n(Target)",
                            fontsize=11, fontweight='bold')
        ax_mag_cl.set_ylim(0, 1)
        _style_mag_ax(ax_mag_cl, n_nodes, r"$\langle\sigma\rangle$")

        def plot_mag_delta(ax, mag_qpu, title):
            delta  = mag_qpu - mag_cl
            colors = ['steelblue' if d >= 0 else 'firebrick' for d in delta]
            ax.bar(x, delta, color=colors, alpha=0.85, linewidth=0)
            ax.axhline(0, color='black', linewidth=1.0, linestyle='--', zorder=3)
            ax.set_title(title, fontsize=11, fontweight='bold')
            lim = max(abs(delta).max() * 1.15, 0.05)
            ax.set_ylim(-lim, lim)
            _style_mag_ax(ax, n_nodes,
                          r"$\Delta\langle\sigma\rangle$ (QPU – Classical)")

        ax_mag_nofdc = fig.add_subplot(gs[3, 1])
        plot_mag_delta(ax_mag_nofdc, no_fdc["magnetization"],
                       "ΔMag: FDC Off – Classical")

        ax_mag_fdc = fig.add_subplot(gs[3, 2])
        plot_mag_delta(ax_mag_fdc, fdc["magnetization"],
                       "ΔMag: FDC On – Classical")
    else:
        for col in [0, 1, 2]:
            ax = fig.add_subplot(gs[3, col])
            ax.text(0.5, 0.5, "Magnetization data\nnot available",
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=10, color='gray')
            ax.axis('off')

    # Colorbars
    cbar_main = fig.add_axes([0.92, 0.50, 0.015, 0.20])
    fig.colorbar(im_main, cax=cbar_main, label="Pearson Correlation")

    cbar_diff = fig.add_axes([0.92, 0.28, 0.015, 0.20])
    fig.colorbar(im_diff, cax=cbar_diff, label="Correlation Delta")

    fig.suptitle(
        "Flux Drift Compensation: Off vs On",
        fontsize=14, fontweight='bold', y=0.98
    )
    plt.tight_layout(rect=[0, 0, 0.9, 0.97])
    plt.show()


def plot_srt_aggregation_comparison(comparison_results):
    """
    Visualises "average all SRT batches" vs "pick the single best SRT batch".

    Layout
    ------
    Row 0 : Error norm bars  |  Chain break bars  |  Summary text
    Row 1 : Classical matrix |  Averaged matrix   |  Best-SRT matrix
    Row 2 : Per-batch errors |  Diff: Avg – Cl    |  Diff: Best – Cl
    Row 3 : Classical mag    |  ΔMag: Averaged    |  ΔMag: Best-SRT
    """
    mat_cl   = comparison_results["classical_matrix"]
    avg      = comparison_results["averaged"]
    best     = comparison_results["best_srt"]
    mag_cl   = comparison_results.get("classical_magnetization")
    per_batch = comparison_results.get("per_batch", [])
    best_idx  = best.get("best_batch_idx", None)

    fig = plt.figure(figsize=(16, 21))
    gs  = fig.add_gridspec(4, 3, height_ratios=[0.15, 0.3, 0.3, 0.25],
                           hspace=0.45, wspace=0.3)

    bar_labels = ["Averaged\nSRTs", "Best\nSRT"]
    bar_colors = ["steelblue", "darkorange"]

    # ── Row 0: Summary Stats ──
    ax_err   = fig.add_subplot(gs[0, 0])
    vals_err = [avg["error_norm"], best["error_norm"]]
    bars     = ax_err.bar(bar_labels, vals_err, color=bar_colors, alpha=0.8, width=0.4)
    ax_err.set_title("Error Norm vs Classical", fontweight='bold')
    ax_err.set_ylabel("||QPU – Classical||")
    ax_err.set_ylim(0, max(vals_err) * 1.2)
    for bar, v in zip(bars, vals_err):
        ax_err.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                    f"{v:.3f}", ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax_brk   = fig.add_subplot(gs[0, 1])
    vals_brk = [avg["break_frac"], best["break_frac"]]
    bars2    = ax_brk.bar(bar_labels, vals_brk, color=bar_colors, alpha=0.8, width=0.4)
    ax_brk.set_title("Chain Break Fraction", fontweight='bold')
    ax_brk.set_ylabel("Fraction of Broken Chains")
    ax_brk.set_ylim(0, max(vals_brk) * 1.35 + 1e-4)
    for bar, v in zip(bars2, vals_brk):
        ax_brk.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.02,
                    f"{v:.1%}", ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax_txt = fig.add_subplot(gs[0, 2])
    ax_txt.axis('off')
    delta_err = avg["error_norm"] - best["error_norm"]
    winner    = "Best SRT" if delta_err > 0 else "Averaged"
    n_batches = comparison_results.get("srt_batches", len(per_batch))
    num_reads = comparison_results.get("num_reads", "?")
    summary   = (
        f"Averaged ({n_batches} × {num_reads} reads):\n"
        f"  Error : {avg['error_norm']:.4f}\n"
        f"  Breaks: {avg['break_frac']:.2%}\n\n"
        f"Best SRT (batch {best_idx} of {n_batches}):\n"
        f"  Error : {best['error_norm']:.4f}\n"
        f"  Breaks: {best['break_frac']:.2%}\n\n"
        f"Δ Error: {abs(delta_err):.4f}\n"
        f"Winner: {winner}"
    )
    ax_txt.text(0.5, 0.5, summary, transform=ax_txt.transAxes, fontsize=11,
                verticalalignment='center', horizontalalignment='center',
                fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow', alpha=0.8))

    # ── Matrix helper ──
    def plot_mat(ax, data, title, diff=False):
        vmin, vmax = (-0.5, 0.5) if diff else (-1, 1)
        im = ax.imshow(data, cmap='seismic', vmin=vmin, vmax=vmax, origin='lower')
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axis('off')
        return im

    # ── Row 1: Absolute correlation matrices ──
    ax_cl = fig.add_subplot(gs[1, 0])
    plot_mat(ax_cl, mat_cl, "Target (Classical RBM)")

    ax_avg = fig.add_subplot(gs[1, 1])
    plot_mat(ax_avg, avg["matrix"],
             f"Averaged SRTs\nErr: {avg['error_norm']:.3f}")

    ax_best = fig.add_subplot(gs[1, 2])
    im_main = plot_mat(ax_best, best["matrix"],
                       f"Best SRT (batch {best_idx})\nErr: {best['error_norm']:.3f}")

    # ── Row 2: Per-batch error plot (col 0) + difference matrices ──
    ax_pb = fig.add_subplot(gs[2, 0])
    if per_batch:
        pb_errors = [pb["error_norm"] for pb in per_batch]
        x_b = np.arange(len(pb_errors))
        bar_cols = ['darkorange' if i == best_idx else 'steelblue' for i in x_b]
        ax_pb.bar(x_b, pb_errors, color=bar_cols, alpha=0.8, linewidth=0)
        ax_pb.axhline(avg["error_norm"], color='dimgray', linewidth=1.5,
                      linestyle='--', label=f"Avg ({avg['error_norm']:.3f})")
        if best_idx is not None:
            ax_pb.scatter([best_idx], [pb_errors[best_idx]], marker='*',
                          s=180, color='darkorange', zorder=5, label=f"Best (batch {best_idx})")
        ax_pb.set_title("Per-Batch Error Norms", fontsize=11, fontweight='bold')
        ax_pb.set_xlabel("Batch Index")
        ax_pb.set_ylabel("||QPU – Classical||")
        ax_pb.legend(fontsize=8)
        ax_pb.grid(axis='y', linestyle='--', alpha=0.4)
        ax_pb.spines['top'].set_visible(False)
        ax_pb.spines['right'].set_visible(False)
    else:
        ax_pb.axis('off')

    ax_diff_avg = fig.add_subplot(gs[2, 1])
    plot_mat(ax_diff_avg, avg["matrix"] - mat_cl,
             "Error: Averaged – Classical", diff=True)

    ax_diff_best = fig.add_subplot(gs[2, 2])
    im_diff = plot_mat(ax_diff_best, best["matrix"] - mat_cl,
                       "Error: Best SRT – Classical", diff=True)

    # ── Row 3: Magnetization ──
    def _style_mag_ax(ax, n_nodes, ylabel):
        ax.set_xlabel("Latent Node Index", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_xlim(-1, n_nodes)
        ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)
        ax.grid(axis='y', linestyle='--', alpha=0.5, zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    if mag_cl is not None and "magnetization" in avg and "magnetization" in best:
        n_nodes = len(mag_cl)
        x = np.arange(n_nodes)

        ax_mag_cl = fig.add_subplot(gs[3, 0])
        ax_mag_cl.bar(x, mag_cl, color='dimgray', alpha=0.75, linewidth=0)
        ax_mag_cl.set_title("Classical Magnetization\n(Target)",
                            fontsize=11, fontweight='bold')
        ax_mag_cl.set_ylim(0, 1)
        _style_mag_ax(ax_mag_cl, n_nodes, r"$\langle\sigma\rangle$")

        def plot_mag_delta(ax, mag_qpu, title):
            delta  = mag_qpu - mag_cl
            colors = ['steelblue' if d >= 0 else 'firebrick' for d in delta]
            ax.bar(x, delta, color=colors, alpha=0.85, linewidth=0)
            ax.axhline(0, color='black', linewidth=1.0, linestyle='--', zorder=3)
            ax.set_title(title, fontsize=11, fontweight='bold')
            lim = max(abs(delta).max() * 1.15, 0.05)
            ax.set_ylim(-lim, lim)
            _style_mag_ax(ax, n_nodes,
                          r"$\Delta\langle\sigma\rangle$ (QPU – Classical)")

        ax_mag_avg = fig.add_subplot(gs[3, 1])
        plot_mag_delta(ax_mag_avg, avg["magnetization"], "ΔMag: Averaged – Classical")

        ax_mag_best = fig.add_subplot(gs[3, 2])
        plot_mag_delta(ax_mag_best, best["magnetization"], "ΔMag: Best SRT – Classical")
    else:
        for col in [0, 1, 2]:
            ax = fig.add_subplot(gs[3, col])
            ax.text(0.5, 0.5, "Magnetization data\nnot available",
                    ha='center', va='center', transform=ax.transAxes,
                    fontsize=10, color='gray')
            ax.axis('off')

    # Colorbars
    cbar_main = fig.add_axes([0.92, 0.50, 0.015, 0.20])
    fig.colorbar(im_main, cax=cbar_main, label="Pearson Correlation")

    cbar_diff = fig.add_axes([0.92, 0.28, 0.015, 0.20])
    fig.colorbar(im_diff, cax=cbar_diff, label="Correlation Delta")

    fig.suptitle(
        "SRT Aggregation: Average All Batches vs Pick Best Batch",
        fontsize=14, fontweight='bold', y=0.98
    )
    plt.tight_layout(rect=[0, 0, 0.9, 0.97])
    plt.show()