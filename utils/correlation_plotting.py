import torch
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from CaloQuVAE import logging
logger = logging.getLogger(__name__)

def compute_voxelwise_correlation_matrix(data):
    """
    Computes the voxel-wise correlation matrix across events.
    
    Parameters:
    - data: [N_events, N_voxels] tensor
    """
    mean = data.mean(dim=0, keepdim=True)
    std = data.std(dim=0, keepdim=True) + 1e-8 # avoid division by 0 errors
    normalized = (data - mean) / std
    corr = torch.matmul(normalized.T, normalized) / (data.size(0) - 1)
    return corr

def frobenius_distance(mat1, mat2):
    """
    Returns the Frobenius norm between two correlation matrices.
    
    Parameters:
    - mat1, mat2: [N_voxels, N_voxels] tensors
    """
    return torch.norm(mat1 - mat2, p='fro').item()


def compute_sparsity_correlation_matrix(data):
    """
    Computes voxel-wise correlation of sparsity (hit = 1, no hit = 0).
    
    Parameters:
    - data: [N_events, N_voxels] tensor where non-zero = hit
    """
    binary = (data > 0).float()  # 1 for hit, 0 for no hit (sparsity voxel wise), layer wise uses other method
    return compute_voxelwise_correlation_matrix(binary)


def compute_layerwise_correlation_matrices(data, relevant_layers, voxels_per_layer):
    """
    Compute voxel-wise correlation matrices for each calorimeter layer.

    Parameters:
    - data: [N_events, N_total_voxels]
    - relevant_layers: list of layer numbers
    - voxels_per_layer: number of voxels per layer
    """
    corr_matrices = {}
    for i, layer in enumerate(relevant_layers):
        start = i * voxels_per_layer
        end = (i + 1) * voxels_per_layer
        corr_matrices[layer] = compute_voxelwise_correlation_matrix(data[:, start:end])
    return corr_matrices


def compute_layerwise_frobenius_distances(corr_gt, corr_prior):
    """
    Compute layer-wise Frobenius distances between correlation matrices.

    Parameters:
    - corr_gt: ground truth correlation matrices {layer: matrix}
    - corr_prior: prior sample correlation matrices {layer: matrix}
    """
    return {
        layer: frobenius_distance(corr_gt[layer], corr_prior[layer])
        for layer in corr_gt}


def collapse_to_layer_energy(data, relevant_layers, voxels_per_layer):
    """
    Collapse full voxel data into layer energy sums.
    
    Parameters:
    - data: [N_events, N_voxels]
    - relevant_layers: list of layer indices
    - voxels_per_layer: voxels per layer
    """
    layer_energies = []
    for i in range(len(relevant_layers)):
        start = i * voxels_per_layer
        end = (i + 1) * voxels_per_layer
        layer_sum = data[:, start:end].sum(dim=1)
        layer_energies.append(layer_sum)
    return torch.stack(layer_energies, dim=1)

def compute_layerwise_sparsity_correlation_matrices(data, relevant_layers, voxels_per_layer):
    """
    Compute voxel-wise sparsity correlation matrices for each layer.

    Parameters:
    - data: [N_events, N_total_voxels]
    - relevant_layers: list of layer indices
    - voxels_per_layer: voxels per layer
    """
    binary_hits = (data > 0).float()
    return compute_layerwise_correlation_matrices(binary_hits, relevant_layers, voxels_per_layer)


def collapse_to_layer_sparsity(data, relevant_layers, voxels_per_layer):
    """
    Gives a [N_events, N_layers] tensor of sparsity values for the layers
    """
    N_events = data.shape[0]
    N_layers = len(relevant_layers)
    layer_sparsity = torch.zeros(N_events, N_layers)

    for i, layer in enumerate(relevant_layers):
        start = i * voxels_per_layer
        end = (i + 1) * voxels_per_layer
        layer_data = data[:, start:end]
        sparsity = (layer_data == 0).float().sum(dim=1) / voxels_per_layer
        layer_sparsity[:, i] = sparsity

    return layer_sparsity

def compute_patch_layer_correlation_matrix(data, patch_coords, relevant_layers, r, phi):
    """
    Creates a correlation matrix over all (layer, patch) combinations.

    Parameters:
    - data: [N_events, N_voxels]
    - patch_coords: list of (r_idx, phi_idx) patches to be used
    - relevant_layers: list of layers used
    - r, phi: number of bins in radial and phi directions
    """
    voxels_per_layer = r * phi
    patch_vectors = []

    for r_idx, phi_idx in patch_coords:
        flat_idx = r_idx * phi + phi_idx
        for i, layer in enumerate(relevant_layers):
            start = i * voxels_per_layer
            idx = start + flat_idx
            patch_vectors.append(data[:, idx:idx+1])  # shape is: [N_events, 1]

    stacked = torch.cat(patch_vectors, dim=1)  # shape is now: [N_events, N_patches * N_layers]
    normed = (stacked - stacked.mean(dim=0, keepdim=True)) / (stacked.std(dim=0, keepdim=True) + 1e-8)
    corr = torch.matmul(normed.T, normed) / (stacked.size(0) - 1)
    return corr

def plot_frobenius_distance_vs_layer(distances, title="Frobenius Distance per Layer", save_path=None):
    """
    Plot the Frobenius distance for each layer.

    Parameters:
    - distances: dictionary of {layer_num: frob_distance}
    """
    layers = list(distances.keys())
    values = list(distances.values())
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(layers, values, color='steelblue', alpha=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Frobenius Metric")
    ax.set_title(title)
    ax.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()

    return fig


def plot_correlation_matrix(matrix, layer=None, title=None, save_path=None):
    """
    Plot a heatmap of a correlation matrix.
    """
    if isinstance(matrix, torch.Tensor):
        matrix = matrix.cpu().numpy()

    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.imshow(matrix, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax, ax=ax)

    if title:
        ax.set_title(title)
    elif layer is not None:
        ax.set_title(f"Correlation Matrix - Layer {layer}")

    ax.set_xlabel("Voxel Index")
    ax.set_ylabel("Voxel Index")

    if save_path:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()
    return fig

def plot_layer_energy_correlation_matrix(corr_matrix, cfg, layer_labels=None, title="Layer Energy Correlation"):
    """
    Plot a heatmap of layer-level correlation matrix.

    Parameters:
    - corr_matrix: [N_layers, N_layers]
    - layer_labels: e.g. [0, 1, 2, 3, 4, 12]
    """
    if isinstance(corr_matrix, torch.Tensor):
        corr_matrix = corr_matrix.detach().cpu().numpy()

    if layer_labels is None:
        layer_labels = [f"Layer {i}" for i in range(corr_matrix.shape[0])]
        
    # since CaloChallenge has 40+ layers the numbers become messy to look at
    if "atlas" not in cfg.data.dataset_name.lower():
        annot=False
    else:
        annot=True

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        corr_matrix,
        xticklabels=layer_labels,
        yticklabels=layer_labels,
        cmap="coolwarm",
        annot=annot,
        fmt=".2f",
        square=True,
        cbar_kws={"label": "Correlation"},
        vmin=-1, vmax=1, 
        ax=ax
    )
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_layerwise_correlation_grid(corr_dict, title_prefix="Correlation Matrix"):
    """
    Plots a grid of correlation heatmaps for each layer with voxel indices on axes.

    Parameters:
        corr_dict: {layer: correlation matrix}
        title_prefix: subplot title prefix
    """

    n_layers = len(corr_dict)
    n_cols = 4
    n_rows = int(np.ceil(n_layers / n_cols))

    fig, axs = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4.5 * n_rows))
    axs = axs.flatten()
    vmin, vmax = -1.0, 1.0
    for i, (layer, matrix) in enumerate(sorted(corr_dict.items())):
        matrix_np = matrix.detach().cpu().numpy()
        ax = axs[i]

        size = matrix_np.shape[0]
        step = max(1, size // 6)
        ticks = np.arange(0, size, step)

        sns.heatmap(
            matrix_np,
            ax=ax,
            cmap="coolwarm",
            center=0,
            cbar=False,
            vmin=vmin,
            vmax=vmax,
            square=True
        )

        ax.set_title(f"{title_prefix} - Layer {layer}", fontsize=10)
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticks, rotation=45, fontsize=8)
        ax.set_yticks(ticks)
        ax.set_yticklabels(ticks, fontsize=8)
        ax.set_xlabel("Voxel Index", fontsize=9)
        ax.set_ylabel("Voxel Index", fontsize=9)

    # hide any unused axes
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    #shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label="Correlation")

    fig.subplots_adjust(right=0.9, wspace=0.4, hspace=0.4)  # Leave space for colorbar and labels
    return fig

def plot_patch_layer_correlation_matrix(corr_matrix, patch_coords, relevant_layers, title_prefix="Patch-Layer Correlation"):
    """
    Plot heatmap of (layer, patch) correlation matrix with axis labels like L0_p(0,0), etc.

    Parameters:
        corr_matrix: [N_total, N_total]
        patch_coords: list of (r, phi) where length = N_patches
        relevant_layers: list of layer indices
        title_prefix
    """
    if isinstance(corr_matrix, torch.Tensor):
        corr_matrix = corr_matrix.detach().cpu().numpy()

    labels = []
    for rphi in patch_coords:
        for layer in relevant_layers:
            labels.append(f"L{layer}_p({rphi[0]},{rphi[1]})")

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        corr_matrix,
        xticklabels=labels,
        yticklabels=labels,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        cbar_kws={"label": "Correlation"},
        ax=ax
    )
    ax.set_title(f"{title_prefix} Patch-Layer Correlation Matrix", fontsize=14)
    ax.tick_params(axis='x', labelsize=2, rotation=90)
    ax.tick_params(axis='y', labelsize=2, rotation=0)
    plt.tight_layout()
    return fig

# function called in the engine to generate the correlation plots and frobenius metrics
def correlation_plots(cfg, incident_energy, showers, showers_prior, epoch):
    target_corr = compute_voxelwise_correlation_matrix(showers) # ground truth
    sampled_corr = compute_voxelwise_correlation_matrix(showers_prior) # prior
    frob_dist = frobenius_distance(target_corr, sampled_corr) # compute frobenius metric
    logger.info(f"Epoch {epoch} - Frobenius Metric (Voxel Level): {frob_dist:.4f}") # move these outside of this function?
    fig_target_corr = plot_correlation_matrix(target_corr, title="Ground Truth Correlation Matrix")
    fig_sampled_corr = plot_correlation_matrix(sampled_corr, title="Sampled Correlation Matrix")

    # Layer-wise results:
    voxels_per_layer = cfg.data.r * cfg.data.phi
    relevant_layers = cfg.data.relevantLayers
                                        
    gt_corr = compute_layerwise_correlation_matrices(showers, relevant_layers, voxels_per_layer)
    prior_corr = compute_layerwise_correlation_matrices(showers_prior, relevant_layers, voxels_per_layer)
    frob_dists = compute_layerwise_frobenius_distances(gt_corr, prior_corr)
    layer_energies = collapse_to_layer_energy(showers, relevant_layers, voxels_per_layer)
    # plots of all active layers:
    fig_gt_grid = plot_layerwise_correlation_grid(gt_corr, title_prefix="GT Correlation")
    fig_prior_grid = plot_layerwise_correlation_grid(prior_corr, title_prefix="Prior Correlation")
    # Plot Frobenius distance between GT and Prior layer-wise
    fig_frob_layerwise = plot_frobenius_distance_vs_layer(frob_dists, title="Layerwise Frobenius Metric (GT vs Prior)")

    # GT layer-wise energy correlation
    layer_energies_gt = collapse_to_layer_energy(showers, relevant_layers, voxels_per_layer)
    layer_corr_matrix_gt = compute_voxelwise_correlation_matrix(layer_energies_gt)
    fig_gt = plot_layer_energy_correlation_matrix(
        layer_corr_matrix_gt,
        cfg,
        layer_labels=[str(l) for l in relevant_layers],
        title="GT Layer Energy Correlation")

    # Prior Layerwise Energy Correlation
    layer_energies_prior = collapse_to_layer_energy(showers_prior, relevant_layers, voxels_per_layer)
    layer_corr_matrix_prior = compute_voxelwise_correlation_matrix(layer_energies_prior)
    fig_prior = plot_layer_energy_correlation_matrix(
        layer_corr_matrix_prior,
        cfg,
        layer_labels=[str(l) for l in relevant_layers],
        title="Prior Layer Energy Correlation")

    frob_dist_energy_corr = frobenius_distance(layer_corr_matrix_gt, layer_corr_matrix_prior)
    logger.info(f"Epoch {epoch} - Frobenius Metric (Layer Level): {frob_dist_energy_corr:.4f}")


    # Sparsity correlations and Frobenius metrics:
    gt_sparsity_corr = compute_sparsity_correlation_matrix(showers)
    sampled_sparsity_corr = compute_sparsity_correlation_matrix(showers_prior)

    # Frobenius metric between sparsity correlation matrices
    sparsity_frob_distance = frobenius_distance(gt_sparsity_corr, sampled_sparsity_corr)
    logger.info(f"Epoch {epoch} - Sparsity Frobenius Metric (Voxel Level): {sparsity_frob_distance:.4f}")

    # Sparsity correlation matrices
    gt_spars_corr = plot_correlation_matrix(gt_sparsity_corr, title="GT Sparsity Correlation")
    prior_spars_corr = plot_correlation_matrix(sampled_sparsity_corr, title="Sampled Sparsity Correlation")

    # Layer-wise sparsity correlation matrices
    gt_sparsity_corr = compute_layerwise_sparsity_correlation_matrices(showers, relevant_layers, voxels_per_layer)
    prior_sparsity_corr = compute_layerwise_sparsity_correlation_matrices(showers_prior, relevant_layers, voxels_per_layer)

    # Plot grids of sparsity correlation matrices (grid of all layers)
    fig_gt_sparsity = plot_layerwise_correlation_grid(gt_sparsity_corr, title_prefix="GT Sparsity Correlation")
    fig_prior_sparsity = plot_layerwise_correlation_grid(prior_sparsity_corr, title_prefix="Prior Sparsity Correlation")
    frob_sparsity_dists = compute_layerwise_frobenius_distances(gt_sparsity_corr, prior_sparsity_corr) # frobenius metric between layers


    layer_sparsity_gt = collapse_to_layer_sparsity(showers, relevant_layers, voxels_per_layer)
    sparsity_corr_matrix_gt = compute_voxelwise_correlation_matrix(layer_sparsity_gt)

    fig_gt_sparsity_corr = plot_layer_energy_correlation_matrix(
        sparsity_corr_matrix_gt,
        cfg,
        layer_labels=[str(l) for l in relevant_layers],
        title="GT Layer Sparsity Correlation"
    )

    layer_sparsity_prior = collapse_to_layer_sparsity(showers_prior, relevant_layers, voxels_per_layer)
    sparsity_corr_matrix_prior = compute_voxelwise_correlation_matrix(layer_sparsity_prior)

    fig_prior_sparsity_corr = plot_layer_energy_correlation_matrix(
        sparsity_corr_matrix_prior,
        cfg,
        layer_labels=[str(l) for l in relevant_layers],
        title="Prior Layer Sparsity Correlation"
    )

    # Patch correlations and Frobenius metrics
    layer_labels = [str(l) for l in relevant_layers]

    # large patch matrix
    patch_coords = [(0, 0)] + [(r_, phi_) for r_ in [4, 10, 15] for phi_ in [0, 3, 6, 9]]
    corr = compute_patch_layer_correlation_matrix(showers, patch_coords, relevant_layers, cfg.data.r, cfg.data.phi)
    corr_prior = compute_patch_layer_correlation_matrix(showers_prior, patch_coords, relevant_layers, cfg.data.r, cfg.data.phi)
    fig_gt_patch = plot_patch_layer_correlation_matrix(corr, patch_coords, relevant_layers, title_prefix="GT")
    fig_prior_patch = plot_patch_layer_correlation_matrix(corr_prior, patch_coords, relevant_layers, title_prefix="Prior")
    frob_patch_layer = frobenius_distance(corr, corr_prior)
    logger.info(f"Epoch {epoch} - Frobenius Metric (Patch-Layer Correlation): {frob_patch_layer:.4f}")

    return (
            fig_target_corr, fig_sampled_corr, fig_gt_grid, fig_prior_grid,
            fig_frob_layerwise, fig_gt, fig_prior,
            gt_spars_corr, prior_spars_corr,
            fig_gt_sparsity, fig_prior_sparsity,
            fig_gt_sparsity_corr, fig_prior_sparsity_corr,
            fig_gt_patch, fig_prior_patch,
            { # dictionary of frob metrics:
                "frob_dist_voxel": frob_dist,
                "frob_dist_energy_corr_layer": frob_dist_energy_corr,
                "sparsity_frob_distance": sparsity_frob_distance,
                "frob_sparsity_dists_layer": frob_sparsity_dists,
                "frob_patch_layer": frob_patch_layer
            })
    
    
