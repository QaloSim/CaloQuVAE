import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import seaborn as sns
import torch
from typing import Tuple, List


def plot_scatter_labels(ax, data_proj, gen_data_proj, proj1, proj2, labels):
    ax.scatter(
        data_proj[:, proj1],
        data_proj[:, proj2],
        color="black",
        s=50,
        label=labels[0],
        zorder=0,
        alpha=0.3,
    )
    ax.scatter(
        gen_data_proj[:, proj1],
        gen_data_proj[:, proj2],
        color="red",
        label=labels[1],
        s=20,
        zorder=2,
        edgecolor="black",
        marker="o",
        alpha=1,
        linewidth=0.4,
    )


def plot_hist(
    ax, data_proj, gen_data_proj, color, proj, labels, orientation="vertical"
):
    ax.hist(
        data_proj[:, proj],
        bins=40,
        color="black",
        histtype="step",
        label=labels[0],
        zorder=0,
        density=True,
        orientation=orientation,
        lw=1,
    )
    ax.hist(
        gen_data_proj[:, proj],
        bins=40,
        color=color,
        histtype="step",
        label=labels[1],
        zorder=1,
        density=True,
        orientation=orientation,
        lw=1.5,
    )
    ax.axis("off")


def plot_PCA(data1, data2, labels, dir1=0, dir2=1):
    fig = plt.figure(dpi=100, figsize=(5, 5))
    gs = GridSpec(4, 4)

    ax_scatter = fig.add_subplot(gs[1:4, 0:3])
    ax_hist_x = fig.add_subplot(gs[0, 0:3])
    ax_hist_y = fig.add_subplot(gs[1:4, 3])

    plot_scatter_labels(ax_scatter, data1, data2, dir1, dir2, labels=labels)
    plot_hist(ax_hist_x, data1, data2, "red", dir1, labels=labels)
    plot_hist(
        ax_hist_y, data1, data2, "red", dir2, orientation="horizontal", labels=labels
    )

    ax_hist_x.legend(fontsize=12, bbox_to_anchor=(1, 1))
    h, l = ax_scatter.get_legend_handles_labels()
    ax_scatter.set_xlabel(f"PC{dir1}")
    ax_scatter.set_ylabel(f"PC{dir2}")


def get_ortho(mat: torch.Tensor):
    """Orthonormalize the column vectors of a matrix.

    Parameters
    ----------
    mat : torch.Tensor
        Matrix to orthonormalized. (a, b)

    Returns
    -------
    torch.Tensor
        Orthonormalized matrix. (a, b)
    """
    res = mat.clone()
    n, d = mat.shape

    u0 = mat[:, 0] / mat[:, 0].norm()
    res[:, 0] = u0
    for i in range(1, d):
        ui = mat[:, i]
        for j in range(i):
            ui -= (ui @ res[:, j]) * res[:, j]
        res[:, i] = ui / ui.norm()
    return res


def compute_U(
    M: torch.Tensor,
    weights: torch.Tensor,
    d: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Compute the first right eigenvector of the dataset.

    Parameters
    ----------
    M : torch.Tensor
        Dataset. (n_sample, n_visible)
    weights : torch.Tensor
        Weights of each sample (n_sample,)
    intrinsic_dimension : int
        Number of principal axis to compute.
    device : torch.device
        Device.
    dtype : torch.dtype
        Dtype

    Returns
    -------
    torch.Tensor
        Right eigenvectors. (n_dim, n_visible)
    """
    M = M * torch.sqrt(weights)
    num_samples, num_visibles = M.shape
    max_iter = 100
    err_threshold = 1e-15
    curr_v = (
        torch.rand(num_samples, d, device=device, dtype=dtype) * 2 - 1
    )
    u = torch.rand(num_visibles, d, device=device, dtype=dtype)
    curr_id_mat = (
        torch.rand(d, d, device=device, dtype=dtype)
        * 2
        - 1
    )
    for n in range(max_iter):
        v = curr_v.clone()
        curr_v = M @ u
        if num_samples < num_visibles:
            id_mat = (v.T @ curr_v) / num_samples
            curr_v = get_ortho(curr_v)
        curr_u = M.T @ curr_v
        if num_visibles <= num_samples:
            id_mat = (u.T @ curr_u) / num_samples
            curr_u = get_ortho(curr_u)
        u = curr_u.clone()
        if (id_mat - curr_id_mat).norm() < err_threshold:
            break
        curr_id_mat = id_mat.clone()
    u = get_ortho(u)
    return u



def load_data(real_path: str, gen_path: str, device: torch.device, incidence_path: str=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    print(f"Loading data...")
    real = torch.load(real_path, map_location=device).to(torch.float32)
    gen = torch.load(gen_path, map_location=device).to(torch.float32)
    if incidence_path is not None:
        print(f"Loading incident energy...")
        incidence_energy = torch.load(incidence_path, map_location=device).to(torch.float32)
        return real, gen, incidence_energy
    else:
        return real, gen

def load_data_features(real_path: str, gen_path: str, recon_features_path: str, classical_features_path: str, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    print(f"Loading data and features...")
    real = torch.load(real_path, map_location=device).to(torch.float32)
    gen = torch.load(gen_path, map_location=device).to(torch.float32)
    recon_features = torch.load(recon_features_path, map_location=device).to(torch.float32)
    classical_features = torch.load(classical_features_path, map_location=device).to(torch.float32)
    return real, gen, recon_features, classical_features

def run_pca_pipeline(
    data_real: torch.Tensor, 
    data_gen: torch.Tensor, 
    n_components: int = 4,
    center_projection: bool = False 
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Args:
        center_projection: 
            If True, projects (X - mean) @ U. (Standard PCA, centered at 0,0)
            If False, projects X @ U. (Your method, preserves origin offset)
    """
    device = data_real.device
    num_samples, num_visibles = data_real.shape
    
    print("Computing Mean and Centering Real Data...")
    mean_vec = data_real.mean(0)
    M = data_real - mean_vec
    
    # Weights assumed uniform based on your snippet
    weights = torch.ones(num_samples, 1, device=device)
    
    print("Computing Eigenvectors (U)...")
    U = compute_U(M, weights, d=n_components, device=device, dtype=torch.float32)
    
    print("Projecting Data...")
    scale = num_visibles**0.5
    
    if center_projection:
        # Standard PCA: Center both datasets
        proj_real = ((data_real - mean_vec) @ U) / scale
        proj_gen = ((data_gen - mean_vec) @ U) / scale
    else:
        # Your Method: Project raw data (Offset preserved)
        proj_real = (data_real @ U) / scale
        proj_gen = (data_gen @ U) / scale
        
    return proj_real.detach().cpu().numpy(), proj_gen.detach().cpu().numpy()




def calculate_variance_explained(data: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
    """
    Calculates the proportion of total variance in 'data' explained by each component in 'U'.
    """
    # Center the data
    mean_vec = data.mean(dim=0)
    centered_data = data - mean_vec
    
    # Total variance is the sum of variances of all individual features (Trace of Covariance)
    total_variance = centered_data.var(dim=0, unbiased=True).sum()
    
    # Project data onto the basis U
    projected_data = centered_data @ U
    
    # Variance captured by each principal component
    explained_variance_per_pc = projected_data.var(dim=0, unbiased=True)
    
    # Return as a ratio
    return explained_variance_per_pc / total_variance

def orchestrate_basis_evaluation(
    data_marginal: torch.Tensor,
    data_c_real: torch.Tensor,
    n_components: int = 4,
) -> dict:
    """
    Evaluates how well the marginalized PCs explain the conditional distribution.
    """
    device = data_marginal.device
    
    print("1. Extracting Marginal Basis (U_m)...")
    M_marg = data_marginal - data_marginal.mean(0)
    w_marg = torch.ones(M_marg.shape[0], 1, device=device)
    U_m = compute_U(M_marg, w_marg, d=n_components, device=device, dtype=torch.float32)
    
    print("2. Extracting Conditional Basis (U_c) for Ground Truth...")
    M_cond = data_c_real - data_c_real.mean(0)
    w_cond = torch.ones(M_cond.shape[0], 1, device=device)
    U_c = compute_U(M_cond, w_cond, d=n_components, device=device, dtype=torch.float32)
    
    print("3. Calculating Subspace Alignment (Cosine Similarity)...")
    # Absolute dot product between eigenvectors (since sign is arbitrary)
    cos_sim_matrix = torch.abs(U_m.T @ U_c).detach().cpu().numpy()
    
    print("4. Calculating Variance Explained on Conditional Data...")
    # How well does the native basis explain the conditional data? (The theoretical max)
    var_explained_native = calculate_variance_explained(data_c_real, U_c).detach().cpu().numpy()
    
    # How well does the marginalized basis explain the conditional data? (Your experiment)
    var_explained_marginal = calculate_variance_explained(data_c_real, U_m).detach().cpu().numpy()
    
    return {
        "cosine_similarity": cos_sim_matrix,
        "var_native": var_explained_native,
        "var_marginal": var_explained_marginal,
        "n_components": n_components
    }