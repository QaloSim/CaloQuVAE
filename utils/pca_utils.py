import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import seaborn as sns
import torch


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