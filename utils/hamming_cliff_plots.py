"""Replot the recovered Hamming Cliff curves without rerunning the sampler.

The companion NPZ contains the two one-dimensional quantities that were shown
in ``paper/Hamming Cliff.png``.  The original sample tensors are not available;
the arrays here are digitized from the rendered PNG and are intended for
restyling, relabeling, and presentation plots.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np


DEFAULT_DATA_PATH = (
    Path(__file__).resolve().parents[1]
    / "paper_plots"
    / "hamming_cliff"
    / "hamming_cliff.npz"
)


def load_from_npz(npz_path: str | Path = DEFAULT_DATA_PATH) -> dict[str, np.ndarray]:
    """Load the recovered Hamming Cliff arrays and metadata.

    The returned dictionary always contains ``indices``, ``binary_drift``, and
    ``gray_drift``.  Additional extraction metadata is returned when present
    in the NPZ.
    """

    with np.load(npz_path, allow_pickle=False) as data:
        required = ("indices", "binary_drift", "gray_drift")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"{npz_path} is missing required keys: {missing}")
        return {key: data[key].copy() for key in data.files}


def replot_from_npz(
    npz_path: str | Path = DEFAULT_DATA_PATH,
    output_path: str | Path | None = None,
    *,
    labels: Sequence[str] = ("Binary Code", "Gray Code"),
    colors: Sequence[str] = ("red", "green"),
    linestyles: Sequence[str] = ("-", "--"),
    line_alphas: Sequence[float] = (1.0, 1.0),
    linewidth: float = 3.0,
    fill: bool = True,
    fill_alphas: Sequence[float] = (0.08, 0.10),
    gray_hatch: str | None = None,
    atlas_style: bool = True,
    atlas_label: str | None = "Preliminary",
    atlas_label_loc: int = 0,
    atlas_label_y: float = 0.945,
    title: str | None = None,
    title_fontsize: float | None = 24.0,
    xlabel: str = "Latent Node Index",
    ylabel: str = r"Magnetization Drift $\Delta \langle \sigma_z \rangle$ (Zero is Ideal)",
    xlabel_coords: tuple[float, float] = (0.5, -0.12),
    ylabel_coords: tuple[float, float] = (-0.13, 1.2),
    figsize: tuple[float, float] = (20, 7),
    ylim: tuple[float, float] | None = None,
    xlim: tuple[float, float] | None = None,
    grid: bool = False,
    legend: bool = True,
    zero_line: bool = True,
    show: bool = False,
    close: bool = False,
    dpi: float | None = None,
):
    """Recreate the Hamming Cliff plot from the recovered arrays.

    Parameters are deliberately presentation-oriented: change labels, colors,
    linestyles, axes, and annotations without touching the extracted data or
    rerunning the RBM sampler.  The ATLAS mplhep style and the repository's
    poster settings are enabled by default; set ``atlas_style=False`` for the
    bare Matplotlib version.  The function returns the Matplotlib ``Figure``;
    use ``fig.axes[0]`` for any additional custom edits.

    ``output_path`` is optional.  When supplied, the figure is saved there and
    the parent directory is created automatically.
    """

    if len(labels) != 2:
        raise ValueError("labels must contain exactly two entries")
    if len(colors) != 2:
        raise ValueError("colors must contain exactly two entries")
    if len(linestyles) != 2:
        raise ValueError("linestyles must contain exactly two entries")
    if len(line_alphas) != 2:
        raise ValueError("line_alphas must contain exactly two entries")
    if len(fill_alphas) != 2:
        raise ValueError("fill_alphas must contain exactly two entries")

    if atlas_style:
        hep.style.use(hep.style.ATLAS)
        plt.rcParams.update({
            "font.size": 16,
            "axes.linewidth": 2.5,
            "xtick.major.width": 2,
            "ytick.major.width": 2,
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "figure.facecolor": "white",
        })

    data = load_from_npz(npz_path)
    indices = np.asarray(data["indices"])
    binary_drift = np.asarray(data["binary_drift"])
    gray_drift = np.asarray(data["gray_drift"])

    if not (indices.ndim == binary_drift.ndim == gray_drift.ndim == 1):
        raise ValueError("indices and drift arrays must be one-dimensional")
    if not (len(indices) == len(binary_drift) == len(gray_drift)):
        raise ValueError("indices and drift arrays must have equal lengths")

    if title is None:
        energy_pair = data.get("energy_pair")
        if energy_pair is not None and len(energy_pair) >= 2:
            e1, e2 = sorted(int(value) for value in energy_pair[:2])
            title = rf"Hamming Cliff Sensitivity: ${e1} \to {e2}$ MeV"
        else:
            title = "Hamming Cliff Sensitivity"

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    ax.step(
        indices,
        binary_drift,
        where="mid",
        color=colors[0],
        linestyle=linestyles[0],
        linewidth=linewidth,
        label=labels[0],
        alpha=line_alphas[0],
    )
    ax.step(
        indices,
        gray_drift,
        where="mid",
        color=colors[1],
        linestyle=linestyles[1],
        linewidth=linewidth,
        label=labels[1],
        alpha=line_alphas[1],
    )

    if fill:
        ax.fill_between(
            indices,
            0,
            binary_drift,
            step="mid",
            color=colors[0],
            alpha=fill_alphas[0],
        )
        ax.fill_between(
            indices,
            0,
            gray_drift,
            step="mid",
            color=colors[1],
            alpha=fill_alphas[1],
            hatch=gray_hatch,
            edgecolor=colors[1] if gray_hatch else None,
        )

    if zero_line:
        ax.axhline(0, color="black", linewidth=1.5, alpha=0.5)

    ax.set_title(title, pad=12, fontsize=title_fontsize)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.xaxis.set_label_coords(*xlabel_coords)
    ax.yaxis.set_label_coords(*ylabel_coords)
    ax.xaxis.label.set_horizontalalignment("center")
    ax.yaxis.label.set_verticalalignment("center")
    if len(indices) < 20:
        ax.set_xticks(indices)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if legend:
        ax.legend(loc="upper right", frameon=False)
    if grid:
        ax.grid(True, linestyle=":", alpha=0.6)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    # The long rotated y label needs more room than tight_layout allocates.
    # Keep a dedicated header band so the ATLAS mark and title do not collide.
    fig.subplots_adjust(left=0.17, right=0.985, bottom=0.18, top=0.84)

    if atlas_label:
        atlas_axis = fig.add_axes([ax.get_position().x0, atlas_label_y, 0.45, 0.01])
        atlas_axis.axis("off")
        hep.atlas.label(
            text=atlas_label,
            data=False,
            rlabel="",
            ax=atlas_axis,
            loc=atlas_label_loc,
        )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi)
    if show:
        plt.show()
    if close:
        plt.close(fig)
    return fig


__all__ = ["DEFAULT_DATA_PATH", "load_from_npz", "replot_from_npz"]
