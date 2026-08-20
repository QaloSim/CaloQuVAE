#!/usr/bin/env python3
"""Make the compact two-panel beta-calibration figure used by the paper.

The original 21 April plot was saved as a raster image rather than with its
history arrays.  The values below are digitized from that plot; the reported
final beta and energy mismatch are retained in the annotation and caption.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / (
    "paper_plots/dwave/beta_search/"
    "beta_optimization_20260421_energy10000_clean.png"
)


def build_figure(
    output: Path = OUTPUT,
    atlas_label: str = "Preliminary",
) -> Path:
    epochs = np.arange(10)
    beta = np.array([1.142, 1.097, 1.059, 1.023, 1.000,
                     0.979, 0.958, 0.940, 0.927, 0.9249])
    # Digitized from the supplied raster plot.  Keep the reported final
    # mismatch in the annotation rather than recomputing it from rounded
    # digitized coordinates.
    rbm_energy = np.full(10, -286.44)
    qpu_energy = np.array([-279.21, -281.60, -282.56, -282.81, -283.90,
                           -284.28, -284.31, -284.48, -285.03, -285.84])

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
    })
    fig, (ax_beta, ax_energy) = plt.subplots(
        2, 1, figsize=(6.8, 5.9), sharex=True,
        gridspec_kw={"height_ratios": (1, 1.15), "hspace": 0.10},
    )

    purple = "#800080"
    target = "#4682B4"
    sampled = "#FF6347"
    grid = dict(color="#B8B8B8", linestyle="--", linewidth=0.7, alpha=0.55)

    ax_beta.plot(epochs, beta, color=purple, marker="o", markersize=4.5,
                 linewidth=1.8, label=r"$\beta$")
    ax_beta.set_ylabel(r"Inverse temperature $\beta$")
    ax_beta.set_title(r"Adaptive $\beta$ calibration (10 GeV)", pad=5)
    ax_beta.grid(True, **grid)
    ax_beta.legend(loc="upper right", frameon=True, handlelength=2.0)
    ax_beta.annotate(
        r"$\beta_{\rm eff}=0.9249$",
        xy=(epochs[-1], beta[-1]), xytext=(-8, 23),
        textcoords="offset points", ha="right", va="bottom",
        fontsize=8,
        arrowprops={"arrowstyle": "-|>", "lw": 0.8, "color": "black"},
    )
    ax_beta.set_ylim(0.915, 1.155)

    ax_energy.plot(epochs, rbm_energy, color=target, linestyle="--",
                   linewidth=1.8, label="RBM target")
    ax_energy.plot(epochs, qpu_energy, color=sampled, marker="x",
                   markersize=5.5, linewidth=1.8, label="QPU sampled")
    ax_energy.set_xlabel("Epoch")
    ax_energy.set_ylabel("Mean joint energy")
    ax_energy.set_title("Energy matching (10 GeV)", pad=5)
    ax_energy.grid(True, **grid)
    ax_energy.legend(loc="upper right", frameon=True, handlelength=2.0)
    ax_energy.set_xlim(-0.45, 9.45)
    ax_energy.set_xticks(np.arange(0, 10, 2))
    ax_energy.set_ylim(-286.8, -278.8)

    for ax in (ax_beta, ax_energy):
        ax.spines["top"].set_visible(True)
        ax.spines["right"].set_visible(True)
        ax.tick_params(direction="out", length=3)

    fig.subplots_adjust(left=0.13, right=0.985, top=0.89, bottom=0.10)
    atlas_ax = fig.add_axes([0.13, 0.935, 0.70, 0.03])
    atlas_ax.axis("off")
    hep.atlas.label(atlas_label, data=False, rlabel="",
                    ax=atlas_ax, loc=0)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--atlas-label", default="Preliminary")
    args = parser.parse_args()
    build_figure(args.output, atlas_label=args.atlas_label)


if __name__ == "__main__":
    main()
