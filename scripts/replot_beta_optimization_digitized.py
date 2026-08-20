#!/usr/bin/env python3
"""Replot beta-optimization histories.

The original notebook did not persist the histogram arrays, so this script
uses the approximate CSV digitization in paper_plots/dwave/beta_search by
default.  It can also plot an exact scalar history recovered from one of the
saved ``beta_info.json`` files, without pretending that its histogram was
persisted.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import mplhep as hep
import numpy as np


DEFAULT_DATA_DIR = (
    Path(__file__).resolve().parents[1]
    / "paper_plots/dwave/beta_search"
)
DEFAULT_OUTPUT = DEFAULT_DATA_DIR / (
    "beta_optimization_20260430_020246_replotted.png"
)
DEFAULT_CONVERGENCE = DEFAULT_DATA_DIR / (
    "beta_optimization_20260430_020246_convergence.csv"
)
DEFAULT_HISTOGRAM = DEFAULT_DATA_DIR / (
    "beta_optimization_20260430_020246_histogram.csv"
)


def _add_atlas_label(fig: plt.Figure, ax: plt.Axes) -> None:
    """Match the ATLAS label placement used by the original plot."""
    dummy = fig.add_axes([ax.get_position().x0, 0.965, 0.45, 0.01])
    dummy.axis("off")
    hep.atlas.label("Preliminary", data=False, rlabel="", ax=dummy, loc=0)


def _load_convergence(path: Path) -> tuple[np.ndarray, ...]:
    """Load either a digitized CSV or a saved ``beta_info.json``."""
    if path.suffix.lower() == ".json":
        info = json.loads(path.read_text())
        beta = np.asarray(info["beta_hist"], dtype=float)
        rbm_energy = np.asarray(info["rbm_e_hist"], dtype=float)
        qpu_energy = np.asarray(info["qpu_e_hist"], dtype=float)
        return np.arange(len(beta), dtype=float), beta, rbm_energy, qpu_energy

    convergence = np.genfromtxt(
        path,
        delimiter=",",
        names=True,
        comments="#",
        skip_header=2,
    )
    return (
        convergence["epoch"],
        convergence["beta"],
        convergence["rbm_energy"],
        convergence["qpu_energy"],
    )


def replot(
    data_dir: Path,
    output: Path,
    convergence_file: Path | None = None,
    histogram_file: Path | None = None,
    show_histogram: bool = True,
    title: str | None = None,
) -> Path:
    convergence_path = convergence_file or data_dir / DEFAULT_CONVERGENCE.name
    epochs, beta, rbm_energy, qpu_energy = _load_convergence(convergence_path)

    histogram = None
    if show_histogram:
        histogram_path = histogram_file or data_dir / DEFAULT_HISTOGRAM.name
        histogram = np.genfromtxt(
            histogram_path,
            delimiter=",",
            names=True,
            comments="#",
            skip_header=2,
        )

    final_beta = float(beta[-1])
    final_delta = abs(float(qpu_energy[-1] - rbm_energy[-1]))

    n_rows = 3 if show_histogram else 2
    fig = plt.figure(figsize=(10, 14 if show_histogram else 8))
    height_ratios = [1, 1, 1.4] if show_histogram else [1, 1]
    gs = GridSpec(n_rows, 1, figure=fig, height_ratios=height_ratios, hspace=0.38)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2]) if show_histogram else None

    ax1.plot(epochs, beta, marker="o", linestyle="-", color="purple", label=r"$\beta$")
    ax1.set_ylabel(r"Inverse Temperature ($\beta$)", fontsize=11)
    ax1.set_title(title or r"$\beta$ Schedule", fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(fontsize=10)
    ax1.annotate(
        f"Final $\\beta$: {final_beta:.4f}",
        xy=(epochs[-1], final_beta),
        xytext=(0, 35),
        textcoords="offset points",
        ha="center",
        arrowprops=dict(facecolor="black", shrink=0.05, width=1, headwidth=6),
        fontsize=9,
    )

    ax2.plot(
        epochs,
        rbm_energy,
        label="RBM (Target)",
        color="steelblue",
        linestyle="--",
        linewidth=2,
    )
    ax2.plot(
        epochs,
        qpu_energy,
        label="QPU (Sampled)",
        color="tomato",
        marker="x",
        linestyle="-",
    )
    ax2.set_xlabel("Epoch", fontsize=11)
    ax2.set_ylabel("Mean Joint Energy", fontsize=11)
    ax2.set_title(f"Energy Convergence  (Final $\\Delta$: {final_delta:.4f})", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(fontsize=10)

    if histogram is not None:
        edges = np.r_[histogram["bin_left"][0], histogram["bin_right"]]
        ax3.stairs(
            histogram["rbm_density"],
            edges,
            baseline=0,
            fill=True,
            alpha=0.50,
            color="steelblue",
            label="RBM (Target)",
        )
        ax3.stairs(
            histogram["qpu_initial_density"],
            edges,
            baseline=None,
            fill=False,
            alpha=0.75,
            color="salmon",
            linewidth=2,
            linestyle="--",
            label="QPU (Initial, epoch 0)",
        )
        ax3.stairs(
            histogram["qpu_final_density"],
            edges,
            baseline=None,
            fill=False,
            alpha=0.75,
            color="tomato",
            linewidth=2,
            label=fr"QPU (Final, $\beta$={final_beta:.3f})",
        )
        ax3.set_xlabel("Joint Energy", fontsize=11)
        ax3.set_ylabel("Density", fontsize=11)
        ax3.set_title("Energy Distribution Shift", fontsize=12)
        ax3.legend(fontsize=10)
        ax3.grid(True, linestyle="--", alpha=0.4)

    _add_atlas_label(fig, ax1)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--convergence-file",
        type=Path,
        default=None,
        help="CSV or beta_info.json containing beta and energy histories.",
    )
    parser.add_argument("--histogram-file", type=Path, default=None)
    parser.add_argument(
        "--no-histogram",
        action="store_true",
        help="Make a clean two-panel plot when histogram data are unavailable.",
    )
    parser.add_argument("--title", default=None)
    args = parser.parse_args()
    replot(
        args.data_dir,
        args.output,
        convergence_file=args.convergence_file,
        histogram_file=args.histogram_file,
        show_histogram=not args.no_histogram,
        title=args.title,
    )


if __name__ == "__main__":
    main()
