#!/usr/bin/env python3
"""Build the four standalone scientific panels used in paper Figure 1.

The script consumes only the recovered NPZ files; it does not rerun the
simulator, AE, RBM, or QPU workflow.  Every panel is constructed natively at
its publication aspect ratio and exported as SVG plus a matching PDF.  Panel
letters, observable titles, the shared photon-energy title, and the model
legend belong to LaTeX rather than to these plot assets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import mplhep as hep
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.shower_wd import calculate_wasserstein_distances


DEFAULT_INPUT_DIR = Path("paper_plots/AtlasCustom2Uniform60k_recovered")
DEFAULT_OUTPUT_DIR = Path("paper/Sim2Science_Workshop/figures")

LAYER_NAMES = {
    0: "PreSamplerB",
    1: "EMB1",
    2: "EMB2",
    3: "EMB3",
    12: "TileBar0",
}

# These values are mirrored exactly by shower_panels.sty.
SERIES = (
    ("data_ref", "Ground Truth (Geant4)", "black", "-"),
    ("Recon", "AE reconstruction", "#e41a1c", "-"),
    ("GPU", "Classical RBM", "#16831a", "--"),
    ("QPU", "QPU RBM", "#f39c12", "-."),
)

UNCERTAINTY_COLOR = "#808080"
REFERENCE_LINEWIDTH = 1.7
MODEL_LINEWIDTH = 1.2
MINI_REFERENCE_LINEWIDTH = 1.15
MINI_MODEL_LINEWIDTH = 0.95


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--atlas-label",
        default="Internal",
        help="Status text in the ATLAS stamp (for example, Preliminary or Internal).",
    )
    parser.add_argument(
        "--ratio-min-count",
        type=int,
        default=5,
        help="Minimum Ground Truth (Geant4) events in a bin before drawing a ratio.",
    )
    return parser.parse_args()


def _add_atlas_label(
    ax: plt.Axes,
    text: str,
    compact: bool = False,
    placement: str = "upper-right",
    x_shift: float = 0.0,
    layer_label: str | None = None,
    layer_label_x_shift: float = 0.0,
) -> None:
    """Add an ATLAS status stamp and optional layer name inside the panel."""

    if compact:
        inset = (
            [0.04 + x_shift, 0.70, 0.54, 0.26]
            if placement == "upper-left"
            else [0.34 + x_shift, 0.70, 0.54, 0.26]
        )
        loc = 2
        fontsize = 5.5
    elif placement == "under-title":
        inset = [0.02, 0.67, 0.74, 0.27]
        loc = 2
        fontsize = 8.0
    else:
        inset = [0.44 + x_shift, 0.69, 0.54, 0.27]
        loc = 2
        fontsize = 8.0

    label_axis = ax.inset_axes(inset, zorder=10)
    label_axis.set_facecolor("none")
    label_axis.patch.set_alpha(0.0)
    label_axis.axis("off")
    hep.atlas.label(
        text=text,
        data=False,
        rlabel="",
        ax=label_axis,
        loc=loc,
        fontsize=fontsize,
    )
    if layer_label is not None:
        # Keep the layer name aligned with the ATLAS wordmark, below both the
        # wordmark and its status line.  The coordinates are expressed in the
        # parent axes so the annotation remains stable when the panel is
        # rescaled by LaTeX.
        ax.text(
            inset[0]
            + (0.06 if compact else 0.045)
            + layer_label_x_shift,
            inset[1] - 0.08 if compact else inset[1] + 0.025,
            layer_label,
            transform=ax.transAxes,
            ha="left",
            va="baseline",
            fontsize=6.5 if compact else 7.0,
            fontweight="bold",
            color="black",
            zorder=11,
            clip_on=False,
        )


def _clean(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    return values[np.isfinite(values)]


def _step(values: np.ndarray) -> np.ndarray:
    return np.append(values, values[-1])


def _load_series(npz: np.lib.npyio.NpzFile) -> list[tuple[str, np.ndarray, str, str]]:
    return [
        (label, _clean(npz[key]), color, linestyle)
        for key, label, color, linestyle in SERIES
    ]


def _draw_histogram_panel(
    fig: plt.Figure,
    input_dir: Path,
    stem: str,
    xlabel: str,
    ratio_min_count: int,
    xlim: tuple[float | None, float | None] | None = None,
    grid_left: float = 0.20,
    grid_right: float = 0.97,
    ylabel_pad: float = 4.0,
) -> tuple[plt.Axes, dict[str, float]]:
    """Draw a standalone log-density/ratio panel."""

    grid = GridSpec(
        2,
        1,
        figure=fig,
        height_ratios=(3, 1),
        hspace=0.04,
        left=grid_left,
        right=grid_right,
        bottom=0.17,
        top=1.0,
    )
    ax_main = fig.add_subplot(grid[0])
    ax_ratio = fig.add_subplot(grid[1], sharex=ax_main)

    with np.load(input_dir / f"{stem}.npz", allow_pickle=False) as npz:
        bins = np.asarray(npz["bins"], dtype=float)
        series = _load_series(npz)

    reference = series[0][1]
    counts_ref, _ = np.histogram(reference, bins=bins)
    density_ref, _ = np.histogram(reference, bins=bins, density=True)
    ref_err = np.zeros_like(density_ref)
    populated = counts_ref > 0
    ref_err[populated] = density_ref[populated] / np.sqrt(counts_ref[populated])

    ax_main.step(
        bins,
        _step(density_ref),
        color=series[0][2],
        linestyle=series[0][3],
        linewidth=REFERENCE_LINEWIDTH,
        where="post",
    )
    ax_main.fill_between(
        bins,
        _step(np.maximum(density_ref - ref_err, 0)),
        _step(density_ref + ref_err),
        color=UNCERTAINTY_COLOR,
        alpha=0.18,
        step="post",
        linewidth=0,
    )

    positive_densities = [density_ref[density_ref > 0]]
    for _, values, color, linestyle in series[1:]:
        density, _ = np.histogram(values, bins=bins, density=True)
        positive_densities.append(density[density > 0])
        ax_main.step(
            bins,
            _step(density),
            color=color,
            linestyle=linestyle,
            linewidth=MODEL_LINEWIDTH,
            where="post",
        )

        ratio = np.full_like(density, np.nan, dtype=float)
        valid = (counts_ref >= ratio_min_count) & (density_ref > 0)
        ratio[valid] = density[valid] / density_ref[valid]
        ax_ratio.step(
            bins,
            _step(ratio),
            color=color,
            linestyle=linestyle,
            linewidth=1.1,
            where="post",
        )

    ax_main.set_yscale("log")
    min_positive = min(
        (float(values.min()) for values in positive_densities if values.size),
        default=1e-6,
    )
    ax_main.set_ylim(bottom=max(min_positive * 0.5, 1e-8))
    ax_main.set_ylabel(
        "Probability density", fontsize=8, labelpad=ylabel_pad
    )
    ax_main.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
        labelsize=7,
    )
    ax_main.tick_params(labelbottom=False)

    ax_ratio.axhline(1, color="0.45", linestyle="--", linewidth=0.8)
    ax_ratio.set_ylim(0.5, 1.5)
    ax_ratio.set_ylabel("Ratio", fontsize=8)
    ax_ratio.set_xlabel(xlabel, fontsize=9)
    ax_ratio.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
        labelsize=7,
    )
    ax_ratio.grid(True, which="both", linestyle=":", linewidth=0.5, alpha=0.6)
    if xlim is not None:
        ax_main.set_xlim(left=xlim[0], right=xlim[1])

    wd_summary = calculate_wasserstein_distances(
        {
            "data_ref": series[0][1],
            **{label: values for label, values, _, _ in series[1:]},
        }
    )
    return ax_main, wd_summary


def _draw_layer_fractions(
    fig: plt.Figure,
    input_dir: Path,
    atlas_label: str,
) -> dict[str, dict[str, float]]:
    """Draw the five layer fractions as one standalone scientific panel."""

    grid = GridSpec(
        2,
        3,
        figure=fig,
        wspace=0.05,
        hspace=0.08,
        left=0.11,
        right=0.99,
        bottom=0.11,
        top=1.0,
    )
    wd_summaries: dict[str, dict[str, float]] = {}

    with (
        np.load(input_dir / "Etot.npz", allow_pickle=False) as total_npz,
        np.load(input_dir / "Etot_over_Einc.npz", allow_pickle=False) as response_npz,
    ):
        for idx, (layer, layer_name) in enumerate(LAYER_NAMES.items()):
            row, col = divmod(idx, 3)
            ax_main = fig.add_subplot(grid[row, col])
            _add_atlas_label(
                ax_main,
                text=atlas_label,
                compact=True,
                placement="upper-left" if layer == 2 else "upper-right",
                x_shift=0.03 if layer == 1 else -0.06 if layer == 2 else 0.0,
                layer_label=layer_name,
                layer_label_x_shift=0.14 if layer == 1 else 0.0,
            )

            layer_series = []
            with np.load(
                input_dir / f"Layer{layer}_Energy.npz", allow_pickle=False
            ) as layer_npz:
                for key, label, color, linestyle in SERIES:
                    total = np.asarray(total_npz[key], dtype=float)
                    response = np.asarray(response_npz[key], dtype=float)
                    energy = np.asarray(layer_npz[key], dtype=float)
                    valid = np.isfinite(total) & np.isfinite(response) & np.isfinite(energy)
                    incident = total[valid] / np.maximum(response[valid], 1e-12)
                    fraction = energy[valid] / np.maximum(incident, 1e-12)
                    layer_series.append((label, fraction, color, linestyle))

            wd_summaries[f"Layer{layer}_EnergyFraction"] = (
                calculate_wasserstein_distances(
                    {
                        "data_ref": layer_series[0][1],
                        **{
                            label: values
                            for label, values, _, _ in layer_series[1:]
                        },
                    }
                )
            )
            all_values = np.concatenate([values for _, values, _, _ in layer_series])
            xmin, xmax = float(np.min(all_values)), float(np.max(all_values))
            bins = np.linspace(xmin, xmax if xmax > xmin else xmin + 1.0, 70)

            reference = layer_series[0][1]
            counts_ref, _ = np.histogram(reference, bins=bins)
            density_ref, _ = np.histogram(reference, bins=bins, density=True)
            ref_err = np.zeros_like(density_ref)
            populated = counts_ref > 0
            ref_err[populated] = density_ref[populated] / np.sqrt(counts_ref[populated])
            ax_main.step(
                bins,
                _step(density_ref),
                color=layer_series[0][2],
                linewidth=MINI_REFERENCE_LINEWIDTH,
                where="post",
            )
            ax_main.fill_between(
                bins,
                _step(np.maximum(density_ref - ref_err, 0)),
                _step(density_ref + ref_err),
                color=UNCERTAINTY_COLOR,
                alpha=0.16,
                step="post",
                linewidth=0,
            )

            positive_densities = [density_ref[density_ref > 0]]
            for _, values, color, linestyle in layer_series[1:]:
                density, _ = np.histogram(values, bins=bins, density=True)
                positive_densities.append(density[density > 0])
                ax_main.step(
                    bins,
                    _step(density),
                    color=color,
                    linestyle=linestyle,
                    linewidth=MINI_MODEL_LINEWIDTH,
                    where="post",
                )

            min_positive = min(
                (float(values.min()) for values in positive_densities if values.size),
                default=1e-6,
            )
            ax_main.set_yscale("log")
            ax_main.set_ylim(bottom=max(min_positive * 0.5, 1e-7))
            ax_main.tick_params(
                axis="both",
                which="both",
                direction="in",
                top=True,
                right=True,
                labelsize=7,
                pad=1,
            )
            if col == 0:
                ax_main.set_ylabel(
                    "Probability\ndensity", fontsize=7, labelpad=2
                )
            else:
                ax_main.tick_params(labelleft=False)
            if row == 1:
                ax_main.set_xlabel(
                    r"$E_{\mathrm{layer}}/E_{\mathrm{inc}}$", fontsize=7, labelpad=2
                )

    return wd_summaries


def _save_panel(
    fig: plt.Figure,
    output_stem: Path,
    pad_inches: float = 0.0,
    tight: bool = True,
) -> None:
    """Save one panel as an SVG review artifact and a LaTeX-ready PDF."""

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    bbox_inches = "tight" if tight else None
    fig.savefig(
        output_stem.with_suffix(".svg"),
        bbox_inches=bbox_inches,
        pad_inches=pad_inches,
    )
    fig.savefig(
        output_stem.with_suffix(".pdf"),
        bbox_inches=bbox_inches,
        pad_inches=pad_inches,
    )
    plt.close(fig)


def plot_total_response(
    input_dir: Path,
    output_stem: Path,
    ratio_min_count: int = 5,
    atlas_label: str = "Internal",
) -> dict[str, float]:
    fig = plt.figure(figsize=(3.4, 3.0))
    ax, wd = _draw_histogram_panel(
        fig,
        input_dir,
        "Etot_over_Einc",
        r"$E_{\mathrm{tot}}/E_{\mathrm{inc}}$",
        ratio_min_count,
    )
    _add_atlas_label(ax, text=atlas_label, placement="under-title")
    _save_panel(fig, output_stem)
    return wd


def plot_layer_fractions(
    input_dir: Path,
    output_stem: Path,
    atlas_label: str = "Internal",
) -> dict[str, dict[str, float]]:
    # Match the height of the neighboring Figure 1(a) panel when this panel
    # is placed at 0.56\linewidth in the manuscript.
    fig = plt.figure(figsize=(4.7, 3.1))
    wd = _draw_layer_fractions(fig, input_dir, atlas_label)
    # The compact grid has no need for export padding; reclaim that space so
    # the two-row panel fills its LaTeX slot.
    _save_panel(fig, output_stem)
    return wd


def plot_emb1_centroid(
    input_dir: Path,
    output_stem: Path,
    ratio_min_count: int = 5,
    atlas_label: str = "Internal",
) -> dict[str, float]:
    fig = plt.figure(figsize=(3.4, 3.0))
    ax, wd = _draw_histogram_panel(
        fig,
        input_dir,
        "Layer1_MeanEta",
        r"$\langle u_\eta \rangle_{\mathrm{EMB1}}$ [mm]",
        ratio_min_count,
        xlim=(-300, 300),
        grid_left=0.12,
        grid_right=0.89,
        ylabel_pad=-1.0,
    )
    _add_atlas_label(ax, text=atlas_label, x_shift=0.08, layer_label="EMB1")
    # Keep c) and d) on the same fixed canvas. Tight-cropping each panel
    # independently makes c)'s longer negative tick labels enlarge its
    # bounding box, so LaTeX scales its actual axes down relative to d).
    _save_panel(fig, output_stem, tight=False)
    return wd


def plot_emb2_width(
    input_dir: Path,
    output_stem: Path,
    ratio_min_count: int = 5,
    atlas_label: str = "Internal",
) -> dict[str, float]:
    fig = plt.figure(figsize=(3.4, 3.0))
    ax, wd = _draw_histogram_panel(
        fig,
        input_dir,
        "Layer2_WidthEta",
        r"$\sigma_{u_\eta,\mathrm{EMB2}}$ [mm]",
        ratio_min_count,
        xlim=(None, 60),
        grid_left=0.12,
        grid_right=0.89,
        ylabel_pad=-1.0,
    )
    _add_atlas_label(ax, text=atlas_label, layer_label="EMB2")
    _save_panel(fig, output_stem, tight=False)
    return wd


def build_figure(
    input_dir: Path,
    output_dir: Path,
    ratio_min_count: int,
    atlas_label: str = "Internal",
    write_metadata: bool = True,
) -> dict:
    """Orchestrate the four independent Figure 1 plot functions."""

    if ratio_min_count < 1:
        raise ValueError("ratio-min-count must be at least 1")
    output_dir.mkdir(parents=True, exist_ok=True)

    wd_a = plot_total_response(
        input_dir, output_dir / "fig1a_total_response", ratio_min_count, atlas_label
    )
    wd_b = plot_layer_fractions(
        input_dir, output_dir / "fig1b_layer_fractions", atlas_label
    )
    wd_c = plot_emb1_centroid(
        input_dir, output_dir / "fig1c_emb1_centroid", ratio_min_count, atlas_label
    )
    wd_d = plot_emb2_width(
        input_dir, output_dir / "fig1d_emb2_width", ratio_min_count, atlas_label
    )

    counts = {}
    with np.load(input_dir / "Etot_over_Einc.npz", allow_pickle=False) as npz:
        for key, label, _, _ in SERIES:
            counts[label] = int(np.asarray(npz[key]).size)

    panel_files = [
        "fig1a_total_response.svg",
        "fig1b_layer_fractions.svg",
        "fig1c_emb1_centroid.svg",
        "fig1d_emb2_width.svg",
    ]
    metadata = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "panel_files": panel_files,
        "atlas_label": atlas_label,
        "ratio_min_fullsim_count": ratio_min_count,
        "ratio_panels": ["a", "c", "d"],
        "energy_aggregation": "E_inc ~ U(1,300) GeV",
        "series_counts": counts,
        "layer_names": LAYER_NAMES,
        "spatial_coordinates": {
            "u_eta": "r cos(alpha)",
            "u_phi": "r sin(alpha)",
        },
        "wasserstein_distances": {
            "Figure 1(a) Etot_over_Einc": wd_a,
            "Figure 1(b) layer-energy fractions": wd_b,
            "Figure 1(c) Layer1_MeanEta": wd_c,
            "Figure 1(d) Layer2_WidthEta": wd_d,
        },
    }
    if write_metadata:
        (output_dir / "fig1_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n"
        )
    return metadata


def main() -> None:
    args = _parse_args()
    metadata = build_figure(
        args.input_dir,
        args.output_dir,
        args.ratio_min_count,
        atlas_label=args.atlas_label,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
