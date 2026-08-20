#!/usr/bin/env python3
"""Export standalone fixed-energy shower-moment panels for Figures 3 and A.5.

Each layer/energy combination is a separate SVG/PDF asset.  Panel letters and
the shared model legend are supplied by LaTeX; Figure 3's particle/energy
label and the layer name are rendered inside each scientific panel below the
ATLAS stamp.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.shower_wd import calculate_wasserstein_distances


DEFAULT_INPUT_ROOT = Path("paper_plots")
DEFAULT_OUTPUT_DIR = Path("paper/Sim2Science_Workshop/figures")
OBSERVABLES = {
    "mean_eta": {
        "stem": "MeanEta",
        "latex": r"$\langle u_\eta \rangle$ [mm]",
        "figure_prefix": "fig3",
        "replot_root": Path("paper_plots/dedicated_mean_eta_replot"),
    },
    "width_eta": {
        "stem": "WidthEta",
        "latex": r"$\sigma_{u_\eta}$ [mm]",
        "figure_prefix": "figA5",
        "replot_root": Path("paper_plots/dedicated_width_eta_replot"),
    },
}

# These values are mirrored exactly by shower_panels.sty.
REFERENCE = ("data_ref", "Ground Truth (Geant4)", "black", "-")
MODEL_SERIES = (
    ("Recon", "AE reconstruction", "#e41a1c", "-"),
    ("GPU", "Classical RBM", "#16831a", "--"),
    ("QPU", "QPU RBM", "#f39c12", "-."),
)
LAYERS = ((1, "EMB1"), (2, "EMB2"))
ENERGIES = (5, 50, 250)
REFERENCE_LINEWIDTH = 1.7
MODEL_LINEWIDTH = 1.25
UNCERTAINTY_COLOR = "#808080"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--atlas-label",
        default="Internal",
        help="Status text in the ATLAS stamp (for example, Preliminary or Internal).",
    )
    parser.add_argument(
        "--observable", choices=tuple(OBSERVABLES), default="mean_eta"
    )
    parser.add_argument(
        "--energy",
        dest="energies",
        type=int,
        nargs=3,
        metavar=("LOW", "MID", "HIGH"),
        default=ENERGIES,
    )
    parser.add_argument("--without-ae", action="store_true")
    parser.add_argument(
        "--replot-output-root",
        type=Path,
        default=None,
        help="Optional directory for standard NPZ audit replots.",
    )
    return parser.parse_args()


def _clean(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    return values[np.isfinite(values)]


def _load_npz(path: Path, include_ae: bool) -> dict:
    with np.load(path, allow_pickle=False) as npz:
        if "bins" not in npz.files or "data_ref" not in npz.files:
            raise ValueError(f"{path} must contain 'bins' and 'data_ref'")
        bins = np.asarray(npz["bins"], dtype=float).reshape(-1)
        if bins.size < 2 or not np.all(np.isfinite(bins)) or not np.all(np.diff(bins) > 0):
            raise ValueError(f"{path} contains invalid histogram bin edges")

        arrays = {"data_ref": _clean(npz["data_ref"])}
        keys = ["GPU", "QPU"]
        if include_ae:
            keys.insert(0, "Recon")
        for key in keys:
            if key not in npz.files:
                raise ValueError(f"{path} is missing required series '{key}'")
            arrays[key] = _clean(npz[key])
    return {"bins": bins, "arrays": arrays}


def _histogram(values: np.ndarray, bins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    counts, _ = np.histogram(values, bins=bins)
    density, _ = np.histogram(values, bins=bins, density=True)
    return counts.astype(float), density.astype(float)


def _step(values: np.ndarray) -> np.ndarray:
    return np.append(values, values[-1])


def _observable_xlabel(observable: str, layer: int) -> str:
    if observable == "mean_eta":
        return rf"$\langle u_\eta \rangle_{{\mathrm{{EMB{layer}}}}}$ [mm]"
    if observable == "width_eta":
        return rf"$\sigma_{{u_\eta,\mathrm{{EMB{layer}}}}}$ [mm]"
    raise ValueError(f"Unknown observable: {observable}")


def _add_atlas_label(
    ax: plt.Axes,
    text: str,
    x_shift: float = 0.0,
    fontsize: float = 6.5,
    inset: list[float] | None = None,
    layer_label: str | None = None,
    energy_label: str | None = None,
    layer_label_x_shift: float = 0.0,
) -> None:
    if inset is None:
        inset = [0.49 + x_shift, 0.70, 0.48 - x_shift, 0.27]
    label_axis = ax.inset_axes(
        inset,
        zorder=10,
    )
    label_axis.set_facecolor("none")
    label_axis.patch.set_alpha(0.0)
    label_axis.axis("off")
    hep.atlas.label(
        text=text,
        data=False,
        rlabel="",
        ax=label_axis,
        loc=2,
        fontsize=fontsize,
    )
    if layer_label is not None:
        # Use the same normalized inset as the ATLAS stamp so the layer name
        # stays directly below it when LaTeX rescales the panel.
        # Match the left edge of the text rendered by ``hep.atlas.label`` in
        # the inset axes above it.
        label_x = inset[0] + 0.0321 + layer_label_x_shift
        layer_y = inset[1] + 0.025
        ax.text(
            label_x,
            layer_y,
            layer_label,
            transform=ax.transAxes,
            ha="left",
            va="baseline",
            fontsize=max(fontsize - 0.25, 5.5),
            fontweight="bold",
            color="black",
            zorder=11,
            clip_on=False,
        )
        if energy_label is not None:
            ax.text(
                label_x,
                layer_y - 0.068,
                energy_label,
                transform=ax.transAxes,
                ha="left",
                va="baseline",
                fontsize=max(fontsize - 0.75, 5.0),
                color="black",
                zorder=11,
                clip_on=False,
            )


def _panel_stem(
    observable: str,
    layer_index: int,
    layer_name: str,
    energy: int,
    energy_index: int,
) -> str:
    panel_letter = chr(ord("a") + layer_index * len(ENERGIES) + energy_index)
    suffix = "" if observable == "mean_eta" else "_width_eta"
    return (
        f"{OBSERVABLES[observable]['figure_prefix']}{panel_letter}_"
        f"{layer_name.lower()}_{energy}gev{suffix}"
    )


def plot_fixed_energy_panel(
    payload: dict,
    output_stem: Path,
    observable: str,
    layer: int,
    atlas_label: str = "Internal",
    atlas_x_shift: float = 0.0,
    layer_name: str | None = None,
    energy: int | None = None,
    layer_label_x_shift: float = 0.0,
) -> None:
    """Plot and save one layer/energy scientific panel."""

    bins = payload["bins"]
    arrays = payload["arrays"]
    reference_counts, reference_density = _histogram(arrays["data_ref"], bins)
    reference_error = np.zeros_like(reference_density)
    populated = reference_counts > 0
    reference_error[populated] = (
        reference_density[populated] / np.sqrt(reference_counts[populated])
    )

    # The six main-text panels must fit above the workshop page limit even
    # after LaTeX adds a local legend and subcaption to every panel.  The
    # appendix width panels have a little more vertical room.
    panel_height = 2.15 if observable == "mean_eta" else 2.5
    fig, ax = plt.subplots(figsize=(3.4, panel_height))
    ax.step(
        bins,
        _step(reference_density),
        color=REFERENCE[2],
        linestyle=REFERENCE[3],
        linewidth=REFERENCE_LINEWIDTH,
        where="post",
    )
    ax.fill_between(
        bins,
        _step(np.maximum(reference_density - reference_error, 0.0)),
        _step(reference_density + reference_error),
        color=UNCERTAINTY_COLOR,
        alpha=0.18,
        step="post",
        linewidth=0,
    )

    positive = [reference_density[reference_density > 0]]
    for key, _, color, linestyle in MODEL_SERIES:
        if key not in arrays:
            continue
        _, density = _histogram(arrays[key], bins)
        if np.any(density > 0):
            positive.append(density[density > 0])
        ax.step(
            bins,
            _step(density),
            color=color,
            linestyle=linestyle,
            linewidth=MODEL_LINEWIDTH,
            where="post",
        )

    positive_values = np.concatenate([values for values in positive if values.size])
    ymin = max(float(np.min(positive_values)) * 0.45, 1e-7)
    ymax = max(float(np.max(positive_values)) * 1.8, ymin * 10.0)
    ax.set_yscale("log")
    ax.set_xlim(float(bins[0]), float(bins[-1]))
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel(_observable_xlabel(observable, layer), fontsize=9)
    ax.set_ylabel("Probability density", fontsize=9)
    ax.tick_params(
        axis="both",
        which="both",
        direction="in",
        top=True,
        right=True,
        labelsize=8,
    )
    ax.grid(True, which="major", linestyle=":", linewidth=0.45, alpha=0.45)
    if observable == "width_eta":
        # Appendix A.6 uses the compact width panels.  Give the status stamp
        # a little more presence while moving its inset down from the frame.
        atlas_fontsize = 7.5
        # Nudge the enlarged stamp a little right while keeping its right
        # margin unchanged across all six fixed-energy panels.
        atlas_inset = [0.49 + atlas_x_shift, 0.64, 0.48 - atlas_x_shift, 0.30]
    else:
        atlas_fontsize = 6.5
        atlas_inset = None
    _add_atlas_label(
        ax,
        atlas_label,
        x_shift=atlas_x_shift,
        fontsize=atlas_fontsize,
        inset=atlas_inset,
        layer_label=layer_name or f"EMB{layer}",
        energy_label=(
            rf"$\gamma,\ E_{{\mathrm{{inc}}}}={energy}\;\mathrm{{GeV}}$"
            if observable == "mean_eta" and energy is not None
            else None
        ),
        layer_label_x_shift=layer_label_x_shift,
    )
    fig.subplots_adjust(left=0.19, right=0.98, bottom=0.19, top=0.97)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    # Keep the six Appendix A.6 panels on an identical canvas.  Tight
    # bounding boxes vary with the x-tick labels and can make identical
    # ATLAS stamps appear at different sizes after LaTeX rescales them.
    bbox_inches = None if observable == "width_eta" else "tight"
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches=bbox_inches)
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches=bbox_inches)
    plt.close(fig)


def _run_npz_replots(
    input_root: Path,
    energies: tuple[int, int, int],
    output_root: Path,
    property_stem: str,
) -> None:
    from utils.atlas_plots import replot_from_npz

    output_root.mkdir(parents=True, exist_ok=True)
    for energy in energies:
        replot_from_npz(
            save_dir=str(input_root / f"AtlasCustom{energy}GeV"),
            output_dir=str(output_root / f"AtlasCustom{energy}GeV"),
            yscale="log",
            xscale="linear",
            colors=[series[2] for series in MODEL_SERIES],
            linestyles=[series[3] for series in MODEL_SERIES],
            make_pdf=True,
            glob_pattern=f"Layer[12]_{property_stem}.npz",
            ratio_min_reference_count=5,
            reference_label=REFERENCE[1],
        )


def build_figure(
    input_root: Path,
    output_dir: Path,
    energies: tuple[int, int, int],
    include_ae: bool,
    replot_output_root: Path | None,
    observable: str,
    atlas_label: str = "Internal",
) -> dict:
    """Orchestrate the six independent fixed-energy plot functions."""

    if len(energies) != 3:
        raise ValueError("Exactly three dedicated energies are required")
    if observable not in OBSERVABLES:
        raise ValueError(f"Unknown observable: {observable}")
    property_stem = OBSERVABLES[observable]["stem"]

    payloads: dict[tuple[int, int], dict] = {}
    for energy in energies:
        for layer, _ in LAYERS:
            source = (
                input_root
                / f"AtlasCustom{energy}GeV"
                / f"Layer{layer}_{property_stem}.npz"
            )
            if not source.is_file():
                raise FileNotFoundError(source)
            payloads[(energy, layer)] = _load_npz(source, include_ae)

    if replot_output_root is not None:
        _run_npz_replots(input_root, energies, replot_output_root, property_stem)

    output_dir.mkdir(parents=True, exist_ok=True)
    panel_files = []
    wd_summaries = {}
    for layer_index, (layer, layer_name) in enumerate(LAYERS):
        for energy_index, energy in enumerate(energies):
            stem = _panel_stem(
                observable, layer_index, layer_name, energy, energy_index
            )
            plot_fixed_energy_panel(
                payloads[(energy, layer)],
                output_dir / stem,
                observable,
                layer,
                atlas_label,
                atlas_x_shift=(
                    0.03
                    if observable == "mean_eta" and layer == 2 and energy == 250
                    else 0.0
                ),
                layer_name=layer_name,
                energy=energy,
                layer_label_x_shift=(
                    0.04
                    if observable == "mean_eta" and layer == 2 and energy == 250
                    else 0.0
                ),
            )
            panel_files.append(f"{stem}.svg")
            wd_summaries[f"{energy} GeV/{layer_name}"] = (
                calculate_wasserstein_distances(
                    payloads[(energy, layer)]["arrays"], include_ae=include_ae
                )
            )

    metadata = {
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "panel_files": panel_files,
        "atlas_label": atlas_label,
        "subplot_count": len(panel_files),
        "energies_GeV": list(energies),
        "layers": {str(layer): name for layer, name in LAYERS},
        "observable": observable,
        "observable_stem": property_stem,
        "observable_latex": OBSERVABLES[observable]["latex"],
        "include_ae_reconstruction": include_ae,
        "wasserstein_distances": wd_summaries,
    }
    (output_dir / f"{OBSERVABLES[observable]['figure_prefix']}_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    return metadata


def main() -> None:
    args = _parse_args()
    metadata = build_figure(
        input_root=args.input_root,
        output_dir=args.output_dir,
        energies=tuple(args.energies),
        include_ae=not args.without_ae,
        replot_output_root=args.replot_output_root,
        observable=args.observable,
        atlas_label=args.atlas_label,
    )
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
