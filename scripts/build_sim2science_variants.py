#!/usr/bin/env python3
"""Build the Preliminary and Internal Sim2Science paper variants.

The manuscript source is shared.  This script is the single entry point for
the figure side of the variant build: it copies non-status assets from the
authoritative paper figure directory and regenerates status-bearing figures
from the same saved NPZ/source artifacts for each requested ATLAS status.

Run from the repository root or from the paper directory::

    python3 scripts/build_sim2science_variants.py

The generated files live in ``paper/Sim2Science_Workshop/figures/{variant}``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import tempfile

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
PAPER_DIR = ROOT / "paper" / "Sim2Science_Workshop"
SOURCE_FIGURES = PAPER_DIR / "figures"
VARIANT_STATUSES = {
    "preliminary": "Preliminary",
    "internal": "Internal",
}

GRID_SOURCES = {
    # Standalone five-layer grid panels for manuscript Appendix Figure A.7.
    "figA6a_250_mean_eta": (
        ROOT / "paper_plots" / "AtlasCustom250GeV",
        "Grid_langle_u_etarangle_[mm]",
        "Layer*_MeanEta.npz",
    ),
    "figA6b_250_mean_phi": (
        ROOT / "paper_plots" / "AtlasCustom250GeV",
        "Grid_langle_u_phirangle_[mm]",
        "Layer*_MeanPhi.npz",
    ),
    "figA6c_250_width_eta": (
        ROOT / "paper_plots" / "AtlasCustom250GeV",
        "Grid_sigma_u_eta_[mm]",
        "Layer*_WidthEta.npz",
    ),
    "figA6d_250_width_phi": (
        ROOT / "paper_plots" / "AtlasCustom250GeV",
        "Grid_sigma_u_phi_[mm]",
        "Layer*_WidthPhi.npz",
    ),
    # Standalone five-layer grid panels for manuscript Appendix Figure A.8.
    "figA7a_uniform_width_eta": (
        ROOT / "paper_plots" / "AtlasCustom2Uniform60k",
        "Grid_sigma_u_eta_[mm]",
        "Layer*_WidthEta.npz",
    ),
    "figA7b_uniform_width_phi": (
        ROOT / "paper_plots" / "AtlasCustom2Uniform60k",
        "Grid_sigma_u_phi_[mm]",
        "Layer*_WidthPhi.npz",
    ),
}

GRID_REFERENCE_LABELS = {
    target: "Ground Truth (Geant4)" for target in GRID_SOURCES
}

# Appendix grid labels are slightly larger than the generic audit stamp. Move
# the default upper-right inset left and down, leaving margins below and to the
# right of the axes frame so the enlarged ATLAS wordmark/status line cannot
# clip at the edges.
GRID_ATLAS_LABEL_INSETS = {
    target: (0.42, 0.64, 0.55, 0.30) for target in GRID_SOURCES
}
GRID_ATLAS_LABEL_FONTSIZES = {
    target: 6.0 for target in GRID_SOURCES
}

GRID_ATLAS_LABEL_LAYER_INSETS = {
    # Layer IDs follow the NPZ geometry keys.  These restore the upper-left
    # exceptions visible in the last committed pre-refactor appendix grids;
    # all other layers retain their historical upper-right placement.
    "figA6a_250_mean_eta": {
        3: (0.03, 0.64, 0.62, 0.30),
        # The old composite had more horizontal room; in the standalone grid
        # the right-side stamp intersects the TileBar0 peak.
        12: (0.03, 0.64, 0.62, 0.30),
    },
    "figA6b_250_mean_phi": {
        1: (0.03, 0.64, 0.62, 0.30),
        # Likewise keep the stamp clear of the compact EMB3 bell curve.
        3: (0.03, 0.64, 0.62, 0.30),
        12: (0.03, 0.64, 0.62, 0.30),
    },
    # Figure A.7(d): move the EMB2 stamp a little farther right in the
    # fixed-250-GeV u_phi-width panel, as approved in the preview.
    "figA6d_250_width_phi": {
        2: (0.47, 0.64, 0.55, 0.30),
    },
}

# A few historical diagnostics are raster-only in this checkout.  They still
# get both statuses by redrawing the status stamp over the same shared source
# image.  Coordinates are normalized (x, y-from-top, width, height).
RASTER_STATUS_BOXES = {
    "atlas_20_mean_eta.png": (0.02, 0.00, 0.28, 0.09),
    "atlas_20_width_eta.png": (0.02, 0.00, 0.28, 0.09),
    "beta_calibration.png": (0.08, 0.00, 0.46, 0.08),
    "qpu_showers.png": (0.01, 0.00, 0.65, 0.11),
    "qpu_srt.png": (0.01, 0.00, 0.25, 0.07),
    "qpu_orbit_aggregation.png": (0.01, 0.00, 0.30, 0.07),
    "atlas_uniform_energy.png": (0.02, 0.00, 0.28, 0.09),
    "atlas_uniform_layer_energy.png": (0.02, 0.00, 0.28, 0.09),
    "validation_layer_energy.png": (0.02, 0.00, 0.28, 0.09),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variant",
        choices=("all", *VARIANT_STATUSES),
        default="all",
        help="Build one variant or both (default: all).",
    )
    return parser.parse_args()


def _copy_shared_assets(output_dir: Path) -> None:
    """Copy the current paper asset bundle before status-specific overrides."""

    output_dir.mkdir(parents=True, exist_ok=True)
    for source in SOURCE_FIGURES.iterdir():
        if source.is_file():
            shutil.copy2(source, output_dir / source.name)


def _reset_plot_style() -> None:
    """Return to the repository's default style between independent plots."""

    import matplotlib.pyplot as plt

    plt.close("all")
    plt.style.use("default")


def _render_npz_grids(status: str, output_dir: Path) -> set[str]:
    """Regenerate grid figures from saved NPZs when the lightweight path works."""

    _reset_plot_style()

    try:
        from utils.atlas_plots import replot_from_npz
    except Exception as exc:  # pragma: no cover - depends on optional user env.
        print(f"[variants] NPZ replot import unavailable: {exc}")
        return set()

    rendered: set[str] = set()
    with tempfile.TemporaryDirectory(prefix="sim2science_replot_") as temp_dir:
        temp_root = Path(temp_dir)
        for target_stem, (input_dir, source_stem, glob_pattern) in GRID_SOURCES.items():
            if not input_dir.is_dir():
                continue

            output_for_source = temp_root / target_stem
            try:
                replot_from_npz(
                    save_dir=str(input_dir),
                    output_dir=str(output_for_source),
                    yscale="log",
                    xscale="linear",
                    colors=["#e41a1c", "#16831a", "#f39c12"],
                    linestyles=["-", "--", "-."],
                    make_pdf=False,
                    glob_pattern=glob_pattern,
                    ratio_min_reference_count=5,
                    reference_label=GRID_REFERENCE_LABELS.get(
                        target_stem, "Ground Truth (Geant4)"
                    ),
                    atlas_label=status,
                    atlas_label_inset=GRID_ATLAS_LABEL_INSETS.get(target_stem),
                    atlas_label_fontsize=GRID_ATLAS_LABEL_FONTSIZES.get(
                        target_stem, 5.0
                    ),
                    atlas_label_layer_insets=GRID_ATLAS_LABEL_LAYER_INSETS.get(
                        target_stem, {}
                    ),
                    grid_show_legend=False,
                    grid_show_title=False,
                    grid_output_formats=("svg", "pdf"),
                    # The enlarged ATLAS stamp needs more horizontal room in
                    # the three-column A.7/A.8 grids.  Keep the displayed
                    # width controlled by LaTeX while widening each layer
                    # panel's aspect ratio.
                    grid_figure_size=(8.0, 5.2),
                )
            except Exception as exc:  # Fall back to the checked-in raster.
                print(f"[variants] NPZ replot failed for {target_stem}: {exc}")
                continue

            for extension in ("svg", "pdf"):
                rendered_path = output_for_source / f"{source_stem}.{extension}"
                if rendered_path.is_file():
                    target_name = f"{target_stem}.{extension}"
                    shutil.copy2(rendered_path, output_dir / target_name)
                    rendered.add(target_name)
    return rendered


def _render_label_overlay(
    source_path: Path,
    output_path: Path,
    status: str,
    box: tuple[float, float, float, float],
) -> None:
    """Replace one rasterized ATLAS stamp while preserving the base figure."""

    import matplotlib.pyplot as plt
    import mplhep as hep

    base = Image.open(source_path).convert("RGBA")
    width, height = base.size
    x, y, box_width, box_height = box
    pixel_box = (
        max(0, int(x * width)),
        max(0, int(y * height)),
        min(width, int((x + box_width) * width)),
        min(height, int((y + box_height) * height)),
    )
    base.paste((255, 255, 255, 255), pixel_box)

    # Keep the text scale close to the original figure family.  The shower
    # display was exported with a much larger title than the diagnostics.
    fontsize = {
        "qpu_showers.png": 26,
    }.get(source_path.name, 13)
    dpi = 100
    label_width = max(1, pixel_box[2] - pixel_box[0])
    label_height = max(1, pixel_box[3] - pixel_box[1])
    fig = plt.figure(
        figsize=(label_width / dpi, label_height / dpi),
        dpi=dpi,
        frameon=False,
    )
    fig.patch.set_alpha(0.0)
    # ``loc=0`` places the label just above its axis.  Render it on a
    # standalone transparent canvas, with that tiny axis at the bottom, then
    # paste the canvas into the masked header region.  This avoids clipping at
    # the top edge of the full figure for the older raster exports.
    label_axis = fig.add_axes([0.0, 0.0, 1.0, 0.01], facecolor="none")
    label_axis.axis("off")
    label_axis.patch.set_alpha(0.0)
    hep.atlas.label(
        status,
        data=False,
        rlabel="",
        ax=label_axis,
        loc=0,
        fontsize=fontsize,
    )
    fig.canvas.draw()
    overlay = Image.frombytes("RGBA", fig.canvas.get_width_height(), fig.canvas.buffer_rgba())
    plt.close(fig)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    base.alpha_composite(overlay, dest=(pixel_box[0], pixel_box[1]))
    base.save(output_path)


def _try_render_dwave_figure(
    source_glob: str,
    plot_function_name: str,
    status: str,
    output_path: Path,
) -> bool:
    """Re-render a D-Wave diagnostic when its optional analysis stack exists."""

    prefix, suffix = source_glob.split("*", 1)
    source_prefix = ROOT / prefix
    source_dir = source_prefix if source_prefix.is_dir() else source_prefix.parent
    source_files = sorted(source_dir.glob("*" + suffix))
    if not source_files:
        return False

    try:
        import matplotlib.pyplot as plt

        if plot_function_name in {
            "plot_srt_aggregation_scatter",
            "plot_srt_aggregation_correlations",
            "plot_orbit_aggregation_scatter",
            "plot_orbit_aggregation_correlations",
            "plot_srt_aggregation_tradeoff",
            "plot_orbit_aggregation_tradeoff",
        }:
            from scripts.plot_sim2science_qpu_aggregation import (
                load_saved_result,
                plot_orbit_aggregation_correlations,
                plot_orbit_aggregation_scatter,
                plot_orbit_aggregation_tradeoff,
                plot_srt_aggregation_correlations,
                plot_srt_aggregation_scatter,
                plot_srt_aggregation_tradeoff,
            )

            result = load_saved_result(source_files[-1])
            plotters = {
                "plot_srt_aggregation_scatter": plot_srt_aggregation_scatter,
                "plot_srt_aggregation_correlations": plot_srt_aggregation_correlations,
                "plot_orbit_aggregation_scatter": plot_orbit_aggregation_scatter,
                "plot_orbit_aggregation_correlations": plot_orbit_aggregation_correlations,
                "plot_srt_aggregation_tradeoff": plot_srt_aggregation_tradeoff,
                "plot_orbit_aggregation_tradeoff": plot_orbit_aggregation_tradeoff,
            }
            plotter = plotters[plot_function_name]
            plotter(
                result,
                atlas_label=status,
                save_path=str(output_path),
            )
        else:
            import torch
            from utils.dwave import plots

            result = torch.load(source_files[-1], map_location="cpu", weights_only=False)
            plot_function = getattr(plots, plot_function_name)
            plot_function(result, atlas_label=status, save_path=str(output_path))
        plt.close("all")
        required_outputs = [output_path]
        if output_path.suffix.lower() == ".svg":
            required_outputs.append(output_path.with_suffix(".pdf"))
        return all(path.is_file() for path in required_outputs)
    except Exception as exc:  # Optional D-Wave dependencies are not required.
        print(f"[variants] D-Wave replot unavailable for {output_path.name}: {exc}")
        return False


def _build_variant(variant: str, status: str) -> None:
    output_dir = SOURCE_FIGURES / variant
    _copy_shared_assets(output_dir)
    _reset_plot_style()

    # These builders all consume the same saved analysis artifacts; only the
    # status text and destination directory differ between variants.
    sys.path.insert(0, str(ROOT))
    from scripts.plot_sim2science_dedicated_width_eta import build_figure as build_dedicated
    from scripts.plot_sim2science_figure2 import build_figure as build_uniform_quality
    from scripts.plot_sim2science_latent_correlations import (
        build_figure as build_latent_correlations,
    )
    from scripts.replot_beta_10gev import build_figure as build_beta
    from utils.hamming_cliff_plots import replot_from_npz as replot_hamming

    _reset_plot_style()
    build_uniform_quality(
        ROOT / "paper_plots" / "AtlasCustom2Uniform60k_recovered",
        output_dir,
        ratio_min_count=5,
        atlas_label=status,
    )

    _reset_plot_style()
    build_latent_correlations(
        bundle_dir=ROOT / "paper/Sim2Science_Workshop/reproducibility",
        output_dir=output_dir,
        atlas_label=status,
    )

    for observable in ("mean_eta", "width_eta"):
        _reset_plot_style()
        build_dedicated(
            input_root=ROOT / "paper_plots",
            output_dir=output_dir,
            energies=(5, 50, 250),
            include_ae=True,
            replot_output_root=None,
            observable=observable,
            atlas_label=status,
        )

    _reset_plot_style()
    replot_hamming(
        npz_path=ROOT / "paper_plots" / "hamming_cliff" / "hamming_cliff.npz",
        output_path=output_dir / "hamming_cliff_atlas.png",
        atlas_label=status,
        dpi=300,
        close=True,
    )

    rendered_grid_names = _render_npz_grids(status, output_dir)

    _reset_plot_style()
    build_beta(output_dir / "beta_calibration.png", atlas_label=status)

    # Prefer the original structured result files where the optional D-Wave
    # stack is installed.  The scatter assets are regenerated as SVG so the
    # paper can retain their vector geometry.
    dwave_sources = {
        "qpu_srt_tradeoff_scatter.svg": (
            "paper/plots/dwave/srt_aggregation_comparison/*174954.pt",
            "plot_srt_aggregation_scatter",
        ),
        "qpu_srt_tradeoff_correlations.png": (
            "paper/plots/dwave/srt_aggregation_comparison/*174954.pt",
            "plot_srt_aggregation_correlations",
        ),
        "qpu_srt_tradeoff_correlations.svg": (
            "paper/plots/dwave/srt_aggregation_comparison/*174954.pt",
            "plot_srt_aggregation_correlations",
        ),
        "qpu_srt_tradeoff_correlations.pdf": (
            "paper/plots/dwave/srt_aggregation_comparison/*174954.pt",
            "plot_srt_aggregation_correlations",
        ),
        "qpu_orbit_aggregation_tradeoff_scatter.svg": (
            "paper/plots/dwave/orbit_sweep/*021102.pt",
            "plot_orbit_aggregation_scatter",
        ),
        "qpu_orbit_aggregation_tradeoff_correlations.png": (
            "paper/plots/dwave/orbit_sweep/*021102.pt",
            "plot_orbit_aggregation_correlations",
        ),
        "qpu_orbit_aggregation_tradeoff_correlations.svg": (
            "paper/plots/dwave/orbit_sweep/*021102.pt",
            "plot_orbit_aggregation_correlations",
        ),
        "qpu_orbit_aggregation_tradeoff_correlations.pdf": (
            "paper/plots/dwave/orbit_sweep/*021102.pt",
            "plot_orbit_aggregation_correlations",
        ),
    }
    for filename, (source_glob, function_name) in dwave_sources.items():
        _try_render_dwave_figure(
            source_glob, function_name, status, output_dir / filename
        )

    # Raster-only assets and historical figures are generated from the same
    # base image for both statuses.  NPZ-backed grids are skipped after a
    # successful replot above.
    for filename, box in RASTER_STATUS_BOXES.items():
        if filename in {
            "atlas_uniform_quality.png",
            "atlas_dedicated_mean_eta.png",
            "atlas_dedicated_width_eta.png",
            "hamming_cliff_atlas.png",
            "beta_calibration.png",
        }:
            continue
        source_path = SOURCE_FIGURES / filename
        if source_path.is_file():
            _render_label_overlay(source_path, output_dir / filename, status, box)

    manifest = {
        "variant": variant,
        "atlas_status_text": status,
        "rendered_from_shared_sources": True,
        "figure_directory": str(output_dir),
    }
    (output_dir / "variant_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[variants] Built {variant}: {output_dir}")


def main() -> None:
    args = _parse_args()
    variants = VARIANT_STATUSES if args.variant == "all" else {args.variant: VARIANT_STATUSES[args.variant]}
    for variant, status in variants.items():
        _build_variant(variant, status)


if __name__ == "__main__":
    main()
