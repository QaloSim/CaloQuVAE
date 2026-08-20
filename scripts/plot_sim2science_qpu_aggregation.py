#!/usr/bin/env python3
"""Lightweight paper-facing plots for saved QPU aggregation diagnostics.

The saved aggregation artifacts are ``torch.save`` zip archives. This module
keeps the paper build able to replot them without importing the full optional
D-Wave analysis stack. When PyTorch is available, its loader is preferred;
otherwise the NumPy metadata and unused-tensor payloads are read directly.
"""

from __future__ import annotations

import os
import io
import pickle
from pathlib import Path
import zipfile

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import mplhep as hep
import numpy as np


def load_saved_result(path: str | os.PathLike[str]):
    """Load a saved aggregation result with or without PyTorch installed."""

    path = Path(path)

    try:
        import torch
    except Exception:
        torch = None

    if torch is not None:
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:
            # Older PyTorch versions do not accept ``weights_only``.
            return torch.load(path, map_location="cpu")
        except Exception:
            # The direct loader below handles the NumPy-only artifacts used by
            # the paper even when the optional torch runtime is incomplete.
            pass

    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as archive:
            payload_names = [
                name for name in archive.namelist() if name.endswith("data.pkl")
            ]
            if not payload_names:
                raise ValueError(f"No pickle payload found in {path}")
            return _load_pickle_payload(archive.read(payload_names[0]))

    with path.open("rb") as stream:
        return _TorchFreeUnpickler(stream).load()


class _TorchFreeUnpickler(pickle.Unpickler):
    """Load result metadata while discarding unused serialized torch tensors."""

    def find_class(self, module, name):
        if module == "torch._utils" and name in {
            "_rebuild_tensor",
            "_rebuild_tensor_v2",
            "_rebuild_tensor_v3",
        }:
            return lambda *args: None
        if module == "torch" and name.endswith("Storage"):
            return object
        return super().find_class(module, name)

    def persistent_load(self, saved_id):
        if isinstance(saved_id, tuple) and saved_id and saved_id[0] == "storage":
            return None
        raise pickle.UnpicklingError(f"Unsupported persistent id: {saved_id!r}")


def _load_pickle_payload(payload: bytes):
    return _TorchFreeUnpickler(io.BytesIO(payload)).load()


def _record_break_fraction(record: dict | None) -> float | None:
    if record is None:
        return None
    for key in ("break_frac", "chain_break_frac"):
        value = record.get(key)
        if value is not None:
            return float(value)
    return None


def _add_atlas_label(fig, text: str, y: float = 0.965) -> None:
    axes = fig.axes
    if not axes:
        return
    x0 = axes[0].get_position().x0
    label_axis = fig.add_axes([x0, y, 0.45, 0.01])
    label_axis.axis("off")
    hep.atlas.label(text, data=False, rlabel="", ax=label_axis)


def _as_matrix(value) -> np.ndarray | None:
    """Convert a saved matrix to a two-dimensional NumPy array."""

    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    matrix = np.asarray(value, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        return None
    return matrix


def _draw_correlation_matrix(ax, matrix, title: str, *, vmin: float, vmax: float):
    """Draw one label-free correlation heatmap."""

    if matrix is None:
        ax.text(
            0.5,
            0.5,
            "Correlation data\nunavailable",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=9,
            color="0.45",
        )
        ax.set_title(title, fontsize=11, fontweight="bold", pad=3)
        ax.set_axis_off()
        return None

    image = ax.imshow(
        matrix,
        cmap="seismic",
        vmin=vmin,
        vmax=vmax,
        origin="lower",
        interpolation="none",
        aspect="equal",
    )
    ax.set_title(title, fontsize=12, fontweight="bold", pad=5)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def _draw_correlation_row(
    fig,
    *,
    classical_matrix,
    aggregate_record: dict | None,
    best_record: dict | None,
    atlas_label: str | None = None,
    matrix_y: float = 0.08,
    matrix_size: float = 0.28,
    matrix_width: float | None = None,
    matrix_x: tuple[float, float, float] = (0.090, 0.375, 0.675),
    target_cbar_x: float = 0.040,
    delta_cbar_x: float = 0.960,
    show_x_axis: bool = False,
):
    """Draw a Target/ΔC_single,min/ΔC_agg strip on an existing figure.

    The strip owns its axes and colorbars.  It is intentionally independent of
    the scatter axes so the two can be composed by LaTeX at the size needed by
    the paper.  ``atlas_label`` is retained for the legacy combined renderer;
    the standalone matrix exports leave the status stamp to the scatter plot.
    """

    target = _as_matrix(classical_matrix)
    aggregate = _as_matrix((aggregate_record or {}).get("matrix"))
    best = _as_matrix((best_record or {}).get("matrix"))

    if matrix_width is None:
        matrix_width = matrix_size
    target_axis = fig.add_axes([matrix_x[0], matrix_y, matrix_width, matrix_size])
    best_axis = fig.add_axes([matrix_x[1], matrix_y, matrix_width, matrix_size])
    aggregate_axis = fig.add_axes([matrix_x[2], matrix_y, matrix_width, matrix_size])

    def error_text(record: dict | None) -> str:
        value = (record or {}).get("error_norm")
        return "n/a" if value is None else f"{float(value):.3f}"

    target_image = _draw_correlation_matrix(
        target_axis,
        target,
        "Target",
        vmin=-1.0,
        vmax=1.0,
    )
    best_image = _draw_correlation_matrix(
        best_axis,
        None if target is None or best is None else best - target,
        f"$\\Delta C_{{\\mathrm{{single,min}}}}$\nε={error_text(best_record)}",
        vmin=-0.5,
        vmax=0.5,
    )
    aggregate_image = _draw_correlation_matrix(
        aggregate_axis,
        None if target is None or aggregate is None else aggregate - target,
        f"$\\Delta C_{{\\mathrm{{agg}}}}$\nε={error_text(aggregate_record)}",
        vmin=-0.5,
        vmax=0.5,
    )

    if show_x_axis:
        # All three matrices use the same 141 binary-variable ordering.  Keep
        # the ticks sparse enough to remain readable when the strip is placed
        # beside the scatter plot in the paper.
        binary_index_ticks = (0, 50, 100, 140)
        for matrix_axis in (target_axis, aggregate_axis, best_axis):
            matrix_axis.set_xticks(binary_index_ticks)
            matrix_axis.minorticks_off()
            matrix_axis.tick_params(
                axis="x",
                labelsize=7.5,
                width=0.45,
                length=2.0,
                pad=2,
            )
            for tick_label in matrix_axis.get_xticklabels():
                tick_label.set_fontweight("normal")
        fig.supxlabel(
            "Binary variable index",
            x=0.5,
            y=0.045,
            fontsize=9,
            fontweight="normal",
        )

    delta_image = aggregate_image if aggregate_image is not None else best_image

    # The target scale is attached to the left of Target; the shared delta
    # scale is attached to the right of ΔC_agg.  Both bars span the matrix
    # height so they remain legible after the figure is reduced in the paper.
    def style_colorbar(
        colorbar,
        label: str,
        *,
        label_position: str = "right",
    ) -> None:
        colorbar.outline.set_linewidth(0.45)
        colorbar.outline.set_edgecolor("0.25")
        colorbar.ax.minorticks_off()
        colorbar.ax.tick_params(
            labelsize=7,
            width=0.45,
            length=1.8,
            colors="0.25",
            pad=1.5,
        )
        for tick_label in colorbar.ax.get_yticklabels():
            tick_label.set_fontweight("normal")
        colorbar.set_label(
            label,
            fontsize=8,
            fontweight="normal",
            labelpad=3,
            color="0.25",
        )
        colorbar.ax.yaxis.set_label_position(label_position)

    if target_image is not None:
        target_cbar = fig.add_axes([target_cbar_x, matrix_y, 0.012, matrix_size])
        target_bar = fig.colorbar(target_image, cax=target_cbar)
        style_colorbar(target_bar, "Correlation", label_position="left")
    if delta_image is not None:
        delta_cbar = fig.add_axes([delta_cbar_x, matrix_y, 0.012, matrix_size])
        delta_bar = fig.colorbar(delta_image, cax=delta_cbar)
        style_colorbar(delta_bar, r"Correlation Error ($\Delta C$)")

    if atlas_label is not None:
        # This path is only used by the legacy combined renderer.  Standalone
        # matrix figures intentionally omit the duplicate status stamp.
        label_axis = fig.add_axes(
            [0.62, matrix_y + matrix_size + 0.02, 0.34, 0.030], zorder=10
        )
        label_axis.set_facecolor("none")
        label_axis.patch.set_alpha(0.0)
        label_axis.axis("off")
        hep.atlas.label(
            atlas_label,
            data=False,
            rlabel="",
            ax=label_axis,
            loc=1,
            fontsize=11,
        )


def _configure_aggregation_style() -> None:
    """Apply the paper style before rendering one standalone diagnostic."""

    hep.style.use(hep.style.ATLAS)
    plt.rcParams.update(
        {
            "font.size": 16,
            "axes.linewidth": 2.5,
            "xtick.major.width": 2,
            "ytick.major.width": 2,
            "font.weight": "bold",
            "axes.labelweight": "bold",
            "figure.facecolor": "white",
        }
    )


def _draw_tradeoff_scatter(
    ax,
    individual_records,
    *,
    point_specs,
    individual_label: str,
    title: str,
    legend_ncol: int,
    legend_loc: str = "upper right",
    legend_bbox_to_anchor=None,
) -> None:
    """Draw the error-versus-chain-break scatter on ``ax``."""

    individual_records = list(individual_records or [])
    individual_points = []
    for record in individual_records:
        if record.get("error_norm") is None:
            continue
        break_frac = _record_break_fraction(record)
        if break_frac is not None:
            individual_points.append((float(record["error_norm"]), break_frac))

    highlighted_points = []
    for record, *_ in point_specs:
        if record is None or record.get("error_norm") is None:
            continue
        break_frac = _record_break_fraction(record)
        if break_frac is not None:
            highlighted_points.append((float(record["error_norm"]), break_frac))

    all_points = individual_points + highlighted_points
    if not all_points:
        raise ValueError("No error/chain-break records were available")

    if individual_points:
        individual_x, individual_y = zip(*individual_points)
        ax.scatter(
            individual_x,
            individual_y,
            s=34,
            color="gray",
            alpha=0.68,
            linewidths=0,
            label=individual_label,
            zorder=2,
        )

    for record, color, marker, label, size in point_specs:
        if record is None or record.get("error_norm") is None:
            continue
        break_frac = _record_break_fraction(record)
        if break_frac is None:
            continue
        error_norm = float(record["error_norm"])
        ax.scatter(
            [error_norm],
            [break_frac],
            s=size,
            color=color,
            marker=marker,
            edgecolors="black",
            linewidths=0.65,
            label=label,
            zorder=4,
        )

    ax.set_title(title, fontsize=20, fontweight="bold", pad=12)
    ax.set_xlabel(r"Error norm vs classical RBM ($\leftarrow$ lower is better)")
    ax.set_ylabel("Chain-break fraction")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=1))
    ax.grid(True, linestyle="--", linewidth=0.7, alpha=0.4, zorder=0)
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)

    x_values, y_values = zip(*all_points)
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    x_pad = max((x_max - x_min) * 0.12, 0.05)
    y_pad = max((y_max - y_min) * 0.22, 0.00025)
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(max(0.0, y_min - y_pad), y_max + y_pad)

    legend_kwargs = {
        "loc": legend_loc,
        "fontsize": 14,
        "frameon": True,
        "framealpha": 0.0,
        "borderpad": 1.0,
        "handlelength": 1.6,
        "handleheight": 1.4,
        "handletextpad": 0.9,
        "labelspacing": 0.8,
        "columnspacing": 1.0,
        "markerscale": 1.0,
        "ncol": legend_ncol,
    }
    if legend_bbox_to_anchor is not None:
        legend_kwargs["bbox_to_anchor"] = legend_bbox_to_anchor
    ax.legend(**legend_kwargs)


def _save_aggregation_figure(fig, save_path: str | os.PathLike[str] | None):
    """Save a standalone aggregation asset when requested and return it."""

    if save_path is not None:
        output_path = Path(save_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200)
        if output_path.suffix.lower() == ".svg":
            # Tectonic's graphicx backend cannot include raw SVG files. Keep
            # the SVG for editing/publication and emit a matching vector PDF
            # companion for the paper build.
            fig.savefig(output_path.with_suffix(".pdf"))
    return fig


def _plot_error_break_scatter(
    individual_records,
    *,
    best_record: dict | None,
    aggregate_record: dict | None,
    secondary_record: dict | None = None,
    individual_label: str,
    best_label: str,
    aggregate_label: str,
    secondary_label: str = "Secondary aggregate",
    legend_ncol: int = 1,
    legend_loc: str = "upper right",
    legend_bbox_to_anchor=None,
    title: str,
    atlas_label: str,
    save_path: str | os.PathLike[str] | None = None,
):
    """Render only the scatter component of an aggregation diagnostic."""

    _configure_aggregation_style()
    point_specs = (
        (best_record, "green", "*", best_label, 190),
        (aggregate_record, "orange", "D", aggregate_label, 150),
        (secondary_record, "purple", "s", secondary_label, 140),
    )

    fig = plt.figure(figsize=(8.5, 4.6))
    ax = fig.add_axes([0.20, 0.20, 0.77, 0.57])
    _draw_tradeoff_scatter(
        ax,
        individual_records,
        point_specs=point_specs,
        individual_label=individual_label,
        title=title,
        legend_ncol=legend_ncol,
        legend_loc=legend_loc,
        legend_bbox_to_anchor=legend_bbox_to_anchor,
    )
    _add_atlas_label(fig, atlas_label, y=0.965)
    return _save_aggregation_figure(fig, save_path)


def _plot_error_break_tradeoff(
    individual_records,
    *,
    classical_matrix,
    best_record: dict | None,
    aggregate_record: dict | None,
    secondary_record: dict | None = None,
    individual_label: str,
    best_label: str,
    aggregate_label: str,
    secondary_label: str = "Secondary aggregate",
    legend_ncol: int = 1,
    title: str,
    atlas_label: str,
    save_path: str | os.PathLike[str] | None = None,
):
    """Render the historical Matplotlib composite for backwards compatibility."""

    _configure_aggregation_style()

    point_specs = (
        (best_record, "green", "*", best_label, 190),
        (aggregate_record, "orange", "D", aggregate_label, 150),
        (secondary_record, "purple", "s", secondary_label, 140),
    )

    fig = plt.figure(figsize=(8.5, 9.2))
    # The scatter keeps its generous left margin for the vertical label, but
    # the matrix row below is laid out independently at nearly full width.
    ax = fig.add_axes([0.20, 0.49, 0.77, 0.34])
    _draw_tradeoff_scatter(
        ax,
        individual_records,
        point_specs=point_specs,
        individual_label=individual_label,
        title=title,
        legend_ncol=legend_ncol,
    )

    _draw_correlation_row(
        fig,
        classical_matrix=classical_matrix,
        aggregate_record=aggregate_record,
        best_record=best_record,
    )
    _add_atlas_label(fig, atlas_label, y=0.965)
    return _save_aggregation_figure(fig, save_path)


def _plot_error_break_correlations(
    *,
    classical_matrix,
    aggregate_record: dict | None,
    best_record: dict | None,
    save_path: str | os.PathLike[str] | None = None,
):
    """Render only the correlation-matrix strip of an aggregation diagnostic."""

    _configure_aggregation_style()
    fig = plt.figure(figsize=(8.5, 3.3))
    _draw_correlation_row(
        fig,
        classical_matrix=classical_matrix,
        aggregate_record=aggregate_record,
        best_record=best_record,
        matrix_y=0.18,
        matrix_size=0.64,
        matrix_width=0.235,
        matrix_x=(0.095, 0.380, 0.665),
        target_cbar_x=0.025,
        delta_cbar_x=0.940,
        show_x_axis=True,
    )
    return _save_aggregation_figure(fig, save_path)


def plot_srt_aggregation_scatter(
    comparison_results: dict,
    *,
    atlas_label: str = "Preliminary",
    save_path: str | os.PathLike[str] | None = None,
):
    """Render the standalone SRT error-versus-chain-break scatter."""

    per_batch = list(comparison_results.get("per_batch", []))
    n_batches = comparison_results.get("srt_batches", len(per_batch))
    return _plot_error_break_scatter(
        per_batch,
        best_record=comparison_results["best_srt"],
        aggregate_record=comparison_results["averaged"],
        individual_label="Individual SRTs",
        best_label="Best SRT",
        aggregate_label="SRT aggregate",
        legend_loc="center",
        legend_bbox_to_anchor=(0.35, 0.5),
        title=f"SRT aggregation ({n_batches} gauges)",
        atlas_label=atlas_label,
        save_path=save_path,
    )


def plot_srt_aggregation_correlations(
    comparison_results: dict,
    *,
    atlas_label: str = "Preliminary",
    save_path: str | os.PathLike[str] | None = None,
):
    """Render the standalone SRT target and residual correlation matrices."""

    # ``atlas_label`` is accepted for parity with the scatter renderer.  The
    # matrix strip is status-neutral because LaTeX places it beside the
    # status-bearing scatter asset.
    del atlas_label
    return _plot_error_break_correlations(
        classical_matrix=comparison_results.get("classical_matrix"),
        aggregate_record=comparison_results["averaged"],
        best_record=comparison_results["best_srt"],
        save_path=save_path,
    )


def plot_orbit_aggregation_scatter(
    sweep_results: dict,
    *,
    atlas_label: str = "Preliminary",
    save_path: str | os.PathLike[str] | None = None,
    include_srt_aggregate: bool = True,
):
    """Render the standalone embedding-orbit error scatter."""

    metrics = sweep_results.get("perm_metrics", [])
    best_run = sweep_results.get("best_orbit")
    aggregate_run = sweep_results.get("aggregated_orbit")
    srt_run = sweep_results.get("default_srt_aggregated")

    anneal_time = sweep_results.get("anneal_time")
    if anneal_time is None:
        title = "Embedding-orbit aggregation"
    else:
        title = f"Embedding-orbit aggregation (anneal time {anneal_time:g} μs)"

    return _plot_error_break_scatter(
        metrics,
        best_record=best_run,
        aggregate_record=aggregate_run,
        secondary_record=srt_run if include_srt_aggregate else None,
        individual_label="Individual orbits",
        best_label="Best orbit",
        aggregate_label="All-orbit aggregate",
        secondary_label="Best-orbit SRT aggregate",
        legend_ncol=1,
        legend_loc="center left",
        legend_bbox_to_anchor=(0.10, 0.62),
        title=title,
        atlas_label=atlas_label,
        save_path=save_path,
    )


def plot_orbit_aggregation_correlations(
    sweep_results: dict,
    *,
    atlas_label: str = "Preliminary",
    save_path: str | os.PathLike[str] | None = None,
):
    """Render the standalone orbit target and residual correlation matrices."""

    del atlas_label
    return _plot_error_break_correlations(
        classical_matrix=sweep_results.get("classical_matrix"),
        aggregate_record=sweep_results.get("aggregated_orbit"),
        best_record=sweep_results.get("best_orbit"),
        save_path=save_path,
    )


def plot_srt_aggregation_tradeoff(
    comparison_results: dict,
    *,
    atlas_label: str = "Preliminary",
    save_path: str | os.PathLike[str] | None = None,
):
    """Render the SRT scatter and its compact correlation-delta row."""

    per_batch = list(comparison_results.get("per_batch", []))
    n_batches = comparison_results.get("srt_batches", len(per_batch))
    return _plot_error_break_tradeoff(
        per_batch,
        classical_matrix=comparison_results.get("classical_matrix"),
        best_record=comparison_results["best_srt"],
        aggregate_record=comparison_results["averaged"],
        individual_label="Individual SRTs",
        best_label="Best SRT",
        aggregate_label="SRT aggregate",
        title=f"SRT aggregation ({n_batches} gauges)",
        atlas_label=atlas_label,
        save_path=save_path,
    )


def plot_orbit_aggregation_tradeoff(
    sweep_results: dict,
    *,
    atlas_label: str = "Preliminary",
    save_path: str | os.PathLike[str] | None = None,
    include_srt_aggregate: bool = True,
):
    """Render the orbit scatter and its compact correlation-delta row."""

    metrics = sweep_results.get("perm_metrics", [])
    best_run = sweep_results.get("best_orbit")
    aggregate_run = sweep_results.get("aggregated_orbit")
    srt_run = sweep_results.get("default_srt_aggregated")

    secondary_label = "Best-orbit SRT aggregate"

    anneal_time = sweep_results.get("anneal_time")
    if anneal_time is None:
        title = "Embedding-orbit aggregation"
    else:
        title = f"Embedding-orbit aggregation (anneal time {anneal_time:g} μs)"

    return _plot_error_break_tradeoff(
        metrics,
        classical_matrix=sweep_results.get("classical_matrix"),
        best_record=best_run,
        aggregate_record=aggregate_run,
        secondary_record=srt_run if include_srt_aggregate else None,
        individual_label="Individual orbits",
        best_label="Best orbit",
        aggregate_label="All-orbit aggregate",
        secondary_label=secondary_label,
        legend_ncol=2,
        title=title,
        atlas_label=atlas_label,
        save_path=save_path,
    )
