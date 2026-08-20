#!/usr/bin/env python3
"""Render the four-panel learned-latent correlation audit for Sim2Science.

The paper bundle contains ``torch.save`` archives whose tensors were saved on
CUDA.  The small loader below reads the CPU-independent storage records
directly, so this paper-facing plot does not require importing PyTorch or
re-running the shower pipeline.
"""

from __future__ import annotations

import argparse
import io
import json
import pickle
import zipfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = ROOT / "paper/Sim2Science_Workshop/reproducibility"
CONDITION_BITS = 36
LEARNED_BITS = 141
VISIBLE_BITS = CONDITION_BITS + LEARNED_BITS


@dataclass(frozen=True)
class _StorageRef:
    dtype: np.dtype
    key: str
    nitems: int


@dataclass(frozen=True)
class _TensorRef:
    storage: _StorageRef
    storage_offset: int
    size: tuple[int, ...]
    stride: tuple[int, ...]


def _storage_dtype(storage_type) -> np.dtype:
    name = getattr(storage_type, "__name__", str(storage_type)).split(".")[-1]
    mapping = {
        "BoolStorage": np.bool_,
        "ByteStorage": np.uint8,
        "CharStorage": np.int8,
        "ShortStorage": np.int16,
        "IntStorage": np.int32,
        "LongStorage": np.int64,
        "HalfStorage": np.float16,
        "FloatStorage": np.float32,
        "DoubleStorage": np.float64,
    }
    if name not in mapping:
        raise TypeError(f"Unsupported torch storage type: {name}")
    return np.dtype(mapping[name])


def _rebuild_tensor(storage, storage_offset, size, stride, *unused):
    if not isinstance(storage, _StorageRef):
        raise TypeError(f"Unexpected storage record: {storage!r}")
    return _TensorRef(
        storage=storage,
        storage_offset=int(storage_offset),
        size=tuple(int(value) for value in size),
        stride=tuple(int(value) for value in stride),
    )


class _TorchArchiveUnpickler(pickle.Unpickler):
    """Unpickle tensor metadata while retaining references to raw storages."""

    def find_class(self, module, name):
        if module == "torch._utils":
            if name in {"_rebuild_tensor", "_rebuild_tensor_v2", "_rebuild_tensor_v3"}:
                return _rebuild_tensor
            if name == "_rebuild_parameter":
                return lambda data, *unused: data
        if module == "torch" and name.endswith("Storage"):
            # Storage classes only appear inside persistent IDs. Their names
            # are enough to recover the NumPy dtype in persistent_load().
            return name
        return super().find_class(module, name)

    def persistent_load(self, saved_id):
        if not isinstance(saved_id, tuple) or not saved_id or saved_id[0] != "storage":
            raise pickle.UnpicklingError(f"Unsupported persistent id: {saved_id!r}")
        _, storage_type, key, _location, nitems = saved_id
        return _StorageRef(
            dtype=_storage_dtype(storage_type),
            key=str(key),
            nitems=int(nitems),
        )


def _materialize(value, archive: zipfile.ZipFile, prefix: str, storage_cache: dict):
    if isinstance(value, _TensorRef):
        storage = value.storage
        if storage.key not in storage_cache:
            member = f"{prefix}data/{storage.key}"
            raw = archive.read(member)
            storage_cache[storage.key] = np.frombuffer(
                raw, dtype=storage.dtype, count=storage.nitems
            )
        base = storage_cache[storage.key]
        if not value.size:
            return np.empty(value.size, dtype=storage.dtype)
        view = np.ndarray(
            shape=value.size,
            dtype=storage.dtype,
            buffer=base,
            offset=value.storage_offset * storage.dtype.itemsize,
            strides=tuple(step * storage.dtype.itemsize for step in value.stride),
        )
        return view.copy()
    if isinstance(value, dict):
        return {
            key: _materialize(item, archive, prefix, storage_cache)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_materialize(item, archive, prefix, storage_cache) for item in value]
    if isinstance(value, tuple):
        return tuple(_materialize(item, archive, prefix, storage_cache) for item in value)
    return value


def load_torch_archive(path: str | Path):
    """Load tensors and simple dictionaries from an uncompressed torch archive."""

    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        pickle_member = next(
            name for name in archive.namelist() if name.endswith("/data.pkl")
        )
        prefix = pickle_member[: -len("data.pkl")]
        payload = archive.read(pickle_member)
        metadata = _TorchArchiveUnpickler(io.BytesIO(payload)).load()
        return _materialize(metadata, archive, prefix, {})


def _learned_correlation(samples: np.ndarray, name: str) -> np.ndarray:
    samples = np.asarray(samples)
    if samples.ndim != 2 or samples.shape[1] != VISIBLE_BITS:
        raise ValueError(
            f"{name} must have shape [N, {VISIBLE_BITS}], got {samples.shape}"
        )
    learned = samples[:, CONDITION_BITS:].astype(np.float64, copy=False)
    centered = learned - learned.mean(axis=0, keepdims=True)
    norms = np.sqrt(np.sum(centered * centered, axis=0))
    covariance = centered.T @ centered
    denominator = np.outer(norms, norms)
    with np.errstate(divide="ignore", invalid="ignore"):
        correlation = covariance / denominator
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(correlation, 0.0)
    return correlation


def load_uniform_latents(bundle_dir: str | Path = DEFAULT_BUNDLE):
    """Load the 50k posterior/GPU pair and the 60k QPU visible latents."""

    bundle_dir = Path(bundle_dir)
    posterior_path = bundle_dir / "latents/notebook_atlascustom2/post_samples.pt"
    gpu_path = bundle_dir / (
        "latents/notebook_atlascustom2/"
        "rbm_clamped_samples_train_data_final.pt"
    )
    qpu_paths = (
        bundle_dir / "latents/qpu/uniform_range/apr_22/processed_samples.pt",
        bundle_dir / "latents/qpu/uniform_range/apr_23/processed_samples.pt",
    )

    posterior = load_torch_archive(posterior_path)
    gpu = load_torch_archive(gpu_path)
    qpu_parts = []
    for path in qpu_paths:
        payload = load_torch_archive(path)
        if not isinstance(payload, dict) or "v" not in payload:
            raise KeyError(f"QPU archive has no visible latent key 'v': {path}")
        qpu_parts.append(np.asarray(payload["v"]))
    qpu = np.concatenate(qpu_parts, axis=0)
    return np.asarray(posterior), np.asarray(gpu), qpu


def _configure_style() -> None:
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


def _draw_matrix(ax, matrix: np.ndarray, title: str, *, vmin: float, vmax: float):
    image = ax.imshow(
        matrix,
        cmap="seismic",
        vmin=vmin,
        vmax=vmax,
        origin="lower",
        interpolation="none",
        aspect="equal",
    )
    ax.set_title(title, fontsize=11, fontweight="bold", pad=5)
    ax.set_xticks((0, 50, 100, 140))
    ax.set_yticks([])
    ax.minorticks_off()
    ax.tick_params(axis="x", labelsize=7.5, width=0.45, length=2.0, pad=2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def _style_colorbar(colorbar, label: str, *, label_position: str):
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
    colorbar.set_label(
        label,
        fontsize=8,
        fontweight="normal",
        labelpad=3,
        color="0.25",
    )
    colorbar.ax.yaxis.set_label_position(label_position)


def _add_atlas_label(fig, atlas_label: str) -> None:
    # ``loc=0`` anchors the stamp above this invisible axis.  Leaving the
    # axis at 0.925 puts the top of the glyph outside the raster canvas,
    # which is visible after the figure is included in the paper.
    label_axis = fig.add_axes([0.065, 0.90, 0.25, 0.035])
    label_axis.axis("off")
    hep.atlas.label(
        atlas_label,
        data=False,
        rlabel="",
        ax=label_axis,
        loc=0,
        fontsize=10,
    )


def build_figure(
    *,
    bundle_dir: str | Path = DEFAULT_BUNDLE,
    output_dir: str | Path,
    atlas_label: str = "Preliminary",
    output_stem: str = "latent_correlation_row",
) -> dict[str, object]:
    """Build PNG/PDF/SVG outputs and return the computed audit metadata."""

    _configure_style()
    posterior, gpu, qpu = load_uniform_latents(bundle_dir)
    posterior_corr = _learned_correlation(posterior, "posterior")
    gpu_corr = _learned_correlation(gpu, "GPU prior")
    qpu_corr = _learned_correlation(qpu, "QPU")
    correlation_error = qpu_corr - gpu_corr
    error_norm = float(np.linalg.norm(correlation_error))
    error_max_abs = float(np.max(np.abs(correlation_error)))
    error_scale = max(0.5, error_max_abs)

    fig = plt.figure(figsize=(14.2, 3.8))
    matrix_y = 0.18
    matrix_size = 0.64
    matrix_width = 0.17
    matrix_x = (0.070, 0.285, 0.500, 0.715)
    matrices = (
        (posterior_corr, "Posterior"),
        (gpu_corr, "RBM GPU"),
        (qpu_corr, "RBM QPU"),
        (
            correlation_error,
            r"$\Delta C$ ($C_{\mathrm{QPU}} - C_{\mathrm{GPU}}$)"
            + f"\n$\\epsilon_{{\\mathrm{{F}}}}={error_norm:.3f}$",
        ),
    )
    images = []
    for (matrix, title), x in zip(matrices, matrix_x):
        axis = fig.add_axes([x, matrix_y, matrix_width, matrix_size])
        if matrix is correlation_error:
            image = _draw_matrix(
                axis,
                matrix,
                title,
                vmin=-error_scale,
                vmax=error_scale,
            )
        else:
            image = _draw_matrix(axis, matrix, title, vmin=-1.0, vmax=1.0)
        images.append(image)

    correlation_cbar = fig.add_axes([0.035, matrix_y, 0.012, matrix_size])
    _style_colorbar(
        fig.colorbar(images[0], cax=correlation_cbar),
        "Correlation",
        label_position="left",
    )
    error_cbar = fig.add_axes([0.935, matrix_y, 0.012, matrix_size])
    _style_colorbar(
        fig.colorbar(images[-1], cax=error_cbar),
        r"Correlation Error ($\Delta C$)",
        label_position="right",
    )
    fig.supxlabel(
        "Binary variable index",
        x=0.50,
        y=0.045,
        fontsize=9,
        fontweight="normal",
    )
    _add_atlas_label(fig, atlas_label)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_base = output_dir / output_stem
    fig.savefig(output_base.with_suffix(".png"), dpi=300)
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)

    metadata = {
        "figure": output_stem,
        "atlas_label": atlas_label,
        "posterior_samples": int(posterior.shape[0]),
        "gpu_samples": int(gpu.shape[0]),
        "qpu_samples": int(qpu.shape[0]),
        "visible_bits": VISIBLE_BITS,
        "condition_bits_excluded": CONDITION_BITS,
        "learned_bits": LEARNED_BITS,
        "diagonal_zeroed": True,
        "correlation_error": "C_QPU - C_GPU",
        "correlation_error_frobenius": error_norm,
        "correlation_error_max_abs": error_max_abs,
        "correlation_error_color_scale": [-error_scale, error_scale],
        "qpu_sources": [
            "latents/qpu/uniform_range/apr_22/processed_samples.pt",
            "latents/qpu/uniform_range/apr_23/processed_samples.pt",
        ],
    }
    (output_dir / f"{output_stem}_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle-dir", type=Path, default=DEFAULT_BUNDLE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--atlas-label", default="Preliminary")
    args = parser.parse_args()
    build_figure(
        bundle_dir=args.bundle_dir,
        output_dir=args.output_dir,
        atlas_label=args.atlas_label,
    )


if __name__ == "__main__":
    main()
