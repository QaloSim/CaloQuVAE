"""Wasserstein-distance helpers for the paper's one-dimensional showers.

The paper plots are backed by ``.npz`` files containing raw, unbinned
observable samples.  Keeping the calculation here makes the figure builders,
the generic NPZ replotter, and the appendix evidence report use the same
definition and model-pair labels.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
from scipy.stats import wasserstein_distance


_SERIES_ALIASES = {
    "data_ref": "data_ref",
    "GEANT4": "data_ref",
    "ATLAS FullSim": "data_ref",
    "Recon": "Recon",
    "AE reconstruction": "Recon",
    "GPU": "GPU",
    "Classical RBM": "GPU",
    "QPU": "QPU",
    "QPU RBM": "QPU",
}


def clean_samples(values) -> np.ndarray:
    """Return finite one-dimensional samples as floating-point values."""

    array = np.asarray(values, dtype=float).reshape(-1)
    return array[np.isfinite(array)]


def _find_series(series: Mapping[str, object], canonical_key: str) -> np.ndarray | None:
    """Find a canonical series in an NPZ-like mapping, if it is present."""

    for key, values in series.items():
        if _SERIES_ALIASES.get(key, key) == canonical_key:
            return clean_samples(values)
    return None


def calculate_wasserstein_distances(
    series: Mapping[str, object],
    *,
    include_ae: bool = True,
) -> dict[str, float]:
    """Calculate raw one-dimensional WDs for the available model series.

    The reference is the ``data_ref``/GEANT4 series.  Distances are evaluated
    directly on finite samples, not on the plotted histogram bins.  Missing
    model series are skipped so the helper also works with reduced figures.
    Keys in the returned mapping are the labels used by the appendix table.
    """

    reference = _find_series(series, "data_ref")
    if reference is None or reference.size == 0:
        raise ValueError("A non-empty data_ref/GEANT4 series is required")

    models = {
        "AE--G4": _find_series(series, "Recon"),
        "Classical--G4": _find_series(series, "GPU"),
        "QPU--G4": _find_series(series, "QPU"),
    }
    if not include_ae:
        models.pop("AE--G4")

    distances: dict[str, float] = {}
    for label, values in models.items():
        if values is not None and values.size:
            distances[label] = float(wasserstein_distance(reference, values))

    classical = _find_series(series, "GPU")
    qpu = _find_series(series, "QPU")
    if classical is not None and qpu is not None and classical.size and qpu.size:
        distances["QPU--Classical"] = float(wasserstein_distance(classical, qpu))
    return distances


def calculate_npz_wasserstein_distances(
    path: str | Path,
    *,
    include_ae: bool = True,
) -> dict[str, float]:
    """Calculate WDs from one saved histogram artifact."""

    with np.load(path, allow_pickle=False) as npz:
        if "data_ref" not in npz.files and "GEANT4" not in npz.files:
            raise ValueError(f"{path} has no data_ref/GEANT4 reference series")
        return calculate_wasserstein_distances(npz, include_ae=include_ae)


def sample_counts(series: Mapping[str, object]) -> dict[str, int]:
    """Return finite sample counts using the paper-facing series names."""

    counts: dict[str, int] = {}
    for canonical_key, display_key in (
        ("data_ref", "GEANT4"),
        ("Recon", "AE"),
        ("GPU", "Classical"),
        ("QPU", "QPU"),
    ):
        values = _find_series(series, canonical_key)
        if values is not None:
            counts[display_key] = int(values.size)
    return counts
