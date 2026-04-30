"""
Persistence helpers for DWave experiment results.

Results are stored as Python dicts containing numpy arrays, torch tensors,
and plain Python scalars.  torch.save/load handles all of these transparently
via pickle under the hood.

Usage
-----
    from utils.dwave.results_io import save_result, load_result

    result = run_srt_aggregation_comparison(...)
    path = save_result(result, "srt_aggregation", output_dir="results/dwave")

    # Later, in a fresh session:
    result = load_result(path)
    plot_srt_aggregation_comparison(result, save_path="plots/srt_agg.pdf")
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path

import torch


def save_result(
    result: dict,
    name: str,
    output_dir: str = "results/dwave",
    timestamp: bool = True,
) -> str:
    """
    Saves an experiment result dict to disk.

    Parameters
    ----------
    result      : dict returned by any ``run_*`` experiment function.
    name        : short label used in the filename (e.g. ``"srt_aggregation"``).
    output_dir  : directory to write into (created if it does not exist).
    timestamp   : if True, appends ``_YYYYMMDD_HHMMSS`` to the filename.

    Returns
    -------
    str : full path to the saved file.
    """
    os.makedirs(output_dir, exist_ok=True)
    suffix = f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}" if timestamp else ""
    path = os.path.join(output_dir, f"{name}{suffix}.pt")
    torch.save(result, path)
    print(f"[results_io] Saved → {path}")
    return path


def load_result(path: str, map_location: str = "cpu") -> dict:
    """
    Loads an experiment result previously saved with :func:`save_result`.

    Parameters
    ----------
    path         : full path to the ``.pt`` file.
    map_location : where to load tensors (``"cpu"`` is always safe; move to
                   GPU afterwards if needed).

    Returns
    -------
    dict : the original result dict, with tensors on *map_location*.
    """
    path = str(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"[results_io] Result file not found: {path}")
    result = torch.load(path, map_location=map_location, weights_only=False)
    print(f"[results_io] Loaded ← {path}")
    return result


def list_results(output_dir: str = "results/dwave") -> list[str]:
    """Returns sorted list of ``.pt`` files in *output_dir*."""
    p = Path(output_dir)
    if not p.is_dir():
        return []
    return sorted(str(f) for f in p.glob("*.pt"))
