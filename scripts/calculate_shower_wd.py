#!/usr/bin/env python3
"""Calculate WDs for every shower-distribution panel in the paper.

The script consumes the saved NPZ artifacts, so it does not rerun feature
extraction, the autoencoder, the RBM, or the QPU workflow.  It writes a
machine-readable evidence ledger containing the raw per-panel values used by
the compact table in Appendix A.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.shower_wd import (
    calculate_wasserstein_distances,
    sample_counts,
)


DEFAULT_UNIFORM_DIR = ROOT / "paper_plots" / "AtlasCustom2Uniform60k_recovered"
DEFAULT_FIXED_ROOT = ROOT / "paper_plots"
DEFAULT_OUTPUT = (
    ROOT / "paper" / "Sim2Science_Workshop" / "evidence" / "shower_wd.json"
)
DEFAULT_TABLE_OUTPUT = (
    ROOT / "paper" / "Sim2Science_Workshop" / "evidence" / "shower_wd_table.tex"
)

LAYERS = ((0, "PreSamplerB"), (1, "EMB1"), (2, "EMB2"), (3, "EMB3"), (12, "TileBar0"))
FIXED_ENERGIES = (5, 50, 250)

FAMILY_INFO = {
    "response": (r"$E_{\mathrm{tot}}/E_{\mathrm{inc}}$", "dimensionless"),
    "layer_fraction": (r"$E_{\mathrm{layer}}/E_{\mathrm{inc}}$", "dimensionless"),
    "mean_eta": (r"$\mu_{u_\eta}$", "mm"),
    "width_eta": (r"$\sigma_{u_\eta}$", "mm"),
    "mean_phi": (r"$\mu_{u_\phi}$", "mm"),
    "width_phi": (r"$\sigma_{u_\phi}$", "mm"),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uniform-dir", type=Path, default=DEFAULT_UNIFORM_DIR)
    parser.add_argument("--fixed-root", type=Path, default=DEFAULT_FIXED_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--table-output", type=Path, default=DEFAULT_TABLE_OUTPUT)
    return parser.parse_args()


def _relative(path: Path) -> str:
    """Represent evidence paths relative to the repository root."""

    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as npz:
        return {key: np.asarray(npz[key]) for key in npz.files if key != "bins"}


def _load_layer_fraction(root: Path, layer: int) -> tuple[dict[str, np.ndarray], list[str]]:
    """Reconstruct the distributions shown in Figure 1(b)."""

    total = _load_npz(root / "Etot.npz")
    response = _load_npz(root / "Etot_over_Einc.npz")
    layer_energy = _load_npz(root / f"Layer{layer}_Energy.npz")
    arrays: dict[str, np.ndarray] = {}
    for key in ("data_ref", "Recon", "GPU", "QPU"):
        total_values = np.asarray(total[key], dtype=float)
        response_values = np.asarray(response[key], dtype=float)
        layer_values = np.asarray(layer_energy[key], dtype=float)
        valid = (
            np.isfinite(total_values)
            & np.isfinite(response_values)
            & np.isfinite(layer_values)
        )
        incident = total_values[valid] / np.maximum(response_values[valid], 1e-12)
        arrays[key] = layer_values[valid] / np.maximum(incident, 1e-12)
    return arrays, [
        _relative(root / "Etot.npz"),
        _relative(root / "Etot_over_Einc.npz"),
        _relative(root / f"Layer{layer}_Energy.npz"),
    ]


def _entry(
    *,
    entry_id: str,
    family: str,
    observable: str,
    figures: list[str],
    source: str | list[str],
    arrays: dict[str, np.ndarray],
) -> dict:
    if family not in FAMILY_INFO:
        raise ValueError(f"Unknown observable family: {family}")
    return {
        "id": entry_id,
        "family": family,
        "observable": observable,
        "unit": FAMILY_INFO[family][1],
        "figures": figures,
        "source": source,
        "sample_counts": sample_counts(arrays),
        "wd": calculate_wasserstein_distances(arrays),
    }


def _add_entry(
    entries: dict[str, dict],
    *,
    entry_id: str,
    family: str,
    observable: str,
    figures: list[str],
    source: str | list[str],
    arrays: dict[str, np.ndarray],
) -> None:
    """Add an entry, merging figure references when a panel is reused."""

    if entry_id not in entries:
        entries[entry_id] = _entry(
            entry_id=entry_id,
            family=family,
            observable=observable,
            figures=list(figures),
            source=source,
            arrays=arrays,
        )
        return

    existing = entries[entry_id]
    existing["figures"] = list(dict.fromkeys(existing["figures"] + figures))


def _add_npz_entry(
    entries: dict[str, dict],
    *,
    path: Path,
    family: str,
    observable: str,
    figures: list[str],
) -> None:
    _add_entry(
        entries,
        entry_id=_relative(path),
        family=family,
        observable=observable,
        figures=figures,
        source=_relative(path),
        arrays=_load_npz(path),
    )


def _family_for_stem(stem: str) -> str:
    suffix = stem.split("_", 1)[1]
    return {
        "MeanEta": "mean_eta",
        "WidthEta": "width_eta",
        "MeanPhi": "mean_phi",
        "WidthPhi": "width_phi",
    }[suffix]


def _observable_for_stem(stem: str) -> str:
    layer, suffix = stem.split("_", 1)
    layer_id = int(layer.removeprefix("Layer"))
    layer_name = dict(LAYERS)[layer_id]
    family = _family_for_stem(stem)
    return f"{FAMILY_INFO[family][0]} ({layer_name})"


def collect_entries(uniform_dir: Path, fixed_root: Path) -> list[dict]:
    """Collect exactly the distributions used by the shower figures."""

    entries: dict[str, dict] = {}

    _add_npz_entry(
        entries,
        path=uniform_dir / "Etot_over_Einc.npz",
        family="response",
        observable=FAMILY_INFO["response"][0],
        figures=["Figure 1(a)"],
    )

    for layer, layer_name in LAYERS:
        fractions, sources = _load_layer_fraction(uniform_dir, layer)
        _add_entry(
            entries,
            entry_id=f"uniform/Layer{layer}_EnergyFraction",
            family="layer_fraction",
            observable=fr"$E_{{{layer_name}}}/E_{{\mathrm{{inc}}}}$",
            figures=["Figure 1(b)"],
            source=sources,
            arrays=fractions,
        )

    _add_npz_entry(
        entries,
        path=uniform_dir / "Layer1_MeanEta.npz",
        family="mean_eta",
        observable=FAMILY_INFO["mean_eta"][0] + " (EMB1)",
        figures=["Figure 1(c)"],
    )
    _add_npz_entry(
        entries,
        path=uniform_dir / "Layer2_WidthEta.npz",
        family="width_eta",
        observable=FAMILY_INFO["width_eta"][0] + " (EMB2)",
        figures=["Figure 1(d)", "Appendix Fig. A7"],
    )

    # Appendix Fig. A7 contains both uniform-scan width grids.
    for layer, layer_name in LAYERS:
        for suffix, family in (("WidthEta", "width_eta"), ("WidthPhi", "width_phi")):
            _add_npz_entry(
                entries,
                path=uniform_dir / f"Layer{layer}_{suffix}.npz",
                family=family,
                observable=f"{FAMILY_INFO[family][0]} ({layer_name})",
                figures=["Appendix Fig. A7"],
            )

    # Main Figure 3 and Appendix Fig. A5 use the EMB1/EMB2 fixed-energy
    # distributions.  Appendix Fig. A6 adds all four spatial families at
    # 250 GeV across all five retained layers.
    for energy in FIXED_ENERGIES:
        fixed_dir = fixed_root / f"AtlasCustom{energy}GeV"
        for layer, layer_name in ((1, "EMB1"), (2, "EMB2")):
            for suffix, family, figure in (
                ("MeanEta", "mean_eta", "Figure 3"),
                ("WidthEta", "width_eta", "Appendix Fig. A5"),
            ):
                _add_npz_entry(
                    entries,
                    path=fixed_dir / f"Layer{layer}_{suffix}.npz",
                    family=family,
                    observable=f"{FAMILY_INFO[family][0]} ({layer_name}, {energy} GeV)",
                    figures=[figure],
                )

    fixed_250_dir = fixed_root / "AtlasCustom250GeV"
    for layer, layer_name in LAYERS:
        for suffix, family in (
            ("MeanEta", "mean_eta"),
            ("MeanPhi", "mean_phi"),
            ("WidthEta", "width_eta"),
            ("WidthPhi", "width_phi"),
        ):
            _add_npz_entry(
                entries,
                path=fixed_250_dir / f"Layer{layer}_{suffix}.npz",
                family=family,
                observable=f"{FAMILY_INFO[family][0]} ({layer_name}, 250 GeV)",
                figures=["Appendix Fig. A6"],
            )

    return [entries[key] for key in sorted(entries)]


def summarize_by_family(entries: list[dict]) -> dict[str, dict]:
    """Summarize each native-unit observable family without mixing units."""

    summary: dict[str, dict] = {}
    for family, (label, unit) in FAMILY_INFO.items():
        family_entries = [entry for entry in entries if entry["family"] == family]
        if not family_entries:
            continue
        values = {
            distance_label: np.asarray(
                [entry["wd"][distance_label] for entry in family_entries],
                dtype=float,
            )
            for distance_label in ("Classical--G4", "QPU--G4", "QPU--Classical")
        }
        summary[family] = {
            "observable": label,
            "unit": unit,
            "entry_count": len(family_entries),
            "statistic": "median [minimum, maximum]",
        }
        for distance_label, family_values in values.items():
            summary[family][distance_label] = {
                "median": float(np.median(family_values)),
                "minimum": float(np.min(family_values)),
                "maximum": float(np.max(family_values)),
            }
    return summary


TABLE_GROUPS = {
    0: "Main-body EMB1/EMB2 panels (5, 50, and 250 GeV)",
    1: "Uniform scan: response and layer-wise energy fractions",
    2: r"Appendix~A.6: remaining layer-wise spatial panels at 250 GeV",
    3: "Uniform scan: layer-wise spatial panels",
}


def _table_group(entry: dict) -> int:
    figures = set(entry["figures"])
    if "Figure 3" in figures or "Appendix Fig. A5" in figures:
        return 0
    if "Figure 1(a)" in figures or "Figure 1(b)" in figures:
        return 1
    if "Appendix Fig. A6" in figures:
        return 2
    if "Figure 1(c)" in figures or "Figure 1(d)" in figures:
        return 3
    if "Appendix Fig. A7" in figures:
        return 3
    return 4


def _table_layer_order(entry: dict) -> int:
    text = f"{entry['id']} {entry['observable']}"
    match = re.search(r"Layer(\d+)", text)
    if match:
        layer = int(match.group(1))
    else:
        layer = next(
            (layer_id for layer_id, layer_name in LAYERS if layer_name in text),
            99,
        )
    # Put the two electromagnetic barrel layers first in the detailed table.
    return {1: 0, 2: 1, 0: 2, 3: 3, 12: 4}.get(layer, 99)


def _table_energy_order(entry: dict) -> int:
    match = re.search(r"(\d+)\s*GeV", f"{entry['id']} {entry['observable']}")
    return int(match.group(1)) if match else -1


def _table_family_order(entry: dict) -> int:
    return {
        "response": 0,
        "layer_fraction": 1,
        "mean_eta": 2,
        "width_eta": 3,
        "mean_phi": 4,
        "width_phi": 5,
    }[entry["family"]]


def table_entries(entries: list[dict]) -> list[dict]:
    """Order one explicit table row for every displayed distribution."""

    return sorted(
        entries,
        key=lambda entry: (
            _table_group(entry),
            _table_energy_order(entry),
            _table_layer_order(entry),
            _table_family_order(entry),
            entry["id"],
        ),
    )


def _format_wd(value: float) -> str:
    """Format a WD compactly while retaining three significant digits."""

    value = float(value)
    if value == 0.0:
        return "0"
    exponent = int(np.floor(np.log10(abs(value))))
    if -2 <= exponent < 3:
        return f"{value:.3g}"
    mantissa = value / (10.0**exponent)
    return rf"{mantissa:.3g}\times 10^{{{exponent}}}"


def _table_label(entry: dict) -> str:
    label = entry["observable"]
    # EMB1/EMB2 are the main-body comparison, so make them visually prominent
    # in both the text-mode layer labels and the math-mode energy-fraction rows.
    label = label.replace("(EMB1", r"(\textbf{EMB1}")
    label = label.replace("(EMB2", r"(\textbf{EMB2}")
    label = label.replace("_{EMB1}", r"_{\mathbf{EMB1}}")
    label = label.replace("_{EMB2}", r"_{\mathbf{EMB2}}")
    if entry["unit"] == "mm":
        label += r" [$\mathrm{mm}$]"
    return label


def render_wd_table(entries: list[dict]) -> str:
    """Render the appendix table with one row per layer/energy distribution."""

    lines = [
        r"\begin{table*}[t]",
        r"  \centering",
        r"  \caption{One-dimensional Wasserstein distances to Geant4 and between the two",
        r"  RBM sampling pipelines. Each row is one displayed layer- or energy-specific",
        r"  distribution; distances are reported in the native units shown in the row label.",
        r"  EMB1 and EMB2 rows correspond to the main-body spatial comparisons and are",
        r"  listed first.}",
        r"  \label{tab:wd}",
        r"  \scriptsize",
        r"  \begin{tabular}{@{}lccc@{}}",
        r"    \toprule",
        r"    Observable / evaluation & Classical--G4 & QPU--G4 & QPU--Classical \\ ",
        r"    \midrule",
    ]
    previous_group = None
    for entry in table_entries(entries):
        group = _table_group(entry)
        if group != previous_group:
            if previous_group is not None:
                lines.append(r"    \addlinespace[2pt]")
            lines.append(
                rf"    \multicolumn{{4}}{{@{{}}l}}{{\emph{{{TABLE_GROUPS[group]}}}}}\\"
            )
            previous_group = group
        wd = entry["wd"]
        lines.append(
            "    "
            + _table_label(entry)
            + " & $"
            + _format_wd(wd["Classical--G4"])
            + "$ & $"
            + _format_wd(wd["QPU--G4"])
            + "$ & $"
            + _format_wd(wd["QPU--Classical"])
            + r"$ \\"
        )
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def build_report(uniform_dir: Path, fixed_root: Path) -> dict:
    entries = collect_entries(uniform_dir, fixed_root)
    detailed_table_entries = table_entries(entries)
    return {
        "method": "Unbinned one-dimensional Wasserstein distance on finite NPZ samples",
        "reference": "GEANT4 / data_ref",
        "model_series": {
            "AE--G4": "Recon vs data_ref",
            "Classical--G4": "GPU vs data_ref",
            "QPU--G4": "QPU vs data_ref",
            "QPU--Classical": "QPU vs GPU",
        },
        "sources": {
            "uniform": _relative(uniform_dir),
            "fixed_energy": [_relative(fixed_root / f"AtlasCustom{energy}GeV") for energy in FIXED_ENERGIES],
        },
        "entries": entries,
        "summary_by_family": summarize_by_family(entries),
        "table": {
            "statistic": "one row per displayed layer- or energy-specific distribution",
            "entry_count": len(detailed_table_entries),
            "entry_ids": [entry["id"] for entry in detailed_table_entries],
            "emphasized_layers": ["EMB1", "EMB2"],
        },
    }


def main() -> None:
    args = _parse_args()
    report = build_report(args.uniform_dir, args.fixed_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    args.table_output.parent.mkdir(parents=True, exist_ok=True)
    args.table_output.write_text(render_wd_table(report["entries"]))
    print(f"Wrote {len(report['entries'])} shower WD entries to {args.output}")
    print(f"Wrote detailed appendix table to {args.table_output}")


if __name__ == "__main__":
    main()
