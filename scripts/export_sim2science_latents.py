#!/usr/bin/env python3
"""Freeze the binary latent inputs used by the Sim2Science shower plots.

The notebook stores the posterior and GPU prior as ``torch.save`` tensors.  A
QPU run is stored as the dictionary consumed by ``load_qpu_samples`` together
with its saved condition arrays.  This script copies those files into the
paper suite and writes a hash manifest, without importing PyTorch or touching
the expensive shower-generation pipeline.

Run from the repository root with::

    python3 scripts/export_sim2science_latents.py
    python3 scripts/export_sim2science_latents.py --check

The source paths below intentionally name the runs used for the paper plots.
They are recorded in the manifest for provenance; the copied files are the
portable inputs used by a later decoder/replot job.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


QPU_FILES = (
    "processed_samples.pt",
    "incidence_energy.pt",
    "u_samples.pt",
    "E_samples.pt",
)

ENERGY_MEV = {
    "AtlasCustom2GeV": 2_000,
    "AtlasCustom5GeV": 5_000,
    "AtlasCustom10GeV": 10_000,
    "AtlasCustom20GeV": 20_000,
    "AtlasCustom50GeV": 50_000,
    "AtlasCustom100GeV": 100_000,
    "AtlasCustom150GeV": 150_000,
    "AtlasCustom250GeV": 250_000,
}

# These are the fixed-energy RBM runs paired with the fixed-energy artifacts
# used by the paper.  There is no matching saved GPU/posterior pair for the
# 150 GeV QPU directory, so that energy is exported below as QPU-only input.
FIXED_RBM_RUNS = {
    "AtlasCustom2GeV": "run_2026-04-28_21-25-22_RBM_FC_AtlasCustom2GeV",
    "AtlasCustom5GeV": "run_2026-04-28_21-32-09_RBM_FC_AtlasCustom5GeV",
    "AtlasCustom10GeV": "run_2026-04-28_21-38-59_RBM_FC_AtlasCustom10GeV",
    "AtlasCustom20GeV": "run_2026-04-30_00-33-59_RBM_FC_AtlasCustom20GeV",
    "AtlasCustom50GeV": "run_2026-04-28_21-45-52_RBM_FC_AtlasCustom50GeV",
    "AtlasCustom100GeV": "run_2026-04-28_21-52-43_RBM_FC_AtlasCustom100GeV",
    "AtlasCustom250GeV": "run_2026-04-28_21-59-06_RBM_FC_AtlasCustom250GeV",
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _source_label(path: Path, repo_root: Path) -> str:
    try:
        return path.relative_to(repo_root).as_posix()
    except ValueError:
        return str(path)


def _copy_artifact(
    source: Path,
    target: Path,
    *,
    repo_root: Path,
    output_dir: Path,
    artifacts: dict[str, dict[str, object]],
    force: bool,
) -> str:
    source = source.expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"Missing latent source: {source}")

    target.parent.mkdir(parents=True, exist_ok=True)
    target_hash = _sha256(source)
    if target.exists():
        existing_hash = _sha256(target)
        if existing_hash != target_hash:
            if not force:
                raise FileExistsError(
                    f"Refusing to replace changed artifact {target}; use --force"
                )
            shutil.copy2(source, target)
    else:
        shutil.copy2(source, target)

    if _sha256(target) != target_hash:
        raise IOError(f"Hash mismatch after copying {source} to {target}")

    relative_target = target.relative_to(output_dir).as_posix()
    artifacts[relative_target] = {
        "source": str(source),
        "source_label": _source_label(source, repo_root),
        "bytes": source.stat().st_size,
        "sha256": target_hash,
        "serialization": "torch.save",
    }
    return relative_target


def _copy_qpu_run(
    source_dir: Path,
    target_dir: Path,
    *,
    repo_root: Path,
    output_dir: Path,
    artifacts: dict[str, dict[str, object]],
    force: bool,
) -> dict[str, str]:
    files = {}
    for filename in QPU_FILES:
        files[filename] = _copy_artifact(
            source_dir / filename,
            target_dir / filename,
            repo_root=repo_root,
            output_dir=output_dir,
            artifacts=artifacts,
            force=force,
        )
    beta_info = source_dir / "beta_info.json"
    if beta_info.is_file():
        files[beta_info.name] = _copy_artifact(
            beta_info,
            target_dir / beta_info.name,
            repo_root=repo_root,
            output_dir=output_dir,
            artifacts=artifacts,
            force=force,
        )
    return files


def _build_manifest(
    *,
    repo_root: Path,
    output_dir: Path,
    force: bool,
) -> dict[str, object]:
    wandb_root = repo_root / "wandb-outputs"
    qpu_root = Path("/fast_scratch_1/caloqvae/dwave_samples_apr_24")
    notebook_run = wandb_root / "run_2026-04-24_17-43-32_RBM_FC_AtlasCustom2"
    artifacts: dict[str, dict[str, object]] = {}

    notebook_group = {
        "run_dir": str(notebook_run),
        "posterior": _copy_artifact(
            notebook_run / "post_samples.pt",
            output_dir / "latents/notebook_atlascustom2/post_samples.pt",
            repo_root=repo_root,
            output_dir=output_dir,
            artifacts=artifacts,
            force=force,
        ),
        "gpu_prior": _copy_artifact(
            notebook_run / "rbm_clamped_samples_train_data_final.pt",
            output_dir
            / "latents/notebook_atlascustom2/rbm_clamped_samples_train_data_final.pt",
            repo_root=repo_root,
            output_dir=output_dir,
            artifacts=artifacts,
            force=force,
        ),
    }

    fixed_energy: dict[str, dict[str, object]] = {}
    for dataset, run_name in FIXED_RBM_RUNS.items():
        run_dir = wandb_root / run_name
        qpu_dir = qpu_root / f"energy_{ENERGY_MEV[dataset]}"
        fixed_energy[dataset] = {
            "energy_mev": ENERGY_MEV[dataset],
            "rbm_run_dir": str(run_dir),
            "posterior": _copy_artifact(
                run_dir / "post_samples.pt",
                output_dir / f"latents/fixed_energy/{dataset}/post_samples.pt",
                repo_root=repo_root,
                output_dir=output_dir,
                artifacts=artifacts,
                force=force,
            ),
            "gpu_prior": _copy_artifact(
                run_dir / "rbm_clamped_samples_train_data_final.pt",
                output_dir
                / f"latents/fixed_energy/{dataset}/"
                "rbm_clamped_samples_train_data_final.pt",
                repo_root=repo_root,
                output_dir=output_dir,
                artifacts=artifacts,
                force=force,
            ),
            "qpu": {
                "source_dir": str(qpu_dir),
                "files": _copy_qpu_run(
                    qpu_dir,
                    output_dir / f"latents/qpu/fixed_energy/{dataset}",
                    repo_root=repo_root,
                    output_dir=output_dir,
                    artifacts=artifacts,
                    force=force,
                ),
            },
        }

    # The notebook lookup includes 150 GeV, but the expected directory is not
    # present in the local scratch tree.  A similarly named energy_15000
    # directory exists; it is kept out of this bundle because it would be
    # incorrect to label a 15 GeV source as 150 GeV.
    qpu_only = {
        "energy_mev": ENERGY_MEV["AtlasCustom150GeV"],
        "status": "missing",
        "expected_source_dir": str(qpu_root / "energy_150000"),
        "similarly_named_source_dir": str(qpu_root / "energy_15000"),
        "note": "The available energy_15000 directory is not assumed to be 150 GeV.",
    }

    uniform_runs = {}
    for run_label in ("apr_22", "apr_23"):
        source_dir = Path(f"/fast_scratch_1/caloqvae/dwave_samples_{run_label}/uniform_range")
        uniform_runs[run_label] = {
            "source_dir": str(source_dir),
            "files": _copy_qpu_run(
                source_dir,
                output_dir / f"latents/qpu/uniform_range/{run_label}",
                repo_root=repo_root,
                output_dir=output_dir,
                artifacts=artifacts,
                force=force,
            ),
        }

    # The recovered run contains the continuous conditions needed to avoid
    # lossy decoding of the 36 condition bits in the uniform-range replot.
    recovered_run = wandb_root / "recovered_2026-08-04_RBM_FC_AtlasCustom2"
    uniform_range_60k = {
        "qpu_sample_count": 60_000,
        "posterior": notebook_group["posterior"],
        "gpu_prior": notebook_group["gpu_prior"],
        "continuous_conditions": _copy_artifact(
            recovered_run / "continuous_conditions.pt",
            output_dir / "latents/uniform_range/continuous_conditions.pt",
            repo_root=repo_root,
            output_dir=output_dir,
            artifacts=artifacts,
            force=force,
        ),
        "qpu_runs": uniform_runs,
    }
    uniform_range_50k = {
        "qpu_sample_count": 50_000,
        "posterior": notebook_group["posterior"],
        "gpu_prior": notebook_group["gpu_prior"],
        "continuous_conditions": uniform_range_60k["continuous_conditions"],
        "qpu_runs": {"apr_23": uniform_runs["apr_23"]},
    }

    return {
        "schema": "sim2science-latents/v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "latent_layout": {
            "serialization": "torch.save",
            "binary_value_convention": "0/1 entries",
            "visible_width": 177,
            "condition_bits": 36,
            "learned_bits": 141,
            "learned_partitions": 3,
            "partition_width": 47,
            "qpu_processed_samples_visible_key": "v",
            "qpu_processed_samples_other_keys": [
                "h",
                "clean_mask",
                "pattern_indices",
            ],
        },
        "notebook_reference": {
            "posterior_is": "self.post_samples after evaluate_ae",
            "gpu_prior_is": "rbm_clamped_samples_train_data_final.pt",
            "qpu_loader": "utils.dwave.postprocessing.load_qpu_samples",
            "gpu_path_note": (
                "The notebook snippet names the RBM run directory; torch.load "
                "must target the rbm_clamped_samples_train_data_final.pt file inside it."
            ),
            "april_24_atlascustom2": notebook_group,
        },
        "groups": {
            "uniform_range_50k": uniform_range_50k,
            "uniform_range_60k": uniform_range_60k,
            "fixed_energy": fixed_energy,
            "qpu_only": {"AtlasCustom150GeV": qpu_only},
        },
        "artifacts": artifacts,
    }


def _check_manifest(output_dir: Path) -> None:
    manifest_path = output_dir / "latent_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    failures = []
    for relative_path, metadata in manifest["artifacts"].items():
        path = output_dir / relative_path
        if not path.is_file():
            failures.append(f"missing {path}")
            continue
        if path.stat().st_size != metadata["bytes"]:
            failures.append(f"size mismatch {path}")
            continue
        if _sha256(path) != metadata["sha256"]:
            failures.append(f"hash mismatch {path}")
    if failures:
        raise RuntimeError("\n".join(failures))
    print(f"Verified {len(manifest['artifacts'])} latent artifacts from {manifest_path}")


def main() -> None:
    repo_root = _repo_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "paper/Sim2Science_Workshop/reproducibility",
        help="Destination directory for latent_manifest.json and latents/.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Verify the copied files against the existing manifest.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing destination file when its hash differs.",
    )
    args = parser.parse_args()
    output_dir = args.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = repo_root / output_dir
    output_dir = output_dir.resolve()

    if args.check:
        _check_manifest(output_dir)
        return

    manifest = _build_manifest(
        repo_root=repo_root,
        output_dir=output_dir,
        force=args.force,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "latent_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    total_bytes = sum(item["bytes"] for item in manifest["artifacts"].values())
    print(
        f"Copied {len(manifest['artifacts'])} latent artifacts "
        f"({total_bytes / 1024**3:.2f} GiB) to {output_dir}"
    )
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
