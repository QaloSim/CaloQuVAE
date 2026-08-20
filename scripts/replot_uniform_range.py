"""Regenerate the ATLAS uniform-energy validation plots from saved artifacts.

This is deliberately separate from the notebook because the QPU runs and the
RBM run are already complete.  The script reconstructs the three model
inputs used by ``evaluate_and_plot`` and therefore writes the intermediate
``.npz`` files needed by ``replot_from_npz``.  The paper-facing series names
are written directly into the artifacts so later replots retain the same
legend.

The canonical 60k result is obtained by passing both the Apr 22 and Apr 23
uniform-range directories::

    python scripts/replot_uniform_range.py \
      --qpu-dir /fast_scratch_1/caloqvae/dwave_samples_apr_22/uniform_range \
      --qpu-dir /fast_scratch_1/caloqvae/dwave_samples_apr_23/uniform_range \
      --output-dir paper_plots/AtlasCustom2Uniform60k

The preferred workflow is to pass ``--gpu-conditions`` pointing at the
continuous ``E_samples``, ``u_samples``, and ``x0`` saved alongside a fresh
RBM run.  The bit-decoding path remains as an explicitly lossy fallback for
the older Apr 24 run, whose continuous TransFusion conditions were not
serialized.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import hydra
import torch
import wandb
from hydra import compose, initialize
from omegaconf import OmegaConf

from data.dataManager import DataManagerLayersShowers
from scripts.run import setup_model
from utils.HLF.atlasgeo import evaluate_and_plot
from utils.dwave.postprocessing import load_qpu_samples


DEFAULT_GPU_RUN = (
    "/home/leozhu/CaloQuVAE/wandb-outputs/"
    "run_2026-04-24_17-43-32_RBM_FC_AtlasCustom2"
)


def _gray_decode(bits: torch.Tensor) -> torch.Tensor:
    """Decode MSB-first standard Gray-code bits to a float integer tensor."""

    n_bits = bits.shape[1]
    powers = 2 ** torch.arange(n_bits - 1, -1, -1, device=bits.device)
    gray = (bits * powers).sum(dim=1).round().to(torch.int64)
    value = gray.clone()
    shift = 1
    while shift < n_bits:
        value ^= value >> shift
        shift *= 2
    return value.float()


def _decode_uniform_conditions(
    rbm_samples: torch.Tensor,
    engine,
    validation_x0: torch.Tensor,
    energy_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, str]:
    """Recover ``x0``, scaled ``u``, and raw layer energies for RBM samples."""

    model_cfg = engine._config.model
    data_cfg = engine._config.data
    cond_size = int(model_cfg.cond_p_size)
    u_bits = int(model_cfg.u_bits)
    n_layers = int(data_cfg.z)
    energy_bits = cond_size - u_bits * n_layers
    lin_bits = int(model_cfg.lin_bits)
    sqrt_bits = int(model_cfg.sqrt_bits)
    log_bits = int(model_cfg.log_bits)
    if energy_bits != lin_bits + sqrt_bits + log_bits:
        raise ValueError(
            "The saved condition layout does not match this model: "
            f"{energy_bits=} but {lin_bits + sqrt_bits + log_bits=}"
        )
    if getattr(model_cfg, "u_cdf", False):
        raise ValueError("CDF-coded u conditions need their saved bin edges to decode")

    conditions = rbm_samples[:, :cond_size].cpu()
    energy_code = conditions[:, :energy_bits]
    lin_code = energy_code[:, :lin_bits]
    sqrt_code = energy_code[:, lin_bits : lin_bits + sqrt_bits]
    log_code = energy_code[:, lin_bits + sqrt_bits :]

    # These are the inverse scales in gray_einc_with_u_compact().
    e_linear = _gray_decode(lin_code) * 588.0
    e_sqrt = (_gray_decode(sqrt_code) * 35.0 / 4.0).square()
    e_log = 1000.0 * torch.exp(_gray_decode(log_code) * 3.0 / 32.0)

    validation_x0 = validation_x0.cpu()
    code_from_validation = engine.model.encoder.energy_encoding_fct(
        validation_x0.to(engine.device)
    ).detach().cpu()
    exact_rows = (code_from_validation == energy_code).all(dim=1)
    agreement = float((code_from_validation == energy_code).float().mean())

    if len(validation_x0) == len(rbm_samples) and agreement >= 0.99:
        x0 = validation_x0
        x0_source = "AtlasCustom2 validation-loader incident energies"
    else:
        # The linear code is the least nonlinear and is the most stable
        # fallback if the saved RBM run was made with a different ordering.
        x0 = e_linear[:, None]
        x0_source = "decoded linear incident-energy Gray code"

    u_scaled_columns = []
    u_start = energy_bits
    for layer in range(n_layers):
        start = u_start + layer * u_bits
        stop = start + u_bits
        u_scaled_columns.append(_gray_decode(conditions[:, start:stop]) / (2**u_bits - 1))
    u_scaled = torch.stack(u_scaled_columns, dim=1)

    feature_min = engine.model.feature_min.detach().cpu()
    feature_max = engine.model.feature_max.detach().cpu()
    u_raw = u_scaled * (feature_max - feature_min) + feature_min

    # Inverse of transform_dataset() used in layer_AE.ipynb.
    e_total = u_raw[:, 0:1] * (energy_scale * x0 + 1e-7)
    raw_layers = torch.zeros_like(u_raw)
    remaining = e_total
    for layer in range(n_layers - 1):
        raw_layers[:, layer : layer + 1] = u_raw[:, layer + 1 : layer + 2] * remaining
        remaining = remaining - raw_layers[:, layer : layer + 1]
    raw_layers[:, -1:] = remaining

    # Keep the redundant nonlinear decodes available in the metadata without
    # making them part of the generation path.
    _ = (e_sqrt, e_log, exact_rows)
    return x0, u_scaled, raw_layers, agreement, x0_source


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--qpu-dir",
        action="append",
        required=True,
        help="Uniform-range QPU artifact directory; repeat to concatenate runs.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for dynamic plots and their .npz files.",
    )
    parser.add_argument("--dataset", default="AtlasCustom2")
    parser.add_argument("--gpu-run-dir", default=DEFAULT_GPU_RUN)
    parser.add_argument(
        "--gpu-conditions",
        default=None,
        help=(
            "Saved continuous GPU conditions (.pt with E_samples, u_samples, "
            "and x0); avoids lossy reconstruction from the 36 RBM condition bits."
        ),
    )
    parser.add_argument("--energy-scale", type=float, default=1.60)
    parser.add_argument("--gpu-list", type=int, nargs="+", default=None)
    parser.add_argument("--clean-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    os.chdir(repo_root)

    for path in args.qpu_dir:
        required = ["processed_samples.pt", "incidence_energy.pt", "u_samples.pt", "E_samples.pt"]
        missing = [name for name in required if not (Path(path) / name).exists()]
        if missing:
            raise FileNotFoundError(f"{path} is missing {missing}")

    # Match the model and feature statistics selected by layer_AE.ipynb.
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    # Hydra resolves config_path relative to this script's directory.
    with initialize(version_base=None, config_path="../config"):
        outer_cfg = compose(config_name="config_layers.yaml")

    ae_cfg = OmegaConf.load(outer_cfg.config_path)
    if args.gpu_list is not None:
        ae_cfg.gpu_list = args.gpu_list
    else:
        ae_cfg.gpu_list = outer_cfg.gpu_list
    ae_cfg.load_state = 1
    if "wandb" in ae_cfg:
        ae_cfg.wandb.mode = "disabled"

    # setup_model/evaluate_ae call wandb.log through the engine even for this
    # read-only reconstruction, so initialize an explicitly disabled run.
    if wandb.run is None:
        wandb.init(mode="disabled")

    print(f"Loading AE checkpoint: {ae_cfg.run_path}")
    engine = setup_model(ae_cfg)
    engine._model_creator.load_state(ae_cfg.run_path, engine.device)

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with initialize(version_base=None, config_path="../config"):
        val_cfg = compose(
            config_name="config_layers",
            overrides=[f"data={args.dataset}", f"feature_stats_path={outer_cfg.feature_stats_path}"],
        )
    val_manager = DataManagerLayersShowers(val_cfg)
    val_manager.apply_stats_and_build_loaders(
        engine.model.feature_min.cpu(), engine.model.feature_max.cpu()
    )

    print(f"Evaluating GEANT4 and Recon on {args.dataset}...")
    engine.evaluate_ae(val_manager.val_loader, epoch=0)
    validation_x0 = engine.incident_energy.clone()

    gpu_path = Path(args.gpu_run_dir) / "rbm_clamped_samples_train_data_final.pt"
    if not gpu_path.exists():
        raise FileNotFoundError(gpu_path)
    rbm_samples = torch.load(gpu_path, map_location="cpu")
    condition_agreement = None
    if args.gpu_conditions is not None:
        condition_path = Path(args.gpu_conditions)
        if not condition_path.exists():
            raise FileNotFoundError(condition_path)
        condition_payload = torch.load(condition_path, map_location="cpu")
        required = {"x0", "u_samples", "E_samples"}
        missing = required.difference(condition_payload)
        if missing:
            raise KeyError(
                f"{condition_path} is missing continuous GPU condition keys: {sorted(missing)}"
            )
        gpu_x0 = condition_payload["x0"].float()
        gpu_u = condition_payload["u_samples"].float()
        gpu_E = condition_payload["E_samples"].float()
        if not (len(rbm_samples) == len(gpu_x0) == len(gpu_u) == len(gpu_E)):
            raise ValueError(
                "Saved GPU RBM samples and continuous conditions have different lengths: "
                f"{len(rbm_samples)}, {len(gpu_x0)}, {len(gpu_u)}, {len(gpu_E)}"
            )
        cond_size = int(engine._config.model.cond_p_size)
        expected_conditions = torch.cat(
            (
                engine.model.encoder.energy_encoding_fct(gpu_x0.to(engine.device)),
                engine.model.encoder.gray_encoding_fct(gpu_u.to(engine.device)),
            ),
            dim=1,
        ).detach().cpu()
        condition_agreement = float(
            (expected_conditions == rbm_samples[:, :cond_size]).float().mean()
        )
        energy_bits = int(engine._config.model.lin_bits)
        energy_bits += int(engine._config.model.sqrt_bits)
        energy_bits += int(engine._config.model.log_bits)
        agreement = float(
            (expected_conditions[:, :energy_bits] == rbm_samples[:, :energy_bits])
            .float()
            .mean()
        )
        x0_source = f"saved continuous conditions: {condition_path}"
        print(
            f"Loaded {len(rbm_samples)} GPU RBM samples with continuous conditions; "
            f"full-condition bit agreement={condition_agreement:.6f}"
        )
    else:
        gpu_x0, gpu_u, gpu_E, agreement, x0_source = _decode_uniform_conditions(
            rbm_samples, engine, validation_x0, args.energy_scale
        )
    gpu_condition_mode = "loaded" if args.gpu_conditions is not None else "decoded"
    print(
        f"{gpu_condition_mode.capitalize()} {len(rbm_samples)} GPU RBM samples; "
        f"energy-code agreement={agreement:.6f}; x0 source={x0_source}"
    )
    gpu_showers = engine.generate_showers_from_rbm(rbm_samples, gpu_x0, gpu_u, gpu_E)

    qpu_samples, qpu_x0, qpu_u, qpu_E = load_qpu_samples(
        args.qpu_dir, clean_only=args.clean_only
    )
    print(f"Generating showers from {len(qpu_samples)} QPU samples...")
    qpu_showers = engine.generate_showers_from_rbm(qpu_samples, qpu_x0, qpu_u, qpu_E)

    data_to_plot = {
        "ATLAS FullSim": (engine.showers, validation_x0),
        "AE reconstruction": (engine.showers_recon, validation_x0),
        "Classical RBM": (gpu_showers, gpu_x0),
        "QPU RBM": (qpu_showers, qpu_x0),
    }
    output_dir = Path(args.output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    binning_file = val_cfg.data.binning_path
    print(f"Writing dynamic plots and .npz files to {output_dir}")
    evaluate_and_plot(data_to_plot, binning_file, output_dir=str(output_dir), device="cpu")

    narrow_dir = Path(str(output_dir) + "_narrow")
    narrow_ranges = {
        "MeanEta": (-10, 10),
        "WidthEta": (0, 40),
        "MeanPhi": (-10, 10),
        "WidthPhi": (0, 40),
    }
    print(f"Writing narrow plots and .npz files to {narrow_dir}")
    evaluate_and_plot(
        data_to_plot,
        binning_file,
        output_dir=str(narrow_dir),
        device="cpu",
        fixed_bin_ranges=narrow_ranges,
        num_bins=100,
    )

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "qpu_dirs": args.qpu_dir,
        "clean_only": args.clean_only,
        "qpu_samples": int(len(qpu_samples)),
        "gpu_samples": int(len(rbm_samples)),
        "validation_samples": int(len(validation_x0)),
        "gpu_run_dir": args.gpu_run_dir,
        "ae_checkpoint": str(ae_cfg.run_path),
        "binning_file": str(binning_file),
        "energy_scale": args.energy_scale,
        "gpu_energy_code_agreement": agreement,
        "gpu_condition_code_agreement": condition_agreement,
        "gpu_x0_source": x0_source,
        "gpu_conditions": args.gpu_conditions,
        "output_dir": str(output_dir),
        "narrow_output_dir": str(narrow_dir),
    }
    (output_dir / "replot_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Saved metadata to {output_dir / 'replot_metadata.json'}")


if __name__ == "__main__":
    main()
