"""
End-to-end generation pipeline: AE + TransFusion + RBM → calorimeter showers.

Typical usage in a notebook:

    from utils.generation_pipeline import load_ae_engine, load_tfusion_engine, load_rbm, generate_and_evaluate

    ae_engine    = load_ae_engine(ae_cfg)
    tfusion      = load_tfusion_engine(tfusion_cfg)
    rbm          = load_rbm(rbm_cfg, checkpoint_path="/path/to/checkpoint.h5")
    # or for PTT:
    rbm          = load_rbm(rbm_cfg, checkpoint_dir="/path/to/run_dir/")

    results = generate_and_evaluate(
        ae_engine, tfusion, rbm,
        validation_datasets=["AtlasCustom", "AtlasCustom2"],
        tfusion_cfg=tfusion_cfg,
        ae_cfg=ae_cfg,
        output_dir="/path/to/outputs",
    )
"""

import os
import copy
from datetime import datetime
from typing import Union, List, Dict, Optional, Tuple

import torch
from omegaconf import OmegaConf

from data.dataManager import DataManagerLayersShowers, DataManagerLayers
from data.layers import transform_dataset
from model.rbm.rbm_two_partite import RBM_TwoPartite
from scripts.run import setup_model
from scripts.run_transfusion import setup_model as setup_model_transfusion
from scripts.RBM_2P_MNIST import save_clamped_samples_for_vae
from utils.rbm.rbm_utils import save_clamped_PTT_samples, prepare_ptt_checkpoints
from utils.HLF.atlasgeo import evaluate_and_plot

from CaloQuVAE import logging
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def load_ae_engine(config):
    """
    Load a pre-trained layer AE engine from a saved checkpoint.

    Args:
        config: OmegaConf config with `config_path`, `run_path`, and `gpu_list`.
                Typically obtained via:
                    cfg = compose(config_name="config_layers.yaml")
                    config = OmegaConf.load(cfg.config_path)
                    config.gpu_list = cfg.gpu_list

    Returns:
        EngineLayers ready for inference (model loaded, data loaders built).
    """
    config.load_state = 1
    engine = setup_model(config)
    engine._model_creator.load_state(config.run_path, engine.device)
    engine.data_mgr.apply_stats_and_build_loaders(
        engine.model.feature_min.cpu(),
        engine.model.feature_max.cpu(),
    )
    logger.info("AE engine loaded from %s", config.run_path)
    return engine


def load_tfusion_engine(config):
    """
    Load a pre-trained TransFusion engine from a saved checkpoint.

    Args:
        config: OmegaConf config with `run_path` and `gpu_list`.
                Typically obtained via:
                    master = compose(config_name="tfusion_config.yaml")
                    config = OmegaConf.load(master.config_path)
                    config.gpu_list = <ae_cfg>.gpu_list

    Returns:
        EngineTransfusion ready for inference.
    """
    engine = setup_model_transfusion(config)
    engine._model_creator.load_state(config.run_path, engine.device)
    logger.info("TransFusion engine loaded from %s", config.run_path)
    return engine


def load_rbm(
    rbm_cfg,
    checkpoint_path: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
) -> Union["RBM_TwoPartite", List["RBM_TwoPartite"]]:
    """
    Load one RBM checkpoint (standard sampling) or all checkpoints from a
    directory (parallel trajectory tempering).

    Provide exactly one of `checkpoint_path` or `checkpoint_dir`.

    Args:
        rbm_cfg:         OmegaConf config with `rbm.num_visible_nodes`.
        checkpoint_path: Path to a single `.h5` checkpoint file.
        checkpoint_dir:  Directory containing epoch-stamped `.h5` files for PTT.

    Returns:
        Single RBM_TwoPartite  –– when `checkpoint_path` is given.
        List[RBM_TwoPartite]   –– when `checkpoint_dir` is given (sorted by epoch).
    """
    if (checkpoint_path is None) == (checkpoint_dir is None):
        raise ValueError("Provide exactly one of `checkpoint_path` or `checkpoint_dir`.")

    if checkpoint_path is not None:
        device = _device_from_cfg(rbm_cfg)
        dummy = torch.zeros(1, rbm_cfg.rbm.num_visible_nodes, device=device)
        rbm = RBM_TwoPartite(rbm_cfg, data=dummy)
        epoch = rbm.load_checkpoint(checkpoint_path, epoch=None)
        logger.info("Loaded single RBM checkpoint (epoch %d) from %s", epoch, checkpoint_path)
        return rbm

    # PTT: load all checkpoints in directory
    models = prepare_ptt_checkpoints(checkpoint_dir, rbm_cfg)
    logger.info("Loaded %d PTT checkpoints from %s", len(models), checkpoint_dir)
    return models


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

def generate_and_evaluate(
    ae_engine,
    tfusion_engine,
    rbm: Union["RBM_TwoPartite", List["RBM_TwoPartite"]],
    validation_datasets: List[str],
    tfusion_cfg,
    ae_cfg,
    output_dir: str,
    gibbs_steps: int = 5000,
    gen_batch_size: int = 8192,
    decode_batch_size: int = 1024,
    label: str = "GPU",
) -> Dict[str, Dict]:
    """
    Full generation pipeline: for each validation dataset, sample layer energies
    from TransFusion, draw latent codes from the RBM prior (clamped on the AE
    conditional encoding), decode showers with the AE, then compare against
    GEANT4 and AE reconstructions using the HLF/atlasgeo plotter.

    Args:
        ae_engine:           EngineLayers (loaded AE).
        tfusion_engine:      EngineTransfusion (loaded flow model).
        rbm:                 Single RBM_TwoPartite for standard sampling, or a
                             chronologically ordered list for PTT sampling.
        validation_datasets: Dataset names, e.g. ["AtlasCustom", "AtlasCustom2"].
        tfusion_cfg:         OmegaConf config for the TransFusion model.
        ae_cfg:              OmegaConf config for the AE (used to build overrides).
        output_dir:          Root directory for all saved plots / tensors.
        gibbs_steps:         MCMC steps for RBM sampling.
        gen_batch_size:      Batch size passed to the RBM sampler.
        decode_batch_size:   Batch size for the AE decoder.
        label:               Key used for generated showers in comparison plots.

    Returns:
        Dict mapping each dataset name to:
            {
                "showers_gt":    Tensor (N, n_voxels)  – GEANT4 showers,
                "showers_recon": Tensor (N, n_voxels)  – AE reconstructions,
                "showers_gen":   Tensor (N, n_voxels)  – RBM-decoded showers,
                "x0":            Tensor (N, 1)          – incident energies,
                "save_dir":      str                    – where plots were saved,
            }
    """
    use_ptt = isinstance(rbm, list)
    results = {}

    for dataset in validation_datasets:
        logger.info("Processing dataset: %s", dataset)
        save_dir = _make_run_dir(output_dir, dataset)

        # ------------------------------------------------------------------
        # 1. Sample layer energies (E) and incident energies (x0) from
        #    TransFusion on the chosen validation split.
        # ------------------------------------------------------------------
        val_cfg_tfusion = copy.deepcopy(tfusion_cfg)
        val_cfg_tfusion.data = OmegaConf.load(f"config/data/{dataset}.yaml")
        tfusion_loader = DataManagerLayers(val_cfg_tfusion).val_loader

        E_samples, _, x0 = tfusion_engine.sample_tfusion(epoch=0, data_loader=tfusion_loader)
        logger.info("  TransFusion: sampled %d events", E_samples.shape[0])

        # ------------------------------------------------------------------
        # 2. Convert raw layer energies → normalized AE conditioning signal.
        # ------------------------------------------------------------------
        u_samples = transform_dataset(E_samples, x0).to(ae_engine.device)
        u_samples = (
            (u_samples - ae_engine.model.feature_min)
            / (ae_engine.model.feature_max - ae_engine.model.feature_min)
        )
        ae_engine.load_cond_encoding(x0, u_samples)

        # ------------------------------------------------------------------
        # 3. Draw latent codes from the RBM (first `cond_p_size` nodes
        #    clamped to the AE conditional encoding).
        # ------------------------------------------------------------------
        cond_p_size = ae_engine._config.model.cond_p_size

        if use_ptt:
            prior_samples_path = save_clamped_PTT_samples(
                checkpoints_list=rbm,
                input_data=ae_engine.post_cond_samples,
                n_clamped=cond_p_size,
                gibbs_steps=gibbs_steps,
                save_dir=save_dir,
                gen_batch_size=gen_batch_size,
            )
        else:
            prior_samples_path = save_clamped_samples_for_vae(
                rbm,
                input_data=ae_engine.post_cond_samples,
                n_clamped=cond_p_size,
                gibbs_steps=gibbs_steps,
                save_dir=save_dir,
                gen_batch_size=gen_batch_size,
            )

        rbm_samples = torch.load(prior_samples_path)
        logger.info("  RBM samples: %s", tuple(rbm_samples.shape))

        # ------------------------------------------------------------------
        # 4. Decode latent codes → calorimeter showers via the AE decoder.
        # ------------------------------------------------------------------
        showers_gen = ae_engine.generate_showers_from_rbm(
            rbm_samples, x0, u_samples, E_samples, batch_size=decode_batch_size
        )

        # ------------------------------------------------------------------
        # 5. Run the AE on the GT validation data to get reconstructions and
        #    ground-truth showers for comparison.
        # ------------------------------------------------------------------
        val_cfg = _make_val_cfg(ae_cfg, dataset)
        val_dm = DataManagerLayersShowers(val_cfg)
        val_dm.apply_stats_and_build_loaders(
            ae_engine.model.feature_min.cpu(),
            ae_engine.model.feature_max.cpu(),
        )
        ae_engine.evaluate_ae(val_dm.val_loader, epoch=0)

        # ------------------------------------------------------------------
        # 6. Compare GEANT4 / AE reconstruction / generated showers.
        # ------------------------------------------------------------------
        data_to_plot = {
            "GEANT4": (ae_engine.showers, x0),
            "Recon":  (ae_engine.showers_recon, x0),
            label:    (showers_gen, x0),
        }
        evaluate_and_plot(data_to_plot, val_cfg.data.binning_path, output_dir=save_dir)
        logger.info("  Plots saved to %s", save_dir)

        # Persist latent samples for downstream analysis
        torch.save(ae_engine.post_samples, os.path.join(save_dir, "post_samples.pt"))

        results[dataset] = {
            "showers_gt":    ae_engine.showers,
            "showers_recon": ae_engine.showers_recon,
            "showers_gen":   showers_gen,
            "x0":            x0,
            "save_dir":      save_dir,
        }

    return results


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _device_from_cfg(cfg) -> torch.device:
    if getattr(cfg, "device", "cpu") == "gpu" and torch.cuda.is_available():
        gpu_list = getattr(cfg, "gpu_list", [0])
        return torch.device(f"cuda:{gpu_list[0]}")
    return torch.device("cpu")


def _make_run_dir(base: str, dataset: str) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = os.path.join(base, f"gen_{timestamp}_{dataset}")
    os.makedirs(path, exist_ok=True)
    return path


def _make_val_cfg(ae_cfg, dataset: str):
    """Build a per-dataset AE config override using Hydra compose."""
    from hydra import compose
    return compose(
        config_name="config_layers",
        overrides=[
            f"data={dataset}",
            f"feature_stats_path={ae_cfg.feature_stats_path}",
        ],
    )
