import os
import math
import logging
from datetime import datetime

import torch
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torchvision import transforms

# Custom project imports
from model.rbm.rbm_two_partite import RBM_TwoPartite
from utils.latent_dataset import LatentDataset
from CaloQuVAE import logging
logger = logging.getLogger(__name__)

def prep_latent_data(data_path, batch_size, features_paths=None, weight_alpha=2.0,
                     weight_max=500.0, tail_threshold=1e-4):
    """
    Load and binarize the saved latent data.
    Returns a single DataLoader (used for both train/test in this simplified script).

    If features_paths is provided (a path or list of paths), per-sample importance
    weights are computed and returned as the fourth element of each batch.
    """
    # Transform: binarize if > 0.5
    transform = transforms.Lambda(lambda x: (x > 0.5).float())

    dataset = LatentDataset(
        data_path=data_path,
        transform=transform,
        features_paths=features_paths,
        weight_alpha=weight_alpha,
        weight_max=weight_max,
        tail_threshold=tail_threshold,
    )

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True
    )
    return loader

def generate_fantasy_samples(rbm, n_samples, burn_in):
    """
    Generate n_samples from the RBM model by running parallel Gibbs chains.
    Does not affect the RBM's internal training chains.
    """
    # 1. Backup RBM's persistent training chains
    original_chains = {key: val.clone() for key, val in rbm.chains.items()}

    # 2. Initialize new random chains and run burn-in
    # Note: reset_chains() uses the RBM's configured batch size (num_chains)
    rbm.reset_chains()

    for _ in range(burn_in):
        rbm.sample_hidden()
        rbm.sample_visibles()

    # 3. Slice the required number of samples
    # (Assumes rbm.num_chains >= n_samples usually, or we slice what we have)
    samples_tensor = rbm.chains["v"][:n_samples].clone()

    # 4. Restore the original persistent chains
    rbm.chains = original_chains

    return samples_tensor

def save_fantasy_samples_for_vae(rbm, n_samples, burn_in, save_dir):
    """
    Generates n_samples fantasy samples in batches and saves them
    in a format loadable by LatentDataset.
    """
    logger.info(f"Generating {n_samples} fantasy samples for VAE...")

    # Determine batch size from RBM configuration
    gen_batch_size = rbm.chains["v"].shape[0]
    if gen_batch_size == 0:
        raise ValueError("RBM chains are size 0. Cannot generate samples.")

    n_batches = math.ceil(n_samples / gen_batch_size)
    all_samples = []

    for i in range(n_batches):
        if i % 5 == 0:
            logger.info(f"Generating batch {i+1}/{n_batches}...")

        batch_samples = generate_fantasy_samples(
            rbm,
            n_samples=gen_batch_size,
            burn_in=burn_in
        )
        all_samples.append(batch_samples.cpu())

    # Concatenate and truncate
    final_samples = torch.cat(all_samples, dim=0)[:n_samples]

    # Create dummy labels (to satisfy LatentDataset requirements)
    dummy_labels = torch.zeros((final_samples.shape[0], 1), dtype=torch.float32)

    # Save
    data_path = os.path.join(save_dir, "rbm_fantasy_train_data.pt")
    label_path = os.path.join(save_dir, "rbm_fantasy_train_labels.pt")

    torch.save(final_samples, data_path)
    torch.save(dummy_labels, label_path)

    logger.info(f"Saved VAE data ({final_samples.shape}) to: {data_path}")
    return data_path

@hydra.main(version_base=None, config_path="../config", config_name="config_layers")
def main(cfg: DictConfig):
    # --- Device Setup ---
    if cfg.device == "gpu" and torch.cuda.is_available():
        # Handle list or single int for gpu_list
        dev = torch.device(f"cuda:{cfg.gpu_list[0]}")
    else:
        dev = torch.device("cpu")

    logger.info(f"Device: {dev}")
    logger.info("Using 2-partite RBM (Hardcoded).")

    # --- Data Loading ---
    features_paths = getattr(cfg.rbm, "features_paths", None) or None
    weight_alpha   = getattr(cfg.rbm, "weight_alpha",  2.0)
    weight_max     = getattr(cfg.rbm, "data_weight_max", 500.0)
    tail_threshold = getattr(cfg.rbm, "tail_threshold", 1e-4)

    train_loader = prep_latent_data(
        data_path=cfg.rbm.latent_data_path,
        batch_size=cfg.rbm.num_chains,
        features_paths=features_paths,
        weight_alpha=weight_alpha,
        weight_max=weight_max,
        tail_threshold=tail_threshold,
    )

    # --- RBM Initialization ---
    sample_batch, _, _, _ = next(iter(train_loader))
    sample_batch_flat = sample_batch.to(dev).view(sample_batch.size(0), -1)
    rbm = RBM_TwoPartite(cfg, data=sample_batch_flat)
    logger.info(f"Initialized RBM: {rbm.num_visible} visible, {rbm.num_hidden} hidden units")

    # --- Fine-tuning: load checkpoint if specified ---
    load_checkpoint_path = getattr(cfg.rbm, "load_checkpoint_path", "") or ""
    is_finetuning = bool(load_checkpoint_path)
    if is_finetuning:
        logger.info(f"Fine-tuning mode: loading checkpoint from {load_checkpoint_path}")
        loaded_epoch = rbm.load_checkpoint(load_checkpoint_path)
        logger.info(f"Resumed from epoch {loaded_epoch}")

    training_mode = cfg.rbm.method
    if training_mode == "CPCD":
        logger.info("Extracting full dataset for Aggressive PCD chain initialization...")
        all_data = []
        for batch in train_loader:
            x = batch[0] # The data tensor
            all_data.append(x.view(x.size(0), -1))
        full_dataset_flat = torch.cat(all_data, dim=0)
        rbm.init_dataset_chains(full_dataset_flat, n_cond=cfg.model.cond_p_size)
        del all_data, full_dataset_flat
    elif training_mode == "PBRB":
        logger.info("Extracting dataset and labels for Proximity-Based Replay Buffer...")
        all_data = []
        all_labels = []
        for batch in train_loader:
            x = batch[0]
            y = batch[1] # Targets (Incidence energies)
            all_data.append(x.view(x.size(0), -1))
            all_labels.append(y)
        full_dataset_flat = torch.cat(all_data, dim=0)
        full_labels_flat = torch.cat(all_labels, dim=0)
        rbm.init_proximity_buffer(full_dataset_flat, full_labels_flat, n_cond=cfg.model.cond_p_size)
        del all_data, all_labels, full_dataset_flat, full_labels_flat

    logger.info(f"Training method: {training_mode}")

    # --- Weight ceiling (constant, enforced every batch) ---
    weight_max_cfg = getattr(cfg.rbm, "weight_max", float("inf"))
    logger.info(f"Per-weight ceiling: weight_max={weight_max_cfg:.4f}")

    # --- RMS constraint schedule (Projected Gradient Descent) ---
    # Always detect the current RMS from the loaded/initialized weights so
    # annealing starts from reality, not from a stale config value.
    W0 = rbm.params["weight_matrix"]
    weight_rms_start = W0.pow(2).mean().sqrt().item()
    if is_finetuning:
        logger.info(f"Fine-tuning: detected loaded weight RMS = {weight_rms_start:.6f}")
        weight_rms_target        = getattr(cfg.rbm, "weight_rms_target",        float("inf"))
        weight_rms_anneal_epochs = getattr(cfg.rbm, "weight_rms_anneal_epochs", cfg.n_epochs)
    else:
        weight_rms_target        = getattr(cfg.rbm, "weight_rms_target",        float("inf"))
        weight_rms_anneal_epochs = 1  # constant ceiling during training, no schedule
    weight_rms_current = weight_rms_start

    logger.info(f"PGD RMS constraint: {weight_rms_start:.6f} → {weight_rms_target:.6f} over {weight_rms_anneal_epochs} epochs. "
                f"Clamped Nodes: {cfg.model.cond_p_size}. Weight Decay: {cfg.rbm.gamma}")

    # --- Output Directory Setup ---
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join(
        "/home/leozhu/CaloQuVAE/wandb-outputs",
        f"run_{run_timestamp}_{cfg.rbm.model_type}"
    )
    os.makedirs(save_dir, exist_ok=True)
    config_save_path = os.path.join(save_dir, "config.yaml")
    with open(config_save_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
    logger.info(f"Saving results to '{save_dir}'")

    # --- Training Loop ---
    num_epochs = weight_rms_anneal_epochs if is_finetuning else cfg.n_epochs

    for epoch in range(num_epochs):
        # Linearly anneal RMS constraint toward target over weight_rms_anneal_epochs
        if weight_rms_anneal_epochs > 1:
            t = min(epoch / (weight_rms_anneal_epochs - 1), 1.0)
        else:
            t = 1.0
        weight_rms_current = weight_rms_start + t * (weight_rms_target - weight_rms_start)

        logger.info(f"Starting Epoch {epoch+1}/{num_epochs} | rms_τ={weight_rms_current:.6f}")

        if getattr(cfg.rbm, "adaptive_k", False) and epoch % cfg.rbm.adaptive_k_interval == 0:
            adaptive_k = rbm.estimate_mixing_time(sample_batch_flat, max_steps=int(1e5), n_cond=cfg.model.cond_p_size)
            rbm.config.rbm.bgs_steps = adaptive_k
            logger.info(f"Adaptive k updated to: {adaptive_k} MCMC steps for this epoch.")

        for batch_idx, (x, targets, indices, weights) in enumerate(train_loader):
            v_data = x.to(dev).view(x.size(0), -1)

            # 1. Positive phase expectations
            with torch.no_grad():
                mh_data = torch.sigmoid(
                    v_data @ rbm.params["weight_matrix"] + rbm.params["hbias"]
                )

            # 2. Prepare data dict (weights are importance weights if features_paths
            #    was configured, otherwise uniform 1s from LatentDataset)
            data_dict = {
                "v": v_data,
                "mh": mh_data,
                "weights": weights.to(dev),
            }

            # 3. Update
            if training_mode == "CD":
                rbm.chains["v"] = v_data.clone()
                rbm.sample_hidden()
                rbm.fit_batch(data_dict, centered=True)
            elif training_mode == "PCD":
                rbm.fit_batch(data_dict, centered=True)
            elif training_mode == "CCD":
                rbm.fit_batch_ccd(data_dict, n_cond=cfg.model.cond_p_size, centered=True)
            elif training_mode == "CPCD":
                rbm.fit_batch_cpcd(data_dict, indices=indices, n_cond=cfg.model.cond_p_size, centered=True)
            elif training_mode == "PBRB":
                pct = getattr(cfg.rbm, "k_neighbors_percent", 0.10)
                k_val = max(1, int(pct * len(train_loader.dataset)))
                noise = getattr(cfg.rbm, "noise_prob", 0.05)

                rbm.fit_batch_proximity(
                    data_dict=data_dict,
                    batch_labels=targets,
                    n_cond=cfg.model.cond_p_size,
                    k_neighbors=k_val,
                    noise_prob=noise,
                    centered=True
                )
            else:
                raise ValueError(f"Unsupported training method: {training_mode}")

            # 1. Per-weight ceiling (always)
            rbm.params["weight_matrix"] = torch.clamp(
                rbm.params["weight_matrix"], -weight_max_cfg, weight_max_cfg
            )
            # 2. PGD projection onto Frobenius-norm ball: τ = rms_τ * sqrt(N)
            W = rbm.params["weight_matrix"]
            frob = W.norm("fro")
            tau = weight_rms_current * math.sqrt(W.numel())
            if frob > tau:
                rbm.params["weight_matrix"] = W * (tau / frob)

        # --- End of Epoch Logging ---
        logger.info(f"  v_bias: mean={rbm.params['vbias'].mean():.4f}, "
                   f"std={rbm.params['vbias'].std():.4f}")
        logger.info(f"  h_bias: mean={rbm.params['hbias'].mean():.4f}, "
                   f"std={rbm.params['hbias'].std():.4f}")
        W_now = rbm.params["weight_matrix"]
        W_rms_now = W_now.pow(2).mean().sqrt().item()
        logger.info(f"  W: mean={W_now.mean():.4f}, std={W_now.std():.4f}")
        logger.info(f"  W: max={W_now.max():.4f}, min={W_now.min():.4f}, "
                   f"rms={W_rms_now:.6f} (τ={weight_rms_current:.6f})")

        # Save Checkpoint
        current_epoch = epoch + 1
        # A number is a power of 2 if (N & (N - 1)) == 0
        is_power_of_two = (current_epoch & (current_epoch - 1)) == 0

        if is_power_of_two or current_epoch == num_epochs:
            try:
                checkpoint_file = os.path.join(save_dir, f"training_checkpoint_epoch_{current_epoch}.h5")
                rbm.save_checkpoint(checkpoint_file, epoch, cfg)
                logger.info(f"Checkpoint saved to {checkpoint_file}")
            except Exception as e:
                logger.error(f"Failed to save checkpoint: {e}")

    try:
        final_model_file = os.path.join(save_dir, f"training_checkpoint_epoch_{num_epochs}.h5")
        rbm.save_checkpoint(final_model_file, num_epochs, cfg)
        logger.info(f"Final model saved to {final_model_file}")
    except Exception as e:
        logger.error(f"Failed to save final model: {e}")
    logger.info("Training complete!")

if __name__ == "__main__":
    main()
