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

def prep_latent_data(data_path, batch_size):
    """
    Load and binarize the saved latent data.
    Returns a single DataLoader (used for both train/test in this simplified script).
    """
    # Transform: binarize if > 0.5
    transform = transforms.Lambda(lambda x: (x > 0.5).float())
    
    dataset = LatentDataset(data_path=data_path, transform=transform)
    
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

@hydra.main(version_base=None, config_path="../config", config_name="config")
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
    # We strictly use the latent data path from config
    train_loader = prep_latent_data(
        data_path=cfg.rbm.latent_data_path,
        batch_size=cfg.rbm.num_chains
    )

    # --- RBM Initialization ---
    sample_batch, _ = next(iter(train_loader))
    print(type(sample_batch))
    sample_batch_flat = sample_batch.to(dev).view(sample_batch.size(0), -1)
    rbm = RBM_TwoPartite(cfg, data=sample_batch_flat)
    logger.info(f"Initialized RBM: {rbm.num_visible} visible, {rbm.num_hidden} hidden units")
    training_mode = cfg.rbm.method
    logger.info(f"Training method: {training_mode}")

    # --- Output Directory Setup ---
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_dir = os.path.join(
        "/home/leozhu/CaloQuVAE/wandb-outputs", 
        f"run_{run_timestamp}_RBM_TwoPartite"
    )
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Saving results to '{save_dir}'")
    
    checkpoint_file = os.path.join(save_dir, "training_checkpoint.h5")

    # --- Training Loop ---
    num_epochs = cfg.n_epochs
    
    for epoch in range(num_epochs):
        logger.info(f"Starting Epoch {epoch+1}/{num_epochs}")

        for batch_idx, (x, _) in enumerate(train_loader):
            v_data = x.to(dev).view(x.size(0), -1)

            
            # 1. Positive phase expectations
            with torch.no_grad():
                mh_data = torch.sigmoid(
                    v_data @ rbm.params["weight_matrix"] + rbm.params["hbias"]
                )

            # 2. Prepare data dict
            data_dict = {
                "v": v_data,
                "mh": mh_data,
                "weights": torch.ones(v_data.shape[0], device=dev),
            }

            # 3. Update
            if training_mode == "CD":
                rbm.chains["v"] = v_data.clone()
                rbm.sample_hidden()
                rbm.fit_batch(data_dict, centered=True)
            if training_mode == "PCD":
                rbm.fit_batch(data_dict, centered=True)
            if training_mode == "CCD":
                rbm.fit_batch_ccd(data_dict, n_cond=cfg.model.cond_p_size, centered=True)
            else:
                raise ValueError(f"Unsupported training method: {training_mode}")

        # --- End of Epoch Logging ---
        logger.info(f"  v_bias: mean={rbm.params['vbias'].mean():.4f}, "
                   f"std={rbm.params['vbias'].std():.4f}")
        logger.info(f"  h_bias: mean={rbm.params['hbias'].mean():.4f}, "
                   f"std={rbm.params['hbias'].std():.4f}")
        logger.info(f"  W: mean={rbm.params['weight_matrix'].mean():.4f}, "
                   f"std={rbm.params['weight_matrix'].std():.4f}")
        logger.info(f"  W: max={rbm.params['weight_matrix'].max():.4f}, "
                   f"min={rbm.params['weight_matrix'].min():.4f}")

        # Save Checkpoint
        if (epoch + 1) % cfg.rbm.checkpoint_interval == 0 or (epoch + 1) == num_epochs:
            try:
                rbm.save_checkpoint(checkpoint_file, epoch, cfg)
                logger.info(f"Checkpoint saved to {checkpoint_file}")
            except Exception as e:
                logger.error(f"Failed to save checkpoint: {e}")

    logger.info("Training complete!")

if __name__ == "__main__":
    main()