import numpy as np
import matplotlib.pyplot as plt
import torch
from hydra.utils import instantiate
from hydra import initialize, compose
import hydra
import wandb
from omegaconf import OmegaConf

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model.rbm.rbm_two_partite import RBM_TwoPartite, ZephyrRBM_TwoPartite

from CaloQuVAE import logging
logger = logging.getLogger(__name__)

import os
from datetime import datetime
import torch.nn.functional as F
from utils.latent_dataset import LatentDataset
import math


def prep_data(batch_size=5000):
    """Load and binarize MNIST"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x > 0.5).float())
    ])
    train_dataset = datasets.MNIST(
        root='./data', train=True, transform=transform, download=True
    )
    test_dataset = datasets.MNIST(
        root='./data', train=False, transform=transform, download=True
    )
    
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = DataLoader(
        dataset=test_dataset, batch_size=batch_size, shuffle=False
    )
    
    return train_loader, test_loader

def prep_latent_data(data_path, batch_size=5000):
    """
    Load and (optionally) binarize the saved latent data.
    
    Args:
        data_path (string): Path to the *data* file (e.g., ..._train_data_...).
                            The label path will be inferred automatically.
        batch_size (int): Batch size for the DataLoader.
    """
    
    # Binarization transform, just like for MNIST.
    # Set to None if you don't want to binarize your latent samples.
    transform = transforms.Lambda(lambda x: (x > 0.5).float())
    
    # 1. Load the LatentDataset
    dataset = LatentDataset(
        data_path=data_path, 
        transform=None
    )
    
    # 2. Create the DataLoader
    loader = DataLoader(
        dataset=dataset, 
        batch_size=batch_size, 
        shuffle=True  
    )
    
    return loader, loader  # Return same loader for train and test for simplicity

def preprocess_batch_2partite(x, dev):
    """
    Flatten MNIST images for 2-partite RBM.
    x: (B, 1, 28, 28) or (B, 784)
    returns: (B, 784) flattened tensor
    """
    B = x.size(0)
    x = x.to(dev)
    return x.view(B, -1)


def visualize_samples_2partite(samples_tensor, n=8, save_path=None):
    """
    samples_tensor: (B, 784)
    Reshape and visualize as 28x28 images.
    """
    samples_tensor = samples_tensor.cpu()
    fig, axes = plt.subplots(1, n, figsize=(n * 2, 2))
    
    for i in range(n):
        img = samples_tensor[i].numpy().reshape(28, 28)
        axes[i].imshow(img, cmap="gray")
        axes[i].axis("off")
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved samples to {save_path}")
    
    plt.show()
    plt.close(fig)


def visualize_reconstructions(rbm, test_loader, dev, n_images=4, save_dir=None, epoch=0):
    """
    Show original MNIST images vs RBM reconstructions.
    This version safely handles the RBM's persistent chains.
    """
    # --- 1. Backup the RBM's original chains ---
    original_chains = {key: val.clone() for key, val in rbm.chains.items()}

    x_test, _ = next(iter(test_loader))
    x_test = x_test[:n_images]
    originals = preprocess_batch_2partite(x_test, dev)
    
    # --- 2. Perform the reconstruction ---
    # Temporarily overwrite the chains for this specific task
    with torch.no_grad():
        rbm.chains["v"] = originals
        rbm.sample_hidden()
        rbm.sample_visibles()
        reconstructions = rbm.chains["v"]
    
    # --- 3. Restore the original chains ---
    # This ensures the RBM's state is unchanged for subsequent operations
    rbm.chains = original_chains

    # --- Visualization (no changes here) ---
    fig, axes = plt.subplots(2, n_images, figsize=(n_images * 2, 4))
    
    for i in range(n_images):
        img = originals[i].cpu().numpy().reshape(28, 28)
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title("Original")
        axes[0, i].axis('off')
    
    for i in range(n_images):
        img = reconstructions[i].cpu().numpy().reshape(28, 28)
        axes[1, i].imshow(img, cmap='gray')
        axes[1, i].set_title("Reconstruction")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    if save_dir:
        fname = os.path.join(save_dir, f"reconstructions_epoch_{epoch+1}.png")
        plt.savefig(fname)
        print(f"Saved reconstructions to {fname}")
    
    plt.show()
    plt.close(fig)

def generate_fantasy_samples(rbm, n_samples=8, burn_in=100):
    """
    Generate n_samples from the RBM model by running parallel Gibbs chains.
    This version assumes the RBM's internal number of chains is >= n_samples.

    Args:
        rbm: RBM instance.
        n_samples: Number of independent samples to generate.
        burn_in: Number of Gibbs steps for burn-in.

    Returns:
        Tensor of shape (n_samples, n_visibles) with generated samples.
    """
    # --- 1. Backup RBM's persistent training chains ---
    # This ensures the sampling process doesn't interfere with training.
    original_chains = {key: val.clone() for key, val in rbm.chains.items()}

    # --- 2. Initialize new random chains for independent sampling ---
    # This uses the RBM's configured number of chains.
    rbm.reset_chains()

    # --- 3. Run burn-in for all chains simultaneously ---
    # A single, parallel Gibbs sampling process is run.
    for _ in range(burn_in):
        rbm.sample_hidden()
        rbm.sample_visibles()

    # --- 4. Get the required number of samples by slicing ---
    # We take the first n_samples from the batch of generated chains.
    samples_tensor = rbm.chains["v"][:n_samples].clone()

    # --- 5. Restore the original persistent chains for training ---
    rbm.chains = original_chains
    # print(samples_tensor.shape)

    return samples_tensor


def visualize_rbm_samples_grid(samples, label="mnist", save_dir=None):
    """
    Visualize the first 100 RBM-generated samples (10x10 grid) and save to /images/.

    Args:
        samples (numpy.ndarray | torch.Tensor): Generated visible samples, shape [N, 784].
            Each row should represent a flattened 28x28 image.
        label (str): Optional descriptive label to include in the saved filename.
    """
    # Convert torch tensor to numpy if necessary
    if torch.is_tensor(samples):
        samples = samples.detach().cpu().numpy()

    assert len(samples) >= 100, "Need at least 100 samples to create grid."


    filename = f"rbm_samples_{label}_grid.png"
    file_path = os.path.join(save_dir, filename)

    # Create 10x10 grid
    fig, axes = plt.subplots(10, 10, figsize=(10, 10))
    for ax, img in zip(axes.flat, samples[:100]):
        ax.imshow(img.reshape(28, 28), cmap="gray")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(file_path, dpi=150)
    plt.show()

    logger.info(f"Saved sample grid to: {file_path}")


def save_fantasy_samples_for_vae(rbm, n_samples, burn_in, save_dir):
    """
    Generates n_samples fantasy samples and saves them in a format
    loadable by LatentDataset for the VAE.
    
    This runs the RBM's chains in batches to accumulate n_samples,
    regardless of the RBM's internal num_chains.
    
    Args:
        rbm: The trained RBM model.
        n_samples (int): The total number of samples to generate.
        burn_in (int): The number of Gibbs steps for *each batch*.
        save_dir (str): Directory to save the files.
    """
    logger.info(f"Attempting to generate {n_samples} fantasy samples for VAE...")
    
    # Determine the batch size for generation from the RBM's chain count
    # This is the max number of samples we can get per batch.
    gen_batch_size = rbm.chains["v"].shape[0]
    
    if gen_batch_size == 0:
        logger.error("RBM chains have not been initialized or have size 0.")
        raise ValueError("RBM chains are size 0. Cannot generate samples.")

    # Calculate number of batches needed
    n_batches = math.ceil(n_samples / gen_batch_size)
    
    all_samples = []
    
    logger.info(f"RBM num_chains = {gen_batch_size}. "
                f"Will generate {n_samples} samples in {n_batches} batches.")
    
    for i in range(n_batches):
        # We run the independent sampling process 'n_batches' times
        # Each run is independently burned in by generate_fantasy_samples
        logger.info(f"Generating batch {i+1}/{n_batches} (burn-in={burn_in} steps)...")
        
        # generate_fantasy_samples re-initializes, burns in,
        # and returns 'gen_batch_size' samples.
        batch_samples = generate_fantasy_samples(
            rbm, 
            n_samples=gen_batch_size, 
            burn_in=burn_in
        )
        all_samples.append(batch_samples)

    # Concatenate all batches
    all_samples_tensor = torch.cat(all_samples, dim=0)
    
    # Truncate to the exact number of samples requested
    final_samples = all_samples_tensor[:n_samples]

    # --- Save the samples and dummy labels ---
    
    # 1. Define file paths (using the same convention)
    data_filename = f"rbm_fantasy_samples_train_data_final.pt"
    label_filename = f"rbm_fantasy_samples_train_labels_final.pt"
    
    data_path = os.path.join(save_dir, data_filename)
    label_path = os.path.join(save_dir, label_filename)
    
    # 2. Create dummy labels (to satisfy LatentDataset)
    # Shape is (N_samples, 1) to match incident_energy
    dummy_labels = torch.zeros((final_samples.shape[0], 1), dtype=torch.float32)

    # 3. Save both tensors
    torch.save(final_samples.cpu(), data_path)
    torch.save(dummy_labels, label_path)
    
    logger.info("="*50)
    logger.info(f"Saved fantasy samples for VAE to: {data_path}")
    logger.info(f"Saved dummy labels to: {label_path}")
    logger.info(f"Final data shape: {final_samples.shape}")
    logger.info("="*50)
    
    return data_path


def main():
    # Initialize Hydra config
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize(version_base=None, config_path="../config")
    cfg = compose(config_name="config.yaml")
    config = OmegaConf.load(cfg.config_path)
    config.gpu_list = cfg.gpu_list
    if not config.rbm.use_latent_data:
        config.rbm.latent_nodes_per_p = 784  # For MNIST

    if config.device == "gpu" and torch.cuda.is_available():
        dev = torch.device(f"cuda:{config.gpu_list[0]}")
    else:
        dev = torch.device("cpu")

    logger.info("Using 2-partite RBM model with new RBM_TwoPartite class")
    if config.rbm.use_latent_data:
        train_loader, test_loader = prep_latent_data(data_path=config.rbm.latent_data_path)
    else:
        train_loader, test_loader = prep_data()


    # Get a sample batch to initialize RBM parameters
    sample_batch, _ = next(iter(train_loader))
    sample_batch_flat = preprocess_batch_2partite(sample_batch, dev)

    # Initialize your new RBM class
    if config.rbm.fullyconnected:
        rbm = RBM_TwoPartite(config, data=sample_batch_flat)
    else:
        rbm = ZephyrRBM_TwoPartite(config, data=sample_batch_flat)

    logger.info(f"Initialized RBM: {rbm.p_size} visible, {rbm.p_size} hidden units")
    logger.info(f"Device: {rbm.device}")
    logger.info(f"Num chains: {config.rbm.num_chains}")

    # Setup save directory
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_type_name = "RBM_TwoPartite"
    wandb_output_path = config.config_path.split("files")[0] + "files"
    save_dir = os.path.join(
        wandb_output_path, f"run_{run_timestamp}_{model_type_name}"
    )
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Saving results to '{save_dir}'")

    num_epochs = config.n_epochs

    for epoch in range(num_epochs):
        logger.info(f"Starting Epoch {epoch+1}/{num_epochs}")

        # --- Training loop ---
        for batch_idx, (x, _) in enumerate(train_loader):
            if batch_idx % 100 == 0:
                logger.info(
                    f"[Epoch {epoch+1}/{num_epochs}] "
                    f"Batch {batch_idx}/{len(train_loader)}"
                )

            # Preprocess batch
            v_data = preprocess_batch_2partite(x, dev)

            if config.rbm.method == "CD":
                rbm.chains["v"] = v_data.clone()
                rbm.sample_hidden()
            
            # Positive phase: compute hidden expectations
            with torch.no_grad():
                mh_data = torch.sigmoid(
                    v_data @ rbm.params["weight_matrix"] + rbm.params["hbias"]
                )

            # Build data dict
            data_dict = {
                "v": v_data,
                "mh": mh_data,
                "weights": torch.ones(v_data.shape[0], device=dev),
            }

            # Training step
            rbm.fit_batch(data_dict, centered=True)

        # --- End of epoch: generate samples and log ---
        logger.info(f"\nEpoch {epoch+1} completed. Generating samples...")
        
        # Parameter statistics (these are fine to log)
        logger.info(f"  v_bias: mean={rbm.params['vbias'].mean():.4f}, "
                   f"std={rbm.params['vbias'].std():.4f}")
        logger.info(f"  h_bias: mean={rbm.params['hbias'].mean():.4f}, "
                   f"std={rbm.params['hbias'].std():.4f}")
        logger.info(f"  W: mean={rbm.params['weight_matrix'].mean():.4f}, "
                   f"std={rbm.params['weight_matrix'].std():.4f}")
        logger.info(f"  W: max={rbm.params['weight_matrix'].max():.4f}, "
                   f"min={rbm.params['weight_matrix'].min():.4f}")
                   
        # ==========================================================
        # ::: MODIFICATION HERE :::
        # Only run visualizations if we are *not* using latent data (i.e., we are using MNIST)
        if not config.rbm.use_latent_data:
            if epoch % 20 == 0:
                logger.info("Generating MNIST-specific visualizations...")
                # Generate fantasy samples from model
                fantasy_samples = generate_fantasy_samples(rbm, n_samples=8, burn_in=config.rbm.bgs_steps)
                sample_path = os.path.join(
                    save_dir, f"fantasy_samples_epoch_{epoch+1}.png"
                )
                visualize_samples_2partite(
                    fantasy_samples, n=8, save_path=sample_path
                )

                # Generate reconstructions
                visualize_reconstructions(
                    rbm, test_loader, dev, n_images=4, save_dir=save_dir, epoch=epoch
                )
            if epoch % 100 == 0:
                logger.info("Generating MNIST sample grid...")
                fantasy_samples = generate_fantasy_samples(rbm, n_samples=100, burn_in=10000)
                visualize_rbm_samples_grid(fantasy_samples, label=str(epoch), save_dir=save_dir)
    if not config.rbm.use_latent_data:
        logger.info("Generating final MNIST sample grid...")
        fantasy_samples = generate_fantasy_samples(rbm, n_samples=100, burn_in=10000)
        visualize_rbm_samples_grid(fantasy_samples, label="mnist_final", save_dir=save_dir)
    else:
        saved_data_path = save_fantasy_samples_for_vae(
                rbm, 
                n_samples=50000, 
                burn_in=10000, 
                save_dir=save_dir
            )
        logger.info(f"VAE data ready at: {saved_data_path}")
        logger.info("\n" + "="*50)
    logger.info("Training complete!")
    logger.info("="*50)


if __name__ == "__main__":
    logger.info("Starting main executable.")
    main()
    logger.info("Finished running script")