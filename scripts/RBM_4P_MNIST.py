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

from model.rbm.rbm_four_partite import RBM_FourPartite

from CaloQuVAE import logging
logger = logging.getLogger(__name__)

import os
from datetime import datetime
import torch.nn.functional as F


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


def preprocess_batch_4partite(x, dev):
    """
    Flatten MNIST images for 4-partite RBM.
    x: (B, 1, 28, 28)
    returns: (B, 784) flattened tensor (will be split into 4x196)
    """
    B = x.size(0)
    x = x.to(dev)
    return x.view(B, -1)


def stitch_images_4partite(partitions, n_images):
    """
    Stitch 4 partitions back into 28x28 images.
    
    Args:
        partitions: tuple of 4 tensors, each (n_images, 196)
        n_images: number of images to stitch
        
    Returns:
        list of (28, 28) numpy arrays
    """
    images = []
    for i in range(n_images):
        # Concatenate all 4 partitions for this image
        full_img = torch.cat([p[i] for p in partitions], dim=0)
        # Reshape to 28x28
        img = full_img.cpu().numpy().reshape(28, 28)
        images.append(img)
    return images


def generate_aided_samples(rbm, original_batch, clamped_mask, burn_in=100):
    """
    Generate samples with some partitions clamped to original values.
    
    Args:
        rbm: RBM_FourPartite instance
        original_batch: (B, 784) tensor
        clamped_mask: [bool, bool, bool, bool] - which partitions to clamp
        burn_in: number of Gibbs steps
        
    Returns:
        tuple of 4 tensors (p0, p1, p2, p3)
    """
    p_size = rbm.p_size
    batch_size = original_batch.shape[0]
    
    # Split original into 4 partitions
    initial_partitions = tuple(
        original_batch[:, i*p_size:(i+1)*p_size] for i in range(4)
    )
    
    # Use conditional sampling
    result = rbm.sample_conditional(
        original_batch, 
        tuple(clamped_mask), 
        beta=1.0
    )
    
    return result


def visualize_aided_reconstructions_4partite(
    rbm,
    test_loader,
    dev,
    n_images=4,
    save_dir=None,
    epoch=0
):
    """
    Visualize how the RBM completes partial images with clamping.
    Shows: Original, Given TL quadrant, Given Top Half
    """
    # Grab a fresh batch
    x_test, _ = next(iter(test_loader))
    originals = preprocess_batch_4partite(x_test[:n_images], dev)
    
    p_size = rbm.p_size
    
    # Split originals into partitions for visualization
    orig_parts = tuple(
        originals[:, i*p_size:(i+1)*p_size] for i in range(4)
    )
    
    # 4-quadrant case: partition layout
    # p0 = top-left, p1 = top-right, p2 = bottom-left, p3 = bottom-right
    mask_tl = [True, False, False, False]   # Clamp top-left quadrant
    mask_top = [True, True, False, False]   # Clamp top half
    
    # Generate completions
    given_tl = generate_aided_samples(rbm, originals, clamped_mask=mask_tl)
    given_top = generate_aided_samples(rbm, originals, clamped_mask=mask_top)
    
    # Create figure: 3 rows × n_images
    fig, axes = plt.subplots(3, n_images, figsize=(n_images * 2, 6))
    
    # Row 0: originals
    orig_imgs = stitch_images_4partite(orig_parts, n_images)
    for i, img in enumerate(orig_imgs):
        axes[0, i].imshow(img, cmap='gray')
        axes[0, i].set_title("Original")
        axes[0, i].axis('off')
    
    # Row 1: given top-left quadrant
    tl_imgs = stitch_images_4partite(given_tl, n_images)
    for i, img in enumerate(tl_imgs):
        axes[1, i].imshow(img, cmap='gray')
        axes[1, i].set_title("Given TL Quadrant")
        axes[1, i].axis('off')
    
    # Row 2: given top half
    top_imgs = stitch_images_4partite(given_top, n_images)
    for i, img in enumerate(top_imgs):
        axes[2, i].imshow(img, cmap='gray')
        axes[2, i].set_title("Given Top Half")
        axes[2, i].axis('off')
    
    plt.tight_layout()
    
    if save_dir:
        fname = os.path.join(save_dir, f"aided_samples_epoch_{epoch+1}.png")
        plt.savefig(fname)
        logger.info(f"Saved aided samples to {fname}")
    
    plt.show()
    plt.close(fig)


def visualize_samples_4partite(samples_tensor, n=8, save_path=None):
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
        logger.info(f"Saved samples to {save_path}")
    
    plt.show()
    plt.close(fig)


def generate_fantasy_samples_4partite(rbm, n_samples=8, burn_in=100):
    """Generate samples from 4-partite RBM."""
    original_chains = {key: val.clone() for key, val in rbm.chains.items()}
    rbm.reset_chains()

    for _ in range(burn_in):
        for i in range(4):
            rbm.sample_partition(i)

    # Concatenate all 4 partitions
    samples_tensor = torch.cat(
        [rbm.chains[f"p{i}"][:n_samples] for i in range(4)], dim=1
    ).clone()
    
    rbm.chains = original_chains

    return samples_tensor


def visualize_rbm_samples_grid(samples, label="mnist", save_dir=None):
    """
    Visualize the first 100 RBM-generated samples (10x10 grid).
    """
    if torch.is_tensor(samples):
        samples = samples.detach().cpu().numpy()

    assert len(samples) >= 100, "Need at least 100 samples to create grid."
    assert samples.shape[1] == 784, f"Expected 784 features, got {samples.shape[1]}"

    filename = f"rbm_samples_{label}_grid.png"
    file_path = os.path.join(save_dir, filename)

    fig, axes = plt.subplots(10, 10, figsize=(10, 10))
    for ax, img in zip(axes.flat, samples[:100]):
        ax.imshow(img.reshape(28, 28), cmap="gray")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(file_path, dpi=150)
    plt.show()

    logger.info(f"Saved sample grid to: {file_path}")


def main():
    train_loader, test_loader = prep_data()

    # Initialize Hydra config
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize(version_base=None, config_path="../config")
    config = compose(config_name="config.yaml")
    
    # Force 4-partite setup
    # MNIST is 784 pixels = 4 partitions × 196 pixels each
    config.rbm.latent_nodes_per_p = 196
    logger.info("Using 4-partite RBM with 196 nodes per partition")

    if config.device == "gpu" and torch.cuda.is_available():
        dev = torch.device(f"cuda:{config.gpu_list[0]}")
    else:
        dev = torch.device("cpu")

    # Get a sample batch to initialize RBM parameters
    sample_batch, _ = next(iter(train_loader))
    sample_batch_flat = preprocess_batch_4partite(sample_batch, dev)

    # Initialize 4-partite RBM
    if config.rbm.fullyconnected:
        rbm = RBM_FourPartite(config, data=sample_batch_flat)
        model_type_name = "RBM_FourPartite_FC"
    else:
        rbm = ZephyrRBM_FourPartite(config, data=sample_batch_flat)
        model_type_name = "RBM_FourPartite_Zephyr"

    logger.info(f"Initialized {model_type_name}")
    logger.info(f"Partition size: {rbm.p_size} nodes per partition")
    logger.info(f"Total nodes: {4 * rbm.p_size}")
    logger.info(f"Device: {rbm.device}")
    logger.info(f"Num chains: {config.rbm.num_chains}")
    logger.info(f"Gibbs steps: {config.rbm.bgs_steps}")

    # Setup save directory
    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    wandb_output_path = "/home/leozhu/CaloQuVAE/wandb-outputs"
    save_dir = os.path.join(
        wandb_output_path, f"run_{run_timestamp}_{model_type_name}"
    )
    os.makedirs(save_dir, exist_ok=True)
    logger.info(f"Saving results to '{save_dir}'")

    num_epochs = config.n_epochs
    centered = config.rbm.centered

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
            batch_data = preprocess_batch_4partite(x, dev)
            p_size = config.rbm.latent_nodes_per_p
            
            # Handle CD initialization
            if config.rbm.method == "CD":
                # Initialize chains with data for Contrastive Divergence
                for i in range(4):
                    rbm.chains[f"p{i}"] = batch_data[
                        :, i*p_size:(i+1)*p_size
                    ].clone()
                # One Gibbs step for CD-1 (or bgs_steps for CD-k)
                rbm.sample_state()
            
            # Build data dict
            data_dict = {
                f"p{i}": batch_data[:, i*p_size:(i+1)*p_size]
                for i in range(4)
            }
            data_dict["weights"] = torch.ones(batch_data.shape[0], device=dev)

            # Training step
            rbm.fit_batch(data_dict, centered=centered)

        # --- End of epoch: generate samples and log ---
        logger.info(f"\nEpoch {epoch+1} completed. Logging statistics...")
        
        # Parameter statistics
        for i in range(4):
            bias = rbm.params[f"bias_{i}"]
            logger.info(
                f"  bias_{i}: mean={bias.mean():.4f}, "
                f"std={bias.std():.4f}, "
                f"max={bias.max():.4f}, "
                f"min={bias.min():.4f}"
            )
        
        for i in range(4):
            for j in range(i+1, 4):
                weight = rbm.params[f"weight_{i}{j}"]
                logger.info(
                    f"  W_{i}{j}: mean={weight.mean():.4f}, "
                    f"std={weight.std():.4f}, "
                    f"max={weight.max():.4f}, "
                    f"min={weight.min():.4f}"
                )
                   
        if epoch % 20 == 0:
            # Generate fantasy samples from model
            logger.info("Generating fantasy samples...")
            fantasy_samples = generate_fantasy_samples_4partite(
                rbm, n_samples=8, burn_in=config.rbm.bgs_steps
            )
            sample_path = os.path.join(
                save_dir, f"fantasy_samples_epoch_{epoch+1}.png"
            )
            visualize_samples_4partite(
                fantasy_samples, n=8, save_path=sample_path
            )

            # Generate aided reconstructions (with clamping)
            logger.info("Generating aided reconstructions...")
            visualize_aided_reconstructions_4partite(
                rbm, test_loader, dev, n_images=4, 
                save_dir=save_dir, epoch=epoch
            )
            
        if epoch % 100 == 0:
            logger.info("Generating 100-sample grid (long burn-in)...")
            fantasy_samples = generate_fantasy_samples_4partite(
                rbm, n_samples=100, burn_in=10000
            )
            visualize_rbm_samples_grid(
                fantasy_samples, label=f"epoch_{epoch}", save_dir=save_dir
            )
    
    # Final sample generation
    logger.info("\n" + "="*50)
    logger.info("Training complete! Generating final samples...")
    logger.info("="*50)
    
    fantasy_samples = generate_fantasy_samples_4partite(
        rbm, n_samples=100, burn_in=10000
    )
    visualize_rbm_samples_grid(
        fantasy_samples, label="mnist_final", save_dir=save_dir
    )

    logger.info("\nAll results saved to: " + save_dir)
    logger.info("Training complete!")


if __name__ == "__main__":
    logger.info("Starting 4-partite RBM training script.")
    main()
    logger.info("Finished running script")