import numpy as np
import matplotlib.pyplot as plt
import torch
from hydra.utils import instantiate
from hydra import initialize, compose
import hydra
import wandb
from omegaconf import OmegaConf
import copy

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from model.rbm.rbm import RBM, RBM_Hidden, RBM_2Partite
from model.rbm.rbm_torch import RBMtorch
from model.rbm.rbm_fulltorch import RBMTorchFull
from scripts.run import setup_model, load_model_instance

from CaloQuVAE import logging
logger = logging.getLogger(__name__)

import os
from datetime import datetime
import torch.nn.functional as F


def prep_data(batch_size=5000):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x > 0.5).float())
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def preprocess_batch(x, dev, hidden_layer=False, two_partite=False):
    """
    x:   (B, 784) MNIST flat
    dev: torch.device
    hidden_layer: if True, pad width so W%3==0 and chunk into 3 vertical strips
    two_partite: if True, return single flattened tensor for traditional RBM
    returns: tuple of partition tensors, each (B, Nj) or single tensor for 2-partite
    """
    B = x.size(0)
    x = x.to(dev)
    
    if two_partite:
        # For 2-partite RBM, just return the flattened data as a single tensor
        return x.view(B, -1)
    
    x = x.view(B, 1, 28, 28)

    if hidden_layer:
        # pad right so width divisible by 3
        _,_,H,W = x.shape
        pad_w = (3 - (W % 3)) % 3
        # F.pad pads (left, right, top, bottom)
        x = F.pad(x, (0, pad_w, 0, 0), value=0.0)  # now W+pad_w divisible by 3
        # chunk into 3 along width
        parts = torch.chunk(x, 3, dim=3)
    else:
        # original 2×2 split
        top, bottom = torch.chunk(x, 2, dim=2)
        tl, tr = torch.chunk(top,    2, dim=3)
        bl, br = torch.chunk(bottom, 2, dim=3)
        parts = [tl, tr, bl, br]

    # flatten each part
    parts = [p.reshape(B, -1) for p in parts]
    return parts


def stitch_images(parts, n_images, hidden_layer=False, two_partite=False):
    """
    For 2-partite: parts is a single tensor (B, 784)
    For 4-partite: parts is tuple of 4 tensors
    For 3-partite: parts is tuple of 3 tensors  
    returns: list of n_images full numpy images
    """
    samples = []

    if two_partite:
        # parts should be a single tensor (B, 784)
        for i in range(n_images):
            img = parts[i].cpu().numpy().reshape(28, 28)
            samples.append(img)
        return samples

    num_parts = len(parts)
    
    for i in range(n_images):
        if hidden_layer:
            # each part is a vertical strip: height=28, width = Nj/28
            strips = []
            for p in parts[:3]:
                flat = p[i].cpu().numpy()
                H = 28
                W = flat.size // H
                strips.append(flat.reshape(H, W))
            # horizontally concatenate 3 strips
            img = np.concatenate(strips, axis=1)
        else:
            # parts = (tl, tr, bl, br) each 14×14
            tl, tr, bl, br = parts
            tl = tl[i].cpu().numpy().reshape(14,14)
            tr = tr[i].cpu().numpy().reshape(14,14)
            bl = bl[i].cpu().numpy().reshape(14,14)
            br = br[i].cpu().numpy().reshape(14,14)
            top    = np.concatenate([tl, tr], axis=1)
            bottom = np.concatenate([bl, br], axis=1)
            img = np.concatenate([top, bottom], axis=0)

        samples.append(img)
    return samples


def visualize_partitions(parts, n=8, hidden_layer=False, two_partite=False, save_path=None):
    """
    Stitch & show/save n samples from partitions.
    """
    samples = stitch_images(parts, n, hidden_layer, two_partite)
    fig, axes = plt.subplots(1, n, figsize=(n*2, 2))
    for ax, img in zip(axes, samples):
        ax.imshow(img, cmap="gray")
        ax.axis("off")
    if save_path:
        plt.savefig(save_path)
        print(f"Saved samples to {save_path}")
    plt.show()
    plt.close(fig)


def generate_aided_samples(rbm, initial_partitions, clamped_mask):
    """
    Generates samples by clamping some partitions and running Gibbs sampling on the rest.
    This is a wrapper around the RBM's new conditional_gibbs_sampling method.
    """
    # Ensure the RBM class has the new, flexible sampling method
    if not hasattr(rbm, 'conditional_gibbs_sampling'):
        raise NotImplementedError("The RBM model must have a 'conditional_gibbs_sampling' method.")
    
    return rbm.conditional_gibbs_sampling(initial_partitions, clamped_mask)


def visualize_aided_reconstructions(
    rbm,
    test_loader,
    dev,
    n_images=4,
    save_dir=None,
    epoch=0,
    hidden_layer=False,
    two_partite=False
):
    """
    Visualize how the RBM completes partial images or reconstructions.
    """
    # grab a fresh batch
    x_test, _ = next(iter(test_loader))
    originals = preprocess_batch(x_test[:n_images], dev, hidden_layer, two_partite)
    
    if two_partite:
        # For 2-partite RBM, show original vs reconstruction
        reconstructions = rbm.reconstruct(originals)
        
        fig, axes = plt.subplots(2, n_images, figsize=(n_images * 2, 4))
        
        # Row 0: originals
        orig_imgs = stitch_images(originals, n_images, hidden_layer, two_partite)
        for i, img in enumerate(orig_imgs):
            axes[0, i].imshow(img, cmap='gray')
            axes[0, i].set_title("Original")
            axes[0, i].axis('off')

        # Row 1: reconstructions
        recon_imgs = stitch_images(reconstructions, n_images, hidden_layer, two_partite)
        for i, img in enumerate(recon_imgs):
            axes[1, i].imshow(img, cmap='gray')
            axes[1, i].set_title("Reconstruction")
            axes[1, i].axis('off')
            
    else:
        # Original multi-partite logic
        num_parts = len(originals)

        if hidden_layer:
            # clamp just the first strip
            mask1 = [True] + [False] * (num_parts - 1)
            # clamp the first two strips
            mask2 = [True, True] + [False] * (num_parts - 2)

            given_1 = generate_aided_samples(rbm, originals, clamped_mask=mask1)[:3]
            given_2 = generate_aided_samples(rbm, originals, clamped_mask=mask2)[:3]

            # build figure: 3 rows × n_images
            fig, axes = plt.subplots(3, n_images, figsize=(n_images * 2, 6))

            # Row 0: originals
            orig_imgs = stitch_images(originals, n_images, hidden_layer)
            for i, img in enumerate(orig_imgs):
                axes[0, i].imshow(img, cmap='gray')
                axes[0, i].set_title("Original")
                axes[0, i].axis('off')

            # Row 1: given strip 1
            imgs1 = stitch_images(given_1, n_images, hidden_layer)
            for i, img in enumerate(imgs1):
                axes[1, i].imshow(img, cmap='gray')
                axes[1, i].set_title("Given Strip 1")
                axes[1, i].axis('off')

            # Row 2: given strips 1+2
            imgs2 = stitch_images(given_2, n_images, hidden_layer)
            for i, img in enumerate(imgs2):
                axes[2, i].imshow(img, cmap='gray')
                axes[2, i].set_title("Given Strips 1+2")
                axes[2, i].axis('off')

        else:
            # 4-quadrant case
            mask_tl  = [True, False, False, False]
            mask_top = [True, True, False, False]

            given_tl  = generate_aided_samples(rbm, originals,
                                                clamped_mask=mask_tl)
            given_top = generate_aided_samples(rbm, originals,
                                                clamped_mask=mask_top)

            fig, axes = plt.subplots(3, n_images, figsize=(n_images * 2, 6))

            # Row 0: originals
            orig_imgs = stitch_images(originals, n_images, hidden_layer)
            for i, img in enumerate(orig_imgs):
                axes[0, i].imshow(img, cmap='gray')
                axes[0, i].set_title("Original")
                axes[0, i].axis('off')

            # Row 1: clamp top-left
            tl_imgs = stitch_images(given_tl, n_images, hidden_layer)
            for i, img in enumerate(tl_imgs):
                axes[1, i].imshow(img, cmap='gray')
                axes[1, i].set_title("Given TL")
                axes[1, i].axis('off')

            # Row 2: clamp top half
            top_imgs = stitch_images(given_top, n_images, hidden_layer)
            for i, img in enumerate(top_imgs):
                axes[2, i].imshow(img, cmap='gray')
                axes[2, i].set_title("Given Top Half")
                axes[2, i].axis('off')

    plt.tight_layout()

    if save_dir:
        fname = os.path.join(save_dir, f"aided_samples_epoch_{epoch+1}.png")
        plt.savefig(fname)
        print(f"Saved aided samples to {fname}")

    plt.show()
    plt.close(fig)

def main():
    train_loader, test_loader = prep_data()

    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize(version_base=None, config_path="../config")
    config=compose(config_name="config.yaml")

    if config.device == "gpu" and torch.cuda.is_available():
        dev = torch.device(f"cuda:{config.gpu_list[0]}")
    else:
        dev = torch.device("cpu")

    # Check for 2-partite RBM
    two_partite = False
    hidden = False
    
    if config.rbm.two_partite:
        two_partite = True
        config.rbm.latent_nodes_per_p = 784
        logger.info("Using 2-partite RBM model")
        rbm = RBM_2Partite(config).to(dev)
        rbm.initOpt()
    elif config.rbm.hidden_layer:
        config.rbm.latent_nodes_per_p = 280
        hidden = True
        logger.info("Using Hidden RBM model")
        rbm = RBM_Hidden(config).to(dev)
        rbm.initOpt()   
    else:
        config.rbm.latent_nodes_per_p = 196
        hidden = False
        logger.info("Using RBM model")
        rbm = RBM(config).to(dev)
        rbm.initOpt()

    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_type_name = f"{rbm.type()}"
    save_dir = f"generated_samples/run_{run_timestamp}_{model_type_name}"
    wandb_output_path = "/home/leozhu/CaloQuVAE/wandb-outputs"
    save_dir = os.path.join(wandb_output_path, save_dir)
    os.makedirs(save_dir, exist_ok=True)
    print(f"Saving generated images to '{save_dir}'")

    num_epochs = config.n_epochs
    for epoch in range(num_epochs):
        # --- Generate and save full (unconditional) samples ---
        gen_batch_size = 8
        if two_partite:
            full_samples = rbm.sample_visible_2p(gen_batch_size)
        else:
            full_samples = rbm.block_gibbs_sampling(gen_batch_size)
        
        full_sample_path = os.path.join(save_dir, f"full_samples_epoch_{epoch+1}.png")
        visualize_partitions(full_samples, n=gen_batch_size, save_path=full_sample_path, 
                           hidden_layer=hidden, two_partite=two_partite)
        
        # --- Generate and save aided (conditional) samples ---
        test_iterator = iter(test_loader)
        visualize_aided_reconstructions(rbm, test_iterator, dev, n_images=4, save_dir=save_dir, 
                                      epoch=epoch, hidden_layer=hidden, two_partite=two_partite)

        for batch_idx, (x, _) in enumerate(train_loader):
            if batch_idx % 10 == 0:
                logger.info(f"[Epoch {epoch+1}/{num_epochs}] "
                            f"Batch {batch_idx}/{len(train_loader)}")
            
            post_samples = preprocess_batch(x, dev, hidden_layer=hidden, two_partite=two_partite)

            # All models use the same training interface now
            rbm.gradient_rbm_centered(post_samples)
            if config.rbm.SGD:
                rbm.update_params_SGD()
            else:
                rbm.update_params()
                
        print(f"Epoch {epoch+1}/{num_epochs} completed. Generating samples...")
        print(f"  v_bias mean: {rbm._bias_dict['v'].mean():.4f}, std: {rbm._bias_dict['v'].std():.4f}")
        print(f"  h_bias mean: {rbm._bias_dict['h'].mean():.4f}, std: {rbm._bias_dict['h'].std():.4f}")
        print(f"  W mean: {rbm._weight_dict['vh'].mean():.4f}, std: {rbm._weight_dict['vh'].std():.4f}")
        print(f"  W max: {rbm._weight_dict['vh'].max():.4f}, min: {rbm._weight_dict['vh'].min():.4f}")

if __name__=="__main__":
    logger.info("Starting main executable.")
    main()
    logger.info("Finished running script")