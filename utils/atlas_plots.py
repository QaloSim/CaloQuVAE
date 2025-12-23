import wandb
import torch
import importlib
import utils.HighLevelFeatsAtlasReg
importlib.reload(utils.HighLevelFeatsAtlasReg)
from utils.HighLevelFeatsAtlasReg import HighLevelFeatures_ATLAS_regular

from utils.HighLevelFeatures import HighLevelFeatures

def plot_calorimeter_shower(cfg, showers, showers_recon, showers_sampled, epoch, save_dir=None, incidence_energy_choice=None, incidence_energy_gt=None, incidence_energy_generated=None):
    """
    Creates calorimeter slice plots. 
    Returns the figure handles and the specific incidence energy value found for the generated sample.
    """
    # hlf set up:
    dataset_name = cfg.data.dataset_name.lower()
    
    if "atlas" in dataset_name:
        HLF = HighLevelFeatures_ATLAS_regular(
            particle=cfg.data.particle,
            filename=cfg.data.binning_path,
            relevantLayers=cfg.data.relevantLayers
        )
    else:
        HLF = HighLevelFeatures(
            particle=cfg.data.particle,
            filename=cfg.data.binning_path,
            relevantLayers=cfg.data.relevantLayers
        )

    # Variable to store the specific energy found
    found_energy_val = None

    if incidence_energy_choice is not None and incidence_energy_gt is not None and incidence_energy_generated is not None:
            
        # 1. Find index for Ground Truth (and Recon)
        diff_gt = torch.abs(incidence_energy_gt - incidence_energy_choice)
        idx_gt = torch.argmin(diff_gt).item()
        
        # 2. Find index for Sampled (independent search)
        diff_sampled = torch.abs(incidence_energy_generated - incidence_energy_choice)
        idx_sampled = torch.argmin(diff_sampled).item()
        
        # Select the data
        real = showers[idx_gt]
        recon = showers_recon[idx_gt]
        sampled = showers_sampled[idx_sampled]
        
        # Extract values
        e_real_val = incidence_energy_gt[idx_gt].item() if incidence_energy_gt.dim() > 0 else incidence_energy_gt.item()
        e_sample_val = incidence_energy_generated[idx_sampled].item() if incidence_energy_generated.dim() > 0 else incidence_energy_generated.item()
        
        # Save the found energy to return later
        found_energy_val = e_sample_val
        
        title_suffix_real = f" (Incidence Energy: {e_real_val:.1f} MeV)"
        title_suffix_sample = f" (Incidence Energy: {e_sample_val:.1f} MeV)"

    else:
        # Fallback to original logic
        idx = showers.sum(dim=1).argsort()[-2]
        real = showers[idx]
        recon = showers_recon[idx]
        sampled = showers_sampled[idx]
        
        title_suffix_real = ""
        title_suffix_sample = ""
        
        # If no specific energy was searched, we return None or the energy at this fallback index if available
        found_energy_val = None 

    real_avg = showers.mean(dim=0)
    recon_avg = showers_recon.mean(dim=0)
    sampled_avg = showers_sampled.mean(dim=0)
        
    input_path = f"{save_dir}/val_input_epoch{epoch}.png" if save_dir else None
    recon_path = f"{save_dir}/val_recon_epoch{epoch}.png" if save_dir else None
    sample_path = f"{save_dir}/val_sampled_epoch{epoch}.png" if save_dir else None

    # images
    image_input = HLF.DrawSingleShower(real, title=f"Val Input (Epoch {epoch}){title_suffix_real}", filename=input_path, cmap='rainbow')
    image_recon = HLF.DrawSingleShower(recon, title=f"Val Recon (Epoch {epoch}){title_suffix_real}", filename=recon_path, cmap='rainbow')
    image_sample = HLF.DrawSingleShower(sampled, title=f"Val Sampled (Epoch {epoch}){title_suffix_sample}", filename=sample_path, cmap='rainbow')
    
    image_input_avg = HLF.DrawSingleShower(real_avg, title=f"Val Input (Epoch {epoch})", filename=input_path, cmap='rainbow')
    image_recon_avg = HLF.DrawSingleShower(recon_avg, title=f"Val Recon (Epoch {epoch})", filename=recon_path, cmap='rainbow')
    image_sample_avg = HLF.DrawSingleShower(sampled_avg, title=f"Val Sampled (Epoch {epoch})", filename=sample_path, cmap='rainbow')
    
    # Single-layer with highlighted patches (ATLAS only)
    if "atlas" in dataset_name:
        highlight_coords = [(0, 0)] + [(r_, phi_) for r_ in [4, 10, 15] for phi_ in [0, 3, 6, 9]]
        HLF.plot_single_layer_with_highlights(
            data=real,
            layer=0,
            r=cfg.data.r,
            phi=cfg.data.phi,
            highlight_coords=highlight_coords,
            title=f"(Epoch {epoch}) Highlighted Voxels")

    return image_input, image_recon, image_sample, image_input_avg, image_recon_avg, image_sample_avg, found_energy_val