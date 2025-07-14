import wandb
from utils.HighLevelFeatsAtlasReg import HighLevelFeatures_ATLAS_regular
from utils.HighLevelFeatures import HighLevelFeatures

def plot_calorimeter_shower(cfg, showers, showers_recon, showers_sampled, epoch, save_dir=None):
    """
    creates calorimeter slice plots and can be used for either the calo or atlas datasets
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

    # idx = 0  # event index (pick event)
    idx = showers.sum(dim=1).argsort()[-2]
    real = showers[idx]
    recon = showers_recon[idx]
    sampled = showers_sampled[idx]

    real_avg = showers.mean(dim=0)
    recon_avg = showers_recon.mean(dim=0)
    sampled_avg = showers_sampled.mean(dim=0)
        
    input_path = f"{save_dir}/val_input_epoch{epoch}.png" if save_dir else None
    recon_path = f"{save_dir}/val_recon_epoch{epoch}.png" if save_dir else None
    sample_path = f"{save_dir}/val_sampled_epoch{epoch}.png" if save_dir else None

    # images
    image_input = HLF.DrawSingleShower(real, title=f"Val Input (Epoch {epoch})", filename=input_path, cmap='rainbow')
    image_recon = HLF.DrawSingleShower(recon, title=f"Val Recon (Epoch {epoch})", filename=recon_path, cmap='rainbow')
    image_sample = HLF.DrawSingleShower(sampled, title=f"Val Sampled (Epoch {epoch})", filename=sample_path, cmap='rainbow')

    image_input_avg = HLF.DrawSingleShower(real_avg, title=f"Val Input (Epoch {epoch})", filename=input_path, cmap='rainbow')
    image_recon_avg = HLF.DrawSingleShower(recon_avg, title=f"Val Recon (Epoch {epoch})", filename=recon_path, cmap='rainbow')
    image_sample_avg = HLF.DrawSingleShower(sampled_avg, title=f"Val Sampled (Epoch {epoch})", filename=sample_path, cmap='rainbow')


    return image_input, image_recon, image_sample, image_input_avg, image_recon_avg, image_sample_avg