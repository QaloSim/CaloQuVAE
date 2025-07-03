import wandb
from utils.HighLevelFeatsAtlasReg import HighLevelFeatures_ATLAS_regular

def plot_calorimeter_shower(cfg, showers, showers_recon, incident_energy, epoch=0, save_dir=None):
    """
    creates calorimeter layer plots for the energy
    """
    # hlf set up:
    HLF = HighLevelFeatures_ATLAS_regular(
        particle=cfg.data.particle,
        filename=cfg.data.binning_path,
        relevantLayers=cfg.data.relevantLayers,
        wandb=True # set to True for wandb
    )

    idx = 0  # event index (pick event)
    real = showers[idx]
    recon = showers_recon[idx]
        
    input_path = f"{save_dir}/val_input_epoch{epoch}.png" if save_dir else None
    recon_path = f"{save_dir}/val_recon_epoch{epoch}.png" if save_dir else None

    # auto save if filename is given
    HLF.DrawSingleShower(real, title=f"Val Input (Epoch {epoch})", filename=input_path)
    HLF.DrawSingleShower(recon, title=f"Val Recon (Epoch {epoch})", filename=recon_path)

    return input_path, recon_path