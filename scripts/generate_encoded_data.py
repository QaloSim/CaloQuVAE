import numpy as np
# import matplotlib.pyplot as plt
import os
from datetime import datetime

import torch
from hydra.utils import instantiate
from hydra import initialize, compose
import hydra

import wandb

from data.dataManager import DataManager
from model.modelCreator import ModelCreator
from omegaconf import OmegaConf
from scripts.run import setup_model, load_model_instance

# from utils.plots import vae_plots
# from utils.rbm_plots import plot_rbm_histogram, plot_rbm_params, plot_forward_output_v2

from scripts.run import set_device

from CaloQuVAE import logging
logger = logging.getLogger(__name__)

def main():
    initialize(version_base=None, config_path="../config")
    config=compose(config_name="config.yaml")
    wandb.init(tags = [config.data.dataset_name], project=config.wandb.project, 
           entity=config.wandb.entity, config=OmegaConf.to_container(config, resolve=True), 
           mode='disabled')
    

    self = load_model_instance(config)
    self.evaluate_vae(self.data_mgr.val_loader,0)

    if config.rbm.get("encoded_data_prefix") is not None:
        data_dir = config.rbm.encoded_data_prefix
    else:
        data_dir = "/fast_scratch_1/caloqvae/rbm_encoded_data"
    os.makedirs(data_dir, exist_ok=True)
    save_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Save the latent samples (the new "images")
    latent_dir1 = os.path.join(data_dir, f'latent_train_data_{save_timestamp}.pt')
    torch.save(self.post_samples, latent_dir1)
    # Save the incident energy (the new "labels")
    latent_dir2 = os.path.join(data_dir, f'latent_train_labels_{save_timestamp}.pt')
    torch.save(self.incident_energy, latent_dir2)
    logger.info(f"Saved training latents of shape {self.post_samples.shape} to {latent_dir1}")
    logger.info(f"Saved training labels of shape {self.incident_energy.shape} to {latent_dir2}")

    self._config.rbm.latent_data_path = latent_dir1
    config_path = self._config.config_path.split("files")[0] + "files/RBM_config_file.yaml"
    OmegaConf.save(self._config, config_path, resolve=True)
    logger.info(f"Saved RBM config file to {config_path}")



if __name__ == "__main__":
    logger.info("Starting main executable.")
    main()
    logger.info("Finished main executable.")