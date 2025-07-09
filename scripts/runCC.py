"""
Run file adapted for Compute Canada clusters (e.g., Narval, Beluga).

Differences from main.py:
- WandB is disabled entirely
- Device is selected automatically
- Paths should be configured in the Hydra config file
"""

import os
import sys
import torch
import numpy as np
import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from torch import device as torch_device
from torch.cuda import is_available

from CaloQuVAE import logging
logger = logging.getLogger(__name__)

from data.dataManager import DataManager
from model.modelCreator import ModelCreator
from engine.engine import Engine

import wandb
wandb.init(mode="disabled")

@hydra.main(config_path="../config", config_name="configCC", version_base=None)
def main(cfg):
    print(OmegaConf.to_yaml(cfg, resolve=True))
    os.chdir(os.path.dirname(os.path.abspath(__file__)))  # Prevent Hydra from changing dir

    # Setup model and training infrastructure
    engine = setup_model(cfg)
    run(engine)

def set_device():
    """Automatically select CUDA if available, else CPU."""
    dev = torch_device("cuda" if is_available() else "cpu")
    logger.info(f"Using device: {dev}")
    return dev

def setup_model(config):
    dataMgr = DataManager(config)
    modelCreator = ModelCreator(config)

    # Instantiate model
    model = modelCreator.init_model()
    model.create_networks()
    model.print_model_info()
    model.prior._n_batches = len(dataMgr.train_loader) - 1

    # Select device
    dev = set_device()
    model.to(dev)

    # Set up engine
    engine = instantiate(config.engine, config)
    engine.data_mgr = dataMgr
    engine.device = dev
    engine.optimiser = torch.optim.Adam(model.parameters(), lr=config.engine.learning_rate)
    engine.model = model
    engine.model_creator = modelCreator

    # Freeze prior if needed
    for name, param in model.named_parameters():
        if 'prior' in name:
            param.requires_grad = False

    return engine

def run(engine):
    cfg = engine._config

    for epoch in range(cfg.epoch_start, cfg.n_epochs):
        engine.fit(epoch)
        engine.evaluate(engine.data_mgr.val_loader, epoch)

        if cfg.freeze_vae and epoch > cfg.epoch_freeze:
            for name, param in engine.model.named_parameters():
                if 'decoder' in name or 'encoder' in name:
                    param.requires_grad = False
            engine._save_model(name="at_freezing_point")
            engine._config.rbm.method = "PCD"
            logger.info(f"RBM will use {engine._config.model.rbmMethod}")

        if epoch % 10 == 0:
            engine._save_model(name=str(epoch))

    engine.evaluate(engine.data_mgr.test_loader, 0)

if __name__ == "__main__":
    logger.info("Starting CC-compatible run.")
    main()
    logger.info("Finished.")
