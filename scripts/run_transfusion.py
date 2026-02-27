import torch
torch.manual_seed(32)
import numpy as np
np.random.seed(32)
import hydra
from hydra.utils import instantiate

from omegaconf import OmegaConf, open_dict

from torch import device
from torch.nn import DataParallel
from torch.cuda import is_available

import wandb

from CaloQuVAE import logging
logger = logging.getLogger(__name__)

from data.dataManager import DataManagerLayers
from model.modelCreator import ModelCreator
from engine.engine_transfusion import EngineTransfusion
from scripts.run import set_device


@hydra.main(config_path="../config", config_name="tfusion_config", version_base=None)
def main(cfg=None):
    mode = getattr(cfg.wandb, 'mode', 'online')
    wandb.init(tags = [cfg.data.dataset_name], project=cfg.wandb.project, entity=cfg.wandb.entity, config=OmegaConf.to_container(cfg, resolve=True), mode=mode)
    if wandb.run is not None and mode != "disabled":
        save_dir = wandb.run.dir
    else:
        # Fallback to config-specified directory if wandb is not being used
        save_dir = getattr(cfg, 'save_dir', None)
        if save_dir is None:
            raise ValueError("No save directory specified in config, and wandb is not being used. Cannot save model state.")

    cfg.save_dir = save_dir
    engine = setup_model(config=cfg)
    wandb.watch(engine.model)

    run(engine)


def setup_model(config=None):
    """
    Set up transfusion model
    """
    dataMgr = DataManagerLayers(cfg=config)
    model_creator = ModelCreator(cfg=config)
    model = model_creator.init_model()
    dev = set_device(config=config)
    model.to(dev)
    model.net.to(dev)

    engine = EngineTransfusion(cfg=config)
    engine.model = model
    engine.data_mgr = dataMgr
    engine._device = dev
    engine.model_creator = model_creator

    engine.optimizer = torch.optim.AdamW(model.parameters(),
                                    lr=config.engine.learning_rate,
                                    weight_decay=getattr(config.engine, 'weight_decay', 0.0))
    total_epochs = config.n_epochs - config.epoch_start
    total_steps = total_epochs * len(dataMgr.train_loader)
    
    engine.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        engine.optimizer,
        T_max=total_steps,
        eta_min=getattr(config.engine, 'min_lr', 1e-6)
    )

    return engine

def run(engine):
    for epoch in range(engine._config.epoch_start, engine._config.n_epochs):
        engine.fit_tfusion(epoch)
        val_loss = engine.evaluate_tfusion(engine.data_mgr.val_loader, epoch)
        if epoch % 10 == 0 and epoch > 0:
            engine._save_model(name=str(epoch))

if __name__=="__main__":
    logger.info("Starting main executable.")
    main()
    logger.info("Finished running script")
