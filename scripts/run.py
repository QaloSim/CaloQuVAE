"""
Main executable. The run() method steers data loading, model creation, training
and evaluation by calling the respective interfaces.

Authors: The CaloQVAE
Year: 2025
"""

#external libraries
import os

import torch
torch.manual_seed(32)
import numpy as np
np.random.seed(32)
import hydra
from hydra.utils import instantiate

from omegaconf import OmegaConf

# PyTorch imports
from torch import device
from torch.nn import DataParallel
from torch.cuda import is_available
    
# Weights and Biases
import wandb

#self defined imports
from CaloQuVAE import logging
logger = logging.getLogger(__name__)

from data.dataManager import DataManager
from model.modelCreator import ModelCreator
from engine.engine import Engine
# from utils.plotting.plotProvider import PlotProvider
# from utils.stats.partition import get_Zs, save_plot, create_filenames_dict
# from utils.helpers import get_epochs, get_project_id

@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg=None):
    mode = cfg.wandb.mode
    if cfg.load_state:
        logger.info(f"Loading config from {cfg.config_path}")
        engine = load_model_instance(cfg)
        cfg = engine._config
        os.environ["WANDB_DIR"] = cfg.config_path.split("wandb")[0]
        iden = get_project_id(cfg.run_path)
        wandb.init(tags = [cfg.data.dataset_name], project=cfg.wandb.project, entity=cfg.wandb.entity, config=OmegaConf.to_container(cfg, resolve=True), mode=mode,
                resume='allow', id=iden)
        # Log metrics with wandb
        wandb.watch(engine.model)
    else:
        engine = setup_model(config=cfg)
        wandb.init(tags = [cfg.data.dataset_name], project=cfg.wandb.project, entity=cfg.wandb.entity, config=OmegaConf.to_container(cfg, resolve=True), mode=mode)
        wandb.watch(engine.model)
    print(OmegaConf.to_yaml(cfg, resolve=True))

    run(engine, callback)

def set_device(config=None):
    if (config.device == 'gpu') and config.gpu_list:
        logger.info('Requesting GPUs. GPU list :' + str(config.gpu_list))
        devids = ["cuda:{0}".format(x) for x in list(config.gpu_list)]
        logger.info("Main GPU : " + devids[0])
        
        if is_available():
            print(devids[0])
            dev = device(devids[0])
            if len(devids) > 1:
                logger.info(f"Using DataParallel on {devids}")
                model = DataParallel(model, device_ids=list(config.gpu_list))
            logger.info("CUDA available")
        else:
            dev = device('cpu')
            logger.info("CUDA unavailable")
    else:
        logger.info('Requested CPU or unable to use GPU. Setting CPU as device.')
        dev = device('cpu')
    return dev


def setup_model(config=None):
    """
    Run m
    """
    dataMgr = DataManager(config)

    #create model handling object
    modelCreator = ModelCreator(config)

    #instantiate the chosen model
    #loads from file 
    model=modelCreator.init_model()
    #create the NN infrastructure
    model.create_networks()
    model.print_model_info()
    # if not config.engine.train_vae_separate:
    model.prior._n_batches = len(dataMgr.train_loader) - 1

    # Load the model on the GPU if applicable
    dev = set_device(config)
        
    # Send the model to the selected device
    model.to(dev)

    # For some reason, need to use postional parameter cfg instead of named parameter
    # with updated Hydra - used to work with named param but now is cfg=None 
    engine=instantiate(config.engine, config)
    #add dataMgr instance to engine namespace
    engine.data_mgr=dataMgr
    #add device instance to engine namespace
    engine.device=dev    
    #instantiate and register optimisation algorithm
    engine.optimiser = torch.optim.Adam(model.parameters(),
                                        lr=config.engine.learning_rate)
    #add the model instance to the engine namespace
    engine.model = model
    # add the modelCreator instance to engine namespace
    engine.model_creator = modelCreator
    # if 'discriminator' in engine._config.engine.keys() and engine._config.engine.discriminator:
    #     engine.critic.to(dev)
    #     engine.critic_2.to(dev)
    
    for name, param in engine.model.named_parameters():
        if 'prior' in name:
            param.requires_grad = False
        print(name, param.requires_grad)
    
    return engine

def run(engine, _callback=lambda _: False):
    if engine._config.engine.training_mode == "ae":
        logger.info("Training AutoEncoder")
        for epoch in range(engine._config.epoch_start, engine._config.n_epochs):
            engine.fit_ae(epoch)

            total_loss_dict = engine.evaluate_ae(engine.data_mgr.val_loader, epoch)
            engine.track_best_val_loss(total_loss_dict)
            engine.generate_plots(epoch, "ae")
            
            if (epoch+1) % 10 == 0:
                engine._save_model(name=str(epoch))
            
            if _callback(engine, epoch):
                break
            
        engine.evaluate_ae(engine.data_mgr.test_loader, 0)

    if engine._config.engine.training_mode == "vae":
        logger.info("Training Variational AutoEncoder")
        for epoch in range(engine._config.epoch_start, engine._config.n_epochs):

            engine.fit_vae(epoch)
            total_loss_dict = engine.evaluate_vae(engine.data_mgr.val_loader, epoch)
            engine.track_best_val_loss(total_loss_dict)
            engine.generate_plots(epoch, "vae")
            
            if (epoch+1) % 10 == 0:
                engine._save_model(name=str(epoch))

            if _callback(engine, epoch):
                break

        engine.evaluate_vae(engine.data_mgr.test_loader, 0)

    if engine._config.engine.training_mode == "rbm":
        logger.info("Training RBM")
        freeze_vae(engine)
        for epoch in range(engine._config.epoch_start, engine._config.n_epochs):

            engine.fit_rbm(epoch)
            engine.evaluate_vae(engine.data_mgr.val_loader, epoch)
            engine.generate_plots(epoch, "rbm")
            
            if (epoch+1) % 10 == 0:
                engine._save_model(name=str(epoch))

        engine.evaluate_vae(engine.data_mgr.test_loader, 0)

#     if config.save_state:
#         config_string = "_".join(str(i) for i in [config.model.model_type, 
#                                                   config.data.data_type,
#                                                   config.tag, "latest"])
#         modelCreator.save_state(config_string)
        
#     if config.save_partition:
#         config_string = "_".join(str(i) for i in [config.model.model_type, 
#                                                   config.data.data_type,
#                                                   config.tag, "latest"])
#         run_path = os.path.join(wandb.run.dir, "{0}.pth".format(config_string))
#         lnZais_list, lnZrais_list, en_encoded_list = get_Zs(run_path, engine, dev, 10, config.data.entity)
#         save_plot(lnZais_list, lnZrais_list, en_encoded_list, run_path)

#     logger.info("run() finished successfully.")

def freeze_vae(engine):
    for name, param in engine.model.named_parameters():
        if 'decoder' in name or 'encoder' in name:
            param.requires_grad = False
        print(name, param.requires_grad)
    # engine._save_model(name="at_freezing_point")
    # engine._config.rbm.method = "PCD"
    logger.info(f'RBM will use {engine._config.rbm.method}')

def callback(engine, epoch):
    """
    Callback function to be used with the engine.
    """
    logger.info(f"Callback function executed at epoch {epoch}.")
    if engine._config.freeze_vae and epoch > engine._config.epoch_freeze:
        engine.load_best_model(epoch)
        engine._config.engine.training_mode = "rbm"
        return True
    else:
        logger.info("Continuing training in AE mode.")
        return False

def get_project_id(path):
    files = os.listdir(path.split('files')[0])
    b = [ ".wandb" in file for file in files]
    idx = (np.array(range(len(files))) * np.array(b)).sum()
    iden = files[idx].split("-")[1].split(".")[0]
    return iden

def load_model_instance(path, adjust_epoch_start=True):
    config = OmegaConf.load(path)
    if adjust_epoch_start:
        # Adjust the epoch start based on the run_path
        config.epoch_start = int(config.run_path.split("_")[-1].split(".")[0])
    config.gpu_list = cfg.gpu_list
    config.load_state = cfg.load_state
    self = setup_model(config)
    self._model_creator.load_state(config.run_path, self.device)
    return self

if __name__=="__main__":
    logger.info("Starting main executable.")
    main()
    logger.info("Finished running script")
