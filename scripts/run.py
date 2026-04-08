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

from omegaconf import OmegaConf, open_dict

# PyTorch imports
from torch import device
from torch.nn import DataParallel
from torch.cuda import is_available

# Multi GPU support
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
# Weights and Biases
import wandb

#self defined imports
from CaloQuVAE import logging
logger = logging.getLogger(__name__)

from data.dataManager import DataManager, DataManagerLayersShowers
from model.modelCreator import ModelCreator
from engine.engine import Engine



@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg=None):
    mode = cfg.wandb.mode
    if cfg.load_state:
        logger.info(f"Loading config from {cfg.config_path}")
        engine = load_model_instance(cfg)
        cfg = engine._config
        os.environ["WANDB_DIR"] = cfg.config_path.split("wandb")[0]
        iden = get_project_id(cfg.run_path)
        # wandb.init(tags = [cfg.data.dataset_name], project=cfg.wandb.project, entity=cfg.wandb.entity, config=OmegaConf.to_container(cfg, resolve=True), mode=mode,
        #         resume='allow', id=iden)
        wandb.init(tags = [cfg.data.dataset_name], project=cfg.wandb.project, entity=cfg.wandb.entity, config=OmegaConf.to_container(cfg, resolve=True), mode=mode)
        # Log metrics with wandb
        wandb.watch(engine.model)
    else:
        engine = setup_model(config=cfg)
        wandb.init(tags = [cfg.data.dataset_name], project=cfg.wandb.project, entity=cfg.wandb.entity, config=OmegaConf.to_container(cfg, resolve=True), mode=mode)
        wandb.watch(engine.model)
    print(OmegaConf.to_yaml(cfg, resolve=True))

    run(engine, callback)

def set_device(config=None):
    if is_distributed():
        local_rank = int(os.environ["LOCAL_RANK"])
        
        torch.cuda.set_device(local_rank) 
        dev = torch.device(f"cuda:{local_rank}")
        logger.info(f"DDP active: Process mapped to {dev}")
        
    elif (config.device == 'gpu') and config.gpu_list and torch.cuda.is_available():
        # Standard Single-GPU Mode fallback
        logger.info('Requesting GPUs. GPU list :' + str(config.gpu_list))
        devids = ["cuda:{0}".format(x) for x in list(config.gpu_list)]
        logger.info("Main GPU : " + devids[0])
        dev = torch.device(devids[0])
        
    else:
        logger.info('Requested CPU or unable to use GPU. Setting CPU as device.')
        dev = torch.device('cpu')
        
    return dev

def setup_model(config=None):
    if is_distributed():
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
    else:
        local_rank = 0

    if getattr(config, "use_u", False):
        dataMgr = DataManagerLayersShowers(config)
    else:
        dataMgr = DataManager(config)

    modelCreator = ModelCreator(config)
    model = modelCreator.init_model()
    model.create_networks()
    model.print_model_info()


    dev = set_device(config)
    model.to(dev)

    if not config.load_state and getattr(config, "use_u", False):
        logger.info("Handling initial stats from raw dataset...")
        
        #  isolate data init to rank 0
        if local_rank == 0:
            raw_min, raw_max = dataMgr.get_raw_feature_ranges() 
            # Move to device immediately so they can be copied to model buffers
            model.feature_min.copy_(raw_min.to(dev))
            model.feature_max.copy_(raw_max.to(dev))

            u_edges = dataMgr.get_u_bin_edges(config.model.u_bits, model.feature_min, model.feature_max)
            model.encoder.u_bin_edges.copy_(u_edges.to(dev))
        
        if is_distributed():
            # NCCL requires tensors to be on the GPU to broadcast
            dist.broadcast(model.feature_min, src=0)
            dist.broadcast(model.feature_max, src=0)
            dist.broadcast(model.encoder.u_bin_edges, src=0)
        
        dataMgr.apply_stats_and_build_loaders(model.feature_min, model.feature_max)

        if config.model.loss_coeff.get("geom_loss", 0.0) > 0.0:
            geom_dict = dataMgr.get_geom_features()
            for key, value in geom_dict.items():
                model.register_buffer(f"geom_{key}", value.to(dev))
        

    if hasattr(model, "prior") and model.prior is not None:
        model.prior._n_batches = len(dataMgr.train_loader) - 1

    if is_distributed():
        model = DDP(model, device_ids=[local_rank])
        base_model = model.module
    else:
        base_model = model

    engine = instantiate(config.engine, config)
    engine.data_mgr = dataMgr
    engine.device = dev    

    params = list(base_model.encoder.parameters()) + list(base_model.decoder.parameters())
    params = [p for p in params if p.requires_grad]
    
    engine.optimiser = torch.optim.AdamW(params, lr=config.engine.learning_rate)
    if hasattr(base_model, "prior") and base_model.prior is not None:
        base_model.prior.initOpt()
    
    engine.model = model
    engine.model_creator = modelCreator
    return engine

def run(engine, _callback=lambda _: False):
    if engine._config.engine.training_mode == "ae":
        logger.info("Training AutoEncoder")
        for epoch in range(engine._config.epoch_start, engine._config.n_epochs):
            engine.fit_ae(epoch)

            if is_master():
                total_loss_dict = engine.evaluate_ae(engine.data_mgr.val_loader, epoch)
                chi2 = engine.generate_plots(epoch, "ae")
                
                if epoch > 10:
                    engine.track_best_val_loss(total_loss_dict, chi2, epoch)
                
                if (epoch+1) % 10 == 0:
                    engine._save_model(name=str(epoch))
        
            # Force all other GPUs to wait until master finishes evaluating/saving
            if torch.distributed.is_initialized():
                torch.distributed.barrier()            
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
        # freeze_vae(engine)
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
    logger.info(f'RBM will use {engine._config.rbm.method}')

def callback(engine, epoch):
    """
    Callback function to be used with the engine.
    """
    logger.info(f"Callback function executed at epoch {epoch}.")
    if engine._config.freeze_vae and epoch + 1 >= engine._config.epoch_freeze:
        if engine._config.engine.training_mode=="ae":
            engine.load_best_model(epoch)
        engine._config.engine.training_mode = "rbm"
        engine._config.epoch_start = epoch + 1
        return True
    else:
        logger.info("Continuing training in current mode.")
        return False

def get_project_id(path):
    files = os.listdir(path.split('files')[0])
    b = [ ".wandb" in file for file in files]
    idx = (np.array(range(len(files))) * np.array(b)).sum()
    iden = files[idx].split("-")[1].split(".")[0]
    return iden

def load_model_instance(cfg, adjust_epoch_start=True):
    config = OmegaConf.load(cfg.config_path)
    if adjust_epoch_start:
        # Adjust the epoch start based on the run_path
        if config.run_path.split("_")[-1].split(".")[0].isdigit():
            config.epoch_start = int(config.run_path.split("_")[-1].split(".")[0])
        else:
            config.epoch_start = cfg.epoch_start
    config.gpu_list = cfg.gpu_list
    config.load_state = cfg.load_state
    if hasattr(cfg.rbm, "no_weights"):
        with open_dict(config):
            config.rbm.no_weights = cfg.rbm.no_weights
    self = setup_model(config)
    self._model_creator.load_state(config.run_path, self.device, vae_opt=self.optimiser, rbm_opt=self.model.prior.opt)
    return self


def is_distributed():
    return "LOCAL_RANK" in os.environ

def is_master():
    return int(os.environ.get("RANK", 0)) == 0



if __name__=="__main__":
    logger.info("Starting main executable.")
    main()
    logger.info("Finished running script")
