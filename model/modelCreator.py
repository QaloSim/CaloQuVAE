"""
ModelCreator - Interface between run scripts and models.

Provides initialisation of models.

CaloQVAE Group
2025
"""

import os
import torch
import wandb

from omegaconf import OmegaConf

from CaloQuVAE import logging
logger = logging.getLogger(__name__)

import torch.nn as nn

#import defined models
from model.dummymodel import MLP
from model.autoencoder.autoencoderbase import AutoEncoderBase, AutoEncoderHidden
from model.autoencoder.ae_separate import AutoEncoderSeparate, AutoEncoderSeparateHidden

_MODEL_DICT={
    "mlp": MLP,
    "autoencoderbase": AutoEncoderBase,
    "autoencoderhidden": AutoEncoderHidden,
    "ae_separate": AutoEncoderSeparate,
    "ae_hidden": AutoEncoderSeparateHidden
}

class ModelCreator():

    def __init__(self, cfg=None):
        self._config=cfg
        self._model=None
    
    def init_model(self):
        logger.info("::Creating Model")
        self._model = _MODEL_DICT[self._config.model.model_name](cfg=self._config)
        return self._model

    @property
    def model(self):
        assert self._model is not None, "Model is not defined."
        return self._model

    @model.setter
    def model(self,model):
        self._model=model
    
        
    def save_state(self, cfg_string='test', vae_opt=None, rbm_opt=None):
        # Use wandb.run.dir if available, else fallback to local directory
        if wandb.run is not None and self._config.wandb.mode != "disabled" and self._config.load_state == 0:
            save_dir = wandb.run.dir
            logger.info(f"Using WandB run directory: {save_dir}")
        elif wandb.run is not None and self._config.wandb.mode != "disabled" and self._config.load_state == 1:
            save_dir = self._config.run_path.split('files')[0] + 'files/'
        else:
            # e.g., set a folder relative to current working directory or config run_path
            if hasattr(self._config, 'run_path'):
                logger.info(f"Using config run directory: {self._config.run_path}")
                save_dir = self._config.run_path
            else:
                logger.warning("No config run directory found, using current working directory.")
                save_dir = os.getcwd()

        os.makedirs(save_dir, exist_ok=True)

        path = os.path.join(save_dir, f"{cfg_string}.pth")
        logger.info(f"Saving model state to {path}")

        modules = list(self._model._modules.keys())
        state_dict = {module: getattr(self._model, module).state_dict() for module in modules}

        torch.save(state_dict, path)

        if vae_opt is not None:
            torch.save(vae_opt.state_dict(), os.path.join(save_dir, f"{cfg_string}_opt_model.pth"))
        if rbm_opt is not None:
            torch.save(rbm_opt.state_dict(), os.path.join(save_dir, f"{cfg_string}_opt_rbm.pth"))


        config_path = os.path.join(save_dir, f"{cfg_string}_config.yaml")
        self._config.run_path = path
        self._config.config_path = config_path
        OmegaConf.save(self._config, config_path, resolve=True)
        return config_path

        
    def save_RBM_state(self, cfg_string='test', encoded_data_energy=None):
        logger.info("Saving RBM state")
        if not os.path.exists(os.path.join(wandb.run.dir, "RBM")):
            # Create the directory
            os.makedirs(os.path.join(wandb.run.dir, "RBM"))
        pathW = os.path.join(wandb.run.dir, "RBM", "{0}.pth".format(cfg_string + '_weights'))
        pathB = os.path.join(wandb.run.dir, "RBM", "{0}.pth".format(cfg_string + '_biases'))
        pathE = os.path.join(wandb.run.dir, "RBM", "{0}.pth".format(cfg_string + '_EncEn'))
        
        # Save the dictionary to a file
        torch.save(self._model.prior._weight_dict, pathW)
        torch.save(self._model.prior._bias_dict, pathB)
        torch.save(encoded_data_energy, pathE)

    def load_state(self, run_path, device, vae_opt=None, rbm_opt=None):
        logger.info("Loading state")
        model_loc = run_path
        
        # Open a file in read-binary mode
        with open(model_loc, 'rb') as f:
            # Interpret the file using torch.load()
            checkpoint=torch.load(f, map_location=device)

            logger.info("Loading weights from file : {0}".format(run_path))
            
            local_module_keys=list(self._model._modules.keys())
            for module in checkpoint.keys():
                if module in local_module_keys:
                    print("Loading weights for module = ", module)
                    getattr(self._model, module).load_state_dict(checkpoint[module])

        base_dir = os.path.dirname(run_path)
        cfg_string = os.path.splitext(os.path.basename(run_path))[0]

        opt_vae_path = os.path.join(base_dir, f"{cfg_string}_opt_model.pth")
        opt_rbm_path   = os.path.join(base_dir, f"{cfg_string}_opt_rbm.pth")

        if vae_opt is not None and os.path.exists(opt_vae_path):
            vae_opt.load_state_dict(torch.load(opt_vae_path, map_location=device))
            logger.info("Loaded VAE optimizer state")

        if rbm_opt is not None and os.path.exists(opt_rbm_path):
            rbm_opt.load_state_dict(torch.load(opt_rbm_path, map_location=device))
            logger.info("Loaded RBM optimizer state")

                    
    def load_RBM_state(self, run_path, device):
        logger.info("Loading RBM state")
        model_loc = run_path
        # Load the dictionary with all tensors mapped to the CPU
        loaded_dict = torch.load(model_loc, map_location=device)
        return loaded_dict


if __name__=="__main__":
    logger.info("Start!")
    mm=ModelCreator()
    logger.info("End!")
