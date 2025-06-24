"""
ModelCreator - Interface between run scripts and models.

Provides initialisation of models.

CaloQVAE Group
2025
"""

import os
import torch
import wandb

from CaloQVAE import logging
logger = logging.getLogger(__name__)

import torch.nn as nn

#import defined models
from model.dummymodel import MLP
from model.autoencoder.autoencoderbase import AutoEncoderBase

_MODEL_DICT={
    "mlp": MLP,
    "autoencoderbase": AutoEncoderBase
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
    
    def save_state(self, cfg_string='test'):
        logger.info("Saving state")
        path = os.path.join(wandb.run.dir, "{0}.pth".format(cfg_string))
        
        # Extract modules from the model dict and add to start_dict 
        modules=list(self._model._modules.keys())
        state_dict={module: getattr(self._model, module).state_dict() for module in modules}
        
        # Save the model parameter dict
        torch.save(state_dict, path)
        
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
        
    def load_state(self, run_path, device):
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
