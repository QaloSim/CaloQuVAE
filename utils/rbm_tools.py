import os
import torch
from omegaconf import OmegaConf

from CaloQuVAE import logging
logger = logging.getLogger(__name__)

class RBMTools:

    @staticmethod
    def save_RBM_state(rbm, save_dir, cfg_string='RBM_state', config=None):
        logger.info("Saving RBM state")
        if not os.path.exists(save_dir):
            # Create the directory
            os.makedirs(save_dir)
        pathW = os.path.join(save_dir, "{0}.pth".format(cfg_string + '_weights'))
        pathVbias = os.path.join(save_dir, "{0}.pth".format(cfg_string + '_bias_visible'))
        pathHbias = os.path.join(save_dir, "{0}.pth".format(cfg_string + '_bias_hidden'))

        # Save the dictionary to a file
        torch.save(rbm.params["weight_matrix"], pathW)
        torch.save(rbm.params["vbias"], pathVbias)
        torch.save(rbm.params["hbias"], pathHbias)
        logger.info(f"Saved RBM weights to {pathW}")
        logger.info(f"Saved RBM visible bias to {pathVbias}")
        logger.info(f"Saved RBM hidden bias to {pathHbias}")
        config["weights_path"] = pathW
        config["vbias_path"] = pathVbias
        config["hbias_path"] = pathHbias
        OmegaConf.save(config, config.rbm_config_path, resolve=True)
        logger.info(f"Saved RBM config file to {config.rbm_config_path}")