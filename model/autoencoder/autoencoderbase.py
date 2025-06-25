"""
Base class for Autoencoder frameworks.

Defines basic common methods and variables shared between models.
Each model overwrites as needed. 
This class inherits from torch.nn.Module, ensuring that network parameters
are registered properly. 
"""
import torch
import torch.nn as nn
from model.encoder.encoderhierarchybase import HierarchicalEncoder
from model.decoder.decoder import Decoder

#logging module with handmade settings.
from CaloQVAE import logging
logger = logging.getLogger(__name__)

# Base Class for Autoencoder models
class AutoEncoderBase(nn.Module):
    def __init__(self, cfg):
        super(AutoEncoderBase,self).__init__()
        self._config=cfg

    def type(self):
        """String identifier for current model.

        Returns:
            model_type: "AE", "VAE", etc.
        """
        return self._model_name

    def _create_encoder(self):
        logger.debug("::_create_encoder")
        if self._config.model.encoder == "hierachicalencoder":
            return HierarchicalEncoder(self._config)

    def _create_decoder(self):
        logger.debug("::_create_decoder")
        if self._config.model.decoder == "decoder":
            return Decoder(self._config)

    def _create_sampler(self):
        """
            Define the sampler to be used for sampling from the RBM.

        Returns:
            Instance of baseSampler.
        """
        raise NotImplementedError
    
    def create_networks(self):
        logger.debug("Creating Network Structures")
        self.encoder=self._create_encoder()
        self.decoder=self._create_decoder()
        # self.prior=self._create_prior()
        # self.sampler = self._create_sampler()
        # self.stater = self._create_stat()
        
        # self._qpu_sampler = self.prior._qpu_sampler
        # self.sampling_time_qpu = []
        # self.sampling_time_gpu = []
    
    # def generate_samples(self):
    #     raise NotImplementedError

    # def __repr__(self):
    #     parameter_string="\n".join([str(par.shape) if isinstance(par,torch.Tensor) else str(par)  for par in self.__dict__.items()])
    #     return parameter_string
    
    # def forward(self, x):
    #     """[summary]

    #     Args:
    #         input_data (): [aaa]

    #     Raises:
    #         NotImplementedError: [ccc]
    #     """
    #     raise NotImplementedError

    def print_model_info(self):
        for key,par in self.__dict__.items():
            if isinstance(par,torch.Tensor):
                logger.info("{0}: {1}".format(key, par.shape))
            else:
                logger.debug("{0}: {1}".format(key, par))
        
if __name__=="__main__":
    logger.info("Running autoencoderbase.py directly") 
    logger.info("Success")