import torch.nn as nn
from model.autoencoder.autoencoderbase import AutoEncoderBase

from CaloQVAE import logging
logger = logging.getLogger(__name__)


class MLP(AutoEncoderBase):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.seq = nn.Sequential(
            nn.Linear(self._config.model.input_size,100),
            nn.Linear(100,self._config.model.input_size),
        )
    def forward(self,x):
        return self.seq(x)