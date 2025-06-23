import torch
import h5py
import numpy as np
from torch.utils.data import DataLoader

from CaloQVAE import logging
logger = logging.getLogger(__name__)

class DataManager():
    def __init__(self, cfg=None):
        self._config=cfg
        self.load_dataset()
        self.create_dataloaders()


    def load_dataset(self):
        with h5py.File(self._config.data.path, 'r') as file:
            # List all groups
            self.f = {}
            # print("Keys: %s" % list(file.keys()))
            logger.info("Keys: %s" % list(file.keys()))
            for key in file.keys():
                self.f[key] = torch.tensor(np.array(file[key]))

        logger.info(f'{self.f.keys()}')

    def  create_dataloaders(self):
        tr = int(np.floor(self.f["showers"].shape[0] * self._config.data.frac_train_dataset))
        va = int(np.floor(self.f["showers"].shape[0] * self._config.data.frac_val_dataset))

        self.train_loader = DataLoader(
            self.f["showers"][:tr,:],
            batch_size=self._config.data.batch_size_tr,
            shuffle=True,
            num_workers=self._config.data.num_workers
        )

        self.val_loader = DataLoader(
            self.f["showers"][tr:tr+va,:],
            batch_size=self._config.data.batch_size_val,
            shuffle=False,
            num_workers=self._config.data.num_workers
        )

        self.test_loader = DataLoader(
            self.f["showers"][tr+va:,:],
            batch_size=self._config.data.batch_size_test,
            shuffle=False,
            num_workers=self._config.data.num_workers
        )

        logger.info("{0}: {2} events, {1} batches".format(self.train_loader,len(self.train_loader),len(self.train_loader.dataset)))
        logger.info("{0}: {2} events, {1} batches".format(self.test_loader,len(self.test_loader),len(self.test_loader.dataset)))
        logger.info("{0}: {2} events, {1} batches".format(self.val_loader,len(self.val_loader),len(self.val_loader.dataset)))