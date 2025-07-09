import torch
import h5py
import numpy as np
from torch.utils.data import DataLoader, Dataset
from CaloQuVAE import logging

# for atlas dataset
from data.atlas import get_atlas_dataset

logger = logging.getLogger(__name__)

class CaloDataset(Dataset):
    def __init__(self, dataset):
        self.showers, self.incident_energies = dataset[0], dataset[1]

    def __len__(self):
        return len(self.showers)

    def __getitem__(self, index):
        return self.showers[index, :], self.incident_energies[index, :]

class DataManager():
    def __init__(self, cfg=None):
        self._config = cfg
        self.select_dataset()          # for different datasets
        self.create_dataloaders()      # slice into train/val/test

    def load_dataset(self):
        with h5py.File(self._config.data.path, 'r') as file:
            # List all groups
            self.f = {}
            # print("Keys: %s" % list(file.keys()))
            logger.info("Keys: %s" % list(file.keys()))
            for key in file.keys():
                self.f[key] = torch.tensor(np.array(file[key]))

        logger.info(f'{self.f.keys()}')

    def select_dataset(self):
        dataset_name = self._config.data.dataset_name.lower()

        if "atlas" in dataset_name:
            logger.info(f"Loading ATLAS dataset: {self._config.data.dataset_name}")
            self.f = get_atlas_dataset(self._config)

        else:
            logger.info(f"Loading other dataset: {self._config.data.dataset_name}")
            self.load_dataset()

    def create_dataloaders(self):
        total = self.f["showers"].shape[0]
        frac_train = self._config.data.frac_train_dataset
        frac_val = self._config.data.frac_val_dataset

        tr = int(np.floor(total * frac_train))
        va = int(np.floor(total * frac_val))

        showers = self.f["showers"]
        energies = self.f["incident_energies"]
        
        self.train_loader = DataLoader(
            CaloDataset((showers[:tr, :], energies[:tr, :])),
            batch_size=self._config.data.batch_size_tr,
            shuffle=True,
            num_workers=self._config.data.num_workers
        )

        self.val_loader = DataLoader(
            CaloDataset((showers[tr:tr + va, :], energies[tr:tr + va, :])),
            batch_size=self._config.data.batch_size_val,
            shuffle=False,
            num_workers=self._config.data.num_workers
        )

        self.test_loader = DataLoader(
            CaloDataset((showers[tr + va:, :], energies[tr + va:, :])),
            batch_size=self._config.data.batch_size_test,
            shuffle=False,
            num_workers=self._config.data.num_workers
        )
        
        logger.info("{0}: {2} events, {1} batches".format(self.train_loader, len(self.train_loader), len(self.train_loader.dataset)))
        logger.info("{0}: {2} events, {1} batches".format(self.test_loader, len(self.test_loader), len(self.test_loader.dataset)))
        logger.info("{0}: {2} events, {1} batches".format(self.val_loader, len(self.val_loader), len(self.val_loader.dataset)))
