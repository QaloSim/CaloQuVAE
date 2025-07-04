import torch
import h5py
import numpy as np
from collections import defaultdict
from torch.utils.data import DataLoader, Dataset

from CaloQuVAE import logging
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
        
    def create_dataloaders(self):
        showers = self.f["showers"]
        energies = self.f["incident_energies"].squeeze()

        # np to check how many unique incident energies
        energies_np = energies.numpy()
        unique_energies = np.unique(energies_np)
        is_discrete = len(unique_energies) < 20 # check if discrete or smeared

        # bin by discrete or smeared dataset
        if is_discrete:
            bin_ids = np.digitize(energies_np, unique_energies, right=False)
        else:
            energy_bin_centers = [2 ** i for i in range(8, 23)]  # energy range
            energy_bin_edges = [2 ** (np.log2(c) - 0.5) for c in energy_bin_centers]
            energy_bin_edges.append(2 ** (np.log2(energy_bin_centers[-1]) + 0.5))
            bin_ids = np.digitize(energies_np, energy_bin_edges, right=False)

        # collect indices per bin
        bin_to_indices = defaultdict(list)
        for i, b in enumerate(bin_ids):
            bin_to_indices[b].append(i)

        train_idx, val_idx, test_idx = [], [], []

        for indices in bin_to_indices.values():
            np.random.shuffle(indices)
            n = len(indices)
            n_train = int(self._config.data.frac_train_dataset * n)
            n_val = int(self._config.data.frac_val_dataset * n)
            n_test = n - n_train - n_val

            train_idx.extend(indices[:n_train])
            val_idx.extend(indices[n_train:n_train + n_val])
            test_idx.extend(indices[n_train + n_val:])

        # shuffle final indices to avoid bin grouping in batches
        np.random.shuffle(train_idx)
        np.random.shuffle(val_idx)
        np.random.shuffle(test_idx)

        #create datasets
        self.train_loader = DataLoader(
            CaloDataset((showers[train_idx], energies[train_idx].unsqueeze(1))),
            batch_size=self._config.data.batch_size_tr,
            shuffle=True,
            num_workers=self._config.data.num_workers
        )

        self.val_loader = DataLoader(
            CaloDataset((showers[val_idx], energies[val_idx].unsqueeze(1))),
            batch_size=self._config.data.batch_size_val,
            shuffle=False,
            num_workers=self._config.data.num_workers
        )

        self.test_loader = DataLoader(
            CaloDataset((showers[test_idx], energies[test_idx].unsqueeze(1))),
            batch_size=self._config.data.batch_size_test,
            shuffle=False,
            num_workers=self._config.data.num_workers
        )

        logger.info("{0}: {2} events, {1} batches".format(self.train_loader, len(self.train_loader), len(self.train_loader.dataset)))
        logger.info("{0}: {2} events, {1} batches".format(self.test_loader, len(self.test_loader), len(self.test_loader.dataset)))
        logger.info("{0}: {2} events, {1} batches".format(self.val_loader, len(self.val_loader), len(self.val_loader.dataset)))