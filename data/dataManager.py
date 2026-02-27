import torch
import h5py
import numpy as np
import os
from torch.utils.data import DataLoader, Dataset
from CaloQuVAE import logging

# for atlas dataset
from data.atlas import get_atlas_dataset
from data.layers import get_layer_dataset

logger = logging.getLogger(__name__)

class CaloDataset(Dataset):
    def __init__(self, dataset):
        self.showers, self.incident_energies = dataset[0], dataset[1]

    def __len__(self):
        return len(self.showers)

    def __getitem__(self, index):
        return self.showers[index, :].float(), self.incident_energies[index, :].float()

class LayerDatasets(Dataset):
    """
    Dataset class that takes in outputs from get_layer_datasets
    """
    def __init__(self, dataset):
        self.layer_energies, self.incident_energies = dataset[0], dataset[1]
    def __len__(self):
        return len(self.layer_energies)
    def __getitem__(self, index):
        return self.layer_energies[index, :].float(), self.incident_energies[index, :].float()

class DataManagerLayers():
    def __init__(self, cfg=None):
        self._config = cfg
        self.select_dataset()          # for different datasets
        self.create_dataloaders()      # slice into train/val/test

    def select_dataset(self):
        """
        Selects Layer dataset (only for ATLAS data)
        """
        dataset_name = self._config.data.dataset_name.lower()
        logger.info(f"Loading ATLAS Layer dataset: {self._config.data.dataset_name}")
        self.f = get_layer_dataset(self._config)

    def create_dataloaders(self):
        tr, va = self.f["split_lengths"]
        logger.info(f"Using pre-calculated stratified splits: Tr={tr}, Val={va}")
        layer_energies, incident_energies = self.f["layer_energies"], self.f["incident_energies"]

        # Check if config already has a path to the stats
        stats_path = getattr(self._config, 'feature_stats_path', None)

        if stats_path and os.path.exists(stats_path):
            #Load from the existing file
            logger.info(f"Loading feature statistics from {stats_path}")
            stats = torch.load(stats_path)
            feature_mean = stats['mean']
            feature_std = stats['std']
            
        elif tr > 0:
            # New run: compute and save
            logger.info("Computing new training feature statistics...")
            train_features = layer_energies[:tr, :]
            feature_mean = train_features.mean(dim=0, keepdim=True)
            feature_std = train_features.std(dim=0, keepdim=True)
            feature_std[feature_std == 0] = 1.0 
            
            # Determine where to save it (e.g., using config save_dir)
            save_dir = getattr(self._config, 'save_dir', os.getcwd())
            os.makedirs(save_dir, exist_ok=True) 
            
            new_stats_path = os.path.join(save_dir, "feature_stats.pt")
            
            # Save the tensors natively
            torch.save({'mean': feature_mean, 'std': feature_std}, new_stats_path)
            
            # Update the config with the path so OmegaConf saves it later
            self._config.feature_stats_path = new_stats_path
            logger.info(f"Saved feature statistics to {new_stats_path}")
            
        else:
            raise ValueError("No stats path found in config, and no training data to compute them. Cannot proceed.")

        # Apply standardization to loaded dataset
        layer_energies = (layer_energies - feature_mean) / feature_std        
        if tr > 0:
            self.train_loader = DataLoader(
                LayerDatasets((layer_energies[:tr, :], incident_energies[:tr, :])),
                batch_size=self._config.data.batch_size_tr,
                shuffle=True,
                num_workers=self._config.data.num_workers
            )
            logger.info("{0}: {2} events, {1} batches".format(
                "Train", len(self.train_loader), len(self.train_loader.dataset)))
        else:
            self.train_loader = None
            logger.info("Train Loader: 0 events (Skipped)")
        if va > 0:
            self.val_loader = DataLoader(
                LayerDatasets((layer_energies[tr:tr + va, :], incident_energies[tr:tr + va, :])),
                batch_size=self._config.data.batch_size_val,
                shuffle=False,
                num_workers=self._config.data.num_workers
            )
            logger.info("{0}: {2} events, {1} batches".format(
                "Val", len(self.val_loader), len(self.val_loader.dataset)))
        else:
            self.val_loader = None
            logger.info("Val Loader: 0 events (Skipped)")
        te_len = layer_energies.shape[0] - tr - va
        if te_len > 0:
            self.test_loader = DataLoader(
                LayerDatasets((layer_energies[tr + va:, :], incident_energies[tr + va:, :])),
                batch_size=self._config.data.batch_size_test,
                shuffle=False,
                num_workers=self._config.data.num_workers
            )
            logger.info("{0}: {2} events, {1} batches".format(
                "Test", len(self.test_loader), len(self.test_loader.dataset)))
        else:
            self.test_loader = None
            logger.info("Test Loader: 0 events (Skipped)")


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
            
            # Check if the dataset loader provided explicit split lengths
            if "split_lengths" in self.f:
                tr, va = self.f["split_lengths"]
                logger.info(f"Using pre-calculated stratified splits: Tr={tr}, Val={va}")
            else:
                # Fallback for generic datasets (Global floor)
                frac_train = self._config.data.frac_train_dataset
                frac_val = self._config.data.frac_val_dataset
                tr = int(np.floor(total * frac_train))
                va = int(np.floor(total * frac_val))
            
            # Extract slices 
            showers = self.f["showers"]
            energies = self.f["incident_energies"]
            
            # --- Train Loader ---
            if tr > 0:
                self.train_loader = DataLoader(
                    CaloDataset((showers[:tr, :], energies[:tr, :])),
                    batch_size=self._config.data.batch_size_tr,
                    shuffle=True,
                    num_workers=self._config.data.num_workers
                )
                logger.info("{0}: {2} events, {1} batches".format(
                    "Train", len(self.train_loader), len(self.train_loader.dataset)))
            else:
                self.train_loader = None
                logger.info("Train Loader: 0 events (Skipped)")

            # --- Validation Loader ---
            if va > 0:
                self.val_loader = DataLoader(
                    CaloDataset((showers[tr:tr + va, :], energies[tr:tr + va, :])),
                    batch_size=self._config.data.batch_size_val,
                    shuffle=False,
                    num_workers=self._config.data.num_workers
                )
                logger.info("{0}: {2} events, {1} batches".format(
                    "Val", len(self.val_loader), len(self.val_loader.dataset)))
            else:
                self.val_loader = None
                logger.info("Val Loader: 0 events (Skipped)")

            # --- Test Loader ---
            # Calculate remaining items to avoid index errors
            te_len = total - tr - va
            if te_len > 0:
                self.test_loader = DataLoader(
                    CaloDataset((showers[tr + va:, :], energies[tr + va:, :])),
                    batch_size=self._config.data.batch_size_test,
                    shuffle=False,
                    num_workers=self._config.data.num_workers
                )
                logger.info("{0}: {2} events, {1} batches".format(
                    "Test", len(self.test_loader), len(self.test_loader.dataset)))
            else:
                self.test_loader = None
                logger.info("Test Loader: 0 events (Skipped)")