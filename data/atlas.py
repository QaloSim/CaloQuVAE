import torch
import h5py
import numpy as np
from collections import defaultdict
from CaloQuVAE import logging
logger = logging.getLogger(__name__)


def get_atlas_dataset(cfg):
    with h5py.File(cfg.data.path, 'r') as f:
        showers = torch.tensor(f["showers"][:]).float()
        energies = torch.tensor(f["incident_energies"][:]).float().squeeze()

    energies_np = energies.numpy()
    unique_energies = np.unique(energies_np)
    is_discrete = len(unique_energies) < 20 

    if is_discrete:
        bin_ids = np.digitize(energies_np, unique_energies, right=False)
    else: 
        if "Custom" in cfg.data.dataset_name:
            min_e = energies_np.min()
            max_e = energies_np.max()
            num_bins = 15
            energy_bin_edges = np.linspace(min_e, max_e + 1e-6, num_bins + 1)
            bin_ids = np.digitize(energies_np, energy_bin_edges, right=False)
        else:
            energy_bin_centers = [2**i for i in range(8, 23)]
            energy_bin_edges = [2**(np.log2(c) - 0.5) for c in energy_bin_centers]
            energy_bin_edges.append(2**(np.log2(energy_bin_centers[-1]) + 0.5))
            bin_ids = np.digitize(energies_np, energy_bin_edges, right=False)

    bin_to_indices = defaultdict(list)
    for i, b in enumerate(bin_ids):
        bin_to_indices[b].append(i)

    train_idx, val_idx, test_idx = [], [], []
    
    # accumulate actual lengths
    total_tr_len = 0
    total_val_len = 0

    for indices in bin_to_indices.values():
        n = len(indices)
        n_train = int(cfg.data.frac_train_dataset * n)
        n_val = int(cfg.data.frac_val_dataset * n)
        # n_test is implicit
        
        # Track the actual sizes
        total_tr_len += n_train
        total_val_len += n_val

        train_idx.extend(indices[:n_train])
        val_idx.extend(indices[n_train:n_train + n_val])
        test_idx.extend(indices[n_train + n_val:])

    ordered_indices = np.concatenate([train_idx, val_idx, test_idx])
    
    # Return the exact split lengths along with the data
    return { 
        "showers": showers[ordered_indices],
        "incident_energies": energies[ordered_indices].unsqueeze(1),
        "split_lengths": (total_tr_len, total_val_len) 
    }