import torch
import h5py
import numpy as np
from collections import defaultdict

def get_atlas_dataset(cfg):
    with h5py.File(cfg.data.path, 'r') as f:
        showers = torch.tensor(f["showers"][:]).float()
        energies = torch.tensor(f["incident_energies"][:]).float().squeeze()

    energies_np = energies.numpy()
    unique_energies = np.unique(energies_np)
    is_discrete = len(unique_energies) < 20 # only ~15 unique so just a val bigger

    if is_discrete:
        bin_ids = np.digitize(energies_np, unique_energies, right=False)
    else: # using smeared logic
        energy_bin_centers = [2**i for i in range(8, 23)]
        energy_bin_edges = [2**(np.log2(c) - 0.5) for c in energy_bin_centers]
        energy_bin_edges.append(2**(np.log2(energy_bin_centers[-1]) + 0.5))
        bin_ids = np.digitize(energies_np, energy_bin_edges, right=False)

    bin_to_indices = defaultdict(list)
    for i, b in enumerate(bin_ids):
        bin_to_indices[b].append(i)

    train_idx, val_idx, test_idx = [], [], []
    for indices in bin_to_indices.values():
        n = len(indices)
        n_train = int(cfg.data.frac_train_dataset * n)
        n_val = int(cfg.data.frac_val_dataset * n)
        n_test = n - n_train - n_val

        train_idx.extend(indices[:n_train])
        val_idx.extend(indices[n_train:n_train + n_val])
        test_idx.extend(indices[n_train + n_val:])

    ordered_indices = np.concatenate([train_idx, val_idx, test_idx])
    return { "showers": showers[ordered_indices],
             "incident_energies": energies[ordered_indices].unsqueeze(1)}
