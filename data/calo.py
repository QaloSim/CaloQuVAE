import torch
import h5py
import numpy as np

def get_calo_dataset(cfg):
    with h5py.File(cfg.data.path, 'r') as f:
        showers = torch.tensor(f["showers"][:]).float()
        energies = torch.tensor(f["incident_energies"][:]).float().squeeze()

    total = len(energies)
    n_train = int(cfg.data.frac_train_dataset * total)
    n_val = int(cfg.data.frac_val_dataset * total)
    n_test = total - n_train - n_val

    indices = np.arange(total)
    np.random.shuffle(indices)

    ordered_indices = np.concatenate([
        indices[:n_train],
        indices[n_train:n_train + n_val],
        indices[n_train + n_val:]])

    return { "showers": showers[ordered_indices],
             "incident_energies": energies[ordered_indices].unsqueeze(1)}