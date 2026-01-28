import torch
import h5py
import numpy as np
from collections import defaultdict
from CaloQuVAE import logging

logger = logging.getLogger(__name__)
def filter_anomalies(showers, energies, cfg):
    """
    Filters out events where incident energy > 10,000 but Layer 2 has zero hits.
    
    Args:
        showers (torch.Tensor): Shape [batch_size, num_voxels]
        energies (torch.Tensor): Shape [batch_size]
        cfg: Configuration object containing data.phi and data.r
        
    Returns:
        tuple: (filtered_showers, filtered_energies)
    """
    # 1. Determine the voxel range for Layer 2 (3rd layer, index 2)
    voxels_per_layer = cfg.data.phi * cfg.data.r
    layer_idx = 2
    
    start_idx = layer_idx * voxels_per_layer
    end_idx = (layer_idx + 1) * voxels_per_layer
    
    # Safety check to ensure the tensor is large enough
    if showers.shape[1] < end_idx:
        logger.warning(f"Cannot filter anomalies: Shower dim {showers.shape[1]} is too small for Layer 2 index range {start_idx}-{end_idx}.")
        return showers, energies

    # 2. Extract Layer 2 data
    layer_2_data = showers[:, start_idx:end_idx]
    
    # 3. Create the filter masks
    # "No hits" implies the sum of energy in that layer is 0
    has_no_hits_l2 = layer_2_data.sum(dim=1) == 0
    
    # Incident energy > 10,000
    is_high_energy = energies > 10000
    
    # Identify anomalies: High Energy AND No Layer 2 Hits
    is_anomaly = is_high_energy & has_no_hits_l2
    
    # 4. Filter the data
    num_anomalies = is_anomaly.sum().item()
    if num_anomalies > 0:
        logger.info(f"Filtering {num_anomalies} anomalous events (Energy > 10k & No Layer 2 hits).")
        
        # Keep only non-anomalous events
        keep_mask = ~is_anomaly
        return showers[keep_mask], energies[keep_mask]
    
    return showers, energies

def get_atlas_dataset(cfg):
    with h5py.File(cfg.data.path, 'r') as f:
        showers = torch.tensor(f["showers"][:]).float()
        energies = torch.tensor(f["incident_energies"][:]).float().squeeze()

#   Filter anomalies before processing
    showers, energies = filter_anomalies(showers, energies, cfg)

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