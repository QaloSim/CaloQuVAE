import h5py
import torch
import numpy as np
from data.atlas import filter_anomalies, get_atlas_dataset
import math

def get_layer_dataset(cfg):
    """
    Loads a voxelized dataset and returns layer-wise data.
    """
    f = get_atlas_dataset(cfg)
    showers = f.pop("showers")
    num_layers = cfg.data.z
    # Reshape to (batch, layers, voxels_per_layer) and sum over the voxel dimension (dim=2)
    batch_size = showers.shape[0]
    num_voxels_per_layer = cfg.data.phi * cfg.data.r
    layer_energies = showers.reshape(batch_size, num_layers, num_voxels_per_layer).sum(dim=2) # (batch, layers) energy per layer
    incident_energies = f.pop("incident_energies")
    layer_energies, incident_energies = reduce(layer_energies, incident_energies, showers)
    f["layer_energies"] = layer_energies
    f["incident_energies"] = incident_energies
    return f


def reduce(x, e_inc, showers, e_min=900.0, e_max=310000.0, f=1.30, eps=1e-7):
    """
    Args:
        x: Raw layer energies, shape (batch_size, 5)
        e_inc: Incidence energy in MeV, shape (batch_size, 1)
        showers: Raw voxel data, shape (batch_size, num_voxels)
    Returns:
        u: Transformed energy ratios, shape (batch_size, 5)
        e_inc_norm: Log-normalized incidence energy in [0, 1], shape (batch_size, 1)
    """
    # Log-normalize the incidence energy to [0, 1]
    log_e = torch.log(e_inc + eps)
    log_min = math.log(e_min)
    log_max = math.log(e_max)
    log_range = log_max - log_min
    e_inc_norm = (log_e - log_min) / log_range
    e_inc_norm = torch.clamp(e_inc_norm, 0.0, 1.0)
    
    # Total deposited energy across all 5 layers
    E_tot = x.sum(dim=1, keepdim=True)
    
    # u0: Ratio of total energy to scaled incidence energy
    u0 = E_tot / (f * e_inc + eps)
    
    # Calculate remaining energy sums for the denominator: sum_{j >= i} E_j
    rem_energy = torch.flip(torch.cumsum(torch.flip(x, dims=[1]), dim=1), dims=[1])
    
    # u_i (for i=1 to 4): Ratio of current layer to the remaining energy sum
    # We drop the final layer (E5) fraction because it is implicitly 1.0
    u_fractions = x[:, :-1] / (rem_energy[:, :-1] + eps)
    
    # Concatenate u0 with the 4 layer fractions to maintain exactly 5 features
    u = torch.cat([u0, u_fractions], dim=1)
    # assert torch.all(u >= 0) and torch.all(u <= 1), "u must be non-negative and less than 1"

    
    return u, e_inc_norm

def reduce_inverse(u, e_inc_norm, feature_mean, feature_std, e_min=900.0, e_max=310000.0, f=1.30, eps=1e-7):
    """
    Recovers [E1, ..., E5] and raw E_inc from standardized [u0, ..., u4] and log-normalized E_inc.
    
    Args:
        u: Predicted ratios (standardized), shape (batch_size, 5)
        e_inc_norm: Log-normalized incidence energy, shape (batch_size, 1)
        feature_mean: Mean used for standardizing u, broadcastable to (batch_size, 5)
        feature_std: Standard deviation used for standardizing u, broadcastable to (batch_size, 5)
        
    Returns:
        x_recovered: Recovered absolute layer energies, shape (batch_size, 5)
        e_inc: Recovered absolute incidence energy in MeV, shape (batch_size, 1)
    """
    # Unstandardize the reduced features first
    u_unstd = (u * feature_std) + feature_mean
    
    # Compute the actual log bounds
    log_min = math.log(e_min)
    log_max = math.log(e_max)
    log_range = log_max - log_min
    
    # Denormalize the incidence energy back to MeV
    log_e = e_inc_norm * log_range + log_min
    e_inc = torch.exp(log_e)
    
    # Recover total deposited energy from u0
    E_tot = u_unstd[:, 0:1] * f * e_inc
    
    x_recovered = torch.zeros_like(u_unstd)
    rem = E_tot
    
    # Iteratively recover E1 through E4
    for i in range(u_unstd.shape[1] - 1):
        # The fraction u_i for layer E_{i+1} is at index i+1 in the u tensor
        layer_fraction = u_unstd[:, i+1:i+2]
        
        x_recovered[:, i:i+1] = layer_fraction * rem
        rem = rem - x_recovered[:, i:i+1]
        
    # The final layer (E5) deterministically gets the remainder
    x_recovered[:, -1:] = rem
    
    return x_recovered, e_inc