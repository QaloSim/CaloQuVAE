import torch
import numpy as np
from typing import Tuple

def process_rbm_samples(response, num_visible, num_hidden, device):
    total_nodes = num_visible + num_hidden
    num_samples = len(response.record.sample)
    var_to_col_idx = {var: i for i, var in enumerate(response.variables)}
    
    try:
        ordered_cols = [var_to_col_idx[i] for i in range(total_nodes)]
    except KeyError as e:
        print(f"Error: Missing logical node {e}. Returning zeros.")
        return torch.zeros((num_samples, num_visible), device=device), torch.zeros((num_samples, num_hidden), device=device)

    dwave_spin_samples = response.record.sample[:, ordered_cols]
    dwave_spin_samples_t = torch.tensor(dwave_spin_samples, dtype=torch.float, device=device)
    dwave_binary_samples = torch.where(dwave_spin_samples_t == -1.0, torch.tensor(0.0, device=device), dwave_spin_samples_t)

    return dwave_binary_samples[:, :num_visible], dwave_binary_samples[:, num_visible:]

def calculate_manual_chain_breaks(response, embedding_map):
    label_to_idx = {label: i for i, label in enumerate(response.variables)}
    samples = response.record.sample
    num_samples = samples.shape[0]
    total_chains = len(embedding_map)
    total_broken_chains = 0
    
    for logical_id, chain in embedding_map.items():
        if len(chain) < 2: continue
        chain_col_indices = [label_to_idx[q] for q in chain if q in label_to_idx]
        if len(chain_col_indices) < 2: continue
            
        chain_spins = samples[:, chain_col_indices]
        row_mins = np.min(chain_spins, axis=1)
        row_maxs = np.max(chain_spins, axis=1)
        total_broken_chains += np.sum(row_mins != row_maxs)

    return total_broken_chains / (num_samples * total_chains)

def unembed_raw_samples(response, embedding_map, num_visible, num_hidden, device):
    label_to_idx = {label: i for i, label in enumerate(response.variables)}
    phys_samples_np = response.record.sample 
    num_samples = phys_samples_np.shape[0]
    total_logical = num_visible + num_hidden
    logical_samples = np.zeros((num_samples, total_logical), dtype=np.float32)
    
    for logical_id in range(total_logical):
        if logical_id not in embedding_map: continue
        chain = embedding_map[logical_id]
        chain_indices = [label_to_idx[q] for q in chain if q in label_to_idx]
        if not chain_indices: continue
            
        chain_vals = phys_samples_np[:, chain_indices] 
        chain_sum = np.sum(chain_vals, axis=1) 
        logical_samples[:, logical_id] = np.where(chain_sum > 0, 1.0, 0.0)
        
    t_logical = torch.tensor(logical_samples, device=device, dtype=torch.float32)
    return t_logical[:, :num_visible], t_logical[:, num_visible:]

def process_expanded_rbm_samples(response, num_visible, num_hidden, conditioning_sets, device):
    total_nodes = num_visible + num_hidden
    num_cond = len(conditioning_sets)
    var_to_col_idx = {var: i for i, var in enumerate(response.variables)}
    ordered_cols = []
    
    for i in range(total_nodes):
        if i < num_cond:
            phys_q = sorted(list(conditioning_sets[i]))[0]
            target_label = f"C{i}_{phys_q}"
        else:
            target_label = i
            
        if target_label in var_to_col_idx:
            ordered_cols.append(var_to_col_idx[target_label])
        else:
            ordered_cols.append(0)

    dwave_spin_samples = response.record.sample[:, ordered_cols]
    dwave_spin_samples_t = torch.tensor(dwave_spin_samples, dtype=torch.float, device=device)
    dwave_binary_samples = torch.where(dwave_spin_samples_t == -1.0, torch.tensor(0.0, device=device), dwave_spin_samples_t)

    return dwave_binary_samples[:, :num_visible], dwave_binary_samples[:, num_visible:]


def process_analysis_result(analysis_result, rbm, conditioning_sets):
    """
    Extracts and sorts Visible/Hidden samples from a ChainAnalysisResult object.
    
    Ensures columns are ordered: [v_0, ... v_n, h_0, ... h_m].
    Handles the conversion from Spin (-1/+1) to Binary (0/1).
    """
    # 1. Setup Dimensions
    n_vis = rbm.params["vbias"].shape[0]
    n_hid = rbm.params["hbias"].shape[0]
    total_nodes = n_vis + n_hid
    num_cond = len(conditioning_sets)
    
    # 2. Map variable labels to their column index in the sample tensor
    # logical_samples is shape (n_samples, n_logical_vars)
    # variable_labels is a list of names corresponding to columns
    label_to_col_idx = {name: i for i, name in enumerate(analysis_result.variable_labels)}
    
    ordered_col_indices = []
    
    # 3. Build the index list in the strict (Visible + Hidden) order
    for i in range(total_nodes):
        if i < num_cond:
            # Reconstruct the special label used during embedding
            # e.g., "C0_1234" where 1234 is the physical qubit index
            phys_q = sorted(list(conditioning_sets[i]))[0]
            target_label = f"C{i}_{phys_q}"
        else:
            # Standard nodes are just integer labels
            target_label = i
            
        if target_label in label_to_col_idx:
            ordered_col_indices.append(label_to_col_idx[target_label])
        else:
            # Fallback for missing variables (should rarely happen in rigorous mode)
            print(f"Warning: Missing label {target_label} in analysis result.")
            ordered_col_indices.append(0)

    # 4. Extract and Convert
    # Use the indices to reorder the columns
    # analysis_result.logical_samples is already a Tensor on the correct device
    raw_samples = analysis_result.logical_samples[:, ordered_col_indices]
    
    # Convert Spin (-1/+1) to Binary (0/1)
    # If samples are already 0/1, this logic still works if checks are robust, 
    # but usually D-Wave returns -1/+1.
    binary_samples = torch.where(
        raw_samples == -1.0, 
        torch.tensor(0.0, device=rbm.device), 
        raw_samples
    )
    
    # 5. Split
    v_s = binary_samples[:, :n_vis]
    h_s = binary_samples[:, n_vis:]
    
    return v_s, h_s