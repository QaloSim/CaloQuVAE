import torch
from typing import Dict, Tuple
import numpy as np

def rbm_to_logical_ising(
    rbm, 
    beta: float = 1.0
) -> Tuple[Dict[int, float], Dict[Tuple[int, int], float], int, int]:
    """Converts RBM parameters to Logical Ising parameters."""
    num_visible = rbm.params["weight_matrix"].shape[0]
    num_hidden = rbm.params["weight_matrix"].shape[1]

    W = rbm.params["weight_matrix"].clone()
    vb = rbm.params["vbias"].clone()
    hb = rbm.params["hbias"].clone()
    
    dwave_J_tensor = -W / 4.0 / beta
    s_v = -torch.sum(W, dim=1) / 4.0 / beta
    dwave_h_visible = -vb / 2.0 / beta + s_v
    s_h = -torch.sum(W, dim=0) / 4.0 / beta
    dwave_h_hidden = -hb / 2.0 / beta + s_h
    
    all_biases_np = torch.cat([dwave_h_visible, dwave_h_hidden]).detach().cpu().numpy()
    dwave_J_tensor_np = dwave_J_tensor.detach().cpu().numpy()

    h = {}
    for i in range(num_visible + num_hidden):
        h[i] = float(all_biases_np[i])

    J = {}
    for i in range(num_visible):
        for j in range(num_hidden):
            val = float(dwave_J_tensor_np[i, j])
            J[(i, j + num_visible)] = val
                
    return h, J, num_visible, num_hidden

def rbm_to_expanded_ising(rbm, fragment_map, exp_embedding, adjacency, beta=1.0):
    """
    Converts RBM weights to Expanded Graph format, but ONLY creates edges
    that physically exist on the chip.
    
    Normalizes weights based on the count of VALID fragments only.
    """
    base_h, base_J, n_vis, n_hid = rbm_to_logical_ising(rbm, beta=beta)
    
    expanded_h = {}
    expanded_J = {}
    
    # --- 1. Process Biases (h) ---
    # (Same as before)
    for node, val in base_h.items():
        if node not in fragment_map:
            expanded_h[node] = val
        else:
            # Add all fragments with 0.0 bias (flux handles the rest)
            for frag_id in fragment_map[node]:
                expanded_h[frag_id] = 0.0
            
    # --- 2. Process Couplings (J) ---
    for (u, v), val in base_J.items():
        u_in_map = u in fragment_map
        v_in_map = v in fragment_map
        
        # A. Standard Edge (No fragments)
        if not u_in_map and not v_in_map:
            expanded_J[(u, v)] = val
            continue

        # B. Fragmented Source (u is conditioning)
        if u_in_map and not v_in_map:
            fragments = fragment_map[u]
            target_chain = set(exp_embedding[v]) # Physical qubits of v
            
            # Find which fragments represent physically valid connections
            valid_fragments = []
            for frag_id in fragments:
                # Get the single physical qubit for this fragment
                p_u = exp_embedding[frag_id][0]
                
                # Check if p_u connects to ANY qubit in target_chain
                # Fast check using adjacency
                # Intersect neighbors of p_u with target_chain
                if not set(adjacency[p_u]).isdisjoint(target_chain):
                    valid_fragments.append(frag_id)
            
            # Distribute J only among valid fragments
            if valid_fragments:
                split_val = val / len(valid_fragments)
                for frag_id in valid_fragments:
                    expanded_J[(frag_id, v)] = split_val
                    
        # C. Fragmented Target (v is conditioning)
        elif not u_in_map and v_in_map:
            fragments = fragment_map[v]
            source_chain = set(exp_embedding[u])
            
            valid_fragments = []
            for frag_id in fragments:
                p_v = exp_embedding[frag_id][0]
                
                # Check connection
                if not set(adjacency[p_v]).isdisjoint(source_chain):
                    valid_fragments.append(frag_id)
                    
            if valid_fragments:
                split_val = val / len(valid_fragments)
                for frag_id in valid_fragments:
                    expanded_J[(u, frag_id)] = split_val
                    
    return expanded_h, expanded_J

def joint_energy(rbm, v, h):
    """Computes the Joint Hamiltonian Energy."""
    interaction = (v @ rbm.params["weight_matrix"]) * h 
    interaction_sum = interaction.sum(dim=1)
    v_bias = v @ rbm.params["vbias"]
    h_bias = h @ rbm.params["hbias"]
    return -interaction_sum - v_bias - h_bias



def convert_energy_to_binary(incidence_energy: float, engine, n_cond: int, num_reads: int, device: torch.device):
    e_tensor = torch.tensor([[incidence_energy]], dtype=torch.float32).to(device)
    with torch.no_grad():
        if hasattr(engine, 'model') and hasattr(engine.model, 'encoder'):
            cond_pattern = engine.model.encoder.binary_energy_refactored(e_tensor)[:, :n_cond]
        else:
            raise ValueError("Failed to encode incidence energy")

    target_batch_full = cond_pattern.repeat(num_reads, 1).to(device)
    return target_batch_full


def calculate_rms_chain_strength(J_logical: dict, rho: float = 1.0) -> float:
    """
    Calculates chain strength based on the Root Mean Square (RMS) of the 
    logical couplings, scaled by a factor rho.
    
    Args:
        J_logical (dict): The logical couplings {(u, v): bias, ...}
        rho (float): Scaling factor (typically between 0.5 and 2.0 depending on problem hardness).
    
    Returns:
        float: The calculated chain strength.
    """
    if not J_logical:
        # Fallback if no couplings exist (unlikely in RBMs)
        return 1.0
        
    j_values = np.array(list(J_logical.values()))
    rms = np.sqrt(np.mean(j_values**2))
    
    return rho * rms
