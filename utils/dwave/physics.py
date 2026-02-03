import torch
from typing import Dict, Tuple
import numpy as np
import math
import networkx as nx


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


def get_optimal_chain_strength(rbm, beta=1.0, lambda_0=1.0):
    """
    Calculates chain strength based on the LOGICAL RBM structure,
    adhering strictly to arXiv:2006.04913 Eq (5).
    """
    # 1. Get Logical Structure (Before expansion/fragmentation)
    # You likely already have this function from your snippet
    base_h, base_J, n_vis, n_hid = rbm_to_logical_ising(rbm, beta=beta)
    
    # N is the number of logical variables
    N = n_vis + n_hid
    
    if N < 2:
        return 1.0

    # 2. Calculate Sigma Squared using LOGICAL couplings
    # sum(J^2) over all unique pairs
    sum_J_sq = sum(val**2 for val in base_J.values())
    
    # Variance of the coupling strength
    # Eq (5): sigma^2 = 2 / (N(N-1)) * sum(J_ab^2) 
    sigma_sq = (2 / (N * (N - 1))) * sum_J_sq
    
    # 3. Calculate Lambda
    # Eq (5): lambda = lambda_0 * sqrt(sigma^2 * N) [cite: 103]
    optimal_lambda = lambda_0 * np.sqrt(sigma_sq * N)
    
    return optimal_lambda

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

def convert_energy_to_gray(incidence_energy: float, engine, n_cond: int, num_reads: int, device: torch.device):
    e_tensor = torch.tensor([[incidence_energy]], dtype=torch.float32).to(device)
    with torch.no_grad():
        if hasattr(engine, 'model') and hasattr(engine.model, 'encoder'):
            cond_pattern = engine.model.encoder.gray_energy_encoding(e_tensor, lin_bits=engine._config.model.lin_bits, sqrt_bits=engine._config.model.sqrt_bits, log_bits=engine._config.model.log_bits)[:, :n_cond]
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


def compute_node_susceptibilities(embedding, adjacency):
    """
    Step 1: Calculate the susceptibility of each physical qubit based on intra-chain geometry.
    Equation (13): chi_i = product( exp( -dist(i, j) / (xi * L) ) )
    """
    # Build graph once for distance calculations
    G = nx.Graph(adjacency)
    node_chi = {}

    for logical_id, chain in embedding.items():
        L = len(chain)
        if L == 0: continue
        
        # Heuristic from paper: correlation length (xi) approx equal to chain length (L) [cite: 338]
        # Denominator becomes L * L.
        denominator = (L * L) if L > 0 else 1.0
        
        # Pre-compute distances within the chain
        # We assume the chain is a connected subgraph.
        for q_i in chain:
            total_dist = 0
            # Get shortest paths from q_i to all other nodes on the chip
            # We restrict calculation to q_j in the same chain.
            paths = nx.single_source_shortest_path_length(G, source=q_i, cutoff=L)
            
            for q_j in chain:
                dist = paths.get(q_j, L) # Fallback to L if disconnected (shouldn't happen)
                total_dist += dist
            
            # Convert product of exponentials to exp of sum
            # chi = exp( - sum(dist) / xi*L )
            node_chi[q_i] = math.exp(-total_dist / denominator)
            
    return node_chi

def compute_edge_factors(J_logical, embedding, adjacency, node_chi):
    """
    Step 2: Calculate the effective susceptibility (X_ab) for each logical edge.
    Equation (11): X_ab = Sum over couplers ( chi_u * chi_v )
    """
    edge_factors = {} # Stores X_ab
    edge_counts = {}  # Stores number of physical couplers
    
    for (u, v) in J_logical.keys():
        # Retrieve physical chains
        chain_u = embedding.get(u, [])
        chain_v = embedding.get(v, [])
        
        # Optimization: iterate over the shorter chain to check adjacency
        if len(chain_u) > len(chain_v):
            primary, secondary = chain_v, chain_u
        else:
            primary, secondary = chain_u, chain_v
            
        secondary_set = set(secondary)
        
        X_ab = 0.0
        count = 0
        
        for p_u in primary:
            # Check physical neighbors of p_u
            neighbors = adjacency.get(p_u, [])
            for p_n in neighbors:
                if p_n in secondary_set:
                    # Found a physical coupler between chain u and chain v
                    chi_1 = node_chi.get(p_u, 1.0)
                    chi_2 = node_chi.get(p_n, 1.0)
                    
                    # Add susceptibility contribution [cite: 332]
                    X_ab += chi_1 * chi_2
                    count += 1
        
        # Handle case with no physical connection (should imply J=0 or broken embedding)
        if count == 0:
            X_ab = 1.0 # Avoid division by zero later
            count = 1
            
        edge_factors[(u, v)] = X_ab
        edge_counts[(u, v)] = count
        
    return edge_factors, edge_counts

def apply_j_scaling(J_logical, edge_factors, edge_counts):
    """
    Step 3: Apply the compensation scaling.
    J_new = J_old * N * (count / X_ab)
    """
    # Calculate Normalization Constant N (Geometric Mean of all X_ab) [cite: 331]
    if not edge_factors:
        return J_logical.copy()
        
    log_sum = sum(math.log(x) for x in edge_factors.values())
    N_norm = math.exp(log_sum / len(edge_factors))
    
    J_compensated = {}
    
    for (u, v), j_val in J_logical.items():
        X_ab = edge_factors.get((u, v), 1.0)
        count = edge_counts.get((u, v), 1)
        
        # We scale inversely to X_ab to homogenize effective coupling [cite: 350]
        # We multiply by 'count' because embed_ising divides the input J by 'count'.
        # We want the *sum* of physical couplers to equal the target strength.
        scale_factor = N_norm * (count / X_ab)
        
        J_compensated[(u, v)] = j_val * scale_factor
        
    return J_compensated