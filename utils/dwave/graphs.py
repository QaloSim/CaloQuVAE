from __future__ import annotations  # 1. Must be the very first line!

from typing import Dict, List, Tuple, Union
import numpy as np
from collections import defaultdict
from dwave.system import DWaveSampler
from dwave.embedding.zephyr import find_biclique_embedding
from dwave.system.composites import FixedEmbeddingComposite
from utils.FluxBiases import h_to_fluxbias 
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from utils.dwave.sampling_backend import ChainAnalysisResult

import pulp
import random
import scipy.sparse as sp
import networkx as nx

# --- A. Basic Zephyr ---

def get_sampler_and_biclique_embedding(num_visible, num_hidden, solver_name):
    print(f"--- Finding Zephyr Embedding for K_{num_visible},{num_hidden} ---")
    try:
        raw_sampler = DWaveSampler(solver=solver_name)
    except Exception as e:
        print(f"Error initializing sampler: {e}")
        return None, None, None

    working_graph = raw_sampler.to_networkx_graph()
    try:
        left_dict, right_dict = find_biclique_embedding(
            num_visible, num_hidden, target_graph=working_graph
        )
    except ValueError as e:
        print(f"Error: RBM is too large for this QPU or topology mismatch. {e}")
        return raw_sampler, None, None

    embedding = {**left_dict, **right_dict}
    for k, v in embedding.items(): embedding[k] = list(v)

    print(f"Successfully created embedding for {len(embedding)} nodes.")
    qpu_sampler = FixedEmbeddingComposite(raw_sampler, embedding)
    return raw_sampler, embedding, qpu_sampler

def get_physical_flux_biases(embedding, total_physical_qubits, logical_clamps, clamp_strength_h=30.0):
    physical_flux_biases = [0.0] * total_physical_qubits
    for logical_node, direction in logical_clamps.items():
        if logical_node not in embedding: continue
        chain = embedding[logical_node]
        h_val = direction * clamp_strength_h 
        fb_val = h_to_fluxbias(h_val)
        for q in chain:
            if q < total_physical_qubits: physical_flux_biases[q] += fb_val
    return physical_flux_biases

# --- B. Manual Embedding Helpers ---

def get_physical_connectivity(source_qubits, target_qubits, adjacency):
    connections = []
    target_set = set(target_qubits)
    for u in source_qubits:
        if u in adjacency:
            for v in adjacency[u]:
                if v in target_set:
                    connections.append(tuple(sorted((u, v))))
    return list(set(connections))

def distribute_bias(logical_h, physical_qubits):
    if not physical_qubits: return {}
    val = logical_h / len(physical_qubits)
    return {q: val for q in physical_qubits}

def distribute_coupling(logical_J, physical_edges):
    if not physical_edges: return {}
    val = logical_J / len(physical_edges)
    return {edge: val for edge in physical_edges}

def get_chain_couplings(chain, strength):
    couplings = {}
    if len(chain) < 2: return couplings
    for i in range(len(chain) - 1):
        u, v = chain[i], chain[i+1]
        couplings[tuple(sorted((u, v)))] = -strength
    return couplings

def build_logical_to_physical_map(num_visible, num_hidden, conditioning_sets, left_chains, right_chains):
    embedding_map = {}
    # 1. Conditioning Nodes
    for i, q_set in enumerate(conditioning_sets):
        embedding_map[i] = list(q_set)
    n_cond = len(conditioning_sets)
    # 2. Remaining Visible
    sorted_left = sorted(left_chains.keys())
    for k, key in enumerate(sorted_left):
        if k >= (num_visible - n_cond): break
        embedding_map[n_cond + k] = list(left_chains[key])
    # 3. Hidden
    sorted_right = sorted(right_chains.keys())
    for k, key in enumerate(sorted_right):
        if k >= num_hidden: break
        embedding_map[num_visible + k] = list(right_chains[key])
    return embedding_map

def get_physical_flux_biases_manual(embedding_map, total_physical_qubits, logical_clamps, clamp_strength_h):
    physical_flux_biases = [0.0] * total_physical_qubits
    for logical_node, direction in logical_clamps.items():
        if logical_node not in embedding_map: continue
        chain = embedding_map[logical_node]
        h_val = direction * clamp_strength_h
        fb_val = h_to_fluxbias(h_val)
        for phys_q in chain:
            if phys_q < total_physical_qubits: physical_flux_biases[phys_q] += fb_val
    return physical_flux_biases

def build_manual_embedded_ising(rbm, sampler, conditioning_sets, left_chains, right_chains, beta, chain_strength, logical_h, logical_J):
    emb_map = build_logical_to_physical_map(rbm.params["vbias"].shape[0], rbm.params["hbias"].shape[0], conditioning_sets, left_chains, right_chains)
    h_phys = defaultdict(float)
    J_phys = defaultdict(float)
    adjacency = sampler.adjacency
    
    # Distribute h
    for l_node, bias_val in logical_h.items():
        if l_node not in emb_map: continue
        dist = distribute_bias(bias_val, emb_map[l_node])
        for q, val in dist.items(): h_phys[q] += val
            
    # Distribute J
    for (u_log, v_log), j_val in logical_J.items():
        if u_log not in emb_map or v_log not in emb_map: continue
        valid_edges = get_physical_connectivity(emb_map[u_log], emb_map[v_log], adjacency)
        dist = distribute_coupling(j_val, valid_edges)
        for edge, val in dist.items(): J_phys[edge] += val
            
    # Apply Chain Strength (Skip conditioning nodes)
    n_cond = len(conditioning_sets)
    for l_node, chain in emb_map.items():
        if l_node >= n_cond:
            chain_Js = get_chain_couplings(chain, chain_strength)
            for edge, val in chain_Js.items(): J_phys[edge] += val
    return dict(h_phys), dict(J_phys)

# --- C. Expanded Embedding Helpers ---

def build_expanded_embedding(conditioning_sets, left_chains, right_chains, num_visible, hidden_side='right'):
    expanded_embedding = {}
    fragment_map = {} 
    
    # 1. Dynamic Assignment
    if hidden_side == 'right':
        visible_chain_source = left_chains
        hidden_chain_source = right_chains
    elif hidden_side == 'left':
        visible_chain_source = right_chains
        hidden_chain_source = left_chains
    else:
        raise ValueError(f"hidden_side must be 'left' or 'right', got {hidden_side}")

    sorted_vis_keys = sorted(visible_chain_source.keys())
    sorted_hid_keys = sorted(hidden_chain_source.keys())

    # 2. Conditioning Nodes (Logical 0 to n_cond-1)
    # These use the SPECIAL conditioning sets found by your heuristic (unused qubits).
    # They do NOT consume chains from the visible_chain_source.
    for logical_id, phys_set in enumerate(conditioning_sets):
        fragments = []
        for phys_q in phys_set:
            frag_id = f"C{logical_id}_{phys_q}"
            expanded_embedding[frag_id] = [phys_q]
            fragments.append(frag_id)
        fragment_map[logical_id] = fragments

    # 3. Standard Visible Nodes (Logical n_cond to num_visible-1)
    # These map to the standard biclique chains.
    n_cond = len(conditioning_sets)
    
    for logical_id in range(n_cond, num_visible):
        # CALCULATE THE OFFSET
        # Logical 53 needs Chain Index 0
        # Logical 54 needs Chain Index 1
        chain_idx = logical_id - n_cond
        
        # Validation
        if chain_idx >= len(sorted_vis_keys):
            raise IndexError(
                f"Not enough standard visible chains! "
                f"RBM needs {num_visible - n_cond} standard chains, "
                f"but embedding only has {len(sorted_vis_keys)}."
            )
            
        actual_key = sorted_vis_keys[chain_idx]
        expanded_embedding[logical_id] = list(visible_chain_source[actual_key])

    # 4. Hidden Nodes (Logical num_visible to end)
    # These map to the hidden side chains.
    for k, key in enumerate(sorted_hid_keys):
        global_id = num_visible + k
        expanded_embedding[global_id] = list(hidden_chain_source[key])

    return expanded_embedding, fragment_map


def build_expanded_embedding_rotation(
    conditioning_sets, 
    left_chains, 
    right_chains, 
    num_visible, 
    hidden_side='right', 
    vis_shift=0,   # NEW: Offset for standard visible chains
    hid_shift=0    # NEW: Offset for hidden chains
):
    expanded_embedding = {}
    fragment_map = {} 
    
    # 1. Dynamic Assignment
    if hidden_side == 'right':
        visible_chain_source = left_chains
        hidden_chain_source = right_chains
    elif hidden_side == 'left':
        visible_chain_source = right_chains
        hidden_chain_source = left_chains
    else:
        raise ValueError(f"hidden_side must be 'left' or 'right', got {hidden_side}")

    sorted_vis_keys = sorted(visible_chain_source.keys())
    sorted_hid_keys = sorted(hidden_chain_source.keys())
    
    # Pre-calculate counts for modulo arithmetic
    n_avail_vis_chains = len(sorted_vis_keys)
    n_avail_hid_chains = len(sorted_hid_keys)

    # 2. Conditioning Nodes (Logical 0 to n_cond-1)
    # NOTE: Conditioning nodes are tied to specific heuristic sets (neighbors), so we usually do NOT rotate these physically
    for logical_id, phys_set in enumerate(conditioning_sets):
        fragments = []
        for phys_q in phys_set:
            frag_id = f"C{logical_id}_{phys_q}"
            expanded_embedding[frag_id] = [phys_q]
            fragments.append(frag_id)
        fragment_map[logical_id] = fragments

    # 3. Standard Visible Nodes (Logical n_cond to num_visible-1)
    n_cond = len(conditioning_sets)
    
    for logical_id in range(n_cond, num_visible):
        # Base index (0, 1, 2...)
        base_idx = logical_id - n_cond
        
        # Apply ROTATION (Modulo)
        chain_idx = (base_idx + vis_shift) % n_avail_vis_chains
        
        # Validation: Ensure we aren't wrapping around into used chains 
        # if the number of needed chains == number of available chains.
        # (This check is soft; if you have spare chains, modulo is safe and desirable).
        if base_idx >= n_avail_vis_chains:
             raise IndexError(
                f"Not enough standard visible chains! "
                f"RBM needs {num_visible - n_cond} standard chains, "
                f"but embedding only has {n_avail_vis_chains}."
            )
            
        actual_key = sorted_vis_keys[chain_idx]
        expanded_embedding[logical_id] = list(visible_chain_source[actual_key])

    # 4. Hidden Nodes (Logical num_visible to end)
    for k, key in enumerate(sorted_hid_keys):
        global_id = num_visible + k
        
        # Apply ROTATION (Modulo)
        chain_idx = (k + hid_shift) % n_avail_hid_chains
        
        actual_key = sorted_hid_keys[chain_idx]
        expanded_embedding[global_id] = list(hidden_chain_source[actual_key])

    return expanded_embedding, fragment_map


def build_expanded_embedding_arbitrary(
    conditioning_sets, 
    left_chains, 
    right_chains, 
    num_visible, 
    hidden_side='right', 
    vis_mapping=None,   # List[int]: Indices mapping logical_vis -> physical_chain_index
    hid_mapping=None    # List[int]: Indices mapping logical_hid -> physical_chain_index
):
    expanded_embedding = {}
    fragment_map = {} 
    
    # 1. Determine Source Chains
    if hidden_side == 'right':
        visible_chain_source = left_chains
        hidden_chain_source = right_chains
    elif hidden_side == 'left':
        visible_chain_source = right_chains
        hidden_chain_source = left_chains
    else:
        raise ValueError(f"hidden_side must be 'left' or 'right', got {hidden_side}")

    # Sort keys to ensure index 0 always refers to the same physical chain
    sorted_vis_keys = sorted(visible_chain_source.keys())
    sorted_hid_keys = sorted(hidden_chain_source.keys())
    
    # 2. Handle Conditioning Nodes (Fixed, usually spatial)
    for logical_id, phys_set in enumerate(conditioning_sets):
        fragments = []
        for phys_q in phys_set:
            frag_id = f"C{logical_id}_{phys_q}"
            expanded_embedding[frag_id] = [phys_q]
            fragments.append(frag_id)
        fragment_map[logical_id] = fragments

    # 3. Handle Standard Visible Nodes (Permutable)
    n_cond = len(conditioning_sets)
    n_standard_vis = num_visible - n_cond
    
    # Default to Identity if no mapping provided
    if vis_mapping is None:
        vis_mapping = list(range(len(sorted_vis_keys)))
        
    if len(vis_mapping) < n_standard_vis:
        raise ValueError(f"vis_mapping length ({len(vis_mapping)}) < needed visible nodes ({n_standard_vis})")

    for i in range(n_standard_vis):
        logical_id = n_cond + i
        
        # Use the mapping to select the physical chain index
        phys_chain_idx = vis_mapping[i]
        
        actual_key = sorted_vis_keys[phys_chain_idx]
        expanded_embedding[logical_id] = [int(q) for q in visible_chain_source[actual_key]]

    # 4. Handle Hidden Nodes (Permutable)
    # Default to Identity
    if hid_mapping is None:
        hid_mapping = list(range(len(sorted_hid_keys)))

    for k, phys_chain_idx in enumerate(hid_mapping):
        # Stop if we have mapped all logical hidden nodes required by the RBM?
        # Typically RBM hidden size == number of hidden chains available.
        # If RBM is smaller, we break.
        # Assuming here we map all available in the permutation list:
        logical_id = num_visible + k
        
        actual_key = sorted_hid_keys[phys_chain_idx]
        expanded_embedding[logical_id] = [int(q) for q in hidden_chain_source[actual_key]]

    return expanded_embedding, fragment_map
    

    
def get_expanded_flux_biases(
    logical_clamps, 
    fragment_map, 
    expanded_embedding, 
    total_physical_qubits, 
    clamp_strength_h=20.0
):
    flux_list = [0.0] * total_physical_qubits
    
    for logical_id, spin in logical_clamps.items():
        if logical_id not in fragment_map: continue
        
        # --- FIX: NEGATIVE SIGN ADDED HERE ---
        # To encourage spin S, we apply bias -S * Strength
        h_val = -spin * clamp_strength_h
        
        fb_val = h_to_fluxbias(h_val)
        
        fragments = fragment_map[logical_id]
        for frag_key in fragments:
            if frag_key in expanded_embedding:
                for q in expanded_embedding[frag_key]:
                    if q < total_physical_qubits: 
                        flux_list[q] += fb_val
                        
    return flux_list

def translate_chain_labels(result: ChainAnalysisResult, n_vis: int) -> ChainAnalysisResult:
    """
    Converts raw integer indices in a ChainAnalysisResult into 
    human-readable labels (e.g., 157 -> 'h29').
    """
    new_labels = []
    
    for label in result.variable_labels:
        if isinstance(label, str):
            # It's a fragment (e.g. "C0_1234"), keep it shorter
            # Optional: shorten to "C0" if you don't care about the physical qubit
            new_labels.append(label)
        elif isinstance(label, int):
            if label < n_vis:
                # It is a visible unit
                new_labels.append(f"v{label}")
            else:
                # It is a hidden unit
                new_labels.append(f"h{label - n_vis}")
        else:
            new_labels.append(str(label))
            
    # Update the result object in-place
    result.variable_labels = new_labels
    return result



def run_embedding(left_size, right_size, solver_name):
    """
    Performs the biclique embedding on the target QPU.
    Returns the sampler, the working graph, and the chains.
    """
    print(f"--- 1. Running Embedding for K_{left_size},{right_size} on {solver_name} ---")
    TARGET_SOLVER = solver_name
    try:
        target_sampler = DWaveSampler(solver=TARGET_SOLVER)
    except Exception as e:
        print(f"Error initializing sampler: {e}")
        print("Please ensure you have D-Wave credentials configured.")
        return None, None, None, None, None

    # Use the sampler's graph directly
    working_graph = target_sampler.to_networkx_graph() 
    
    if not working_graph:
        print("Could not fetch working graph from sampler.")
        return target_sampler, None, None, None, None

    print(f"Successfully fetched QPU graph with {len(working_graph.nodes)} nodes.")

    try:
        left_chains, right_chains = find_biclique_embedding(
            left_size, right_size, target_graph=working_graph
        )
    except Exception as e:
        print(f"Error during find_biclique_embedding: {e}")
        print("This often means the graph is too large for the QPU.")
        return target_sampler, working_graph, None, None, None

    all_chains = list(left_chains.values()) + list(right_chains.values())
    max_len = max(len(chain) for chain in all_chains)

    qubits_used = set()
    for chain in all_chains:
        qubits_used.update(chain)

    print(f" -> Max chain length: {max_len}")
    print(f" -> Total qubits used: {len(qubits_used)}")

    return target_sampler, working_graph, qubits_used, left_chains, right_chains



def build_neighbor_sets(target_sampler, target_logical_nodes, qubits_used):
    """
    Builds the "available neighbor" sets, V_i.
    V_i = {all *available* qubits adjacent to logical node i}
    """
    print(f"\n--- Building {len(target_logical_nodes)} Neighbor Sets ---")
    
    # 1. Get all qubits available on the chip
    all_physical_qubits = set(target_sampler.nodelist)
    qubits_avail = all_physical_qubits - qubits_used
    print(f"Total available qubits: {len(qubits_avail)}")

    # 2. Get the adjacency property of the sampler
    qpu_adjacency = target_sampler.adjacency
    
    neighbor_sets = []
    min_adj_size = float('inf')
    
    # 3. Iterate over each of the logical nodes in Side A
    for i, chain in enumerate(target_logical_nodes):
        
        # This set will hold all *available* neighbors for this one logical node
        chain_available_neighbors = set()
        
        # 4. Check neighbors for every physical qubit in the chain
        for q_in_chain in chain:
            for neighbor in qpu_adjacency[q_in_chain]:
                # 5. If the neighbor is in the available set, add it
                if neighbor in qubits_avail:
                    chain_available_neighbors.add(neighbor)
        
        neighbor_sets.append(chain_available_neighbors)
        
        if len(chain_available_neighbors) < min_adj_size:
            min_adj_size = len(chain_available_neighbors)
            
    print(f"All {len(neighbor_sets)} neighbor sets built.")
    print(f"The 'bottleneck' (min neighbors) is: {min_adj_size}")
    print(f"This is the *absolute upper bound* on the number of nodes.")
    
    # This is V_all, the total pool of qubits we can *ever* use
    all_available_neighbors = set().union(*neighbor_sets)
    
    return neighbor_sets, all_available_neighbors

def find_max_disjoint_hitting_sets_heuristic(neighbor_sets, all_available_neighbors):
    """
    Heuristic algorithm to find the maximum number of disjoint hitting sets.
    
    Inputs:
    - neighbor_sets (list of sets): The 76 sets V_i.
    - all_available_neighbors (set): The pool of all qubits we can use.
    
    Returns:
    - N_nodes (int): The estimated max number of conditioning nodes.
    - all_found_nodes (list of sets): The physical qubit sets for each node.
    """
    print(f"\n--- 3. Running Greedy Heuristic for Max Disjoint Hitting Sets ---")
    
    N_nodes_found = 0
    all_found_nodes = []
    
    # Make a copy so we can safely modify it
    qubit_pool = all_available_neighbors.copy()
    num_logical_nodes = len(neighbor_sets)
    
    # --- Outer Greedy Loop ---
    # Try to build nodes one by one
    while True:
        
        # --- Inner Greedy Loop ---
        # Try to build *one* valid conditioning node (a hitting set)
        current_hitting_set = set()
        
        # Indices of the logical nodes (0 to 75) we still need to hit
        sets_to_hit_indices = set(range(num_logical_nodes))
        
        # We need a pool of qubits *available for this node*
        # This pool will shrink as we build the current_hitting_set
        current_qubit_pool = qubit_pool.copy()
        
        while sets_to_hit_indices:
            # 1. Find the "best" qubit to add
            best_qubit = None
            max_hits = -1
            
            # This is the greedy "Set Cover" part:
            # Check every available qubit...
            for q in current_qubit_pool:
                current_hits = 0
                # ...to see how many *un-hit* logical nodes it hits
                for i in sets_to_hit_indices:
                    if q in neighbor_sets[i]:
                        current_hits += 1
                
                if current_hits > max_hits:
                    max_hits = current_hits
                    best_qubit = q
            
            # 2. Check if we failed
            if max_hits == 0:
                # We failed to build a complete hitting set
                # The remaining qubits in current_qubit_pool
                # cannot hit the remaining sets_to_hit_indices.
                # Break the inner loop (this node fails)
                break
                
            # 3. Add the best qubit to our node
            current_hitting_set.add(best_qubit)
            
            # 4. Remove it from the pool for *this node*
            current_qubit_pool.remove(best_qubit)
            
            # 5. Update the list of nodes we still need to hit
            indices_hit = set()
            for i in sets_to_hit_indices:
                if best_qubit in neighbor_sets[i]:
                    indices_hit.add(i)
            
            sets_to_hit_indices.difference_update(indices_hit)
            
        # --- End of Inner Loop ---
        
        if not sets_to_hit_indices:
            # SUCCESS! We hit all 76 nodes.
            N_nodes_found += 1
            all_found_nodes.append(current_hitting_set)
            
            # Now, permanently remove these qubits from the *global* pool
            qubit_pool.difference_update(current_hitting_set)
            
            # print(f"  -> Found conditioning node {N_nodes_found} (size {len(current_hitting_set)})")
        else:
            # FAILURE. We couldn't build a new node.
            # The remaining qubits in qubit_pool are not sufficient.
            # Break the outer loop
            break
            
    # --- End of Outer Loop ---

    
    return N_nodes_found, all_found_nodes

def analyze_target_side(side_name, sampler, target_nodes_chains, qubits_used):
    """
    Runs the full analysis (build sets + run heuristic) for a given side.
    """
    print("\n" + "===" * 15)
    print(f"--- Analyzing Target Side: {side_name} ---")
    print("===" * 15)
    
    # Build the V_i sets
    v_sets, v_all = build_neighbor_sets(sampler, target_nodes_chains, qubits_used)
    
    if not v_all:
        print("\nNo available neighbors found for any target node on this side.")
        print(f"Estimated max conditioning nodes for {side_name}: 0")
        return 0

    # Run the heuristic
    num_nodes, node_qubit_sets = find_max_disjoint_hitting_sets_heuristic(v_sets, v_all)
    
    print("---" * 10)
    print(f"Heuristic Result for {side_name}: {num_nodes}")
    print(f"   (Estimated max number of conditioning nodes)")
    print("---" * 10)
    
    if num_nodes > 0:
        print("\nPhysical qubit set sizes for each found node:")
        print(sum([len(s) for s in node_qubit_sets]))
    
    return num_nodes, node_qubit_sets



def select_optimal_side(sampler, q_used, left_chains, right_chains, verbose=True):
    """
    Analyzes both the Left and Right chain sets to determine which side 
    offers better connectivity (more available nodes) for conditioning.
    
    Returns:
        best_cond_sets (list): The conditioning sets from the winning side.
        visible_side (str): 'left' or 'right'.
    """
    
    # 1. Analyze Left
    target_nodes_left = list(left_chains.values())
    num_left, cond_sets_left = analyze_target_side(
        "Left Chains", sampler, target_nodes_left, q_used
    )

    # 2. Analyze Right
    target_nodes_right = list(right_chains.values())
    num_right, cond_sets_right = analyze_target_side(
        "Right Chains", sampler, target_nodes_right, q_used
    )

    # 3. Compare and Select
    if num_left >= num_right:
        best_cond_sets = cond_sets_left
        visible_side = 'left'
        winner_count = num_left
    else:
        best_cond_sets = cond_sets_right
        visible_side = 'right'
        winner_count = num_right

    # 4. Summary (Optional)
    if verbose:
        print("\n" + "===" * 15)
        print("--- Connectivity Summary ---")
        print(f"Left Capacity:  {num_left}")
        print(f"Right Capacity: {num_right}")
        print(f"Selected Side:  '{visible_side}' with {winner_count} nodes.")
        print("===" * 15)

    return best_cond_sets, visible_side


def find_exact_max_disjoint_hitting_sets(neighbor_sets, all_available_neighbors):
    """
    Exact ILP algorithm to find the mathematically maximum number of disjoint hitting sets.
    """
    # 1. The absolute maximum possible nodes is bottlenecked by the smallest set
    max_possible_sets = min(len(s.intersection(all_available_neighbors)) for s in neighbor_sets)
    print(f"Maximum possible disjoint hitting sets: {max_possible_sets}")
    
    if max_possible_sets == 0:
        return 0, []

    # Initialize the ILP Problem
    prob = pulp.LpProblem("Max_Disjoint_Hitting_Sets", pulp.LpMaximize)

    # Decision Variables
    # y[c] = 1 if hitting set 'c' is active
    y = pulp.LpVariable.dicts("y", range(max_possible_sets), cat=pulp.LpBinary)
    
    # x[q, c] = 1 if qubit 'q' is assigned to hitting set 'c'
    x = pulp.LpVariable.dicts("x", 
                              [(q, c) for q in all_available_neighbors for c in range(max_possible_sets)], 
                              cat=pulp.LpBinary)

    # Objective: Maximize the number of active hitting sets
    prob += pulp.lpSum(y[c] for c in range(max_possible_sets))

    # Constraint 1: Disjoint sets - Each qubit is used at most once across all sets
    for q in all_available_neighbors:
        prob += pulp.lpSum(x[(q, c)] for c in range(max_possible_sets)) <= 1

    # Constraint 2: Hitting property - If set 'c' is active, it MUST hit every neighbor_set
    for c in range(max_possible_sets):
        for i, v_set in enumerate(neighbor_sets):
            valid_qubits = v_set.intersection(all_available_neighbors)
            prob += pulp.lpSum(x[(q, c)] for q in valid_qubits) >= y[c]
            
    # Constraint 3: Symmetry Breaking (forces the solver to pack sets sequentially)
    for c in range(max_possible_sets - 1):
        prob += y[c] >= y[c+1]

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # Extract the results
    all_found_nodes = []
    for c in range(max_possible_sets):
        if pulp.value(y[c]) == 1.0:
            current_set = set()
            for q in all_available_neighbors:
                if pulp.value(x[(q, c)]) == 1.0:
                    current_set.add(q)
            all_found_nodes.append(current_set)

    return all_found_nodes

def evaluate_split_chains(sampler, left_chains, right_chains):
    """
    Splits fully connected biclique chains in half and calculates the 
    resulting logical adjacency matrix based on physical couplers.
    """
    new_left_chains = {}
    new_right_chains = {}
    
    # 1. Split the visible (left) chains
    for i, (logical_node, chain) in enumerate(left_chains.items()):
        midpoint = len(chain) // 2
        new_left_chains[f"v_{i}_A"] = chain[:midpoint]
        new_left_chains[f"v_{i}_B"] = chain[midpoint:]
        
    # 2. Split the hidden (right) chains
    for j, (logical_node, chain) in enumerate(right_chains.items()):
        midpoint = len(chain) // 2
        new_right_chains[f"h_{j}_A"] = chain[:midpoint]
        new_right_chains[f"h_{j}_B"] = chain[midpoint:]

    # 3. Evaluate new logical connectivity
    adjacency = sampler.adjacency
    num_v = len(new_left_chains)
    num_h = len(new_right_chains)
    connectivity_matrix = np.zeros((num_v, num_h))
    
    v_keys = list(new_left_chains.keys())
    h_keys = list(new_right_chains.keys())
    
    for v_idx, v_key in enumerate(v_keys):
        v_chain = new_left_chains[v_key]
        for h_idx, h_key in enumerate(h_keys):
            h_chain = new_right_chains[h_key]
            
            # A logical edge exists ONLY if there is at least one 
            # physical coupler between the two half-chains
            connected = False
            for q_v in v_chain:
                if connected: break
                for q_h in h_chain:
                    if q_h in adjacency[q_v]:
                        connected = True
                        break
            
            if connected:
                connectivity_matrix[v_idx, h_idx] = 1
                
    return new_left_chains, new_right_chains, connectivity_matrix

def analyze_rbm_topology(connectivity_matrix):
    """
    Quantitatively analyzes the logical sparsity pattern of an RBM.
    Expects a 2D numpy array of shape (num_visible, num_hidden).
    """
    # 1. Convert biadjacency matrix to a full bipartite NetworkX graph
    sparse_mat = sp.csr_matrix(connectivity_matrix)
    G = nx.bipartite.from_biadjacency_matrix(sparse_mat)
    
    # Extract node sets
    top_nodes = {n for n, d in G.nodes(data=True) if d['bipartite'] == 0}
    bottom_nodes = set(G) - top_nodes
    
    # 2. Check for Disjoint Subgraphs (The Shatter Test)
    num_components = nx.number_connected_components(G)
    component_sizes = [len(c) for c in nx.connected_components(G)]
    
    # 3. Check for Dead Nodes
    degrees = np.array([d for n, d in G.degree()])
    dead_nodes = np.sum(degrees == 0)
    
    # 4. Analyze Bottlenecks (Degree Distribution)
    min_degree = np.min(degrees[degrees > 0]) if len(degrees[degrees > 0]) > 0 else 0
    max_degree = np.max(degrees)
    mean_degree = np.mean(degrees)
    
    # 5. Graph Density
    density = nx.bipartite.density(G, top_nodes)
    
    print("\n--- RBM Topological Diagnostics ---")
    print(f"Connected Components: {num_components} (Ideal: 1)")
    
    if num_components > 1:
        print(f"CRITICAL WARNING: Graph is shattered into {num_components} disconnected subgraphs.")
        print(f"Component Sizes (Nodes): {component_sizes}")
        
    print(f"Dead Nodes (0 connections): {dead_nodes}")
    print(f"Degree Stats - Min: {min_degree}, Max: {max_degree}, Mean: {mean_degree:.2f}")
    print(f"Overall Density: {density:.3f} (1.0 is fully connected)")
    
    return G, num_components


def select_optimal_side_exact(sampler, q_used, left_chains, right_chains, threshold=62):
    """
    Evaluates both the Left and Right chains using the exact ILP hitting set solver 
    to lock in the maximum number of conditioning nodes.
    """

    v_sets_right, v_all_right = build_neighbor_sets(sampler, list(right_chains.values()), q_used)
    cond_sets_right = find_exact_max_disjoint_hitting_sets(v_sets_right, v_all_right) if v_all_right else []
    if len(cond_sets_right) > threshold:
        return cond_sets_right[:threshold], 'right'

    # 1. Analyze Left Side
    v_sets_left, v_all_left = build_neighbor_sets(sampler, list(left_chains.values()), q_used)
    cond_sets_left = find_exact_max_disjoint_hitting_sets(v_sets_left, v_all_left) if v_all_left else []
    if len(cond_sets_left) > threshold:
        return cond_sets_left[:threshold], 'left'
    
    # 2. Analyze Right Side
    
    print("\n" + "===" * 15)
    print("--- Exact Connectivity Summary ---")
    print(f"Left Exact Capacity:  {len(cond_sets_left)}")
    print(f"Right Exact Capacity: {len(cond_sets_right)}")
    
    # 3. Select Winner
    if len(cond_sets_left) >= len(cond_sets_right):
        print(f"Selected Side: 'left' with {len(cond_sets_left)} conditioning nodes.")
        print("===" * 15)
        return cond_sets_left, 'left'
    else:
        print(f"Selected Side: 'right' with {len(cond_sets_right)} conditioning nodes.")
        print("===" * 15)
        return cond_sets_right, 'right'

def build_target_neighborhoods(target_chains, adjacency):
    """Pre-computes the physical neighbors for every target logical chain for $O(1)$ lookups."""
    neighborhoods = {}
    for key, chain in target_chains.items():
        neighbors = set()
        for q in chain:
            neighbors.update(adjacency[q])
        neighborhoods[key] = neighbors
    return neighborhoods

def generate_candidate_chains(
    adjacency, used_qubits, target_chains, 
    max_len=10, min_hits=35, target_pool_size=1000, 
    min_free_neighbors=0  # NEW: Guarantees qubits are left for conditioning nodes
):
    """
    Rapidly generates valid chain candidates using randomized BFS walks,
    with aggressive prefix pruning and neighbor survival constraints.
    """
    target_neighborhoods = build_target_neighborhoods(target_chains, adjacency)
    available_qubits = list(set(adjacency.keys()) - used_qubits)
    candidates = []
    
    # Run random walks until we fill our candidate pool or hit an iteration limit
    for _ in range(target_pool_size * 20): 
        if len(candidates) >= target_pool_size: 
            break
            
        start_q = random.choice(available_qubits)
        current_chain = [start_q]
        current_qubits = {start_q}
        
        # 1. Grow the chain blindly
        while len(current_chain) < max_len:
            neighbors = set()
            for q in current_chain:
                neighbors.update(adjacency[q])
            
            valid_steps = list(neighbors - used_qubits - current_qubits)
            if not valid_steps:
                break
                
            next_q = random.choice(valid_steps)
            current_chain.append(next_q)
            current_qubits.add(next_q)
        
        # 2. Prefix Pruning (Remove useless trailing qubits)
        best_prefix_len = 1
        max_hits_achieved = 0
        
        for i in range(1, len(current_chain) + 1):
            prefix = current_chain[:i]
            prefix_hits = sum(1 for target_set in target_neighborhoods.values() if set(prefix).intersection(target_set))
            if prefix_hits > max_hits_achieved:
                max_hits_achieved = prefix_hits
                best_prefix_len = i
                
        pruned_chain = current_chain[:best_prefix_len]
        pruned_qubits = set(pruned_chain)
        hits = max_hits_achieved
        
        # 3. Survival & Threshold Evaluation
        if hits >= min_hits:
            # Check how many unused adjacent qubits this chain leaves behind
            chain_neighbors = set()
            for q in pruned_chain:
                chain_neighbors.update(adjacency[q])
            
            free_neighbors = chain_neighbors - used_qubits - pruned_qubits
            
            if len(free_neighbors) >= min_free_neighbors:
                candidates.append((tuple(pruned_chain), hits))
                
    # Deduplicate candidates (we don't want the ILP evaluating identical chains)
    unique_candidates = {frozenset(c[0]): c for c in candidates}.values()
    return list(unique_candidates)
    

def select_optimal_chains_ilp(candidates):
    """
    Selects the maximum disjoint set of chains from the candidate pool.
    candidates: List of tuples -> (chain_tuple, hit_score)
    """
    print(f"Solving ILP for {len(candidates)} candidate chains...")
    prob = pulp.LpProblem("Bipartite_Expansion", pulp.LpMaximize)
    
    # Decision Variables: y[i] = 1 if candidate chain 'i' is kept
    y = pulp.LpVariable.dicts("y", range(len(candidates)), cat=pulp.LpBinary)
    
    # Objective: Maximize total hits (connectivity)
    prob += pulp.lpSum(y[i] * candidates[i][1] for i in range(len(candidates)))
    
    # Constraint Mapping: Which candidates use which qubits?
    qubit_to_cands = {}
    for i, (chain, _) in enumerate(candidates):
        for q in chain:
            if q not in qubit_to_cands:
                qubit_to_cands[q] = []
            qubit_to_cands[q].append(i)
            
    # Constraint: Physical qubits must be mutually disjoint
    for q, cand_list in qubit_to_cands.items():
        prob += pulp.lpSum(y[i] for i in cand_list) <= 1
        
    # Solve
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    
    selected_chains = []
    total_hits = 0
    for i in range(len(candidates)):
        if pulp.value(y[i]) == 1.0:
            selected_chains.append(candidates[i][0])
            total_hits += candidates[i][1]
            
    print(f"ILP selected {len(selected_chains)} new disjoint chains.")
    return selected_chains


def orchestrate_bipartite_expansion(
    sampler, 
    base_left_chains, 
    base_right_chains, 
    locked_qubits=None,
    max_len=10, 
    start_min_hits=35, 
    pool_size=1500, 
    max_iterations=10
):
    """
    Iteratively expands a bipartite embedding on the QPU.
    Alternates between fixing Left to expand Right, and fixing Right to expand Left.
    """
    adjacency = sampler.adjacency
    
    # Create working copies to prevent mutating the original split chains
    left_chains = {k: list(v) for k, v in base_left_chains.items()}
    right_chains = {k: list(v) for k, v in base_right_chains.items()}
    
    # Track globally used physical qubits to enforce disjointness
    used_qubits = set()
    for chain in left_chains.values(): used_qubits.update(chain)
    for chain in right_chains.values(): used_qubits.update(chain)
    if locked_qubits:
        used_qubits.update(locked_qubits)
    
    iteration = 1
    current_min_hits = start_min_hits
    
    while iteration <= max_iterations:
        print(f"\n{'='*40}")
        print(f"--- Expansion Iteration {iteration} | Min Hits Required: {current_min_hits} ---")
        print(f"{'='*40}")
        
        gains_made = False
        
        # --- PASS 1: Expand Right Chains (Hidden Layer) ---
        print(f"\n[Pass 1] Targeting Left Chains ({len(left_chains)}) to generate new Right candidates...")
        candidates_right = generate_candidate_chains(
            adjacency, used_qubits, left_chains, 
            max_len=max_len, min_hits=current_min_hits, target_pool_size=pool_size
        )
        
        if candidates_right:
            new_right = select_optimal_chains_ilp(candidates_right)
            if new_right:
                gains_made = True
                start_idx = len(right_chains)
                for i, chain in enumerate(new_right):
                    # Assign unique logical keys to the new expanded chains
                    new_key = f"h_exp_{start_idx + i}"
                    right_chains[new_key] = list(chain)
                    used_qubits.update(chain)
                print(f" -> Success: Added {len(new_right)} new Right chains.")
                print(f" -> Total Right Chains: {len(right_chains)}")
        else:
            print(" -> No viable Right candidates found under current constraints.")

        # --- PASS 2: Expand Left Chains (Visible Layer) ---
        # Note: We now target the newly expanded right_chains pool
        print(f"\n[Pass 2] Targeting Right Chains ({len(right_chains)}) to generate new Left candidates...")
        candidates_left = generate_candidate_chains(
            adjacency, used_qubits, right_chains, 
            max_len=max_len, min_hits=current_min_hits, target_pool_size=pool_size
        )
        
        if candidates_left:
            new_left = select_optimal_chains_ilp(candidates_left)
            if new_left:
                gains_made = True
                start_idx = len(left_chains)
                for i, chain in enumerate(new_left):
                    new_key = f"v_exp_{start_idx + i}"
                    left_chains[new_key] = list(chain)
                    used_qubits.update(chain)
                print(f" -> Success: Added {len(new_left)} new Left chains.")
                print(f" -> Total Left Chains: {len(left_chains)}")
        else:
            print(" -> No viable Left candidates found under current constraints.")

        # --- Termination & Decay Logic ---
        if not gains_made:
            # If we failed to add chains, try lowering the connectivity standard
            # before completely terminating the algorithm.
            if current_min_hits > 32:
                print(f"\nStagnation reached at min_hits={current_min_hits}. Decaying threshold by 1...")
                current_min_hits -= 1
            else:
                print("\nAbsolute convergence reached. No more chains can be added to the chip.")
                break
        else:
            iteration += 1
            
    if iteration > max_iterations:
        print(f"\nTerminated after reaching max iterations ({max_iterations}).")

    return left_chains, right_chains


def validate_and_repair_chains(sampler, left_chains, right_chains):
    """
    Checks every chain in both dictionaries against the physical sampler.
    Returns:
        valid_left (dict): Only the physically valid left chains.
        valid_right (dict): Only the physically valid right chains.
        report (str): A summary of what was dropped.
    """
    adjacency = sampler.adjacency
    
    def check_chain_set(chain_dict, side_name):
        valid_subset = {}
        broken_count = 0
        first_error = None
        
        # Sort keys to maintain deterministic ordering
        for key in sorted(chain_dict.keys()):
            chain = chain_dict[key]
            is_valid = True
            
            # 1. Check connectivity
            if len(chain) > 1:
                for i in range(len(chain) - 1):
                    u, v = chain[i], chain[i+1]
                    if v not in adjacency[u]:
                        is_valid = False
                        if first_error is None:
                            first_error = f"Chain {key} disjoint at ({u}, {v})"
                        break
            
            # 2. Check qubit existence (sanity check)
            if not is_valid:
                broken_count += 1
            else:
                valid_subset[key] = chain
                
        return valid_subset, broken_count, first_error

    # Run checks
    good_left, left_bad_count, left_err = check_chain_set(left_chains, "Left")
    good_right, right_bad_count, right_err = check_chain_set(right_chains, "Right")
    
    print("\n" + "==="*10)
    print("--- Chain Validation Report ---")
    print(f"Left Chains:  {len(good_left)} valid, {left_bad_count} broken.")
    if left_err: print(f"  -> Example error: {left_err}")
    
    print(f"Right Chains: {len(good_right)} valid, {right_bad_count} broken.")
    if right_err: print(f"  -> Example error: {right_err}")
    print("==="*10 + "\n")
    
    return good_left, good_right



def get_orbit_mappings(
    seed: Union[int, str, None], 
    n_vis: int, 
    n_hid: int
) -> Tuple[List[int], List[int]]:
    """
    Deterministically generates visible and hidden unit mappings based on a seed.
    
    Args:
        seed: Integer seed, 'identity', or None.
        n_vis: Number of visible units (e.g., 75 for left chains).
        n_hid: Number of hidden units (e.g., 75 for right chains).
        
    Returns:
        (vis_mapping, hid_mapping): Lists of indices.
    """
    # Case 1: Identity (Default)
    if seed is None or seed == "identity" or seed == "default":
        return list(range(n_vis)), list(range(n_hid))
    
    # Case 2: Deterministic Shuffle
    if isinstance(seed, int):
        rng = np.random.default_rng(seed)
        
        # We use .tolist() to ensure they are standard Python lists for JSON/Dataclass serialization
        vis_mapping = rng.permutation(n_vis).tolist()
        hid_mapping = rng.permutation(n_hid).tolist()
        
        return vis_mapping, hid_mapping
        
    raise ValueError(f"Unknown seed format: {seed}")