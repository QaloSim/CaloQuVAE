from __future__ import annotations  # 1. Must be the very first line!

from typing import Dict, List, Tuple, Union, Optional
from dataclasses import dataclass, field
import numpy as np
from collections import defaultdict
import minorminer
from dwave.system import DWaveSampler
from dwave.embedding.zephyr import find_biclique_embedding as find_biclique_embedding_zephyr
from dwave.embedding.pegasus import find_biclique_embedding as find_biclique_embedding_pegasus
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
    if "Advantage2" in solver_name:
        find_biclique_embedding = find_biclique_embedding_zephyr
    else:
        find_biclique_embedding = find_biclique_embedding_pegasus
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
    if "Advantage2" in TARGET_SOLVER:
        find_biclique_embedding = find_biclique_embedding_zephyr
    else:
        find_biclique_embedding = find_biclique_embedding_pegasus

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

def augment_cond_sets_from_visible_chains(
    cond_sets, left_chains, right_chains, selected_side, n_extra,
    strategy='last', n_visible=None
):
    """
    When cond_sets is empty or too small, borrow chains from the visible side
    (the side NOT selected as hidden) and promote them to conditioning nodes.

    The visible side chains are standard biclique chains that are guaranteed
    to be connected to the hidden side, so they work as conditioning nodes
    even when no neighbor-qubit conditioning sets could be found.

    Parameters
    ----------
    cond_sets : list of sets
        Existing conditioning sets (may be empty).
    left_chains : dict  {key -> set/list of qubits}
    right_chains : dict {key -> set/list of qubits}
    selected_side : str  'left' or 'right'  (the HIDDEN side returned by select_optimal_side_exact)
    n_extra : int
        How many visible chains to promote to conditioning nodes.
        If n_extra exceeds the number of available visible chains, all are taken.
    strategy : str
        'last'  – borrow from the tail of sorted visible keys (default)
        'first' – borrow from the head
    n_visible : int or None
        If given, trim the returned visible chains dict (after borrowing) to
        exactly this many entries, keeping the first n_visible sorted keys.
        Use this to match the RBM's visible layer size.
        If None, no trimming is applied.

    Returns
    -------
    augmented_cond_sets : list of sets
        Original cond_sets + newly promoted chain sets.
    new_left_chains : dict
        left_chains with any borrowed/trimmed chains removed.
    new_right_chains : dict
        right_chains with any borrowed/trimmed chains removed.
    borrowed_keys : list
        The sorted keys that were removed from the visible side dict
        (includes both conditioning-borrowed and trimmed keys).
    """
    if n_extra <= 0 and n_visible is None:
        return list(cond_sets), dict(left_chains), dict(right_chains), []

    # Identify the visible side (opposite of the hidden/selected side)
    if selected_side == 'right':
        visible_chains = left_chains
        hidden_chains  = right_chains
        visible_label  = 'left'
    elif selected_side == 'left':
        visible_chains = right_chains
        hidden_chains  = left_chains
        visible_label  = 'right'
    else:
        raise ValueError(f"selected_side must be 'left' or 'right', got {selected_side!r}")

    sorted_keys = sorted(visible_chains.keys())
    n_take = min(max(n_extra, 0), len(sorted_keys))

    if strategy == 'last':
        borrowed_keys = sorted_keys[-n_take:] if n_take > 0 else []
    elif strategy == 'first':
        borrowed_keys = sorted_keys[:n_take]
    else:
        raise ValueError(f"strategy must be 'first' or 'last', got {strategy!r}")

    # Build the new conditioning sets from the borrowed chains
    extra_cond_sets = [set(visible_chains[k]) for k in borrowed_keys]
    augmented_cond_sets = list(cond_sets) + extra_cond_sets

    # Remove borrowed keys from the visible dict
    borrowed_set = set(borrowed_keys)
    new_visible = {k: v for k, v in visible_chains.items() if k not in borrowed_set}

    # Trim to n_visible if requested
    trimmed_keys = []
    if n_visible is not None:
        remaining_sorted = sorted(new_visible.keys())
        if len(remaining_sorted) > n_visible:
            trimmed_keys = remaining_sorted[n_visible:]
            trim_set = set(trimmed_keys)
            new_visible = {k: v for k, v in new_visible.items() if k not in trim_set}

    if selected_side == 'right':
        new_left_chains, new_right_chains = new_visible, dict(right_chains)
    else:
        new_left_chains, new_right_chains = dict(left_chains), new_visible

    n_orig = len(cond_sets)
    n_aug  = len(augmented_cond_sets)
    msg = (f"augment_cond_sets_from_visible_chains: "
           f"borrowed {n_take} chains from the '{visible_label}' (visible) side. "
           f"cond_sets: {n_orig} -> {n_aug}. "
           f"Remaining visible chains: {len(new_visible)}")
    if trimmed_keys:
        msg += f" (trimmed {len(trimmed_keys)} extra to match n_visible={n_visible})"
    print(msg + ".")

    all_removed_keys = borrowed_keys + trimmed_keys
    return augmented_cond_sets, new_left_chains, new_right_chains, all_removed_keys

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
    max_iterations=10,
    min_hits_floor=32,  # Set equal to start_min_hits to disable decay
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
            if current_min_hits > min_hits_floor:
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


# --- D. Pareto Frontier Search ---

@dataclass
class EmbeddingPoint:
    """
    A single point on the size-vs-connectivity Pareto frontier.

    Connectivity metrics:
      - density:     fraction of all (n_vis * n_hid) possible edges that are present
      - min_degree:  worst-case node degree (a value of 0 means a dead / isolated unit)
      - mean_degree: average degree across all logical nodes
    """
    density_floor: float       # the sparsity parameter used for this run (0.0–1.0)
    min_hits_used: int         # absolute min_hits = round(density_floor * base_n)
    n_vis: int
    n_hid: int
    density: float
    min_degree: int
    mean_degree: float
    left_chains: Dict  = field(repr=False)
    right_chains: Dict = field(repr=False)

    @property
    def total_nodes(self) -> int:
        return self.n_vis + self.n_hid


def compute_connectivity_matrix(left_chains: Dict, right_chains: Dict, adjacency) -> np.ndarray:
    """
    Computes the biadjacency matrix M where M[i,j]=1 iff left chain i has at least
    one physical coupler to right chain j.

    Returns an (n_vis, n_hid) int8 array.
    """
    left_keys  = sorted(left_chains.keys(),  key=str)
    right_keys = sorted(right_chains.keys(), key=str)

    # Pre-compute physical neighborhoods for every right chain (avoids O(n^2) adjacency lookups)
    right_neighbor_sets = []
    for rk in right_keys:
        nbrs = set()
        for q in right_chains[rk]:
            nbrs.update(adjacency[q])
        right_neighbor_sets.append(nbrs)

    mat = np.zeros((len(left_keys), len(right_keys)), dtype=np.int8)
    for i, lk in enumerate(left_keys):
        lset = set(left_chains[lk])
        for j, nbrs in enumerate(right_neighbor_sets):
            if lset & nbrs:
                mat[i, j] = 1
    return mat


def _connectivity_stats(mat: np.ndarray) -> Tuple[float, int, float]:
    """Returns (density, min_degree, mean_degree) for a biadjacency matrix."""
    n_vis, n_hid = mat.shape
    row_deg = mat.sum(axis=1)  # each visible node's degree
    col_deg = mat.sum(axis=0)  # each hidden node's degree
    all_deg = np.concatenate([row_deg, col_deg])
    density     = float(mat.sum()) / (n_vis * n_hid)
    min_degree  = int(all_deg.min())
    mean_degree = float(all_deg.mean())
    return density, min_degree, mean_degree


def pareto_sweep_expansion(
    sampler,
    base_left_chains: Dict,
    base_right_chains: Dict,
    density_floors: Optional[List[float]] = None,
    locked_qubits=None,
    max_len: int = 10,
    pool_size: int = 1500,
    max_iterations: int = 15,
) -> List[EmbeddingPoint]:
    """
    Sweeps over sparsity thresholds to map the size-vs-connectivity Pareto frontier.

    Each sweep starts fresh from the biclique seed (base_left_chains, base_right_chains)
    and runs expansion with a FIXED min_hits threshold (no decay), so every point is
    independent and comparable.

    Args:
        density_floors: Fractions in (0, 1] relative to the biclique base size.
                        A value of 1.0 means new chains must connect to *every*
                        existing opposite-side chain (maximally dense, fewest additions).
                        A value of 0.3 means new chains need only 30% connectivity
                        (sparser, but more chains can typically be added).
                        Defaults to 9 evenly-spaced values from 0.3 to 1.0.

    Returns:
        List of EmbeddingPoint, one per density_floor value, sorted by density_floor.
    """
    if density_floors is None:
        density_floors = [round(f, 2) for f in np.linspace(0.3, 1.0, 9)]

    # Biclique base size — fixed reference for computing min_hits
    base_n = max(len(base_left_chains), len(base_right_chains))

    results: List[EmbeddingPoint] = []

    for f in sorted(density_floors):
        min_hits = max(1, round(f * base_n))
        print(f"\n{'='*55}")
        print(f"  Pareto sweep: density_floor={f:.2f}  min_hits={min_hits}/{base_n}")
        print(f"{'='*55}")

        # Independent expansion from the biclique seed, no decay
        left, right = orchestrate_bipartite_expansion(
            sampler,
            base_left_chains,
            base_right_chains,
            locked_qubits=locked_qubits,
            max_len=max_len,
            start_min_hits=min_hits,
            pool_size=pool_size,
            max_iterations=max_iterations,
            min_hits_floor=min_hits,  # disables decay
        )

        mat = compute_connectivity_matrix(left, right, sampler.adjacency)
        density, min_deg, mean_deg = _connectivity_stats(mat)

        pt = EmbeddingPoint(
            density_floor=f,
            min_hits_used=min_hits,
            n_vis=len(left),
            n_hid=len(right),
            density=density,
            min_degree=min_deg,
            mean_degree=mean_deg,
            left_chains=left,
            right_chains=right,
        )
        results.append(pt)
        print(f"  -> {pt.n_vis}v × {pt.n_hid}h  |  density={density:.3f}  "
              f"min_deg={min_deg}  mean_deg={mean_deg:.1f}")

    return results


def extract_pareto_frontier(
    results: List[EmbeddingPoint],
    size_metric: str = "total",
    min_degree_threshold: int = 0,
) -> List[EmbeddingPoint]:
    """
    Returns the Pareto-optimal subset of EmbeddingPoints.

    A point A dominates B if:
      size(A) >= size(B)  AND  density(A) >= density(B)
    with strict inequality in at least one dimension.

    Args:
        size_metric: 'total' uses n_vis + n_hid;
                     'min' uses min(n_vis, n_hid) (balanced RBM measure).
        min_degree_threshold: Drop any point where min_degree < this value
                              (e.g., 1 removes points with dead/isolated units).

    Returns:
        Non-dominated points sorted by ascending size.
    """
    def size_of(p: EmbeddingPoint) -> int:
        return p.total_nodes if size_metric == "total" else min(p.n_vis, p.n_hid)

    # Optional filter: remove points with dead units
    candidates = [p for p in results if p.min_degree >= min_degree_threshold]

    pareto = []
    for p in candidates:
        dominated = any(
            size_of(q) >= size_of(p) and q.density >= p.density
            and (size_of(q) > size_of(p) or q.density > p.density)
            for q in candidates if q is not p
        )
        if not dominated:
            pareto.append(p)

    return sorted(pareto, key=size_of)


def _build_sparse_bipartite_logical_graph(
    n_vis: int, n_hid: int, density: float, rng: np.random.Generator
) -> nx.Graph:
    """
    Builds a bipartite logical graph where each visible node is connected to
    exactly round(density * n_hid) randomly chosen hidden nodes.

    Node labelling: visible = 0..n_vis-1, hidden = n_vis..n_vis+n_hid-1.
    At density=1.0 this produces the complete bipartite K_{n_vis, n_hid}.
    """
    G = nx.Graph()
    G.add_nodes_from(range(n_vis),          bipartite=0)
    G.add_nodes_from(range(n_vis, n_vis + n_hid), bipartite=1)

    k = max(1, round(density * n_hid))
    hidden = np.arange(n_vis, n_vis + n_hid)
    for v in range(n_vis):
        neighbours = rng.choice(hidden, size=min(k, n_hid), replace=False)
        for h in neighbours:
            G.add_edge(v, int(h))
    return G


def pareto_sweep_minorminer(
    sampler,
    base_left_chains: Dict,
    base_right_chains: Dict,
    density_floors: Optional[List[float]] = None,
    step: int = 4,
    max_extra: int = 40,
    n_tries: int = 3,
    random_seed: int = 42,
    **miner_kwargs,
) -> List[EmbeddingPoint]:
    """
    Alternative Pareto sweep using minorminer.find_embedding seeded with the
    biclique chains, instead of the BFS + ILP expansion.

    For each density_floor, we incrementally increase (n_vis, n_hid) by `step`
    until minorminer fails, recording every successful embedding as a Pareto
    point.  The biclique chains are passed as initial_chains so minorminer can
    build on top of them rather than starting from scratch.

    Why this differs from the BFS+ILP approach
    -------------------------------------------
    - minorminer optimises globally for *short chains* (fewer physical qubits
      per logical node).  It has no concept of RBM connectivity.
    - Our BFS+ILP approach greedily maximises per-chain connectivity to the
      opposite side, which is exactly what an RBM needs.
    - Sparser logical graphs let minorminer relax chain-adjacency requirements,
      potentially fitting more nodes — but the resulting connectivity is
      whatever the physical layout happens to give, not the maximised value our
      method targets.

    Args:
        density_floors: Target logical graph densities to sweep. Defaults to
                        [0.4, 0.6, 0.8, 1.0].
        step:       How many nodes to add to each side per increment.
        max_extra:  Maximum additional nodes beyond the biclique base to try.
        n_tries:    Number of minorminer attempts per configuration (best kept).
        random_seed: For reproducible logical graph generation.
        **miner_kwargs: Forwarded to minorminer.find_embedding (e.g.,
                        max_no_improvement=10, timeout=30).
    """
    qpu_graph = sampler.to_networkx_graph()
    adjacency  = sampler.adjacency

    base_left_keys  = sorted(base_left_chains.keys(),  key=str)
    base_right_keys = sorted(base_right_chains.keys(), key=str)
    base_n = len(base_left_chains)
    base_m = len(base_right_chains)

    if density_floors is None:
        density_floors = [0.4, 0.6, 0.8, 1.0]

    rng = np.random.default_rng(random_seed)
    results: List[EmbeddingPoint] = []

    for f in sorted(density_floors):
        print(f"\n{'='*55}")
        print(f"  minorminer Pareto sweep  density_floor={f:.2f}")
        print(f"{'='*55}")

        for extra in range(0, max_extra + 1, step):
            n_vis = base_n + extra
            n_hid = base_m + extra

            # ── 1. Build sparse logical graph ────────────────────────────────
            G = _build_sparse_bipartite_logical_graph(n_vis, n_hid, f, rng)

            # ── 2. Seed from biclique chains ─────────────────────────────────
            # Visible nodes 0..base_n-1 get biclique left chains.
            # Hidden  nodes n_vis..n_vis+base_m-1 get biclique right chains.
            # Extra nodes beyond the biclique get no initial chain.
            initial_chains: Dict = {}
            for i in range(base_n):
                initial_chains[i] = list(base_left_chains[base_left_keys[i]])
            for j in range(base_m):
                initial_chains[n_vis + j] = list(base_right_chains[base_right_keys[j]])

            # ── 3. Run minorminer (multiple tries) ───────────────────────────
            best_emb = None
            best_total_len = float("inf")
            for _ in range(n_tries):
                emb = minorminer.find_embedding(
                    G, qpu_graph,
                    initial_chains=initial_chains,
                    **miner_kwargs,
                )
                if emb:
                    total_len = sum(len(c) for c in emb.values())
                    if total_len < best_total_len:
                        best_emb = emb
                        best_total_len = total_len

            if not best_emb:
                print(f"  -> Failed at {n_vis}v × {n_hid}h — stopping this density floor.")
                break

            # ── 4. Extract chains and measure actual physical connectivity ───
            left_chains  = {i:        list(best_emb[i])             for i in range(n_vis)}
            right_chains = {n_vis + j: list(best_emb[n_vis + j])    for j in range(n_hid)}

            mat = compute_connectivity_matrix(left_chains, right_chains, adjacency)
            density, min_deg, mean_deg = _connectivity_stats(mat)

            pt = EmbeddingPoint(
                density_floor=f,
                min_hits_used=round(f * max(n_vis, n_hid)),
                n_vis=n_vis,
                n_hid=n_hid,
                density=density,
                min_degree=min_deg,
                mean_degree=mean_deg,
                left_chains=left_chains,
                right_chains=right_chains,
            )
            results.append(pt)
            print(f"  -> {n_vis}v × {n_hid}h  |  density={density:.3f}  "
                  f"min_deg={min_deg}  mean_deg={mean_deg:.1f}")

    return results