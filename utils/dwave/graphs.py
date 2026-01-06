from __future__ import annotations  # 1. Must be the very first line!

from typing import Dict, List, Tuple
from collections import defaultdict
from dwave.system import DWaveSampler
from dwave.embedding.zephyr import find_biclique_embedding
from dwave.system.composites import FixedEmbeddingComposite
from utils.FluxBiases import h_to_fluxbias 
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from utils.dwave.sampling_backend import ChainAnalysisResult

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



def run_embedding(rbm_size, solver_name):
    """
    Performs the biclique embedding on the target QPU.
    Returns the sampler, the working graph, and the chains.
    """
    print(f"--- 1. Running Embedding for K_{rbm_size},{rbm_size} on {solver_name} ---")
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
            rbm_size, rbm_size, target_graph=working_graph
        )
    except Exception as e:
        print(f"Error during find_biclique_embedding: {e}")
        print("This often means the graph is too large for the QPU.")
        return target_sampler, working_graph, None, None, None

    all_chains = list(left_chains.values()) + list(right_chains.values())
    max_len = max(len(chain) for chain in all_chains)

    # This is the set of ALL qubits used in the 76x76 embedding
    qubits_used = set()
    for chain in all_chains:
        qubits_used.update(chain)

    print(f" -> Max chain length: {max_len}")
    print(f" -> Total qubits used: {len(qubits_used)}")


    # DIAGNOSTIC BLOCK
    print("--- Diagnosing Chains ---")
    # Grab the first chain from the left side
    test_key = next(iter(left_chains))
    test_chain = left_chains[test_key]
    print(f"Chain {test_key}: {test_chain}")

    # Check connectivity on the actual sampler
    valid = True
    for i in range(len(test_chain)-1):
        u, v = test_chain[i], test_chain[i+1]
        if u not in target_sampler.adjacency[v]:
            print(f"❌ CRITICAL FAILURE: {u} and {v} are NOT connected on the QPU!")
            valid = False
    if valid:
        print("✅ Chain is physically connected.")
    else:
        print("⚠️  The input chains are invalid. Stop here.")
    return target_sampler, working_graph, qubits_used, left_chains, right_chains



def build_neighbor_sets(target_sampler, target_logical_nodes, qubits_used):
    """
    Builds the 76 "available neighbor" sets, V_i.
    V_i = {all *available* qubits adjacent to logical node i}
    """
    print(f"\n--- 2. Building {len(target_logical_nodes)} Neighbor Sets ---")
    
    # 1. Get all qubits available on the chip
    all_physical_qubits = set(target_sampler.nodelist)
    qubits_avail = all_physical_qubits - qubits_used
    print(f"Total available qubits: {len(qubits_avail)}")

    # 2. Get the adjacency property of the sampler
    qpu_adjacency = target_sampler.adjacency
    
    neighbor_sets = []
    min_adj_size = float('inf')
    
    # 3. Iterate over each of the 76 logical nodes in Side A
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
            
    print(f"All 76 neighbor sets built.")
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