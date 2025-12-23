import torch
from typing import Dict, Tuple, List, Optional
import time
import numpy as np
from dwave.system import DWaveSampler
from dwave.embedding.zephyr import find_biclique_embedding
from dwave.system.composites import FixedEmbeddingComposite
from utils.FluxBiases import h_to_fluxbias
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict



def rbm_to_logical_ising(
    rbm, 
    beta: float = 1.0
) -> Tuple[Dict[int, float], Dict[Tuple[int, int], float], int, int]:
    """
    Converts RBM parameters to Logical Ising parameters without hardware clamping.
    """
    num_visible = rbm.params["weight_matrix"].shape[0]
    num_hidden = rbm.params["weight_matrix"].shape[1]

    # Clone parameters
    W = rbm.params["weight_matrix"].clone()
    vb = rbm.params["vbias"].clone()
    hb = rbm.params["hbias"].clone()
    
    # 1. Calculate Couplers (J)
    # J_ij = W_ij / 4 * beta
    dwave_J_tensor = -W / 4.0 / beta
    
    # 2. Calculate Visible Bias (h_visible)
    # h_i = v_i/2 + sum_j(W_ij/4)
    s_v = -torch.sum(W, dim=1) / 4.0 / beta
    dwave_h_visible = -vb / 2.0 / beta + s_v

    # 3. Calculate Hidden Bias (h_hidden)
    # h_j = h_j/2 + sum_i(W_ij/4)
    s_h = -torch.sum(W, dim=0) / 4.0 / beta
    dwave_h_hidden = -hb / 2.0 / beta + s_h
    
    # --- NO CLAMPING HERE ---
    # We return the raw, mathematically correct values. 
    # Clamping is now the sampler's responsibility.

    all_biases_np = torch.cat([dwave_h_visible, dwave_h_hidden]).detach().cpu().numpy()
    dwave_J_tensor_np = dwave_J_tensor.detach().cpu().numpy()

    h = {}
    for i in range(num_visible + num_hidden):
        h[i] = float(all_biases_np[i])

    J = {}
    for i in range(num_visible):
        for j in range(num_hidden):
            val = float(dwave_J_tensor_np[i, j])
            # Shift hidden indices by num_visible
            J[(i, j + num_visible)] = val
                
    return h, J, num_visible, num_hidden

def get_sampler_and_biclique_embedding(
    num_visible: int, 
    num_hidden: int, 
    solver_name: str
) -> Tuple[object, Dict[int, List[int]]]:
    
    print(f"--- Finding Zephyr Embedding for K_{num_visible},{num_hidden} ---")

    try:
        raw_sampler = DWaveSampler(solver=solver_name)
    except Exception as e:
        print(f"Error initializing sampler: {e}")
        return None, None

    # 1. Get the hardware graph
    # This ensures we don't use broken qubits specific to this chip
    working_graph = raw_sampler.to_networkx_graph()
    
    # 2. Generate the embedding
    # Note: We pass num_visible (int) and num_hidden (int).
    # The docs say: "If both a and b are integers, the right shore will be labelled [a, a+b-1]."
    # This means the keys will automatically be 0..n_vis-1 and n_vis..total-1.
    try:
        left_dict, right_dict = find_biclique_embedding(
            num_visible, 
            num_hidden, 
            target_graph=working_graph
        )
    except ValueError as e:
        print(f"Error: RBM is too large for this QPU or topology mismatch. {e}")
        return raw_sampler, None

    # 3. Merge the dictionaries
    # We don't need to wrap values in lists; find_biclique_embedding 
    # already returns chains (iterables) as values.
    embedding = {**left_dict, **right_dict}
    
    # Convert all chains to lists just to be safe (sometimes they are sets/tuples)
    for k, v in embedding.items():
        embedding[k] = list(v)

    print(f"Successfully created embedding for {len(embedding)} nodes.")
    qpu_sampler = FixedEmbeddingComposite(raw_sampler, embedding)

    return raw_sampler, embedding, qpu_sampler


def sample_logical_ising(
    qpu_sampler,
    h: Dict[int, float],
    J: Dict[Tuple[int, int], float],
    num_samples: int = 64,
    measure_time: bool = False,
    chain_strength: float = None  # Changed default to None
) -> Tuple[object, Optional[float]]:
    
    # --- 1. Dynamic Hardware Limit Detection ---
    h_min, h_max = qpu_sampler.child.properties['h_range']
    j_min, j_max = qpu_sampler.child.properties['extended_j_range']

    # --- 2. Clamp Parameters ---
    # Create new dicts to ensure we don't mutate the inputs
    h_clamped = {k: max(h_min, min(v, h_max)) for k, v in h.items()}
    J_clamped = {k: max(j_min, min(v, j_max)) for k, v in J.items()}
    
    # --- 3. Handle Chain Strength ---
    # If not provided, we can now set it based on the max of the sampler limits
    if chain_strength is None:
        chain_strength = j_max
        
    # --- 4. Submit to QPU ---
    sampling_time = None
    
    # Common args
    sample_kwargs = {
        'num_reads': num_samples,
        'answer_mode': 'raw',
        'auto_scale': False, # We handled scaling/clamping manually
        'chain_break_fraction': True, 
        'chain_strength': chain_strength
    }

    if measure_time:
        start = time.perf_counter()
        response = qpu_sampler.sample_ising(h_clamped, J_clamped, **sample_kwargs)
        sampling_time = time.perf_counter() - start
    else:
        response = qpu_sampler.sample_ising(h_clamped, J_clamped, **sample_kwargs)

    # Optional: Alert if clamping was severe (debug only)
    # print(f"  Sent h range: [{min(h_clamped.values()):.2f}, {max(h_clamped.values()):.2f}]")
    
    if hasattr(response, 'record'):
        print(f"Fraction of broken chains: {np.mean(response.record.chain_break_fraction):.4f}")

    return response, sampling_time

def sample_ising_flux_bias(
    qpu_sampler,
    h: Dict[int, float],
    J: Dict[Tuple[int, int], float],
    flux_biases: List[float],
    num_samples: int = 1,
    measure_time: bool = False,
    chain_strength: float = None
) -> object:
    """
    Submits an Ising problem to the QPU with specific Flux Biases applied.
    automatically handles hardware clamping and drift compensation flags.
    """
    # --- 1. Dynamic Hardware Limit Detection ---

    h_min, h_max = qpu_sampler.child.properties['h_range']
    j_min, j_max = qpu_sampler.child.properties['extended_j_range']
    # --- 2. Clamp Parameters ---
    h_clamped = {k: max(h_min, min(v, h_max)) for k, v in h.items()}
    J_clamped = {k: max(j_min, min(v, j_max)) for k, v in J.items()}
    
    # --- 3. Handle Chain Strength ---
    if chain_strength is None:
        chain_strength = j_max
    # --- 4. Submit with Flux Flags ---
    # flux_drift_compensation MUST be False when providing flux_biases
    sample_kwargs = {
        'num_reads': num_samples,
        'answer_mode': 'raw',
        'auto_scale': False,
        'chain_break_fraction': True, 
        'chain_strength': chain_strength,
        'flux_biases': flux_biases,
        'flux_drift_compensation': False
    }
    if measure_time:
        start = time.perf_counter()
        response = qpu_sampler.sample_ising(h_clamped, J_clamped, **sample_kwargs)
        sampling_time = time.perf_counter() - start
    else:
        response = qpu_sampler.sample_ising(h_clamped, J_clamped, **sample_kwargs)
        sampling_time = None

    return response, sampling_time

def sample_rbm_qpu(
    rbm,    
    qpu_sampler,
    beta: float = 1.0,
    num_samples: int = 64,
    measure_time: bool = False,
    chain_strength: float = None
):
    """
    Samples from the RBM using the D-Wave QPU sampler.
    1. Converts RBM to Logical Ising (Math)
    2. Samples from QPU (Hardware Clamping happens here)
    3. Processes samples back to RBM tensors
    """
    # 1. Math Conversion
    h, J, num_visible, num_hidden = rbm_to_logical_ising(rbm, beta=beta)

    # 2. Hardware Sampling
    response, sampling_time = sample_logical_ising(
        qpu_sampler,
        h,
        J,
        num_samples=num_samples,
        measure_time=measure_time,
        chain_strength=chain_strength
    )
        
    # 3. Tensor Conversion
    visible_samples, hidden_samples = process_rbm_samples(
        response,
        num_visible,
        num_hidden,
        device=rbm.device
    )

    return visible_samples, hidden_samples, sampling_time

def joint_energy(rbm, v, h):
    """
    Computes the Joint Hamiltonian Energy E(v,h) = -vWh - bv - ch.
    Calculates the specific energy of the configuration, not Free Energy.
    """
    # 1. Interaction term: sum( (v @ W) * h )
    # v @ W -> [batch, n_hidden], then element-wise multiply by h
    interaction = (v @ rbm.params["weight_matrix"]) * h 
    interaction_sum = interaction.sum(dim=1)
    
    # 2. Bias terms
    v_bias = v @ rbm.params["vbias"]
    h_bias = h @ rbm.params["hbias"]
    
    # 3. Total Energy (negative sum)
    return -interaction_sum - v_bias - h_bias


def process_rbm_samples(
    response, # dimod.SampleSet object
    num_visible: int,
    num_hidden: int,
    device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Converts raw D-Wave spin samples {-1, 1} into binary {0, 1} RBM
    samples, splits them, and moves them to the target device.

    Args:
        response (dimod.SampleSet): The raw output from the D-Wave sampler.
        num_visible (int): Number of visible nodes.
        num_hidden (int): Number of hidden nodes.
        device (torch.device): The torch device (e.g., 'cuda' or 'cpu')
                               to put the final tensors on.

    Returns:
        tuple:
            - visible_samples (torch.Tensor): Samples for visible nodes.
            - hidden_samples (torch.Tensor): Samples for hidden nodes.
    """
    total_nodes = num_visible + num_hidden
    num_samples = len(response.record.sample)

    # --- 5. Process Samples ---
    var_to_col_idx = {var: i for i, var in enumerate(response.variables)}
    
    try:
        ordered_cols = [var_to_col_idx[i] for i in range(total_nodes)]
    except KeyError as e:
        print(f"Error: A logical node index '{e.args[0]}' was missing from the D-Wave response variables.")
        print("This can happen if a node has 0 bias and 0 couplers (is disconnected).")
        # Handle this by creating a default (e.g., all 0) sample array
        visible_samples = torch.zeros((num_samples, num_visible), device=device, dtype=torch.float)
        hidden_samples = torch.zeros((num_samples, num_hidden), device=device, dtype=torch.float)
        return visible_samples, hidden_samples

    # Get all samples and re-order them
    dwave_spin_samples = response.record.sample[:, ordered_cols]
    
    # Convert from numpy array to torch tensor
    dwave_spin_samples_t = torch.tensor(dwave_spin_samples, dtype=torch.float, device=device)

    # --- 6. Convert spin samples {-1, 1} to binary {0, 1} ---
    _ZERO = torch.tensor(0., dtype=torch.float, device=device)
    _MINUS_ONE = torch.tensor(-1., dtype=torch.float, device=device)
    
    dwave_binary_samples = torch.where(
        dwave_spin_samples_t == _MINUS_ONE,
        _ZERO,
        dwave_spin_samples_t
    )

    # --- 7. Split into Visible and Hidden Partitions ---
    visible_samples = dwave_binary_samples[:, :num_visible]
    hidden_samples = dwave_binary_samples[:, num_visible:]
    
    return visible_samples, hidden_samples




def find_beta(
    rbm, 
    qpu_sampler,
    num_reads: int = 128,
    beta_init: float = 1.0,
    lr: float = 0.1,
    num_epochs: int = 15,
    adaptive: bool = False,
    lr_cap: float = 0.5
):
    beta = beta_init
    beta_history = []
    rbm_energy_history = []
    qpu_energy_history = []
    
    # --- 1. Get RBM Baseline Samples ---
    print("Sampling baseline from RBM chains...")
    rbm.sample_state(beta=1.0) # Ensure mixed
    rbm_v = rbm.chains["v"][:num_reads]
    rbm_h = rbm.chains["h"][:num_reads]
    
    with torch.no_grad():
        e_rbm_batch = joint_energy(rbm, rbm_v, rbm_h)
        mean_rbm_energy = e_rbm_batch.mean().item()
    
    print(f"Target (RBM) Mean Energy: {mean_rbm_energy:.4f}")

    for epoch in range(num_epochs):
        
        # --- A. Sample from QPU (Modular Call) ---
        # We pass chain_strength=None so sample_logical_ising calculates it 
        # dynamically based on the clamped J values for this specific beta.
        dwave_v, dwave_h, _ = sample_rbm_qpu(
            rbm, 
            qpu_sampler, 
            beta=beta, 
            num_samples=num_reads,
            chain_strength=None 
        )
        
        # --- B. Energy & Variance ---
        with torch.no_grad():
            e_dwave_batch = joint_energy(rbm, dwave_v, dwave_h)
            mean_dwave_energy = e_dwave_batch.mean().item()
            var_dwave_energy = torch.var(e_dwave_batch).item()
            
        # --- C. Adaptive Learning Rate ---
        if adaptive:
            safe_var = max(1e-6, var_dwave_energy)
            calc_lr = (beta**2) / safe_var
            current_lr = min(lr_cap, max(lr, calc_lr))
        else:
            current_lr = lr

        # --- D. Update Beta ---
        # Heuristic: 
        # J ~ 1/beta. 
        # If QPU Energy > RBM Energy (too hot/random), we need stronger interactions (Higher J).
        # Higher J means LOWER Beta.
        diff = mean_dwave_energy - mean_rbm_energy
        beta -= current_lr * diff        
        
        # Safety clamp
        beta = max(1e-2, beta)
        
        # Logging
        beta_history.append(beta)
        rbm_energy_history.append(mean_rbm_energy)
        qpu_energy_history.append(mean_dwave_energy)
        
        print(f"Epoch {epoch}: Beta={beta:.4f} | QPU_E={mean_dwave_energy:.2f} vs RBM_E={mean_rbm_energy:.2f} | Diff={diff:.2f}")
        
        if abs(diff) < 0.5:
            print("Converged: Energy difference is below threshold.")
            break
            
    return beta, beta_history, rbm_energy_history, qpu_energy_history


def get_physical_flux_biases(
    embedding: Dict[int, List[int]],
    total_physical_qubits: int,
    logical_clamps: Dict[int, float],
    clamp_strength_h: float = 30.0
) -> List[float]:
    """
    Generates the physical flux bias list. 
    Matches the logical_clamps to physical qubits via the embedding.
    """
    physical_flux_biases = [0.0] * total_physical_qubits
    
    for logical_node, direction in logical_clamps.items():
        if logical_node not in embedding:
            continue
            
        # All physical qubits in a chain get the same bias
        chain = embedding[logical_node]
        h_val = direction * clamp_strength_h 
        fb_val = h_to_fluxbias(h_val)
        
        for physical_qubit_idx in chain:
            if physical_qubit_idx < total_physical_qubits:
                physical_flux_biases[physical_qubit_idx] += fb_val

    return physical_flux_biases


def sample_flux_conditioned_batch(
    rbm,
    qpu_sampler,
    embedding: Dict[int, List[int]],
    binary_patterns_batch: torch.Tensor,
    logical_idx_map: List[int],
    beta: float = 1.0,
    clamp_strength_h: float = 20.0,
    chain_strength: float = None
):
    """
    Iterates through a batch of patterns, applies flux clamps to specific nodes,
    and samples the remaining RBM nodes from the QPU.
    """
    # --- 1. Math Conversion ---
    base_h, base_J, n_vis, n_hid = rbm_to_logical_ising(rbm, beta=beta)


    total_qubits = qpu_sampler.child.properties['num_qubits']

    SAFE_FLUX_CAP = 0.01 
    
    # 3. Process Batch
    visible_batch_list = []
    hidden_batch_list = []
    
    # NEW: List to track chain breaks across the loop
    chain_break_history = []
    
    import copy
    
    print(f"--- Processing Flux Batch ({len(binary_patterns_batch)} samples) ---")
    
    for i, row in enumerate(binary_patterns_batch):
        row_np = row.detach().cpu().numpy()
        
        # A. Determine Clamps
        logical_clamps = {}
        current_h = copy.deepcopy(base_h)
        
        for bit_idx, bit_val in enumerate(row_np):
            target_node = logical_idx_map[bit_idx]
            spin = 1 if bit_val > 0.5 else -1
            logical_clamps[target_node] = spin
            
            # Zero out logical h so it doesn't fight the flux
            if target_node in current_h:
                current_h[target_node] = 0.0

        # B. Calculate Raw Flux List
        raw_fb_list = get_physical_flux_biases(
            embedding=embedding,
            total_physical_qubits=total_qubits,
            logical_clamps=logical_clamps,
            clamp_strength_h=clamp_strength_h 
        )
        
        # C. APPLY SAFETY CLIP
        safe_fb_list = []
        for fb in raw_fb_list:
            if fb > SAFE_FLUX_CAP:
                safe_fb_list.append(SAFE_FLUX_CAP)
            elif fb < -SAFE_FLUX_CAP:
                safe_fb_list.append(-SAFE_FLUX_CAP)
            else:
                safe_fb_list.append(fb)
        
        # D. Submit
        response, _ = sample_ising_flux_bias(
            qpu_sampler,
            h=current_h,
            J=base_J,
            flux_biases=safe_fb_list, # Send the clipped list
            num_samples=1, 
            chain_strength=chain_strength
        )

        # NEW: Track chain breaks for this sample
        if hasattr(response, 'record'):
            chain_break_history.append(np.mean(response.record.chain_break_fraction))
        
        # E. Process
        v_s, h_s = process_rbm_samples(response, n_vis, n_hid, rbm.device)
        visible_batch_list.append(v_s)
        hidden_batch_list.append(h_s)
        
        if i % 10 == 0:
            print(f"  Sampled {i}/{len(binary_patterns_batch)}", end='\r')

    all_v = torch.cat(visible_batch_list, dim=0)
    all_h = torch.cat(hidden_batch_list, dim=0)
    
    print("\nBatch Complete.")
    
    # NEW: Print average chain break stats for the whole batch
    if chain_break_history:
        print(f"Mean chain break fraction (entire batch): {np.mean(chain_break_history):.4f}")

    return all_v, all_h


def sample_flux_conditioned_fast(
    rbm,
    qpu_sampler,
    embedding: Dict[int, List[int]],
    binary_patterns_batch: torch.Tensor,
    logical_idx_map: List[int],
    beta: float = 1.0,
    clamp_strength_h: float = 20.0,
    chain_strength: float = None
):
    """
    Fast version of flux conditioning.
    1. Takes ONLY the first pattern from binary_patterns_batch.
    2. Configures the QPU flux biases for that ONE pattern.
    3. Requests 'batch_size' reads from the QPU.
    """
    batch_size = binary_patterns_batch.shape[0]
    
    # 1. Math Conversion
    base_h, base_J, n_vis, n_hid = rbm_to_logical_ising(rbm, beta=beta)

    # 2. Hardware Setup
    total_qubits = qpu_sampler.child.properties['num_qubits']

    # --- 3. Setup for the FIRST pattern only ---
    row_np = binary_patterns_batch[0].detach().cpu().numpy()
    
    logical_clamps = {}
    current_h = base_h.copy() 
    
    for bit_idx, bit_val in enumerate(row_np):
        target_node = logical_idx_map[bit_idx]
        spin = 1 if bit_val > 0.5 else -1
        logical_clamps[target_node] = spin
        
        # Zero out logical h so it doesn't fight the flux
        if target_node in current_h:
            current_h[target_node] = 0.0

    # --- 4. Calculate Flux (With Safety Clipping) ---
    SAFE_FLUX_CAP = 0.01 
    
    raw_fb_list = get_physical_flux_biases(
        embedding=embedding,
        total_physical_qubits=total_qubits,
        logical_clamps=logical_clamps,
        clamp_strength_h=clamp_strength_h 
    )
    
    # Clip limits
    safe_fb_list = [
        max(-SAFE_FLUX_CAP, min(val, SAFE_FLUX_CAP)) 
        for val in raw_fb_list
    ]
    
    # --- 5. Single Batch Submission ---
    response, _ = sample_ising_flux_bias(
        qpu_sampler,
        h=current_h,
        J=base_J,
        flux_biases=safe_fb_list,
        num_samples=batch_size, 
        chain_strength=chain_strength
    )

    if hasattr(response, 'record'):
        print(f"Fraction of broken chains: {np.mean(response.record.chain_break_fraction):.4f}")
    
    # --- 6. Process ---
    v_s, h_s = process_rbm_samples(response, n_vis, n_hid, rbm.device)
    
    return v_s, h_s

def find_beta_flux_bias(
    rbm, 
    qpu_sampler,
    embedding: Dict[int, List[int]],
    binary_patterns_batch: torch.Tensor, 
    logical_idx_map: List[int],          
    rbm_gibbs_steps: int = 1000,           
    beta_init: float = 1.0,
    lr: float = 0.1,
    num_epochs: int = 15,
    adaptive: bool = False,
    lr_cap: float = 0.5
):
    beta = beta_init
    beta_history = []
    rbm_energy_history = []
    qpu_energy_history = []
    n_clamped = binary_patterns_batch.shape[1]
    batch_size = binary_patterns_batch.shape[0]

    print(f"--- Starting Fast Flux-Conditioned Beta Optimization ---")
    
    # --- 1. Get RBM Baseline (Homogeneous) ---
    print("Calculating RBM Conditional Energy for the PRIMARY pattern...")
    
    # CRITICAL: We create a batch where the FIRST pattern is repeated.
    # This ensures RBM energy is calculated on the same problem the QPU is solving.
    primary_pattern = binary_patterns_batch[0].unsqueeze(0) # [1, n_clamped]
    homogeneous_batch = primary_pattern.repeat(batch_size, 1) # [batch, n_clamped]
    
    full_v_rbm = rbm.sample_v_given_v_clamped(
        clamped_v=homogeneous_batch,
        n_clamped=n_clamped,
        gibbs_steps=rbm_gibbs_steps,
        beta=1.0 
    )
    full_h_rbm, _ = rbm._sample_h_given_v(full_v_rbm, beta=1.0)
    
    with torch.no_grad():
        e_rbm_batch = joint_energy(rbm, full_v_rbm, full_h_rbm)
        mean_rbm_energy = e_rbm_batch.mean().item()
        
    print(f"Target (RBM) Mean Energy: {mean_rbm_energy:.4f}")

    # --- 2. Optimization Loop ---
    for epoch in range(num_epochs):
        
        # A. Sample Batch from QPU (Fast Mode)
        # This will use binary_patterns_batch[0] to configure QPU, 
        # then return 'batch_size' samples.
        dwave_v, dwave_h = sample_flux_conditioned_fast(
            rbm,
            qpu_sampler,
            embedding,
            binary_patterns_batch, # Function internally takes row 0
            logical_idx_map,
            beta=beta,
            clamp_strength_h=25.0 
        )
        
        # B. Compute Energy
        with torch.no_grad():
            e_dwave_batch = joint_energy(rbm, dwave_v, dwave_h)
            mean_dwave_energy = e_dwave_batch.mean().item()
            var_dwave_energy = torch.var(e_dwave_batch).item()

        # C. Update Beta
        if adaptive:
            safe_var = max(1e-6, var_dwave_energy)
            calc_lr = (beta**2) / safe_var
            current_lr = min(lr_cap, max(lr, calc_lr))
        else:
            current_lr = lr

        diff = mean_dwave_energy - mean_rbm_energy
        beta -= current_lr * diff
        beta = max(1e-2, beta)

        beta_history.append(beta)
        rbm_energy_history.append(mean_rbm_energy)
        qpu_energy_history.append(mean_dwave_energy)
        
        print(f"Epoch {epoch}: Beta={beta:.4f} | QPU_E={mean_dwave_energy:.2f} vs RBM_E={mean_rbm_energy:.2f} | Diff={diff:.2f}")
        
        if abs(diff) < 0.5: 
            print("Converged.")
            break
            
    return beta, beta_history, rbm_energy_history, qpu_energy_history





def get_physical_connectivity(source_qubits, target_qubits, adjacency):
    """
    Finds all active physical couplers (edges) connecting two sets of qubits.
    Returns a list of tuples [(u, v), ...].
    """
    connections = []
    # Convert target to set for O(1) lookup
    target_set = set(target_qubits)
    
    for u in source_qubits:
        if u in adjacency:
            for v in adjacency[u]:
                if v in target_set:
                    # Sort to ensure (u, v) is consistent (e.g., always smaller first)
                    edge = tuple(sorted((u, v)))
                    connections.append(edge)
    
    # Remove duplicates
    return list(set(connections))

def distribute_bias(logical_h, physical_qubits):
    """
    Spreads a logical bias h evenly over N physical qubits.
    h_i = h_logical / N
    """
    if not physical_qubits:
        return {}
    
    val = logical_h / len(physical_qubits)
    return {q: val for q in physical_qubits}

def distribute_coupling(logical_J, physical_edges):
    """
    Spreads a logical coupling J evenly over M physical edges.
    J_ij = J_logical / M
    """
    if not physical_edges:
        return {}
        
    val = logical_J / len(physical_edges)
    return {edge: val for edge in physical_edges}

def get_chain_couplings(chain, strength):
    """
    Generates the ferromagnetic couplings for a SINGLE physical chain.
    Assumes the chain is connected in the hardware graph.
    """
    couplings = {}
    if len(chain) < 2:
        return couplings
    
    # We assume the chain provided by Zephyr/Minorminer is implicitly connected.
    # We connect adjacent elements in the list or use a full path.
    # For safety with D-Wave chains, we usually couple q[i] to q[i+1].
    for i in range(len(chain) - 1):
        u, v = chain[i], chain[i+1]
        edge = tuple(sorted((u, v)))
        couplings[edge] = -strength
        
    return couplings


def build_logical_to_physical_map(
    num_visible, 
    num_hidden, 
    conditioning_sets, 
    left_chains, 
    right_chains
):
    """
    Constructs a dictionary mapping: 
       RBM_Logical_Node_ID (int) -> List_of_Physical_Qubits (list)
    
    Respects the segmentation: [Conditioning] -> [Remaining Vis] -> [Hidden]
    """
    embedding_map = {}
    
    # --- 1. Conditioning Nodes (0 to n_cond - 1) ---
    # These correspond to the first len(conditioning_sets) visible nodes
    for i, q_set in enumerate(conditioning_sets):
        embedding_map[i] = list(q_set)
        
    n_cond = len(conditioning_sets)
    
    # --- 2. Remaining Visible Nodes (n_cond to num_vis - 1) ---
    # We assume 'left_chains' keys are sorted and map sequentially 
    # to the remaining visible slots.
    sorted_left_keys = sorted(left_chains.keys())
    
    # Safety check
    expected_rem_vis = num_visible - n_cond
    if len(sorted_left_keys) < expected_rem_vis:
        print(f"Warning: Not enough Left Chains ({len(sorted_left_keys)}) for remaining visible nodes ({expected_rem_vis})!")
    
    for k, key in enumerate(sorted_left_keys):
        if k >= expected_rem_vis: break # Stop if we have extra chains
        
        logical_id = n_cond + k
        embedding_map[logical_id] = list(left_chains[key])

    # --- 3. Hidden Nodes (num_vis to num_vis + num_hidden - 1) ---
    # RBM logical indices for hidden nodes start at num_visible.
    sorted_right_keys = sorted(right_chains.keys())
    
    for k, key in enumerate(sorted_right_keys):
        if k >= num_hidden: break
        
        logical_id = num_visible + k
        embedding_map[logical_id] = list(right_chains[key])
        
    return embedding_map

def build_expanded_embedding(conditioning_sets, left_chains, right_chains, num_visible):
    """
    Creates an embedding where Conditioning Nodes are 'exploded' into 
    individual fragment nodes (one per physical qubit).
    
    Returns:
        expanded_embedding: Dict mapping Expanded_ID -> [Physical_Qubit]
        fragment_map: Dict mapping Original_Logical_ID -> List of Expanded_IDs
    """
    expanded_embedding = {}
    
    # Track which new IDs belong to which original conditioning node
    # logical_id (int) -> list of fragment_ids (str)
    fragment_map = {} 
    
    # 1. Handle Conditioning Nodes (Indices 0 to len(sets)-1)
    for logical_id, phys_set in enumerate(conditioning_sets):
        fragments = []
        for phys_q in phys_set:
            # Create a unique ID for this fragment. 
            # Format: "C{logical_id}_{phys_qubit}"
            frag_id = f"C{logical_id}_{phys_q}"
            
            # The embedding for this fragment is just the single qubit
            expanded_embedding[frag_id] = [phys_q]
            fragments.append(frag_id)
            
        fragment_map[logical_id] = fragments

    # 2. Handle Standard Visible Nodes (Left Chains)
    # They start after the conditioning nodes
    n_cond = len(conditioning_sets)
    sorted_left = sorted(left_chains.keys())
    
    for k, key in enumerate(sorted_left):
        if k >= (num_visible - n_cond): break
        
        logical_id = n_cond + k
        # These keep their integer ID
        expanded_embedding[logical_id] = list(left_chains[key])

    # 3. Handle Hidden Nodes (Right Chains)
    # They start after all visible nodes
    sorted_right = sorted(right_chains.keys())
    
    for k, key in enumerate(sorted_right):
        # If RBM has 100 vis and 50 hid, hidden indices are 100..149
        
        global_id = num_visible + k
        expanded_embedding[global_id] = list(right_chains[key])

    return expanded_embedding, fragment_map


def rbm_to_expanded_ising(rbm, fragment_map, beta=1.0):
    """
    Converts RBM weights to the Expanded Graph format.
    Splits weights for conditioning nodes across their fragments.
    """
    # 1. Get Standard Logical Ising
    # (Reuse your existing function)
    base_h, base_J, n_vis, n_hid = rbm_to_logical_ising(rbm, beta=beta)
    
    expanded_h = {}
    expanded_J = {}
    
    # 2. Process Biases (h)
    for node, val in base_h.items():
        if node in fragment_map:
            expanded_h = 0.0
            
    # 3. Process Couplings (J)
    for (u, v), val in base_J.items():
        # Check if u or v are conditioning nodes
        u_is_cond = u in fragment_map
        v_is_cond = v in fragment_map
        
        if not u_is_cond and not v_is_cond:
            # Standard Edge: Keep as is
            expanded_J[(u, v)] = val
            
        elif u_is_cond and not v_is_cond:
            # U is conditioning (Source). Fan out to V.
            fragments = fragment_map[u]
            split_val = val / len(fragments)
            for frag_id in fragments:
                expanded_J[(frag_id, v)] = split_val
                
        elif not u_is_cond and v_is_cond:
            # V is conditioning (Target). Fan out from U.
            fragments = fragment_map[v]
            split_val = val / len(fragments)
            for frag_id in fragments:
                expanded_J[(u, frag_id)] = split_val
                
        # (Case where both are conditioning nodes doesn't exist in RBMs/Bipartite graphs)

    return expanded_h, expanded_J


def get_physical_flux_biases_manual(
    embedding_map: dict,
    total_physical_qubits: int,
    logical_clamps: dict,
    clamp_strength_h: float
) -> list:
    """
    Generates the physical flux bias list using the Manual Map.
    Ensures list is exactly 'total_physical_qubits' long.
    """
    # Initialize with 0.0 for every qubit on the chip
    physical_flux_biases = [0.0] * total_physical_qubits
    

    for logical_node, direction in logical_clamps.items():
        if logical_node not in embedding_map:
            continue
            
        # Get physical qubits
        chain = embedding_map[logical_node]
        
        # Calculate flux
        h_val = direction * clamp_strength_h
        fb_val = h_to_fluxbias(h_val)
        
        # Apply to all physical qubits in this chain/set
        for phys_q in chain:
            if phys_q < total_physical_qubits:
                physical_flux_biases[phys_q] += fb_val
                
    return physical_flux_biases



def sample_manual_ising(
    raw_sampler,
    h_phys: dict,
    J_phys: dict,
    flux_biases: list,
    num_reads: int = 100,
):
    """
    Submits a manually embedded Ising problem to the Raw QPU Sampler.
    
    CRITICAL FEATURES:
    1. Detects physical hardware limits (h_range, j_range).
    2. Clamps manual h and J values to these limits to prevent API errors.
    3. Handles the specific kwargs for flux biasing (auto_scale=False, etc).
    """
    # --- 1. Get Hardware Properties ---
    # We look for 'extended_j_range' first, as it allows wider values for chains.
    # Fallback to 'j_range' if extended is missing.
    props = raw_sampler.properties
    h_min, h_max = props['h_range']
    j_min, j_max = props['extended_j_range']

    h_clamped = {k: max(h_min, min(v, h_max)) for k, v in h_phys.items()}
    J_clamped = {k: max(j_min, min(v, j_max)) for k, v in J_phys.items()}
    
    # Note: No 'chain_strength' argument here. The chains are already in J_clamped.
    response = raw_sampler.sample_ising(
        h_clamped,
        J_clamped,
        flux_biases=flux_biases,
        flux_drift_compensation=False,
        num_reads=num_reads,
        answer_mode='raw',
        auto_scale=False
    )
    
    return response

def calculate_manual_chain_breaks(response, embedding_map):
    """
    Manually checks if physical qubits in a chain agree with each other.
    Fixed to handle dense packed sample matrices.
    """
    # 1. Map Qubit Label -> Column Index
    # response.variables contains the qubit IDs in the order they appear in the matrix
    label_to_idx = {label: i for i, label in enumerate(response.variables)}
    
    samples = response.record.sample
    num_samples = samples.shape[0]
    total_chains = len(embedding_map)
    total_broken_chains = 0
    
    for logical_id, chain in embedding_map.items():
        if len(chain) < 2:
            continue
            
        # 2. Convert physical qubit IDs to matrix column indices
        # We filter out qubits that might not be in the response (inactive/disconnected)
        chain_col_indices = [label_to_idx[q] for q in chain if q in label_to_idx]
        
        if len(chain_col_indices) < 2:
            # If a chain has < 2 active qubits reported, it can't be broken
            continue
            
        # 3. Extract using column indices
        chain_spins = samples[:, chain_col_indices]
        
        # Check consistency (min == max means all values are same)
        row_mins = np.min(chain_spins, axis=1)
        row_maxs = np.max(chain_spins, axis=1)
        
        broken_mask = (row_mins != row_maxs)
        total_broken_chains += np.sum(broken_mask)

    avg_broken = total_broken_chains / (num_samples * total_chains)
    return avg_broken

def unembed_raw_samples(response, embedding_map, num_visible, num_hidden, device):
    """
    Converts Raw Physical Samples -> Logical RBM Tensors via Majority Vote.
    Fixed to handle dense packed sample matrices.
    """
    # 1. Map Qubit Label -> Column Index
    label_to_idx = {label: i for i, label in enumerate(response.variables)}
    
    phys_samples_np = response.record.sample 
    num_samples = phys_samples_np.shape[0]
    total_logical = num_visible + num_hidden
    
    logical_samples = np.zeros((num_samples, total_logical), dtype=np.float32)
    
    for logical_id in range(total_logical):
        if logical_id not in embedding_map:
            continue
            
        chain = embedding_map[logical_id]
        
        # 2. Get column indices for this chain
        chain_indices = [label_to_idx[q] for q in chain if q in label_to_idx]
        
        if not chain_indices:
            continue
            
        # 3. Extract columns
        chain_vals = phys_samples_np[:, chain_indices] 
        
        # 4. Majority Vote
        chain_sum = np.sum(chain_vals, axis=1)
        logical_samples[:, logical_id] = np.where(chain_sum > 0, 1.0, 0.0)
        
    t_logical = torch.tensor(logical_samples, device=device, dtype=torch.float32)
    v_s = t_logical[:, :num_visible]
    h_s = t_logical[:, num_visible:]
    
    return v_s, h_s



def process_expanded_rbm_samples(
    response, 
    num_visible: int,
    num_hidden: int,
    conditioning_sets: list, # We need this to reconstruct the string labels
    device: torch.device
):
    total_nodes = num_visible + num_hidden
    num_samples = len(response.record.sample)
    num_cond = len(conditioning_sets)

    # 1. Map Response Labels to Column Indices
    var_to_col_idx = {var: i for i, var in enumerate(response.variables)}
    
    ordered_cols = []
    
    # 2. Build the Column List Order [0, 1, ..., Total-1]
    for i in range(total_nodes):
        target_label = None
        
        if i < num_cond:
            # --- CASE A: Conditioning Node ---
            # The composite renamed "0" to multiple fragments like "C0_12", "C0_45".
            # Since they are flux-clamped to be identical, we just grab the FIRST one.
            # We must reconstruct the label string exactly as we defined it in the embedding.
            
            # Grab first physical qubit in the set
            # (Note: sets are unordered, so we sort or convert to list to match build logic)
            phys_q = sorted(list(conditioning_sets[i]))[0]
            target_label = f"C{i}_{phys_q}"
            
        else:
            # --- CASE B: Standard Node ---
            # FixedEmbeddingComposite automatically handled the chains for these.
            # They appear as standard integers in the response.
            target_label = i
            
        # 3. Find the column
        if target_label in var_to_col_idx:
            ordered_cols.append(var_to_col_idx[target_label])
        else:
            print(f"Error: Missing label '{target_label}' in D-Wave response.")
            # Fallback: append 0 (or handle gracefully)
            ordered_cols.append(0)

    # --- 4. Fast Numpy Slicing (Same as your old code) ---
    # We now have the columns in the exact order [Logical_0 ... Logical_End]
    dwave_spin_samples = response.record.sample[:, ordered_cols]
    
    dwave_spin_samples_t = torch.tensor(dwave_spin_samples, dtype=torch.float, device=device)

    # --- 5. Convert -1/+1 to 0/1 ---
    _ZERO = torch.tensor(0., dtype=torch.float, device=device)
    _MINUS_ONE = torch.tensor(-1., dtype=torch.float, device=device)
    
    dwave_binary_samples = torch.where(
        dwave_spin_samples_t == _MINUS_ONE,
        _ZERO,
        dwave_spin_samples_t
    )

    return dwave_binary_samples[:, :num_visible], dwave_binary_samples[:, num_visible:]


from utils.FluxBiases import h_to_fluxbias

def get_expanded_flux_biases(
    logical_clamps: dict,     # {0: +1, 1: -1, ...}
    fragment_map: dict,       # {0: ["C0_12", "C0_45"], ...}
    expanded_embedding: dict, # {"C0_12": [12], "C0_45": [45], ...}
    total_physical_qubits: int,
    clamp_strength_h: float = 20.0
) -> list:
    """
    Generates physical flux biases for the Expanded Graph approach.
    Traverses: Logical Node -> Fragment Keys -> Physical Qubits.
    """
    # 1. Initialize empty flux list
    flux_list = [0.0] * total_physical_qubits
    
    # 2. Iterate only over the nodes we intend to clamp
    for logical_id, spin in logical_clamps.items():
        
        # If this logical node isn't in our fragment map (e.g. it's a standard node),
        # we can't flux bias it this way. Skip.
        if logical_id not in fragment_map:
            continue
            
        # 3. Calculate the flux value once
        h_val = spin * clamp_strength_h
        fb_val = h_to_fluxbias(h_val)
        
        # 4. Fan-out to all fragments
        # fragment_map[0] -> ["C0_12", "C0_45", ...]
        fragments = fragment_map[logical_id]
        
        for frag_key in fragments:
            # Look up the physical qubit(s) for this fragment
            # usually expanded_embedding["C0_12"] -> [12]
            if frag_key in expanded_embedding:
                phys_qubits = expanded_embedding[frag_key]
                
                for q in phys_qubits:
                    if q < total_physical_qubits:
                        flux_list[q] += fb_val
                        
    return flux_list


def sample_expanded_flux_conditioned(
    rbm,
    raw_sampler,
    conditioning_sets,
    left_chains,
    right_chains,
    binary_patterns_batch,
    beta=1.0,
    chain_strength=None, # Passed to Composite now!
    clamp_strength_h=20.0
):
    batch_size = binary_patterns_batch.shape[0]
    total_qubits = raw_sampler.properties['num_qubits']

    # --- 1. Build Expanded Embedding ---
    rbm_num_visible = rbm.parameters["vbias"].shape[0]
    rbm_num_hidden = rbm.parameters["hbias"].shape[0]
    exp_embedding, fragment_map = build_expanded_embedding(
        conditioning_sets, left_chains, right_chains, rbm_num_visible
    )
    
    # --- 2. Initialize Composite Sampler ---
    composite_sampler = FixedEmbeddingComposite(raw_sampler, exp_embedding)
    
    # --- 3. Build Expanded Weights ---
    h_exp, J_exp = rbm_to_expanded_ising(rbm, fragment_map, beta)
    
    # --- 4. Setup Flux Biases  ---
    row_np = binary_patterns_batch[0].detach().cpu().numpy()
    
    # Build the logical intent: "I want Node 0 to be +1, Node 1 to be -1"
    logical_clamps = {}
    for i, bit_val in enumerate(row_np):
        if i >= len(conditioning_sets): 
            break
        logical_clamps[i] = 1 if bit_val > 0.5 else -1

    # Generate the physical list using the new cleaner helper
    raw_fb_list = get_expanded_flux_biases(
        logical_clamps=logical_clamps,
        fragment_map=fragment_map,
        expanded_embedding=exp_embedding,
        total_physical_qubits=total_qubits,
        clamp_strength_h=clamp_strength_h
    )    
    # Safety Clip
    SAFE_FLUX_CAP = 0.01
    safe_fb_list = [max(-SAFE_FLUX_CAP, min(v, SAFE_FLUX_CAP)) for v in raw_fb_list]

    # --- 5. Submit ---
    print(f"Submitting Expanded Graph (n={batch_size})...")
    
    # We can finally use the high-level API safely
    response = composite_sampler.sample_ising(
        h_exp, 
        J_exp,
        flux_biases=safe_fb_list,
        flux_drift_compensation=False,
        num_reads=batch_size,
        chain_strength=chain_strength, # Now handled by D-Wave!
        auto_scale=False, # Still False because we manually scaled H/J for fragments
        return_embedding=True # Useful for debug
    )
    
    # --- 6. Process Results ---
    # The response comes back with "Expanded" labels (Strings and Ints).
    # We need to coalesce them back to RBM Logical Tensors.
    
    v_s, h_s = process_expanded_rbm_samples(
        response,
        num_visible=rbm_num_visible,
        num_hidden=rbm_num_hidden,
        conditioning_sets=conditioning_sets,
        device=rbm.device
    )
    
    return v_s, h_s

def sample_manual_flux_conditioned_fast(
    rbm,
    raw_sampler,
    conditioning_sets: list,
    left_chains: dict,
    right_chains: dict,
    binary_patterns_batch: torch.Tensor,
    beta: float = 1.0,
    chain_strength: float = 3.0,
    clamp_strength_h: float = 20.0
):
    """
    Fast manual sampling with implicit identity mapping.
    Assumes binary_patterns_batch column [i] corresponds to Logical Node [i].
    """
    batch_size = binary_patterns_batch.shape[0]
    total_qubits = raw_sampler.properties['num_qubits']
    
    # --- 1. Build Map & Static Graph ---
    # We still need the full map to unembed the results later
    rbm_num_visible = rbm.params["vbias"].shape[0]
    rbm_num_hidden = rbm.params["hbias"].shape[0]
    emb_map = build_logical_to_physical_map(
        rbm_num_visible, rbm_num_hidden,
        conditioning_sets, left_chains, right_chains
    )
    
    h_phys, J_phys = build_manual_embedded_ising(
        rbm, raw_sampler, conditioning_sets, 
        left_chains, right_chains, beta, chain_strength
    )

    # --- 2. Setup Pattern (First Row Only) ---
    # We take the first pattern to configure the QPU
    row_np = binary_patterns_batch[0].detach().cpu().numpy()
    
    # Validation: Ensure we don't have more data columns than conditioning nodes
    if len(row_np) > len(conditioning_sets):
        print(f"Warning: Batch has {len(row_np)} columns but only {len(conditioning_sets)} conditioning sets.")
        print("Ignoring extra columns.")

    logical_clamps = {}
    
    # SIMPLIFIED LOOP: Identity Mapping
    # Column 0 -> Node 0, Column 1 -> Node 1, etc.
    for i, bit_val in enumerate(row_np):
        if i >= len(conditioning_sets): 
            break
            
        spin = 1 if bit_val > 0.5 else -1
        logical_clamps[i] = spin
        
        # Zero out logical h in the physical graph so it doesn't fight the flux
        if i in emb_map:
            for q in emb_map[i]:
                h_phys[q] = 0.0

    # --- 3. Calculate Flux Biases ---
    raw_fb_list = get_physical_flux_biases_manual(
        embedding_map=emb_map,
        total_physical_qubits=total_qubits,
        logical_clamps=logical_clamps,
        clamp_strength_h=clamp_strength_h 
    )
    
    # Safety Clip
    SAFE_FLUX_CAP = 0.01
    safe_fb_list = [
        max(-SAFE_FLUX_CAP, min(val, SAFE_FLUX_CAP)) 
        for val in raw_fb_list
    ]

    # --- 4. Submit ---
    print(f"Submitting fast batch (n={batch_size}) for Beta={beta:.2f}...")
    
    response = sample_manual_ising(
        raw_sampler=raw_sampler,
        h_phys=h_phys,
        J_phys=J_phys,
        flux_biases=safe_fb_list,
        num_reads=batch_size,   
    )    

    # --- 5. Manual Chain Break Check ---
    break_frac = calculate_manual_chain_breaks(response, emb_map)
    print(f"  -> Manual Chain Break Fraction: {break_frac:.4f}")

    # --- 6. Unembed ---
    v_s, h_s = unembed_raw_samples(
        response, 
        emb_map, 
        rbm_num_visible, 
        rbm_num_hidden, 
        rbm.device
    )
    
    return v_s, h_s