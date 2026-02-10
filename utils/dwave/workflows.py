import torch
import numpy as np
import copy
from dwave.system.composites import FixedEmbeddingComposite
from datetime import datetime
import os
from typing import Tuple, List, Optional, Union

# --- Explicit Local Imports ---
from .physics import (
    rbm_to_logical_ising,
    joint_energy,
    rbm_to_expanded_ising,
    convert_energy_to_binary,
    calculate_rms_chain_strength,
    get_latent_correlation,
)
from .graphs import (
    get_physical_flux_biases,
    build_logical_to_physical_map,
    build_manual_embedded_ising,
    get_physical_flux_biases_manual,
    build_expanded_embedding,
    get_expanded_flux_biases,
    build_expanded_embedding_rotation,
    build_expanded_embedding_arbitrary,
    get_orbit_mappings,
)
from .sampling_backend import (
    sample_logical_ising,
    sample_ising_flux_bias,
    sample_physical_with_analysis,
    sample_physical_with_analysis_srt,
    sample_manual_ising,
    sample_physical_arbitrary
    )
from .postprocessing import (
    process_rbm_samples,
    calculate_manual_chain_breaks,
    unembed_raw_samples,
    process_expanded_rbm_samples,
    process_analysis_result
)

from .plots import plot_energy_comparison

# --- A. Basic RBM Sampling ---

def sample_rbm_qpu(
    rbm, 
    qpu_sampler, 
    beta=1.0, 
    num_samples=64, 
    measure_time=False, 
    chain_strength=None
):
    h, J, n_vis, n_hid = rbm_to_logical_ising(rbm, beta=beta)
    
    response, s_time = sample_logical_ising(
        qpu_sampler, h, J, num_samples, measure_time, chain_strength
    )
    
    v_s, h_s = process_rbm_samples(response, n_vis, n_hid, rbm.device)
    return v_s, h_s, s_time


def find_beta(
    rbm, 
    qpu_sampler, 
    num_reads=128, 
    beta_init=1.0, 
    lr=0.1, 
    num_epochs=15, 
    adaptive=False
):
    beta = beta_init
    beta_hist, rbm_e_hist, qpu_e_hist = [], [], []
    
    # RBM Baseline
    rbm.sample_state(beta=1.0)
    with torch.no_grad():
        base_v = rbm.chains["v"][:num_reads]
        base_h = rbm.chains["h"][:num_reads]
        mean_rbm_energy = joint_energy(rbm, base_v, base_h).mean().item()
    
    print(f"Target (RBM) Mean Energy: {mean_rbm_energy:.4f}")

    for epoch in range(num_epochs):
        dwave_v, dwave_h, _ = sample_rbm_qpu(
            rbm, qpu_sampler, beta=beta, num_samples=num_reads
        )
        
        with torch.no_grad():
            e_dwave = joint_energy(rbm, dwave_v, dwave_h)
            mean_dwave = e_dwave.mean().item()
            var_dwave = torch.var(e_dwave).item()
            
        # Update Beta
        if adaptive:
            safe_var = max(1e-6, var_dwave)
            calc_lr = (beta**2) / safe_var
            current_lr = min(0.5, max(lr, calc_lr))
        else:
            current_lr = lr

        diff = mean_dwave - mean_rbm_energy
        beta = max(1e-2, beta - current_lr * diff)
        
        beta_hist.append(beta)
        rbm_e_hist.append(mean_rbm_energy)
        qpu_e_hist.append(mean_dwave)
        
        print(f"Epoch {epoch}: Beta={beta:.4f} | Diff={diff:.2f}")
        if abs(diff) < 0.5: 
            break
            
    return beta, beta_hist, rbm_e_hist, qpu_e_hist


# --- B. Basic Flux Conditioning ---

def sample_flux_conditioned_batch(
    rbm, 
    qpu_sampler, 
    embedding, 
    binary_patterns_batch, 
    logical_idx_map, 
    beta=1.0, 
    clamp_strength_h=20.0, 
    chain_strength=None
):
    base_h, base_J, n_vis, n_hid = rbm_to_logical_ising(rbm, beta=beta)
    total_qubits = qpu_sampler.child.properties['num_qubits']
    visible_batch_list, hidden_batch_list = [], []
    
    for i, row in enumerate(binary_patterns_batch):
        row_np = row.detach().cpu().numpy()
        logical_clamps = {}
        current_h = copy.deepcopy(base_h)
        
        for bit_idx, bit_val in enumerate(row_np):
            target_node = logical_idx_map[bit_idx]
            logical_clamps[target_node] = 1 if bit_val > 0.5 else -1
            if target_node in current_h: 
                current_h[target_node] = 0.0

        raw_fb = get_physical_flux_biases(
            embedding, total_qubits, logical_clamps, clamp_strength_h
        )
        safe_fb = [max(-0.01, min(v, 0.01)) for v in raw_fb]
        
        response, _ = sample_ising_flux_bias(
            qpu_sampler, current_h, base_J, safe_fb, 
            num_samples=1, chain_strength=chain_strength
        )
        
        v_s, h_s = process_rbm_samples(response, n_vis, n_hid, rbm.device)
        visible_batch_list.append(v_s)
        hidden_batch_list.append(h_s)

    return torch.cat(visible_batch_list, dim=0), torch.cat(hidden_batch_list, dim=0)


def sample_flux_conditioned_fast(
    rbm, 
    qpu_sampler, 
    embedding, 
    binary_patterns_batch, 
    logical_idx_map, 
    beta=1.0, 
    clamp_strength_h=20.0, 
    chain_strength=None
):
    batch_size = binary_patterns_batch.shape[0]
    base_h, base_J, n_vis, n_hid = rbm_to_logical_ising(rbm, beta=beta)
    total_qubits = qpu_sampler.child.properties['num_qubits']

    # Configure for the FIRST pattern only
    row_np = binary_patterns_batch[0].detach().cpu().numpy()
    logical_clamps = {}
    current_h = base_h.copy() 
    
    for bit_idx, bit_val in enumerate(row_np):
        target_node = logical_idx_map[bit_idx]
        logical_clamps[target_node] = 1 if bit_val > 0.5 else -1
        if target_node in current_h: 
            current_h[target_node] = 0.0

    raw_fb = get_physical_flux_biases(
        embedding, total_qubits, logical_clamps, clamp_strength_h
    )
    safe_fb = [max(-0.01, min(v, 0.01)) for v in raw_fb]
    
    response, _ = sample_ising_flux_bias(
        qpu_sampler, current_h, base_J, safe_fb, 
        num_samples=batch_size, chain_strength=chain_strength
    )
    
    v_s, h_s = process_rbm_samples(response, n_vis, n_hid, rbm.device)
    return v_s, h_s


def find_beta_flux_bias(
    rbm, 
    qpu_sampler, 
    embedding, 
    binary_patterns_batch, 
    logical_idx_map, 
    rbm_gibbs_steps=1000, 
    beta_init=1.0, 
    lr=0.1, 
    num_epochs=15, 
    adaptive=False
):
    beta = beta_init
    beta_hist, rbm_e_hist, qpu_e_hist = [], [], []
    n_clamped = binary_patterns_batch.shape[1]
    
    # Calculate RBM Conditional Energy (Homogeneous)
    primary_pattern = binary_patterns_batch[0].unsqueeze(0)
    homogeneous_batch = primary_pattern.repeat(binary_patterns_batch.shape[0], 1)
    
    full_v_rbm = rbm.sample_v_given_v_clamped(
        clamped_v=homogeneous_batch, 
        n_clamped=n_clamped, 
        gibbs_steps=rbm_gibbs_steps, 
        beta=1.0
    )
    full_h_rbm, _ = rbm._sample_h_given_v(full_v_rbm, beta=1.0)
    
    with torch.no_grad(): 
        mean_rbm_energy = joint_energy(rbm, full_v_rbm, full_h_rbm).mean().item()

    for epoch in range(num_epochs):
        dwave_v, dwave_h = sample_flux_conditioned_fast(
            rbm, qpu_sampler, embedding, binary_patterns_batch, 
            logical_idx_map, beta=beta, clamp_strength_h=25.0
        )
        
        with torch.no_grad():
            e_dwave = joint_energy(rbm, dwave_v, dwave_h)
            mean_dwave = e_dwave.mean().item()
            var_dwave = torch.var(e_dwave).item()

        if adaptive:
            safe_var = max(1e-6, var_dwave)
            current_lr = min(0.5, max(lr, (beta**2)/safe_var))
        else:
            current_lr = lr
            
        diff = mean_dwave - mean_rbm_energy
        beta = max(1e-2, beta - current_lr * diff)
        
        beta_hist.append(beta)
        rbm_e_hist.append(mean_rbm_energy)
        qpu_e_hist.append(mean_dwave)
        
        print(f"Epoch {epoch}: Beta={beta:.4f} | Diff={diff:.2f}")
        if abs(diff) < 0.5: 
            break
            
    return beta, beta_hist, rbm_e_hist, qpu_e_hist


# --- C. Manual Flux Conditioning ---

def sample_manual_flux_conditioned_fast(
    rbm, 
    raw_sampler, 
    conditioning_sets, 
    left_chains, 
    right_chains, 
    binary_patterns_batch, 
    beta=1.0, 
    chain_strength=3.0, 
    clamp_strength_h=20.0
):
    batch_size = binary_patterns_batch.shape[0]
    total_qubits = raw_sampler.properties['num_qubits']
    
    # 1. Math & Graph
    h_log, J_log, n_vis, n_hid = rbm_to_logical_ising(rbm, beta=beta)
    
    emb_map = build_logical_to_physical_map(
        n_vis, n_hid, conditioning_sets, left_chains, right_chains
    )
    h_phys, J_phys = build_manual_embedded_ising(
        rbm, raw_sampler, conditioning_sets, left_chains, right_chains, 
        beta, chain_strength, h_log, J_log
    )

    # 2. Configure Flux (First Pattern)
    row_np = binary_patterns_batch[0].detach().cpu().numpy()
    logical_clamps = {}
    
    for i, bit_val in enumerate(row_np):
        if i >= len(conditioning_sets): break
        logical_clamps[i] = 1 if bit_val > 0.5 else -1
        # Zero out logical h on graph
        if i in emb_map:
            for q in emb_map[i]: 
                h_phys[q] = 0.0

    raw_fb = get_physical_flux_biases_manual(
        emb_map, total_qubits, logical_clamps, clamp_strength_h
    )
    safe_fb = [max(-0.01, min(val, 0.01)) for val in raw_fb]

    # 3. Submit
    response = sample_manual_ising(
        raw_sampler, h_phys, J_phys, safe_fb, num_reads=batch_size
    )
    
    break_frac = calculate_manual_chain_breaks(response, emb_map)
    print(f"  -> Manual Chain Break Fraction: {break_frac:.4f}")

    v_s, h_s = unembed_raw_samples(response, emb_map, n_vis, n_hid, rbm.device)
    return v_s, h_s


# --- D. Expanded Flux Conditioning ---

def sample_expanded_flux_conditioned(
    rbm, 
    raw_sampler, 
    conditioning_sets, 
    left_chains, 
    right_chains, 
    binary_patterns_batch, 
    beta=1.0, 
    chain_strength=None, 
    clamp_strength_h=50.0,
    fast_sampling=True
):
    """
    Samples from the QPU using flux biases derived from binary_patterns_batch.
    
    Args:
        fast_sampling (bool): 
            If True, calculates flux bias from batch[0] and repeats for num_reads=batch_size.
            If False, iterates through batch row-by-row, updating flux bias for every sample.
    """
    batch_size = binary_patterns_batch.shape[0]
    total_qubits = raw_sampler.properties['num_qubits']
    n_vis = rbm.params["vbias"].shape[0]
    n_hid = rbm.params["hbias"].shape[0]

    # 1. Graph Construction (Shared for both modes)
    # Topology and Ising weights depend on Beta/Weights, not the specific row data.
    exp_embedding, fragment_map = build_expanded_embedding(
        conditioning_sets, left_chains, right_chains, n_vis
    )
    
    composite_sampler = FixedEmbeddingComposite(raw_sampler, exp_embedding)
    h_exp, J_exp = rbm_to_expanded_ising(
            rbm, 
            fragment_map, 
            exp_embedding, 
            raw_sampler.adjacency, 
            beta
        )    

    # 2. Sampling Execution
    if fast_sampling:
        # --- FAST MODE: Homogeneous/Repeated Sampling ---
        # Takes the first row, assumes the rest are identical (or we want to repeat the first)
        row_np = binary_patterns_batch[0].detach().cpu().numpy()
        logical_clamps = {}
        for i, bit_val in enumerate(row_np):
            if i >= len(conditioning_sets): break
            logical_clamps[i] = 1 if bit_val > 0.5 else -1

        raw_fb = get_expanded_flux_biases(
            logical_clamps, fragment_map, exp_embedding, total_qubits, clamp_strength_h
        )
        safe_fb = [max(-0.01, min(v, 0.01)) for v in raw_fb]

        print(f"Submitting Fast Batch (n={batch_size})...")
        response, _ = sample_ising_flux_bias(
            composite_sampler, h_exp, J_exp, safe_fb, 
            num_samples=batch_size, chain_strength=chain_strength
        )
        v_s, h_s = process_expanded_rbm_samples(
            response, n_vis, n_hid, conditioning_sets, rbm.device
        )
        return v_s, h_s

    else:
        # --- SLOW MODE: Heterogeneous/Iterative Sampling ---
        # Iterates row by row, recalculating flux for every specific pattern
        qpu_v_list = []
        qpu_h_list = []
        
        print(f"Submitting Slow/Heterogeneous Batch (n={batch_size})...")
        
        # We suppress inner print breaks to avoid spamming console
        for i in range(batch_size):
            row_np = binary_patterns_batch[i].detach().cpu().numpy()
            
            # Calculate Logical Clamps for this specific row
            logical_clamps = {}
            for k_idx, bit_val in enumerate(row_np):
                if k_idx >= len(conditioning_sets): break
                logical_clamps[k_idx] = 1 if bit_val > 0.5 else -1

            # Calculate Physical Flux Biases
            raw_fb = get_expanded_flux_biases(
                logical_clamps, fragment_map, exp_embedding, total_qubits, clamp_strength_h
            )
            safe_fb = [max(-0.01, min(v, 0.01)) for v in raw_fb]

            # Submit single sample
            response, _ = sample_ising_flux_bias(
                composite_sampler, h_exp, J_exp, safe_fb, 
                num_samples=1, chain_strength=chain_strength, print_breaks=False
            )
            
            # Decode
            v_single, h_single = process_expanded_rbm_samples(
                response, n_vis, n_hid, conditioning_sets, rbm.device
            )
            qpu_v_list.append(v_single)
            qpu_h_list.append(h_single)

        return torch.cat(qpu_v_list, dim=0), torch.cat(qpu_h_list, dim=0)


def sample_expanded_flux_conditioned_rigorous(
    rbm, 
    raw_sampler, 
    conditioning_sets, 
    left_chains, 
    right_chains, 
    binary_patterns_batch, 
    hidden_side='right',
    beta=1.0, 
    chain_strength=1.0, 
    rho = 28.0,
    clamp_strength_h=50.0,
    source=None,
    save_dir="/home/leozhu/CaloQuVAE/wandb-outputs/dwave_misc/"
):
    """
    Samples from the D-Wave QPU using an expanded embedding where conditioning sets 
    are attached to the Hidden Layer.
    
    Args:
        hidden_side (str): 'left' or 'right'. The side that contains the conditioning 
                           sets and acts as the Hidden layer.
    """
    batch_size = binary_patterns_batch.shape[0]
    total_qubits = raw_sampler.properties['num_qubits']
    n_vis = rbm.params["vbias"].shape[0]
    n_hid = rbm.params["hbias"].shape[0]
    n_cond = len(conditioning_sets) # Calculated here to fix scope error

    # 1. Determine Visible Side based on Hidden Side
    # If Hidden is Right, Visible must be Left (and vice versa).
    if hidden_side == 'right':
        visible_side = 'left' 
    elif hidden_side == 'left':
        visible_side = 'right'
    else:
        raise ValueError("hidden_side must be 'left' or 'right'")

    # 2. Graph Construction
    # We pass visible_side to the builder. The builder assigns the OTHER side to hidden/conditioning.
    exp_embedding, fragment_map = build_expanded_embedding(
        conditioning_sets, 
        left_chains, 
        right_chains, 
        num_visible=n_vis, 
        hidden_side=hidden_side
    )
    
    # 3. Ising Formulation
    # Note: rbm_to_expanded_ising handles the mix of integer (logical) 
    # and string (fragment) keys in exp_embedding.
    h_exp, J_exp = rbm_to_expanded_ising(
        rbm, fragment_map, exp_embedding, raw_sampler.adjacency, beta
    )    

    # 4. Flux Calculation
    # Assuming the first row of the batch dictates the clamps
    row_np = binary_patterns_batch[0].detach().cpu().numpy()
    
    # Create logical clamps for the conditioning nodes (indices 0 to n_cond-1)
    logical_clamps = {i: (1 if v > 0.5 else -1) for i, v in enumerate(row_np) if i < n_cond}

    raw_fb = get_expanded_flux_biases(
        logical_clamps, fragment_map, exp_embedding, total_qubits, clamp_strength_h
    )
    
    # Safety clamp to prevent hardware errors
    safe_fb = [max(-0.01, min(v, 0.01)) for v in raw_fb]

    if chain_strength is None:
            # Calculate based on the specific couplings of this batch/model
            calc_strength = calculate_rms_chain_strength(J_exp, rho=rho)
            
            # Optional: Clip it to hardware limits to be safe
            max_j = raw_sampler.properties.get('extended_j_range', [None, 2.0])[1]
            chain_strength = min(calc_strength, max_j)
            
            print(f"Dynamic Chain Strength calculated: {chain_strength:.2f} (rho={rho:.2f})")
    
    run_metadata = {
        'timestamp': str(datetime.now()),
        'n_vis': n_vis,
        'n_hid': n_hid,
        'n_cond': n_cond,
        'beta': beta,
        'hidden_side': hidden_side,
        'visible_side': visible_side,
        "chain_strength": chain_strength,
        "chain_method": 'rms_dynamic' if chain_strength is None else 'manual',
    }
    if source is not None:
        run_metadata['source'] = source


    # 5. Sample with Analysis
    print(f"Submitting Batch: (n={batch_size})...")    
    analysis_result = sample_physical_with_analysis(
        raw_sampler=raw_sampler,
        h_logical=h_exp,
        J_logical=J_exp,
        embedding=exp_embedding,
        flux_biases=safe_fb,
        num_samples=batch_size,
        chain_strength=chain_strength,
        device=rbm.device,
        metadata=run_metadata,
        save_dir=save_dir
    )
    
    # 6. Post-Process Reporting
    print(f"Clean samples: {analysis_result.total_clean_fraction:.2%}")
    # Only try to print breaks if we actually have data
    if hasattr(analysis_result, 'breaks_per_variable') and len(analysis_result.breaks_per_variable) > 0:
        print(f"Top 5 broken vars: {np.argsort(analysis_result.breaks_per_variable)[-5:]}")
    
    return analysis_result


def sample_expanded_flux_conditioned_rigorous_srt(
    rbm, 
    raw_sampler, 
    conditioning_sets, 
    left_chains, 
    right_chains, 
    binary_patterns_batch, 
    hidden_side='right',
    beta=1.0, 
    chain_strength=1.0, 
    rho = 28.0,
    clamp_strength_h=50.0,
    source=None,
    save_dir="/home/leozhu/CaloQuVAE/wandb-outputs/dwave_misc/",
    use_srt=True,
    additive_flux_offsets=None, # Tensor or List of shape [total_qubits]
    # --- NEW ORBIT ARGUMENTS ---
    vis_shift=0, # Cyclic shift for visible chains
    hid_shift=0 # Cyclic shift for hidden chains
):
    """
    Samples from the D-Wave QPU using expanded embedding with ORBIT ROTATION support.
    
    """
    batch_size = binary_patterns_batch.shape[0]
    total_qubits = raw_sampler.properties['num_qubits']
    n_vis = rbm.params["vbias"].shape[0]
    n_hid = rbm.params["hbias"].shape[0]
    n_cond = len(conditioning_sets)

    # 1. Determine Sides (for logging/logic)
    if hidden_side == 'right':
        visible_side = 'left' 
    elif hidden_side == 'left':
        visible_side = 'right'
    else:
        raise ValueError("hidden_side must be 'left' or 'right'")

    print(f"--- Orbit Config: VisShift={vis_shift}, HidShift={hid_shift} ---")

    # 2. Graph Construction (WITH ROTATION)
    # Replaced 'build_expanded_embedding' with your new function
    exp_embedding, fragment_map = build_expanded_embedding_rotation(
        conditioning_sets, 
        left_chains, 
        right_chains, 
        num_visible=n_vis, 
        hidden_side=hidden_side,
        vis_shift=vis_shift,
        hid_shift=hid_shift
    )
    
    # 3. Ising Formulation
    # This automatically uses the rotated 'exp_embedding' to map logical h/J to physical QPU
    h_exp, J_exp = rbm_to_expanded_ising(
        rbm, fragment_map, exp_embedding, raw_sampler.adjacency, beta
    )    

    # 4. Flux Calculation
    row_np = binary_patterns_batch[0].detach().cpu().numpy()
    logical_clamps = {i: (1 if v > 0.5 else -1) for i, v in enumerate(row_np) if i < n_cond}

    # This also respects rotation because it looks up 'logical_id' in 'exp_embedding'
    raw_fb = get_expanded_flux_biases(
        logical_clamps, fragment_map, exp_embedding, total_qubits, clamp_strength_h
    )

    # --- INJECT CALIBRATION SHIMS ---
    if additive_flux_offsets is not None:
        if hasattr(additive_flux_offsets, 'cpu'):
            additive_flux_offsets = additive_flux_offsets.cpu().numpy()
        
        if len(additive_flux_offsets) != total_qubits:
             raise ValueError(f"Shim dimension {len(additive_flux_offsets)} != QPU qubits {total_qubits}")
        
        # Additive application: Final = Logical + Calibration
        raw_fb = [r + s for r, s in zip(raw_fb, additive_flux_offsets)]
    # --------------------------------
    
    safe_limit = 0.01
    safe_fb = [max(-1*safe_limit, min(v, safe_limit)) for v in raw_fb]

    if chain_strength is None:
            calc_strength = calculate_rms_chain_strength(J_exp, rho=rho)
            # Use 'extended_j_range' if available, otherwise default to typical 2.0
            max_j = raw_sampler.properties.get('extended_j_range', [None, 2.0])[1]
            chain_strength = min(calc_strength, max_j)
            print(f"Dynamic Chain Strength calculated: {chain_strength:.2f} (rho={rho:.2f})")
    
    run_metadata = {
        'timestamp': str(datetime.now()),
        'chain_strength': chain_strength,
        'calibrated': (additive_flux_offsets is not None),
        'vis_shift': vis_shift,
        'hid_shift': hid_shift
    }
    if source is not None:
        run_metadata['source'] = source

    # 5. Sample
    print(f"Submitting Batch: (n={batch_size}, SRT={use_srt})...")    
    analysis_result = sample_physical_with_analysis_srt(
        raw_sampler=raw_sampler,
        h_logical=h_exp,
        J_logical=J_exp,
        embedding=exp_embedding,
        flux_biases=safe_fb,
        num_samples=batch_size,
        chain_strength=chain_strength,
        device=rbm.device,
        metadata=run_metadata,
        save_dir=save_dir,
        use_srt=use_srt 
    )
    
    return analysis_result


def sample_expanded_flux_arbitrary(
    rbm,
    raw_sampler,
    conditioning_sets,
    left_chains,
    right_chains,
    binary_patterns_batch,
    hidden_side='right',
    beta=1.0,
    chain_strength=None,
    rho=28.0,
    clamp_strength_h=50.0,
    source=None,
    save_dir="/home/leozhu/CaloQuVAE/wandb-outputs/dwave_misc/",
    use_srt=True,
    additive_flux_offsets=None, # Tensor or List of shape [total_qubits]
    vis_mapping=None, 
    hid_mapping=None,
    perm_seed=None # For logging purposes
):
    """
    Samples using arbitrary permutations (shuffles) of chain assignments.
      - Calls sample_physical_arbitrary
      - Passes orbit data as explicit arguments, not metadata
    """
    batch_size = binary_patterns_batch.shape[0]
    total_qubits = raw_sampler.properties['num_qubits']
    n_vis = rbm.params["vbias"].shape[0]
    
    # 1. Build Embedding using the NEW Arbitrary function
    exp_embedding, fragment_map = build_expanded_embedding_arbitrary(
        conditioning_sets, 
        left_chains, 
        right_chains, 
        num_visible=n_vis, 
        hidden_side=hidden_side,
        vis_mapping=vis_mapping,
        hid_mapping=hid_mapping
    )

    # 2. Convert RBM to Ising
    h_exp, J_exp = rbm_to_expanded_ising(
        rbm, fragment_map, exp_embedding, raw_sampler.adjacency, beta
    )    

    # 3. Flux Calculation
    row_np = binary_patterns_batch[0].detach().cpu().numpy()
    n_cond = len(conditioning_sets)
    
    # Create logical clamps
    logical_clamps = {i: (1 if v > 0.5 else -1) for i, v in enumerate(row_np) if i < n_cond}
    
    # Calculate Flux Biases (respects rotation via fragment_map)
    raw_fb = get_expanded_flux_biases(
        logical_clamps, fragment_map, exp_embedding, total_qubits, clamp_strength_h
    )

    # --- INJECT CALIBRATION SHIMS ---
    if additive_flux_offsets is not None:
        if hasattr(additive_flux_offsets, 'cpu'):
            additive_flux_offsets = additive_flux_offsets.cpu().numpy()
        
        if len(additive_flux_offsets) != total_qubits:
             raise ValueError(f"Shim dimension {len(additive_flux_offsets)} != QPU qubits {total_qubits}")
        
        # Additive application: Final = Logical + Calibration
        raw_fb = [r + s for r, s in zip(raw_fb, additive_flux_offsets)]
    # --------------------------------
    
    # Safety Clamp
    safe_limit = 0.01
    safe_fb = [max(-1*safe_limit, min(v, safe_limit)) for v in raw_fb]

    # Chain Strength Default
    if chain_strength is None:
        chain_strength = raw_sampler.properties.get('extended_j_range', [None, 2.0])[1]
    
    # 4. Minimal Metadata (Orbit info is now passed as args)
    run_metadata = {
        'timestamp': str(datetime.now()),
        'chain_strength': chain_strength,
        'calibrated': (additive_flux_offsets is not None)
    }
    if source is not None:
        run_metadata['source'] = source

    # 5. Sample (Using new signature)
    analysis_result = sample_physical_arbitrary(
        raw_sampler=raw_sampler,
        h_logical=h_exp,
        J_logical=J_exp,
        embedding=exp_embedding,
        flux_biases=safe_fb,
        num_samples=batch_size,
        chain_strength=chain_strength,
        device=rbm.device,
        metadata=run_metadata,
        save_dir=save_dir,
        use_srt=use_srt,
        # --- Explicit Orbit Passing ---
        orbit_seed=perm_seed,
        vis_mapping=vis_mapping,
        hid_mapping=hid_mapping
    )
    
    return analysis_result

def find_beta_rigorous(
    rbm, 
    qpu_sampler, 
    conditioning_sets,
    left_chains,
    right_chains,
    binary_patterns_batch,
    hidden_side='right',
    num_reads=128,
    rbm_gibbs_steps=1000, 
    beta_init=3.0, 
    lr=0.1, 
    num_epochs=15, 
    tolerance=0.1,
    adaptive=False,
    use_fast_sampling=True,
    validate_beta_heterogeneous=False,
    use_srt=True
):
    """
    Optimizes Beta using the rigorous expanded sampling method.
    Uses process_analysis_result for clean unpacking.
    """
    beta = beta_init
    beta_hist, rbm_e_hist, qpu_e_hist = [], [], []
    n_clamped = binary_patterns_batch.shape[1]
    
    # --- 1. Prepare Target Batch ---
    if use_fast_sampling:
        print(f"Mode: Fast Estimation (Homogeneous Batch) | Side: {hidden_side}")
        primary_pattern = binary_patterns_batch[0].unsqueeze(0)
        target_batch = primary_pattern.repeat(num_reads, 1)
    else:
        print(f"Mode: Slow Estimation (Heterogeneous Batch) | Side: {hidden_side}")
        target_batch = binary_patterns_batch[:num_reads]

    # --- 2. RBM Baseline ---
    print("Calculating RBM Baseline Energy...")
    full_v_rbm = rbm.sample_v_given_v_clamped(
        clamped_v=target_batch, 
        n_clamped=n_clamped, 
        gibbs_steps=rbm_gibbs_steps, 
        beta=1.0
    )
    full_h_rbm, _ = rbm._sample_h_given_v(full_v_rbm, beta=1.0)
    
    with torch.no_grad(): 
        energies_rbm = joint_energy(rbm, full_v_rbm, full_h_rbm)
        mean_rbm_energy = energies_rbm.mean().item()
        print(f"Target (RBM) Mean Energy: {mean_rbm_energy:.4f}")

    energies_qpu_final = None
    raw_sampler_obj = qpu_sampler.child if hasattr(qpu_sampler, 'child') else qpu_sampler

    # --- 3. Optimization Loop ---
    for epoch in range(num_epochs):
        
        # A. Sample
        analysis_result = sample_expanded_flux_conditioned_rigorous_srt(
            rbm=rbm,
            raw_sampler=raw_sampler_obj,
            conditioning_sets=conditioning_sets,
            left_chains=left_chains,
            right_chains=right_chains,
            binary_patterns_batch=target_batch,
            hidden_side=hidden_side,
            beta=beta,
            source="beta-estimate",
            use_srt=use_srt
        )
        
        # B. Unpack (New clean function call)
        dwave_v, dwave_h = process_analysis_result(
            analysis_result, 
            rbm, 
            conditioning_sets
        )
        
        # C. Calculate Metrics
        with torch.no_grad():
            e_dwave = joint_energy(rbm, dwave_v, dwave_h)
            mean_dwave = e_dwave.mean().item()
            var_dwave = torch.var(e_dwave).item()
            energies_qpu_final = e_dwave

        # D. Update Step
        if adaptive:
            safe_var = max(1e-6, var_dwave)
            current_lr = min(0.5, max(lr, (beta**2)/safe_var))
        else:
            current_lr = lr
            
        diff = mean_dwave - mean_rbm_energy
        beta = max(1e-2, beta - current_lr * diff)
        
        beta_hist.append(beta)
        rbm_e_hist.append(mean_rbm_energy)
        qpu_e_hist.append(mean_dwave)
        
        print(f"Epoch {epoch}: Beta={beta:.4f} | Diff={diff:.2f} | Clean={analysis_result.total_clean_fraction:.2%}")
        
        if abs(diff) < tolerance: 
            print("Converged within tolerance.")
            break
            
    if energies_qpu_final is not None:
        plot_energy_comparison(energies_rbm, energies_qpu_final, beta)

    # --- 4. Conditional Validation ---
    if use_fast_sampling and validate_beta_heterogeneous:
        print("\n--- Running Heterogeneous Validation ---")
        
        val_batch = binary_patterns_batch[:num_reads//2]
        
        # RBM Ref
        val_v_rbm = rbm.sample_v_given_v_clamped(
            clamped_v=val_batch, n_clamped=n_clamped, gibbs_steps=rbm_gibbs_steps, beta=1.0
        )
        val_h_rbm, _ = rbm._sample_h_given_v(val_v_rbm, beta=1.0)
        mean_val_rbm = joint_energy(rbm, val_v_rbm, val_h_rbm).mean().item()
        
        # QPU Ref
        val_analysis = sample_expanded_flux_conditioned_rigorous(
            rbm=rbm,
            raw_sampler=raw_sampler_obj,
            conditioning_sets=conditioning_sets,
            left_chains=left_chains,
            right_chains=right_chains,
            binary_patterns_batch=val_batch,
            hidden_side=hidden_side,
            beta=beta,
            clamp_strength_h=50.0,
            save_dir=save_dir
        )
        
        # Unpack Validation
        val_v, val_h = process_analysis_result(val_analysis, rbm, conditioning_sets)
        
        mean_val_qpu = joint_energy(rbm, val_v, val_h).mean().item()
        val_diff = mean_val_qpu - mean_val_rbm
        
        print(f"Validation Diff: {val_diff:.4f}")
        if abs(val_diff) > 1.0:
            print("WARNING: Fast estimation diverged from heterogeneous validation.")
        else:
            print("SUCCESS: Beta generalizes to heterogeneous batch.")

    return beta, beta_hist, rbm_e_hist, qpu_e_hist


def find_beta_experimental(
    rbm, 
    qpu_sampler, 
    conditioning_sets,
    left_chains,
    right_chains,
    binary_patterns_batch,
    hidden_side='right',
    num_reads=128,
    rbm_gibbs_steps=1000, 
    beta_init=3.0, 
    lr=0.1, 
    num_epochs=15, 
    tolerance=0.01,
    adaptive=False,
    use_fast_sampling=True,
    validate_beta_heterogeneous=False
):
    """
    Optimizes Beta by matching the VARIANCE of the Energy distribution
    using ONLY clean (unbroken) samples.
    
    Logic:
    - If Var(QPU) > Var(Target), QPU is too hot.
    - To cool QPU, we need larger physical params.
    - We achieve this by DECREASING beta (assuming H_phys = H_rbm / beta).
    """
    beta = beta_init
    beta_hist, rbm_var_hist, qpu_var_hist = [], [], []
    n_clamped = binary_patterns_batch.shape[1]
    
    # --- 1. Prepare Target Batch ---
    if use_fast_sampling:
        print(f"Mode: Fast Estimation (Homogeneous Batch) | Side: {hidden_side}")
        primary_pattern = binary_patterns_batch[0].unsqueeze(0)
        target_batch = primary_pattern.repeat(num_reads, 1)
    else:
        print(f"Mode: Slow Estimation (Heterogeneous Batch) | Side: {hidden_side}")
        target_batch = binary_patterns_batch[:num_reads]

    # --- 2. RBM Baseline (Target Variance) ---
    print("Calculating RBM Baseline Energy Statistics...")
    full_v_rbm = rbm.sample_v_given_v_clamped(
        clamped_v=target_batch, 
        n_clamped=n_clamped, 
        gibbs_steps=rbm_gibbs_steps, 
        beta=1.0
    )
    full_h_rbm, _ = rbm._sample_h_given_v(full_v_rbm, beta=1.0)
    
    with torch.no_grad(): 
        energies_rbm = joint_energy(rbm, full_v_rbm, full_h_rbm)
        mean_rbm_energy = energies_rbm.mean().item()
        # VARIANCE TARGET
        var_rbm_energy = torch.var(energies_rbm).item()
        print(f"Target (RBM) Mean: {mean_rbm_energy:.4f} | Variance: {var_rbm_energy:.4f}")

    energies_qpu_final = None
    raw_sampler_obj = qpu_sampler.child if hasattr(qpu_sampler, 'child') else qpu_sampler

    # --- 3. Optimization Loop ---
    for epoch in range(num_epochs):
        
        # A. Sample
        analysis_result = sample_expanded_flux_conditioned_rigorous(
            rbm=rbm,
            raw_sampler=raw_sampler_obj,
            conditioning_sets=conditioning_sets,
            left_chains=left_chains,
            right_chains=right_chains,
            binary_patterns_batch=target_batch,
            hidden_side=hidden_side,
            beta=beta,
            chain_strength=1.0, 
            source="beta-estimate",
        )
        
        # B. Unpack Raw Tensors
        dwave_v_full, dwave_h_full = process_analysis_result(
            analysis_result, 
            rbm, 
            conditioning_sets
        )
        
        # C. Filter Clean Samples
        # Check rows where sum of breaks is 0
        is_clean_numpy = ~analysis_result.break_matrix.any(axis=1)
        clean_indices = torch.tensor(is_clean_numpy, device=rbm.device, dtype=torch.bool)
        n_clean = clean_indices.sum().item()
        clean_frac = n_clean / len(dwave_v_full)
        
        if n_clean > 2:
            dwave_v = dwave_v_full[clean_indices]
            dwave_h = dwave_h_full[clean_indices]
        else:
            print(f"WARNING: Low clean sample count ({n_clean}). Using full batch to prevent crash.")
            dwave_v = dwave_v_full
            dwave_h = dwave_h_full

        # D. Calculate Metrics (Variance)
        with torch.no_grad():
            e_dwave = joint_energy(rbm, dwave_v, dwave_h)
            var_dwave = torch.var(e_dwave).item()
            energies_qpu_final = e_dwave

        # E. Update Step (Variance Matching)
        diff_var = var_dwave - var_rbm_energy
        
        if adaptive:
            # Prevent exploding updates if variance difference is massive
            safe_denom = max(1e-6, abs(diff_var))
            current_lr = min(0.5, max(lr, 0.1 / safe_denom))
        else:
            current_lr = lr
            
        # UPDATE LOGIC:
        # If Var_QPU > Var_RBM (Diff is +), QPU is too hot.
        # We need to Decrease Beta to increase params (cool down).
        beta = max(1e-2, beta - current_lr * diff_var)
        
        beta_hist.append(beta)
        rbm_var_hist.append(var_rbm_energy)
        qpu_var_hist.append(var_dwave)
        
        print(f"Epoch {epoch}: Beta={beta:.4f} | Var_QPU={var_dwave:.4f} | Diff={diff_var:.4f} | Clean={clean_frac:.2%}")
        
        if abs(diff_var) < tolerance: 
            print("Converged within tolerance.")
            break
            
    if energies_qpu_final is not None:
        plot_energy_comparison(energies_rbm, energies_qpu_final, beta)

    # --- 4. Conditional Validation ---
    if use_fast_sampling and validate_beta_heterogeneous:
        print("\n--- Running Heterogeneous Validation ---")
        val_batch = binary_patterns_batch[:num_reads//2]
        
        # RBM Ref
        val_v_rbm = rbm.sample_v_given_v_clamped(
            clamped_v=val_batch, n_clamped=n_clamped, gibbs_steps=rbm_gibbs_steps, beta=1.0
        )
        val_h_rbm, _ = rbm._sample_h_given_v(val_v_rbm, beta=1.0)
        val_var_rbm = torch.var(joint_energy(rbm, val_v_rbm, val_h_rbm)).item()
        
        # QPU Ref
        val_analysis = sample_expanded_flux_conditioned_rigorous(
            rbm=rbm,
            raw_sampler=raw_sampler_obj,
            conditioning_sets=conditioning_sets,
            left_chains=left_chains,
            right_chains=right_chains,
            binary_patterns_batch=val_batch,
            hidden_side=hidden_side,
            beta=beta,
            clamp_strength_h=50.0
        )
        
        # Unpack & Clean
        val_v_full, val_h_full = process_analysis_result(val_analysis, rbm, conditioning_sets)
        val_is_clean = ~val_analysis.break_matrix.any(axis=1)
        val_clean_indices = torch.tensor(val_is_clean, device=rbm.device, dtype=torch.bool)
        
        if val_clean_indices.sum() > 2:
            val_v = val_v_full[val_clean_indices]
            val_h = val_h_full[val_clean_indices]
        else:
            val_v = val_v_full
            val_h = val_h_full
            
        val_var_qpu = torch.var(joint_energy(rbm, val_v, val_h)).item()
        
        val_diff = val_var_qpu - val_var_rbm
        print(f"Validation Var Diff: {val_diff:.4f}")
        
        if abs(val_diff) > (var_rbm_energy * 0.5):
            print("WARNING: Fast estimation diverged from heterogeneous validation.")
        else:
            print("SUCCESS: Beta generalizes to heterogeneous batch.")

    return beta, beta_hist, rbm_var_hist, qpu_var_hist

def generate_conditioned_shower(
    engine,
    rbm,
    raw_sampler,
    conditioning_sets,
    left_chains,
    right_chains,
    target_energy,
    beta=1.0,
    chain_strength=None,
    clamp_strength_h=50.0
):
    """
    Orchestrates the QPU sampling and Classical Decoding.
    Returns a tensor of the generated shower (shape [1, input_size]).
    """
    
    # 1. Prepare Energy Tensor
    device = engine.device
    energy_tensor = torch.tensor([[target_energy]], dtype=torch.float32).to(device)

    # 2. Encode Energy to Binary Pattern
    with torch.no_grad():
        binary_energy_encoded = engine.model.encoder.binary_energy_refactored(energy_tensor)
    
    # 3. Sample from QPU (calling the function already in this file)
    v_s, h_s = sample_expanded_flux_conditioned(
        rbm=rbm,
        raw_sampler=raw_sampler,
        conditioning_sets=conditioning_sets,
        left_chains=left_chains,
        right_chains=right_chains,
        binary_patterns_batch=binary_energy_encoded,
        beta=beta,
        chain_strength=chain_strength,
        clamp_strength_h=clamp_strength_h,
        fast_sampling=False 
    )

    # 4. Construct RBM Samples and Decode
    rbm_samples_qpu = v_s.to(device)
    engine.generate_showers_from_rbm(rbm_samples_qpu, energy_tensor)
    
    # 5. Extract result
    return engine.showers_prior_generated[0]


def find_beta_flux_bias_expanded(
    rbm, 
    qpu_sampler, 
    embedding, 
    conditioning_sets,
    left_chains,
    right_chains,
    binary_patterns_batch, 
    num_reads=128,
    rbm_gibbs_steps=1000, 
    beta_init=3.0, 
    lr=0.1, 
    num_epochs=15, 
    tolerance=0.1,
    adaptive=False,
    use_fast_sampling=True,
    validate_beta_heterogeneous=False
):
    """
    Optimizes Beta.
    
    Args:
        use_fast_sampling (bool): 
            If True: Creates a homogeneous batch from row [0] for speed. Runs validation at end.
            If False: Uses the actual input batch (heterogeneous). Skips validation at end.
    """
    beta = beta_init
    beta_hist, rbm_e_hist, qpu_e_hist = [], [], []
    n_clamped = binary_patterns_batch.shape[1]
    
    # 1. Prepare Target Batch and RBM Baseline
    if use_fast_sampling:
        print("Mode: Fast Estimation (Homogeneous Batch)")
        # Create artificial homogeneous batch based on first row
        primary_pattern = binary_patterns_batch[0].unsqueeze(0)
        target_batch = primary_pattern.repeat(num_reads, 1)
    else:
        print("Mode: Slow Estimation (Heterogeneous Batch)")
        # Use the actual diverse data
        target_batch = binary_patterns_batch[:num_reads]

    # Calculate RBM Conditional Energy (Baseline)
    # We must sample the RBM using the *same* target_batch structure we defined above
    full_v_rbm = rbm.sample_v_given_v_clamped(
        clamped_v=target_batch, 
        n_clamped=n_clamped, 
        gibbs_steps=rbm_gibbs_steps, 
        beta=1.0
    )
    full_h_rbm, _ = rbm._sample_h_given_v(full_v_rbm, beta=1.0)
    
    with torch.no_grad(): 
        energies_rbm = joint_energy(rbm, full_v_rbm, full_h_rbm)
        mean_rbm_energy = energies_rbm.mean().item()

    energies_qpu_final = None

    # 2. Optimization Loop
    for epoch in range(num_epochs):
        # Call the sampler with the appropriate mode
        dwave_v, dwave_h = sample_expanded_flux_conditioned(
            rbm=rbm, 
            raw_sampler=qpu_sampler.child, 
            conditioning_sets=conditioning_sets, 
            left_chains=left_chains, 
            right_chains=right_chains, 
            binary_patterns_batch=target_batch, 
            beta=beta, 
            clamp_strength_h=25.0,
            fast_sampling=use_fast_sampling # Pass the flag down
        )
        
        with torch.no_grad():
            e_dwave = joint_energy(rbm, dwave_v, dwave_h)
            mean_dwave = e_dwave.mean().item()
            var_dwave = torch.var(e_dwave).item()
            energies_qpu_final = e_dwave

        if adaptive:
            safe_var = max(1e-6, var_dwave)
            current_lr = min(0.5, max(lr, (beta**2)/safe_var))
        else:
            current_lr = lr
            
        diff = mean_dwave - mean_rbm_energy
        beta = max(1e-2, beta - current_lr * diff)
        
        beta_hist.append(beta)
        rbm_e_hist.append(mean_rbm_energy)
        qpu_e_hist.append(mean_dwave)
        
        print(f"Epoch {epoch}: Beta={beta:.4f} | Diff={diff:.2f}")
        if abs(diff) < tolerance: 
            break
            
    if energies_qpu_final is not None:
        plot_energy_comparison(energies_rbm, energies_qpu_final, beta)

    # 3. Conditional Validation
    if use_fast_sampling and validate_beta_heterogeneous:
        # If we optimized on a fast homogeneous approximation, we MUST validate 
        # on the real heterogeneous data to ensure Beta generalizes.
        diff, rbm_e, qpu_e = validate_beta_heterogeneous(
            rbm, 
            qpu_sampler, 
            conditioning_sets, 
            left_chains, 
            right_chains, 
            binary_patterns_batch, # The full heterogeneous batch
            num_reads=num_reads//2,
            beta=beta
        )
        if abs(diff) > 1.0:
            print("WARNING: Fast estimation diverged from heterogeneous validation.")
        else:
            print("SUCCESS: Beta generalizes to heterogeneous batch.")
    else:
        # If we optimized using the slow heterogeneous method, 
        # the last epoch WAS the validation.
        print("Optimization performed on full heterogeneous batch. Skipping post-validation.")

            
    return beta, beta_hist, rbm_e_hist, qpu_e_hist


def find_beta_arbitrary(
    rbm,
    raw_sampler,
    conditioning_sets,
    left_chains,
    right_chains,
    binary_patterns_batch,
    vis_mapping: list,
    hid_mapping: list,
    orbit_seed: int = None,
    num_reads=1024,
    rbm_gibbs_steps=5000,
    beta_init=3.0, 
    lr=0.01, 
    num_epochs=15, 
    tolerance=0.1,
    use_srt=True,
    save_dir=None
):
    """
    Orbit aware beta optimization
    """
    beta = beta_init
    beta_hist, rbm_e_hist, qpu_e_hist = [], [], []
    n_clamped = binary_patterns_batch.shape[1]

    primary_pattern = binary_patterns_batch[0].unsqueeze(0)
    target_batch = primary_pattern.repeat(num_reads, 1)

    print(f"Calculating RBM Baseline Energy for Orbit Seed {orbit_seed}...")
    full_v_rbm = rbm.sample_v_given_v_clamped(
        clamped_v=target_batch, 
        n_clamped=n_clamped, 
        gibbs_steps=rbm_gibbs_steps, 
        beta=1.0 # Classical RBM is always beta=1.0
    )
    full_h_rbm, _ = rbm._sample_h_given_v(full_v_rbm, beta=1.0)
    
    with torch.no_grad(): 
        energies_rbm = joint_energy(rbm, full_v_rbm, full_h_rbm)
        mean_rbm_energy = energies_rbm.mean().item()
        print(f"   Target Mean Energy: {mean_rbm_energy:.4f}")
    
    for epoch in range(num_epochs):
        
        # A. Sample using Arbitrary Mapping
        analysis_result = sample_expanded_flux_arbitrary(
            rbm=rbm,
            raw_sampler=raw_sampler,
            conditioning_sets=conditioning_sets,
            left_chains=left_chains,
            right_chains=right_chains,
            binary_patterns_batch=target_batch,
            hidden_side=hidden_side,
            beta=beta,
            source=f"beta_est_orbit{orbit_seed}",
            use_srt=use_srt,
            save_dir=save_dir,
            # --- Pass Orbit ---
            vis_mapping=vis_mapping,
            hid_mapping=hid_mapping,
            perm_seed=orbit_seed
        )
        
        dwave_v, dwave_h = process_analysis_result(
            analysis_result, 
            rbm, 
            conditioning_sets
        )
        
        # C. Calculate Metrics
        with torch.no_grad():
            # Calculate energy of QPU samples using RBM weights
            e_dwave = joint_energy(rbm, dwave_v, dwave_h)
            mean_dwave = e_dwave.mean().item()

        # D. Update Step
        diff = mean_dwave - mean_rbm_energy

        beta_new = max(1e-2, beta - lr * diff)

        print(f"Epoch {epoch}: Beta={beta:.4f} -> {beta_new:.4f} | Diff={diff:.2f} | Clean={analysis_result.total_clean_fraction:.2%}")

        beta = beta_new
        beta_hist.append(beta)
        rbm_e_hist.append(mean_rbm_energy)
        qpu_e_hist.append(mean_dwave)

        if abs(diff) < tolerance: 
            print("Converged within tolerance.")
            break
    
    return beta, beta_hist, rbm_e_hist, qpu_e_hist




def find_beta_multi_energy(
    incidence_energies: list, # List of energies, e.g., [1, 10, 50]
    energy_patterns_dict: dict, # Map: {energy_val: binary_patterns_tensor}
    rbm,
    raw_sampler,
    conditioning_sets,
    left_chains,
    right_chains,
    hidden_side: str = "right",
    orbit_seed=None, # Default orbit seed
    num_reads=1024,
    rbm_gibbs_steps=5000,
    beta_init=3.0, 
    lr=0.01, 
    num_epochs=15, 
    tolerance=0.1,
    use_srt=True,
    save_dir=None
):
    """
    Orbit aware beta optimization across multiple incidence energies.
    Optimizes a single scalar Beta to minimize the energy divergence 
    between RBM and QPU across the provided energy spectrum.
    """
    beta = beta_init
    beta_hist, rbm_e_hist, qpu_e_hist = [], [], []
    
    # --- 1. Calculate Global RBM Baseline (Target) ---
    print(f"Calculating Global RBM Baseline across {len(incidence_energies)} energies...")
    
    per_energy_rbm_means = []
    
    # We pre-calculate the target batch for each energy to save time in the loop
    target_batches = {} 

    for energy in incidence_energies:
        if energy not in energy_patterns_dict:
            raise ValueError(f"Energy {energy} missing from energy_patterns_dict")
            
        # Prepare batch for this energy
        primary_pattern = energy_patterns_dict[energy][0].unsqueeze(0)
        target_batch = primary_pattern.repeat(num_reads, 1)
        target_batches[energy] = target_batch
        n_clamped = target_batch.shape[1]

        # Classical Sampling (Beta=1.0)
        full_v_rbm = rbm.sample_v_given_v_clamped(
            clamped_v=target_batch, 
            n_clamped=n_clamped, 
            gibbs_steps=rbm_gibbs_steps, 
            beta=1.0 
        )
        full_h_rbm, _ = rbm._sample_h_given_v(full_v_rbm, beta=1.0)
        
        with torch.no_grad(): 
            energies_rbm = joint_energy(rbm, full_v_rbm, full_h_rbm)
            mean_e = energies_rbm.mean().item()
            per_energy_rbm_means.append(mean_e)
            print(f"   Energy {energy}GeV Baseline: {mean_e:.4f}")

    # The target is the average energy across all incidence types
    global_target_energy = np.mean(per_energy_rbm_means)
    print(f"Global Target Mean Energy: {global_target_energy:.4f}")
    print("-" * 60)

    # get orbit mappings for the given seed
    if hidden_side == 'right':
        n_vis, n_hid = len(left_chains), len(right_chains)
    else:
        n_vis, n_hid = len(right_chains), len(left_chains)

    vis_mapping, hid_mapping = get_orbit_mappings(orbit_seed, n_vis, n_hid)

    # --- 2. Optimization Loop ---
    for epoch in range(num_epochs):
        
        epoch_qpu_means = []
        epoch_clean_fractions = []
        
        # Iterate through all incidence energies using CURRENT Beta
        for energy in incidence_energies:
            
            target_batch = target_batches[energy]
            
            # A. Sample QPU (Arbitrary Mapping)
            # Note: We use the same orbit_seed for all checks to minimize embedding noise
            analysis_result = sample_expanded_flux_arbitrary(
                rbm=rbm,
                raw_sampler=raw_sampler,
                conditioning_sets=conditioning_sets,
                left_chains=left_chains,
                right_chains=right_chains,
                binary_patterns_batch=target_batch,
                hidden_side=hidden_side,
                beta=beta,
                source=f"beta_est_E{energy}_orbit{orbit_seed}",
                use_srt=use_srt,
                save_dir=save_dir,
                vis_mapping=vis_mapping,
                hid_mapping=hid_mapping,
                perm_seed=orbit_seed 
            )
            
            # B. Process Results
            dwave_v, dwave_h = process_analysis_result(
                analysis_result, 
                rbm, 
                conditioning_sets
            )
            
            # C. Calculate Metrics for this specific energy batch
            with torch.no_grad():
                e_dwave = joint_energy(rbm, dwave_v, dwave_h)
                epoch_qpu_means.append(e_dwave.mean().item())
                epoch_clean_fractions.append(analysis_result.total_clean_fraction)

        # --- Aggregation & Update ---
        global_qpu_mean = np.mean(epoch_qpu_means)
        avg_clean_frac = np.mean(epoch_clean_fractions)
        
        # D. Update Step
        # Diff = QPU - RBM. If QPU energy is too high, we increase Beta (lower temp).
        diff = global_qpu_mean - global_target_energy
        beta_new = max(1e-2, beta - lr * diff)

        print(f"Epoch {epoch}: Beta={beta:.4f} -> {beta_new:.4f} | "
              f"Global Diff={diff:.2f} | Avg Clean={avg_clean_frac:.2%}")

        beta = beta_new
        
        # Store History
        beta_hist.append(beta)
        rbm_e_hist.append(global_target_energy)
        qpu_e_hist.append(global_qpu_mean)

        if abs(diff) < tolerance: 
            print("Converged within tolerance.")
            break
    
    return beta, beta_hist, rbm_e_hist, qpu_e_hist



def validate_beta_heterogeneous(
    rbm, 
    qpu_sampler, 
    conditioning_sets, 
    left_chains, 
    right_chains, 
    binary_patterns_batch, 
    beta, 
    num_reads=128,
    chain_strength=None, 
    clamp_strength_h=50.0,
    rbm_gibbs_steps=1000
):
    """
    Validates a specific Beta by sampling the batch row-by-row (heterogeneously)
    on the QPU and comparing the average energy to an RBM Gibbs chain.
    """
    # Slice the batch
    binary_patterns_batch = binary_patterns_batch[:num_reads]
    batch_size = binary_patterns_batch.shape[0]

    print(f"\n--- Starting Beta Validation (Batch Size: {batch_size}) ---")

    # 1. RBM Baseline (Heterogeneous)
    print("Generating RBM baseline samples...")
    full_v_rbm = rbm.sample_v_given_v_clamped(
        clamped_v=binary_patterns_batch, 
        n_clamped=binary_patterns_batch.shape[1], 
        gibbs_steps=rbm_gibbs_steps, 
        beta=1.0
    )
    full_h_rbm, _ = rbm._sample_h_given_v(full_v_rbm, beta=1.0)
    
    with torch.no_grad(): 
        energies_rbm = joint_energy(rbm, full_v_rbm, full_h_rbm)
        mean_rbm_energy = energies_rbm.mean().item()

    # 2. QPU Sampling (Heterogeneous via fast_sampling=False)
    # We reuse the logic in sample_expanded_flux_conditioned for the loop
    dwave_v, dwave_h = sample_expanded_flux_conditioned(
        rbm=rbm, 
        raw_sampler=qpu_sampler.child, 
        conditioning_sets=conditioning_sets, 
        left_chains=left_chains, 
        right_chains=right_chains, 
        binary_patterns_batch=binary_patterns_batch, 
        beta=beta, 
        chain_strength=chain_strength,
        clamp_strength_h=clamp_strength_h,
        fast_sampling=False  # <--- Forces row-by-row sampling
    )

    with torch.no_grad():
        energies_qpu = joint_energy(rbm, dwave_v, dwave_h)
        mean_dwave_energy = energies_qpu.mean().item()
        
    diff = mean_dwave_energy - mean_rbm_energy
    
    print(f"Validation Results for Beta={beta:.4f}:")
    print(f"  RBM Mean Energy:   {mean_rbm_energy:.4f}")
    print(f"  QPU Mean Energy:   {mean_dwave_energy:.4f}")
    print(f"  Difference:        {diff:.4f}")

    plot_energy_comparison(energies_rbm, energies_qpu, beta)
    
    return diff, mean_rbm_energy, mean_dwave_energy



def mass_sample_dwave(
    incidence_energy: float,
    total_samples_needed: int,
    save_dir: str,
    # System Objects
    engine,
    rbm,
    raw_sampler,
    conditioning_sets,
    left_chains,
    right_chains,
    # Hyperparameters
    n_cond: int = 53,
    initial_beta: float = 3.0,
    batch_size: int = 1024,
    drift_tolerance: float = 1.0,  # Max allowed energy deviation before recalibration
    hidden_side: str = 'right',
    device: str = 'cpu',
    use_srt: bool = True
):
    """
    Production sampling loop for D-Wave. 
    Continuously samples until total_samples_needed is reached.
    Monitors energy drift and recalibrates beta if necessary, but KEEPS the drifted batch.
    
    Returns:
        tuple: (path_clean, path_dirty, path_energies)
    """
    # Ensure directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    print(f"\n=== Mass Sampling Start: {total_samples_needed} samples @ {incidence_energy} MeV ===")
    
    # 1. Setup & Classical Baseline
    print("-> Establishing Classical Energy Baseline...")
    
    # Create the binary condition pattern for this energy
    base_pattern = convert_energy_to_binary(
        incidence_energy=incidence_energy, 
        engine=engine, 
        n_cond=n_cond, 
        num_reads=1, 
        device=device
    )
    # Expand to a reasonable baseline batch size (e.g., 512) for accurate energy est
    baseline_batch = base_pattern.repeat(512, 1)
    
    # Run Classical Gibbs to get baseline energy
    v_base = rbm.sample_v_given_v_clamped(
        clamped_v=baseline_batch, 
        n_clamped=n_cond, 
        gibbs_steps=2000, 
        beta=1.0 # Classical RBM always runs at beta=1.0
    )
    h_base, _ = rbm._sample_h_given_v(v_base, beta=1.0)
    
    with torch.no_grad():
        baseline_energies = joint_energy(rbm, v_base, h_base)
        target_energy_mean = baseline_energies.mean().item()
    
    print(f"-> Target Classical Energy: {target_energy_mean:.4f}")

    # 2. Initialize Accumulators
    clean_samples_accum = []
    dirty_samples_accum = []
    
    samples_collected = 0
    current_beta = initial_beta
    
    # 3. Main Production Loop
    while samples_collected < total_samples_needed:
        
        # Determine actual batch size (don't over-sample on the last batch)
        remaining = total_samples_needed - samples_collected
        current_batch_size = min(batch_size, remaining)
        
        # Prepare batch of conditions
        batch_patterns = base_pattern.repeat(current_batch_size, 1)
        
        # --- A. Call QPU ---
        try:
            analysis_result = sample_expanded_flux_conditioned_rigorous_srt(
                rbm=rbm,
                raw_sampler=raw_sampler,
                conditioning_sets=conditioning_sets,
                left_chains=left_chains,
                right_chains=right_chains,
                binary_patterns_batch=batch_patterns,
                hidden_side=hidden_side,
                beta=current_beta,
                source=f"prod_E{int(incidence_energy)}",
                use_srt=use_srt
            )
        except Exception as e:
            print(f"!! QPU Call Failed: {e}. Retrying batch...")
            continue

        # --- B. Unpack & Process ---
        # Unpack raw bits to RBM format
        batch_v, batch_h = process_analysis_result(analysis_result, rbm, conditioning_sets)
        batch_mask = analysis_result.clean_mask.cpu()
        
        # --- C. Drift Check ---
        # Calculate energy of the raw QPU samples using the CLASSICAL RBM weights
        with torch.no_grad():
            # Move to GPU for calculation, then back to CPU
            qpu_energies = joint_energy(rbm, batch_v.to(rbm.device), batch_h.to(rbm.device))
            qpu_energy_mean = qpu_energies.mean().item()
            
        energy_diff = qpu_energy_mean - target_energy_mean
        
        print(f"   Batch {samples_collected}/{total_samples_needed} | "
              f"Beta: {current_beta:.3f} | "
              f"E_qpu: {qpu_energy_mean:.2f} (Diff: {energy_diff:.2f}) | "
              f"Clean: {analysis_result.total_clean_fraction:.1%}")

        if abs(energy_diff) > drift_tolerance:
            print(f"!! DRIFT DETECTED (Diff {energy_diff:.2f} > {drift_tolerance}). Recalibrating Beta...")
            
            # --- D. Recalibrate Beta ---
            # Using your tweaked hyperparameters (lr=0.025, num_reads=1024)
            new_beta, _, _, _ = find_beta_rigorous(
                rbm=rbm,
                qpu_sampler=raw_sampler, 
                conditioning_sets=conditioning_sets,
                left_chains=left_chains,
                right_chains=right_chains,
                binary_patterns_batch=batch_patterns[:128], # Use a subset for calibration
                hidden_side=hidden_side,
                beta_init=current_beta,
                lr=0.025,
                num_reads=1024,
                tolerance=0.1, 
                use_srt=use_srt
            )
            
            print(f"-> Beta updated: {current_beta:.3f} -> {new_beta:.3f}")
            current_beta = new_beta
            
            print("-> Keeping drifted batch and continuing.")
            # We explicitly DO NOT 'continue' here, so the code falls through to step E
            
        # --- E. Store Valid Batch ---
        batch_v_cpu = batch_v.cpu()
        
        # Split clean/dirty
        clean_batch = batch_v_cpu[batch_mask]
        dirty_batch = batch_v_cpu[~batch_mask]
        
        if len(clean_batch) > 0:
            clean_samples_accum.append(clean_batch)
        if len(dirty_batch) > 0:
            dirty_samples_accum.append(dirty_batch)
            
        samples_collected += current_batch_size

    # 4. Finalize & Save
    print("\n-> Post-processing and saving...")
    
    # Concatenate lists
    final_clean = torch.cat(clean_samples_accum, dim=0) if clean_samples_accum else torch.empty(0)
    final_dirty = torch.cat(dirty_samples_accum, dim=0) if dirty_samples_accum else torch.empty(0)
    
    # Create incidence energy tensor (matches total count of clean + dirty)
    total_count = final_clean.shape[0] + final_dirty.shape[0]
    final_energies = torch.full((total_count, 1), incidence_energy, dtype=torch.float32)
    
    # Define Paths using os.path.join
    filename_base = f"samples_E{int(incidence_energy)}_SRT{int(use_srt)}"
    
    path_clean = os.path.join(save_dir, f"{filename_base}_clean.pt")
    path_dirty = os.path.join(save_dir, f"{filename_base}_dirty.pt")
    path_energies = os.path.join(save_dir, f"{filename_base}_energies.pt")
    
    # Save
    torch.save(final_clean, path_clean)
    torch.save(final_dirty, path_dirty)
    torch.save(final_energies, path_energies)
    
    print(f"Saved {final_clean.shape[0]} clean and {final_dirty.shape[0]} dirty samples.")
    print(f"Outputs:\n  {path_clean}\n  {path_dirty}\n  {path_energies}")
    
    return path_clean, path_dirty, path_energies


def scan_orbits_monte_carlo(
    rbm, raw_sampler, conditioning_sets, left_chains, right_chains,
    target_batch, target_corr_matrix, 
    beta: float,
    num_permutations: int = 20,
    reads_per_perm: int = 500,
    hidden_side: str = 'right',
    use_srt: bool = True,
    start_seed: Union[int, str, None] = None # <-- ADDED ARGUMENT
):
    """
    Scans the provided start_seed plus N random orbits.
    Returns the seed that minimizes Correlation Error.
    """
    print(f"\n--- Phase 2: Monte Carlo Orbit Scan (Baseline + {num_permutations} randoms) ---")
    
    n_vis = len(left_chains)
    n_hid = len(right_chains)
    
    results = []
    
    # 1. Build the list of seeds to test
    # Always test the start_seed first (Identity or specific seed)
    seeds_to_test = [start_seed] 
    
    # Then add N random seeds
    for _ in range(num_permutations):
        seeds_to_test.append(np.random.randint(0, 1000000))

    # 2. Execution Loop
    for i, seed in enumerate(seeds_to_test):
        
        # Labeling for clarity
        if i == 0:
            label = f"Baseline ({seed})"
        else:
            label = f"Random {i}/{num_permutations}"

        # Get Mappings (works for None, "identity", or int)
        v_map, h_map = get_orbit_mappings(seed, n_vis, n_hid)
        
        # Sample
        analysis = sample_expanded_flux_arbitrary(
            rbm=rbm, raw_sampler=raw_sampler,
            conditioning_sets=conditioning_sets,
            left_chains=left_chains, right_chains=right_chains,
            binary_patterns_batch=target_batch[:reads_per_perm],
            hidden_side=hidden_side, beta=beta,
            vis_mapping=v_map, hid_mapping=h_map, perm_seed=seed,
            source=f"scan_{i}_{seed}", use_srt=use_srt
        )
        
        # Analyze
        batch_v, _ = process_analysis_result(analysis, rbm, conditioning_sets)
        
        # Calculate Correlation            
        qpu_corr = get_latent_correlation(batch_v, n_cond=len(conditioning_sets))
        
        # Score (Frobenius Norm)
        error = np.linalg.norm(qpu_corr - target_corr_matrix)
        
        results.append({'seed': seed, 'error': error})
        print(f"   {label}: Seed {seed} | Error: {error:.4f}")

    # 3. Pick Winner
    if not results:
        print("!! All scans failed. Defaulting to start_seed.")
        return start_seed

    # Sort by error (ascending)
    best_result = sorted(results, key=lambda x: x['error'])[0]
    
    # Check if we beat the baseline
    baseline_error = results[0]['error']
    improvement = baseline_error - best_result['error']
    
    print(f"-> Winner: Seed {best_result['seed']} (Error {best_result['error']:.4f})")
    if improvement > 1e-4:
        print(f"   (Improved over baseline by {improvement:.4f})")
    else:
        print(f"   (Baseline was optimal or matched)")
    
    return best_result['seed']


def mass_sample_dwave_orbit_aware(
    incidence_energy: float,
    total_samples_needed: int,
    save_dir: str,
    # System Objects
    engine, rbm, raw_sampler, conditioning_sets, left_chains, right_chains,
    # Orbit Configuration
    default_orbit_seed: int = None, # Initial guess (None = Identity)
    perform_orbit_scan: bool = True,
    scan_permutations: int = 20,    # How many MC steps if scanning
    # Hyperparameters
    n_cond: int = 52,
    initial_beta: float = 3.0,
    batch_size: int = 1024,
    drift_tolerance: float = 1.0,  
    hidden_side: str = 'right',
    device: str = 'cpu',
):
    """
    Orbit-Aware Production Sampling Pipeline.
    
    Flow:
    1. Baselines (Energy & Correlation)
    2. Coarse Beta (Default Orbit)
    3. Orbit Scan -> Find Best Seed
    4. Fine Beta (Best Orbit)
    5. Mass Sampling (with Drift Correction on Best Orbit)
    
    Returns:
        (path_clean, path_dirty, path_energies, best_orbit_seed)
    """
    os.makedirs(save_dir, exist_ok=True)
    n_vis = len(left_chains)
    n_hid = len(right_chains)
    
    print(f"\n=== Mass Sampling (Orbit Aware) Start: {total_samples_needed} samples @ {incidence_energy} MeV ===")

    # --- Phase 0: Baselines ---
    print("-> Phase 0: Establishing Classical Baselines...")
    
    # Generate Target Batch (Classical)
    base_pattern = convert_energy_to_gray(
        incidence_energy=incidence_energy, engine=engine, n_cond=n_cond, 
        num_reads=1, device=device
    )
    baseline_batch = base_pattern.repeat(batch_size, 1)

    # Run Classical Gibbs
    v_base = rbm.sample_v_given_v_clamped(
        clamped_v=baseline_batch, n_clamped=n_cond, gibbs_steps=5000, beta=1.0 
    )
    h_base, _ = rbm._sample_h_given_v(v_base, beta=1.0)
    
    with torch.no_grad():
        baseline_energies = joint_energy(rbm, v_base, h_base)
        target_energy_mean = baseline_energies.mean().item()
    # Calculate Correlation Matrix (for Orbit Scoring)
    target_corr_matrix = get_latent_correlation(v_base, n_cond)
        
    # --- Phase 1: Coarse Beta Estimation ---
    # We need a rough beta to make the orbit scan valid
    print(f"\n-> Phase 1: Coarse Beta Est. (Default Orbit: {default_orbit_seed})...")
    
    vis_def, hid_def = get_orbit_mappings(default_orbit_seed, n_vis, n_hid)
    
    beta_coarse = find_beta_arbitrary(
        rbm, raw_sampler, conditioning_sets, left_chains, right_chains,
        binary_patterns_batch=baseline_batch[:128],
        vis_mapping=vis_def, hid_mapping=hid_def, orbit_seed=default_orbit_seed,
        beta_init=initial_beta,
        use_srt=True
    )
    print(f"Coarse Beta: {beta_coarse:.3f}")

    # --- Phase 2: Orbit Scan ---
    best_orbit_seed = default_orbit_seed
    
    if perform_orbit_scan:
        best_orbit_seed = scan_orbits_monte_carlo(
            rbm, raw_sampler, conditioning_sets, left_chains, right_chains,
            target_batch=baseline_batch,
            target_corr_matrix=target_corr_matrix,
            beta=beta_coarse,
            num_permutations=scan_permutations,
            reads_per_perm=batch_size,
            use_srt=True,
            start_seed=default_orbit_seed
        )
    else:
        print("\n-> Phase 2: Orbit Scan Skipped (Using Default).")

    # Get final mappings
    vis_best, hid_best = get_orbit_mappings(best_orbit_seed, n_vis, n_hid)

    # --- Phase 3: Fine Beta Tuning ---
    # Now that we have the physical layout locked, refine the temperature
    print(f"\n-> Phase 3: Fine Beta Tuning (Best Orbit: {best_orbit_seed})...")
    
    beta_final = find_beta_arbitrary(
        rbm, raw_sampler, conditioning_sets, left_chains, right_chains,
        binary_patterns_batch=baseline_batch[:128],
        vis_mapping=vis_best, hid_mapping=hid_best, orbit_seed=best_orbit_seed,
        beta_init=beta_coarse, # Start from coarse estimate
        use_srt=True
    )
    print(f"Final Beta: {beta_final:.3f}")

    # --- Phase 4: Production Sampling ---
    print(f"\n-> Phase 4: Production Sampling...")
    
    clean_accum = []
    dirty_accum = []
    samples_collected = 0
    current_beta = beta_final
    
    while samples_collected < total_samples_needed:
        remaining = total_samples_needed - samples_collected
        current_batch_size = min(batch_size, remaining)
        
        batch_patterns = base_pattern.repeat(current_batch_size, 1)
        
        # A. Sample
        try:
            analysis_result = sample_expanded_flux_arbitrary(
                rbm=rbm, raw_sampler=raw_sampler,
                conditioning_sets=conditioning_sets,
                left_chains=left_chains, right_chains=right_chains,
                binary_patterns_batch=batch_patterns,
                hidden_side=hidden_side, beta=current_beta,
                vis_mapping=vis_best, hid_mapping=hid_best, perm_seed=best_orbit_seed,
                source=f"prod_E{int(incidence_energy)}", use_srt=use_srt
            )
        except Exception as e:
            print(f"!! QPU Fail: {e}. Retrying...")
            continue
            
        # B. Unpack
        batch_v, batch_h = process_analysis_result(analysis_result, rbm, conditioning_sets)
        batch_mask = analysis_result.clean_mask.cpu()
        
        # C. Drift Check
        with torch.no_grad():
            qpu_energies = joint_energy(rbm, batch_v.to(rbm.device), batch_h.to(rbm.device))
            qpu_mean = qpu_energies.mean().item()
            
        diff = qpu_mean - target_energy_mean
        
        print(f"   Batch {samples_collected}/{total_samples_needed} | Beta: {current_beta:.3f} | "
              f"E_qpu: {qpu_mean:.2f} (Diff: {diff:.2f}) | Clean: {analysis_result.total_clean_fraction:.1%}")

        if abs(diff) > drift_tolerance:
            print(f"!! DRIFT ({diff:.2f} > {drift_tolerance}). Recalibrating...")
            
            # Recalibrate using the BEST orbit
            new_beta = find_beta_arbitrary(
                rbm, raw_sampler, conditioning_sets, left_chains, right_chains,
                binary_patterns_batch=batch_patterns,
                vis_mapping=vis_best, hid_mapping=hid_best, orbit_seed=best_orbit_seed,
                beta_init=current_beta,num_reads=batch_size, use_srt=True
            )
            print(f"   -> Beta updated: {current_beta:.3f} -> {new_beta:.3f}")
            current_beta = new_beta

        # D. Store
        batch_v_cpu = batch_v.cpu()
        clean_batch = batch_v_cpu[batch_mask]
        dirty_batch = batch_v_cpu[~batch_mask]
        
        if len(clean_batch) > 0: clean_accum.append(clean_batch)
        if len(dirty_batch) > 0: dirty_accum.append(dirty_batch)
            
        samples_collected += current_batch_size

    # --- Save & Finish ---
    print("\n-> Saving Results...")
    final_clean = torch.cat(clean_accum, dim=0) if clean_accum else torch.empty(0)
    final_dirty = torch.cat(dirty_accum, dim=0) if dirty_accum else torch.empty(0)
    
    total_count = final_clean.shape[0] + final_dirty.shape[0]
    final_energies = torch.full((total_count, 1), incidence_energy, dtype=torch.float32)
    
    # Filename now includes Orbit Seed
    fname = f"samples_E{int(incidence_energy)}_ORB{best_orbit_seed}"
    path_clean = os.path.join(save_dir, f"{fname}_clean.pt")
    path_dirty = os.path.join(save_dir, f"{fname}_dirty.pt")
    path_energies = os.path.join(save_dir, f"{fname}_energies.pt")
    
    torch.save(final_clean, path_clean)
    torch.save(final_dirty, path_dirty)
    torch.save(final_energies, path_energies)
    
    print(f"Done. Best Orbit: {best_orbit_seed}")
    return path_clean, path_dirty, path_energies, best_orbit_seed



def mass_sample_dwave_multi_energy(
    incidence_energies: list,
    energy_patterns_dict: dict,
    save_dir: str,
    # System Objects
    engine, rbm, raw_sampler, conditioning_sets, left_chains, right_chains,
    # Hyperparameters
    num_srt_batches: int=8,
    n_cond: int = 49,
    beta: float = 3.0,
    batch_size: int = 1024,
    hidden_side: str = 'right',
    device: str = 'cpu',
    orbit_seed = None,
):
    """
    Sampling across multiple incidence energies in a single run.
    Default orbit is used with no drift detection
    Calls are made with multiple batches of SRTs
    """
    os.makedirs(save_dir, exist_ok=True)
    n_vis = len(left_chains)
    n_hid = len(right_chains)
    
    print(f"\n=== Mass Sampling Start: {len(incidence_energies) * batch_size * num_srt_batches} samples across {len(incidence_energies)} energies ===")

    # 1. Prepare Target Batches for Each Energy
    target_batches = {}
    for energy in incidence_energies:
        if energy not in energy_patterns_dict:
            raise ValueError(f"Energy {energy} missing from energy_patterns_dict")
        base_pattern = energy_patterns_dict[energy][0].unsqueeze(0)
        target_batches[energy] = base_pattern.repeat(batch_size, 1)

    # 2. Sample Each Energy Sequentially
    qpu_samples = []
    if hidden_side == 'right':
        n_vis, n_hid = len(left_chains), len(right_chains)
    else:
        n_vis, n_hid = len(right_chains), len(left_chains)

    vis_mapping, hid_mapping = get_orbit_mappings(orbit_seed, n_vis, n_hid)

    for _ in range(num_srt_batches):    
        for energy in incidence_energies:
            print(f"\n-> Sampling for Energy {energy} MeV...")
            
            batch_patterns = target_batches[energy]
            
            try:
                analysis_result = sample_expanded_flux_arbitrary(
                    rbm=rbm, raw_sampler=raw_sampler,
                    conditioning_sets=conditioning_sets,
                    left_chains=left_chains, right_chains=right_chains,
                    binary_patterns_batch=batch_patterns,
                    hidden_side=hidden_side, beta=beta,
                    source=f"prod_E{int(energy)}", use_srt=True, 
                    save_dir=save_dir, 
                    vis_mapping=vis_mapping, hid_mapping=hid_mapping,
                    perm_seed=orbit_seed
                )
            except Exception as e:
                print(f"!! QPU Fail for Energy {energy}: {e}. Skipping...")
                continue
            qpu_samples.append(analysis_result)

    return qpu_samples
