import torch
import numpy as np
import copy
from dwave.system.composites import FixedEmbeddingComposite
from datetime import datetime
import dimod
from typing import Dict, Any

# --- Explicit Local Imports ---
from .physics import (
    rbm_to_logical_ising,
    joint_energy,
    rbm_to_expanded_ising,
    convert_energy_to_binary,
    convert_energy_to_gray,
    calculate_rms_chain_strength,
    compute_node_susceptibilities,
    compute_edge_factors,
    apply_j_scaling
)
from .graphs import (
    get_physical_flux_biases,
    build_logical_to_physical_map,
    build_manual_embedded_ising,
    get_physical_flux_biases_manual,
    build_expanded_embedding,
    build_expanded_embedding_arbitrary,
    get_expanded_flux_biases,
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

from .workflows import (
    sample_expanded_flux_conditioned_rigorous,
    sample_expanded_flux_conditioned_rigorous_srt,
    sample_expanded_flux_arbitrary,
    find_beta_arbitrary
)

from .plots import plot_energy_comparison


def run_chain_break_experiment(
    incidence_energy: float,
    engine,
    rbm,
    qpu_sampler,
    conditioning_sets,
    left_chains,
    right_chains,
    n_cond: int = 53,
    beta: float = 3.0,
    num_reads: int = 1000,
    batch_size: int = 1024,
    hidden_side: str = 'right',
    device: str = 'cpu'
):
    print(f"--- Starting Chain Break Experiment (Energy = {incidence_energy} MeV) ---")
    target_batch_full = convert_energy_to_binary(incidence_energy=incidence_energy, engine=engine, n_cond=n_cond, num_reads=num_reads, device=device)


    # --- 2. QPU Sampling (Batched) ---
    raw_sampler_obj = qpu_sampler.child if hasattr(qpu_sampler, 'child') else qpu_sampler
    
    qpu_v_list = []
    qpu_h_list = []
    clean_mask_list = []

    total_samples = target_batch_full.shape[0]
    
    # Handle batching
    for i in range(0, total_samples, batch_size):
        current_batch_end = min(i + batch_size, total_samples)
        current_batch = target_batch_full[i : current_batch_end]
        
        # QPU Call
        analysis_result = sample_expanded_flux_conditioned_rigorous(
            rbm=rbm,
            raw_sampler=raw_sampler_obj,
            conditioning_sets=conditioning_sets,
            left_chains=left_chains,
            right_chains=right_chains,
            binary_patterns_batch=current_batch,
            hidden_side=hidden_side,
            beta=beta,
            source=f"chain_break_exp_E{int(incidence_energy)}",
        )

        # Unpack V and H (We need H for accurate Joint Energy)
        batch_v, batch_h = process_analysis_result(analysis_result, rbm, conditioning_sets)
        batch_mask = analysis_result.clean_mask
        
        # Move to CPU immediately to free GPU mem
        qpu_v_list.append(batch_v.cpu())
        qpu_h_list.append(batch_h.cpu())
        clean_mask_list.append(batch_mask.cpu())

    # --- 3. Aggregate & Compute QPU Energies ---
    qpu_v = torch.cat(qpu_v_list, dim=0)
    qpu_h = torch.cat(qpu_h_list, dim=0)
    full_clean_mask = torch.cat(clean_mask_list, dim=0).bool()
    
    # Compute Energies ONCE here, using the correct paired (v, h)
    # We move to RBM device for calculation, then back to CPU numpy
    with torch.no_grad():
        all_energies = joint_energy(rbm, qpu_v.to(device), qpu_h.to(device)).cpu().numpy()

    # Split Data (Energies and Samples)
    clean_energies = all_energies[full_clean_mask]
    dirty_energies = all_energies[~full_clean_mask]
    
    clean_samples = qpu_v[full_clean_mask]
    dirty_samples = qpu_v[~full_clean_mask]

    clean_count = len(clean_energies)
    dirty_count = len(dirty_energies)
    clean_frac = clean_count / total_samples if total_samples > 0 else 0

    print(f"QPU Result: {clean_count} Clean | {dirty_count} Dirty ({clean_frac:.1%})")

    # --- 4. Classical Baseline ---
    print("Generating Classical Baseline...")
    v_rbm = rbm.sample_v_given_v_clamped(
        clamped_v=target_batch_full, 
        n_clamped=target_batch_full.shape[1], 
        gibbs_steps=2000, 
        beta=1.0 
    )
    h_rbm, _ = rbm._sample_h_given_v(v_rbm, beta=1.0)
    
    with torch.no_grad():
        classical_energies = joint_energy(rbm, v_rbm, h_rbm).cpu().numpy()

    # --- 5. Pack Results ---
    return {
        "incidence_energy": incidence_energy,
        
        # Energies (for Histogram Plotting)
        "classical": classical_energies,
        "clean": clean_energies,
        "dirty": dirty_energies,
        
        # Raw Samples (for Correlation Matrices)
        "classical_samples": v_rbm.cpu(),
        "clean_samples": clean_samples,
        "dirty_samples": dirty_samples,
        
        "stats": {
            "n_clean": clean_count,
            "n_dirty": dirty_count,
            "n_total": total_samples,
            "pct_clean": clean_frac * 100
        }
    }



def run_spin_gauge_experiment(
    incidence_energy: float,
    engine,
    rbm,
    raw_sampler,  # <--- MUST be the raw DWaveSampler
    conditioning_sets,
    left_chains,
    right_chains,
    n_cond: int = 53,
    beta: float = 3.0,
    num_reads: int = 1000,
    batch_size: int = 1024,
    hidden_side: str = 'right',
    device: str = 'cpu',
    use_srt: bool = True  # <--- New Toggle
):
    print(f"--- Starting Spin Gauge Experiment (Energy = {incidence_energy} MeV, SRT={use_srt}) ---")
    
    # 1. Prepare Target Data
    target_batch_full = convert_energy_to_binary(
        incidence_energy=incidence_energy, 
        engine=engine, 
        n_cond=n_cond, 
        num_reads=num_reads, 
        device=device
    )

    qpu_v_list = []
    qpu_h_list = []
    clean_mask_list = []

    total_samples = target_batch_full.shape[0]
    
    # 2. Loop Through Batches
    for i in range(0, total_samples, batch_size):
        current_batch_end = min(i + batch_size, total_samples)
        current_batch = target_batch_full[i : current_batch_end]
        
        # --- QPU Call ---
        # We pass raw_sampler and use_srt directly.
        # The inner function now handles the masking and wrapping.
        analysis_result = sample_expanded_flux_conditioned_rigorous_srt(
            rbm=rbm,
            raw_sampler=raw_sampler,
            conditioning_sets=conditioning_sets,
            left_chains=left_chains,
            right_chains=right_chains,
            binary_patterns_batch=current_batch,
            hidden_side=hidden_side,
            beta=beta,
            source=f"spin_gauge_exp_E{int(incidence_energy)}",
            use_srt=use_srt  # <--- Passing the flag
        )

        # Unpack Results
        batch_v, batch_h = process_analysis_result(analysis_result, rbm, conditioning_sets)
        batch_mask = analysis_result.clean_mask
        
        qpu_v_list.append(batch_v.cpu())
        qpu_h_list.append(batch_h.cpu())
        clean_mask_list.append(batch_mask.cpu())

    # --- 3. Aggregate Results ---
    qpu_v = torch.cat(qpu_v_list, dim=0)
    qpu_h = torch.cat(qpu_h_list, dim=0)
    full_clean_mask = torch.cat(clean_mask_list, dim=0).bool()
    
    # Compute Energies
    with torch.no_grad():
        all_energies = joint_energy(rbm, qpu_v.to(device), qpu_h.to(device)).cpu().numpy()

    clean_energies = all_energies[full_clean_mask]
    dirty_energies = all_energies[~full_clean_mask]
    
    clean_count = len(clean_energies)
    dirty_count = len(dirty_energies)
    clean_frac = clean_count / total_samples if total_samples > 0 else 0

    print(f"Result: {clean_count} Clean | {dirty_count} Dirty ({clean_frac:.1%})")

    # --- 4. Classical Baseline (Optional, same as before) ---
    print("Generating Classical Baseline...")
    v_rbm = rbm.sample_v_given_v_clamped(
        clamped_v=target_batch_full, 
        n_clamped=target_batch_full.shape[1], 
        gibbs_steps=2000, 
        beta=1.0 
    )
    h_rbm, _ = rbm._sample_h_given_v(v_rbm, beta=1.0)
    with torch.no_grad():
        classical_energies = joint_energy(rbm, v_rbm, h_rbm).cpu().numpy()

    return {
            "incidence_energy": incidence_energy,
            "classical": classical_energies,
            "clean": clean_energies,
            "dirty": dirty_energies,
            "classical_samples": v_rbm.cpu(),
            "clean_samples": qpu_v[full_clean_mask],
            "dirty_samples": qpu_v[~full_clean_mask],
            "stats": {
                "n_clean": clean_count,
                "n_dirty": dirty_count,
                "n_total": total_samples,
                "pct_clean": clean_frac * 100,
                "use_srt": use_srt
            }
        }


def quantize_rbm_simple(source_rbm, n_steps: int = 100):
    """
    Creates a new RBM where weights and biases are snapped to a linear grid.
    
    Args:
        source_rbm: The original continuous RBM.
        n_steps (int): The number of available distinct values (quantization levels)
                       across the range of each parameter tensor.
    """
    discrete_rbm = copy.deepcopy(source_rbm)
    
    def linear_quantize(tensor, steps):
        if steps <= 0: return tensor
        
        # 1. Determine the range
        # We assume a symmetric range [-max, max] to preserve zero-centering
        max_val = torch.abs(tensor).max()
        if max_val == 0: return tensor
        
        # 2. Calculate step size (resolution)
        # Range covers 2 * max_val
        step_size = (2 * max_val) / steps
        
        # 3. Snap values to the grid
        # divide by step -> round -> multiply by step
        return torch.round(tensor / step_size) * step_size

    # Quantize W, vbias, and hbias independently
    # This assumes the DACs for weights and biases adjust to their dynamic range
    discrete_rbm.params["weight_matrix"] = linear_quantize(
        discrete_rbm.params["weight_matrix"], n_steps
    )
    discrete_rbm.params["vbias"] = linear_quantize(
        discrete_rbm.params["vbias"], n_steps
    )
    discrete_rbm.params["hbias"] = linear_quantize(
        discrete_rbm.params["hbias"], n_steps
    )
    
    return discrete_rbm



def run_spin_gauge_experiment_discretized(
    incidence_energy: float,
    engine,
    rbm,
    raw_sampler,
    conditioning_sets,
    left_chains,
    right_chains,
    n_cond: int = 53,
    beta: float = 3.0,
    num_reads: int = 1000,
    batch_size: int = 1024,
    hidden_side: str = 'right',
    device: str = 'cpu',
    use_srt: bool = True,
    n_quantization_steps: int = 100  # <--- New Linear Parameter
):
    print(f"--- Starting Discretized Exp (E={incidence_energy}, Steps={n_quantization_steps}) ---")
    
    # 1. Create the Low-Precision RBM
    print(f"-> Quantizing RBM parameters to {n_quantization_steps} linear steps...")
    rbm_discrete = quantize_rbm_simple(rbm, n_steps=n_quantization_steps)
    
    # 2. Prepare Target Data
    target_batch_full = convert_energy_to_binary(
        incidence_energy=incidence_energy, 
        engine=engine, 
        n_cond=n_cond, 
        num_reads=num_reads, 
        device=device
    )

    qpu_v_list = []
    qpu_h_list = []
    clean_mask_list = []

    total_samples = target_batch_full.shape[0]
    
    # 3. Loop Through Batches
    for i in range(0, total_samples, batch_size):
        current_batch_end = min(i + batch_size, total_samples)
        current_batch = target_batch_full[i : current_batch_end]
        
        # --- QPU Call ---
        # We pass the DISCRETE RBM.
        # The QPU will now receive an embedding based on these "chunky" weights.
        analysis_result = sample_expanded_flux_conditioned_rigorous_srt(
            rbm=rbm_discrete, 
            raw_sampler=raw_sampler,
            conditioning_sets=conditioning_sets,
            left_chains=left_chains,
            right_chains=right_chains,
            binary_patterns_batch=current_batch,
            hidden_side=hidden_side,
            beta=beta,
            source=f"discrete_{n_quantization_steps}steps_E{int(incidence_energy)}",
            use_srt=use_srt
        )

        batch_v, batch_h = process_analysis_result(analysis_result, rbm_discrete, conditioning_sets)
        batch_mask = analysis_result.clean_mask
        
        qpu_v_list.append(batch_v.cpu())
        qpu_h_list.append(batch_h.cpu())
        clean_mask_list.append(batch_mask.cpu())

    # --- 4. Aggregate Results ---
    qpu_v = torch.cat(qpu_v_list, dim=0)
    qpu_h = torch.cat(qpu_h_list, dim=0)
    full_clean_mask = torch.cat(clean_mask_list, dim=0).bool()
    
    # Compute Energies (Using the DISCRETE model for consistency)
    with torch.no_grad():
        all_energies = joint_energy(rbm_discrete, qpu_v.to(device), qpu_h.to(device)).cpu().numpy()

    clean_energies = all_energies[full_clean_mask]
    dirty_energies = all_energies[~full_clean_mask]
    
    clean_count = len(clean_energies)
    dirty_count = len(dirty_energies)
    clean_frac = clean_count / total_samples if total_samples > 0 else 0

    print(f"Result: {clean_count} Clean | {dirty_count} Dirty ({clean_frac:.1%})")

    # --- 5. Classical Baseline ---
    # We sample from the same rbm_discrete to see if the QPU matches 
    # a "perfectly sampled" low-precision model.
    print("Generating Classical Baseline (from Discretized RBM)...")
    v_rbm = rbm_discrete.sample_v_given_v_clamped(
        clamped_v=target_batch_full, 
        n_clamped=target_batch_full.shape[1], 
        gibbs_steps=2000, 
        beta=1.0 
    )
    h_rbm, _ = rbm_discrete._sample_h_given_v(v_rbm, beta=1.0)
    
    with torch.no_grad():
        classical_energies = joint_energy(rbm_discrete, v_rbm, h_rbm).cpu().numpy()

    return {
            "incidence_energy": incidence_energy,
            "classical": classical_energies,
            "clean": clean_energies,
            "dirty": dirty_energies,
            "classical_samples": v_rbm.cpu(),
            "clean_samples": qpu_v[full_clean_mask],
            "dirty_samples": qpu_v[~full_clean_mask],
            "stats": {
                "n_clean": clean_count,
                "n_dirty": dirty_count,
                "n_total": total_samples,
                "pct_clean": clean_frac * 100,
                "use_srt": use_srt,
                "n_quantization_steps": n_quantization_steps
            }
        }

def run_bgs_experiment(
    incidence_energy: float,
    engine,
    rbm,
    raw_sampler,
    conditioning_sets,
    left_chains,
    right_chains,
    n_cond: int = 53,
    beta: float = 3.0,     # Beta for QPU scaling
    num_reads: int = 1000,
    batch_size: int = 1024,
    hidden_side: str = 'right',
    device: str = 'cpu',
    use_srt: bool = True
):
    print(f"--- Starting Refined Spin Gauge Experiment (Energy = {incidence_energy} MeV, SRT={use_srt}) ---")
    
    # 1. Prepare Target Data
    target_batch_full = convert_energy_to_binary(
        incidence_energy=incidence_energy, 
        engine=engine, 
        n_cond=n_cond, 
        num_reads=num_reads, 
        device=device
    )

    # Storage for Raw QPU results
    qpu_v_list = []
    qpu_h_list = []
    clean_mask_list = []
    
    # Storage for Refined results
    refined_v_list = []

    total_samples = target_batch_full.shape[0]
    
    # 2. Loop Through Batches
    for i in range(0, total_samples, batch_size):
        current_batch_end = min(i + batch_size, total_samples)
        current_batch = target_batch_full[i : current_batch_end]
        
        # --- A. QPU Sampling ---
        analysis_result = sample_expanded_flux_conditioned_rigorous_srt(
            rbm=rbm,
            raw_sampler=raw_sampler,
            conditioning_sets=conditioning_sets,
            left_chains=left_chains,
            right_chains=right_chains,
            binary_patterns_batch=current_batch,
            hidden_side=hidden_side,
            beta=beta,
            source=f"spin_gauge_exp_E{int(incidence_energy)}",
            use_srt=use_srt
        )

        # Unpack QPU Results
        batch_v, batch_h = process_analysis_result(analysis_result, rbm, conditioning_sets)
        batch_mask = analysis_result.clean_mask
        
        # Store Raw
        qpu_v_list.append(batch_v.cpu())
        qpu_h_list.append(batch_h.cpu())
        clean_mask_list.append(batch_mask.cpu())

        # --- B. Refinement Step (GPU) ---
        # We take the noisy QPU samples and run 1 step of Gibbs Sampling
        # to "pull" them towards the RBM's learned manifold.
        with torch.no_grad():
            # Move QPU samples to GPU for RBM processing
            v_input = batch_v.to(rbm.device)
            
            # 1. Sample Hidden given QPU Visible
            # Note: We use beta=1.0 for software RBM steps to match training conditions
            h_refined, _ = rbm._sample_h_given_v(v_input, beta=1.0)
            
            # 2. Sample Visible given Refined Hidden
            v_refined, _ = rbm._sample_v_given_h(h_refined, beta=1.0)
            
            # 3. CRITICAL: Re-clamp the conditional bits
            # The Gibbs step might have flipped the energy encoding bits. 
            # We must force them back to the requested energy to ensure fair comparison.
            v_refined[:, :n_cond] = v_input[:, :n_cond]
            
            refined_v_list.append(v_refined.cpu())

    # --- 3. Aggregate Results ---
    qpu_v = torch.cat(qpu_v_list, dim=0)
    qpu_h = torch.cat(qpu_h_list, dim=0)
    refined_v = torch.cat(refined_v_list, dim=0)
    full_clean_mask = torch.cat(clean_mask_list, dim=0).bool()
    
    # Compute Energies (Raw QPU)
    with torch.no_grad():
        raw_energies = joint_energy(rbm, qpu_v.to(rbm.device), qpu_h.to(rbm.device)).cpu().numpy()
        
        # Compute Energies (Refined)
        # We need corresponding hiddens for the refined visibles to get joint energy
        h_refined_final, _ = rbm._sample_h_given_v(refined_v.to(rbm.device), beta=1.0)
        refined_energies = joint_energy(rbm, refined_v.to(rbm.device), h_refined_final).cpu().numpy()

    # Split Raw by Cleanliness
    clean_energies = raw_energies[full_clean_mask]
    dirty_energies = raw_energies[~full_clean_mask]
    
    # Split Refined by Cleanliness (based on the original QPU chain breaks)
    clean_refined_energies = refined_energies[full_clean_mask]
    dirty_refined_energies = refined_energies[~full_clean_mask]
    
    clean_count = len(clean_energies)
    dirty_count = len(dirty_energies)
    clean_frac = clean_count / total_samples if total_samples > 0 else 0

    print(f"Result: {clean_count} Clean | {dirty_count} Dirty ({clean_frac:.1%})")

    # --- 4. Classical Baseline ---
    print("Generating Classical Baseline...")
    v_rbm = rbm.sample_v_given_v_clamped(
        clamped_v=target_batch_full, 
        n_clamped=n_cond, # Use n_cond here to be explicit
        gibbs_steps=2000, 
        beta=1.0 
    )
    h_rbm, _ = rbm._sample_h_given_v(v_rbm, beta=1.0)
    with torch.no_grad():
        classical_energies = joint_energy(rbm, v_rbm, h_rbm).cpu().numpy()

    return {
            "incidence_energy": incidence_energy,
            # Energies
            "classical": classical_energies,
            "clean_raw": clean_energies,
            "dirty_raw": dirty_energies,
            "clean": clean_refined_energies,
            "dirty": dirty_refined_energies,
            
            # Samples (Tensors)
            "classical_samples": v_rbm.cpu(),
            "clean_raw_samples": qpu_v[full_clean_mask],
            "dirty_raw_samples": qpu_v[~full_clean_mask],
            "clean_samples": refined_v[full_clean_mask],
            "dirty_samples": refined_v[~full_clean_mask],
            
            # Stats
            "stats": {
                "n_clean": clean_count,
                "n_dirty": dirty_count,
                "n_total": total_samples,
                "pct_clean": clean_frac * 100,
                "use_srt": use_srt
            }
        }


def run_hamming_cliff_experiment(
    energy_pair: tuple, # e.g. (131071, 131072)
    engine,
    rbm,
    raw_sampler,
    conditioning_sets,
    left_chains,
    right_chains,
    n_cond: int = 53,
    beta: float = 3.0,
    num_reads: int = 10000,
    batch_size: int = 1024,
    hidden_side: str = 'right',
    device: str = 'cpu',
    use_srt: bool = True
):
    results = {}
    
    print(f"--- Starting Hamming Cliff Experiment (Raw QPU): {energy_pair} ---")
    
    for energy_val in energy_pair:
        print(f"\nProcessing Energy: {energy_val} MeV")
        
        # --- 1. Prepare Target Data ---
        target_batch = convert_energy_to_binary(
            incidence_energy=energy_val, 
            engine=engine, 
            n_cond=n_cond, 
            num_reads=num_reads, 
            device=device
        )
        
        # --- 2. Classical RBM Baseline ---
        print(f"  > Sampling Classical RBM...")
        v_rbm = rbm.sample_v_given_v_clamped(
            clamped_v=target_batch, 
            n_clamped=n_cond, 
            gibbs_steps=2000, 
            beta=1.0 
        )
        # Get energies for Classical
        h_rbm, _ = rbm._sample_h_given_v(v_rbm, beta=1.0)
        with torch.no_grad():
            classical_energies = joint_energy(rbm, v_rbm, h_rbm).cpu().numpy()

        # --- 3. QPU Sampling (Raw) ---
        print(f"  > Sampling QPU (No Refinement)...")
        qpu_v_list = []
        total_samples = target_batch.shape[0]

        for i in range(0, total_samples, batch_size):
            current_batch_end = min(i + batch_size, total_samples)
            current_batch = target_batch[i : current_batch_end]
            
            # A. QPU Call
            analysis_result = sample_expanded_flux_conditioned_rigorous_srt(
                rbm=rbm,
                raw_sampler=raw_sampler,
                conditioning_sets=conditioning_sets,
                left_chains=left_chains,
                right_chains=right_chains,
                binary_patterns_batch=current_batch,
                hidden_side=hidden_side,
                beta=beta,
                source=f"hamming_exp_E{int(energy_val)}",
                use_srt=use_srt
            )

            # B. Unpack Raw Samples (No Gibbs steps applied)
            # process_analysis_result handles the majority voting/decoding from chains
            batch_v, _ = process_analysis_result(analysis_result, rbm, conditioning_sets)
            
            qpu_v_list.append(batch_v.cpu())

        # Concatenate QPU results
        qpu_v = torch.cat(qpu_v_list, dim=0)
        
        # Calculate QPU Energies (Raw)
        # We need hiddens to calculate energy, so we sample H given Raw V once.
        # This is strictly for energy calculation, not for refining V.
        h_qpu_calc, _ = rbm._sample_h_given_v(qpu_v.to(rbm.device), beta=1.0)
        with torch.no_grad():
            qpu_energies = joint_energy(rbm, qpu_v.to(rbm.device), h_qpu_calc).cpu().numpy()

        # --- 4. Store Results ---
        results[energy_val] = {
            "classical_samples": v_rbm.cpu(),
            "classical_energies": classical_energies,
            "qpu_samples": qpu_v,
            "qpu_energies": qpu_energies
        }
        
    return results




def run_flux_calibration_experiment(
    incidence_energy: float,       # <--- New Argument
    engine,                        # <--- New Argument (for conversion)
    rbm_structure,     
    raw_sampler,
    conditioning_sets, 
    left_chains, 
    right_chains,
    n_cond: int = 53,              # <--- Default for CaloQVAE
    beta: float = 3.0,
    iterations: int = 50,       
    samples_per_iter: int = 500,
    learning_rate: float = 1e-5,  
    device: str = 'cpu'
):
    print(f"--- Starting Zero-Field Flux Calibration (Energy={incidence_energy} MeV) ---")
    
    # 1. Zero-Field RBM 
    # We keep weights/biases at 0. This ensures that any magnetization we see 
    # is purely due to hardware drift or crosstalk from the conditioning clamps,
    # not the RBM's learned physics.
    zero_rbm = copy.deepcopy(rbm_structure)
    with torch.no_grad():
        zero_rbm.params['weight_matrix'].fill_(0.0)
        zero_rbm.params['vbias'].fill_(0.0)
        zero_rbm.params['hbias'].fill_(0.0)
    
    total_qubits = raw_sampler.properties['num_qubits']
    current_shims = np.zeros(total_qubits)

    # 2. Generate Conditioning Batch
    # Instead of a dummy zero batch, we use the actual bit pattern for this energy.
    # This allows us to compensate for crosstalk specifically induced by these active clamps.
    conditioning_batch = convert_energy_to_binary(
        incidence_energy=incidence_energy, 
        engine=engine, 
        n_cond=n_cond, 
        num_reads=samples_per_iter, 
        device=device
    )

    # 3. Build Masks & Chain Maps
    
    # A. Identify Clamped Qubits (to ignore in updates)
    clamped_phys_qubits = set()
    for c_set in conditioning_sets:
        clamped_phys_qubits.update(c_set)
        
    # B. Build Chain List (to gang updates)
    all_chains_list = []
    for chain_dict in [left_chains, right_chains]:
        if chain_dict:
            for _, phys_list in chain_dict.items():
                if len(phys_list) > 0:
                    all_chains_list.append(phys_list)
    
    history = {'shims': [], 'magnetizations': [], 'std_dev': [], 'rmse': [], 'iterations': iterations}

    for k in range(iterations):
        print(f"Calibration Iteration {k+1}/{iterations}...", end='\r')
        
        # A. Sample (Standard Config: Strong Chains, Active Clamps)
        analysis_result = sample_expanded_flux_conditioned_rigorous_srt(
            rbm=zero_rbm,
            raw_sampler=raw_sampler,
            conditioning_sets=conditioning_sets, 
            left_chains=left_chains,
            right_chains=right_chains,
            binary_patterns_batch=conditioning_batch, # <--- Passing the Energy Pattern
            hidden_side='right',
            beta=beta,
            chain_strength=1.0,     
            source=f"flux_calib_iter_{k}_E{int(incidence_energy)}",
            use_srt=False, 
            additive_flux_offsets=current_shims 
        )
        
        # B. Extract Physical Samples
        response = analysis_result.physical_response
        raw_samples = response.record.sample.astype(float)
        
        if raw_samples.min() >= 0:
             spins = 2.0 * raw_samples - 1.0
        else:
             spins = raw_samples

        # C. Compute Magnetization (Full Map)
        m_physical = np.mean(spins, axis=0) 
        active_indices = np.array(list(response.variables), dtype=int)
        
        full_m_map = np.zeros(total_qubits)
        full_m_map[active_indices] = m_physical

        # D. Update Logic (With Masking)
        update_vector = np.zeros(total_qubits)
        
        # 1. Update Chains (if not clamped)
        for phys_chain in all_chains_list:
            # Check if this chain overlaps with clamps
            if any(q in clamped_phys_qubits for q in phys_chain):
                continue 
                
            chain_mags = [full_m_map[q] for q in phys_chain if q in active_indices]
            if chain_mags:
                avg_chain_mag = np.mean(chain_mags)
                for q in phys_chain:
                    update_vector[q] = avg_chain_mag
                    
        # 2. Update Singletons (if not clamped)
        chain_qubits = set([q for chain in all_chains_list for q in chain])
        for q in active_indices:
            if q not in chain_qubits and q not in clamped_phys_qubits:
                update_vector[q] = full_m_map[q]

        # E. Update Rule (-=)
        current_shims -= learning_rate * update_vector
        
        # Safety Clip
        np.clip(current_shims, -0.01, 0.01, out=current_shims)
        
        # F. History
        history['shims'].append(current_shims.copy())
        
        # For stats, filtering out the clamped ones gives a better view of convergence
        free_indices = [q for q in active_indices if q not in clamped_phys_qubits]
        if free_indices:
            free_mags = full_m_map[free_indices]
            history['magnetizations'].append(free_mags) 
            history['std_dev'].append(np.std(free_mags))
            history['rmse'].append(np.sqrt(np.mean(free_mags**2)))
        else:
            # Fallback if everything is clamped (unlikely)
            history['magnetizations'].append(full_m_map[active_indices])
            history['std_dev'].append(0)
            history['rmse'].append(0)
        
    print(f"\nCalibration Complete. Final Free-Qubit RMSE: {history['rmse'][-1]:.4f}")
    
    return history, current_shims

def run_shim_verification_experiment(
    incidence_energy: float,
    final_shims: np.ndarray,
    engine,
    rbm,
    raw_sampler,
    conditioning_sets,
    left_chains,
    right_chains,
    n_cond: int = 53,
    beta: float = 3.0,
    num_reads: int = 10000,
    batch_size: int = 1024,
    device: str = 'cpu'
):
    """
    Runs a comparative experiment to verify the impact of flux shims.
    
    1. Generates Target Data (Energy encoding).
    2. Generates Classical Baseline (Gibbs sampling).
    3. Runs QPU WITHOUT Shims (Control).
    4. Runs QPU WITH Shims (Experiment).
    5. Returns all samples for correlation analysis.
    """
    print(f"--- Starting Shim Verification Experiment (Energy = {incidence_energy} MeV) ---")
    print(f"--- Beta: {beta} | SRT: True ---")

    # 1. Prepare Target Data
    target_batch_full = convert_energy_to_binary(
        incidence_energy=incidence_energy, 
        engine=engine, 
        n_cond=n_cond, 
        num_reads=num_reads, 
        device=device
    )
    
    total_samples = target_batch_full.shape[0]

    # Containers
    results = {
        "classical": [],
        "no_shim": [],
        "shimmed": [],
        "clean_mask_no_shim": [],
        "clean_mask_shimmed": []
    }

    # 2. Classical Baseline
    print("Generating Classical Baseline...")
    v_rbm = rbm.sample_v_given_v_clamped(
        clamped_v=target_batch_full, 
        n_clamped=n_cond, 
        gibbs_steps=2000, 
        beta=1.0 
    )
    results["classical"] = v_rbm.cpu()

    # 3. QPU Batched Loop
    print(f"Sampling QPU (Total Reads: {total_samples})...")
    
    for i in range(0, total_samples, batch_size):
        current_batch_end = min(i + batch_size, total_samples)
        current_batch = target_batch_full[i : current_batch_end]
        
        # --- A. Control Run: NO SHIMS ---
        # Note: additive_flux_offsets is None
        res_no_shim = sample_expanded_flux_conditioned_rigorous_srt(
            rbm=rbm,
            raw_sampler=raw_sampler,
            conditioning_sets=conditioning_sets,
            left_chains=left_chains,
            right_chains=right_chains,
            binary_patterns_batch=current_batch,
            hidden_side='right',
            beta=beta,
            source=f"verify_NO_shim_E{int(incidence_energy)}",
            use_srt=True,               # Explicitly ON
            additive_flux_offsets=None  # Explicitly None
        )
        
        v_no_shim, _ = process_analysis_result(res_no_shim, rbm, conditioning_sets)
        results["no_shim"].append(v_no_shim.cpu())
        results["clean_mask_no_shim"].append(res_no_shim.clean_mask.cpu())

        # --- B. Experiment Run: WITH SHIMS ---
        # Note: additive_flux_offsets = final_shims
        res_shimmed = sample_expanded_flux_conditioned_rigorous_srt(
            rbm=rbm,
            raw_sampler=raw_sampler,
            conditioning_sets=conditioning_sets,
            left_chains=left_chains,
            right_chains=right_chains,
            binary_patterns_batch=current_batch,
            hidden_side='right',
            beta=beta,
            source=f"verify_WITH_shim_E{int(incidence_energy)}",
            use_srt=True,                   # Explicitly ON
            additive_flux_offsets=final_shims # Passing the calibration result
        )

        v_shimmed, _ = process_analysis_result(res_shimmed, rbm, conditioning_sets)
        results["shimmed"].append(v_shimmed.cpu())
        results["clean_mask_shimmed"].append(res_shimmed.clean_mask.cpu())

    # 4. Aggregate
    final_data = {
        "incidence_energy": incidence_energy,
        "classical_samples": results["classical"],
        "no_shim_samples": torch.cat(results["no_shim"], dim=0),
        "shimmed_samples": torch.cat(results["shimmed"], dim=0),
        "no_shim_mask": torch.cat(results["clean_mask_no_shim"], dim=0).bool(),
        "shimmed_mask": torch.cat(results["clean_mask_shimmed"], dim=0).bool(),
    }
    
    # Simple Yield Stats Output
    ns_clean = final_data["no_shim_mask"].sum().item()
    s_clean = final_data["shimmed_mask"].sum().item()
    print(f"Yield Report | No Shim: {ns_clean}/{total_samples} | Shimmed: {s_clean}/{total_samples}")
    
    return final_data



def run_orbit_verification_experiment(
    incidence_energy: float,
    engine,
    rbm,
    raw_sampler,
    conditioning_sets,
    left_chains,
    right_chains,
    n_cond: int = 53,
    beta: float = 3.0,
    num_reads: int = 10000,
    batch_size: int = 1024,
    orbit_step: int = 16, # The size of the shift per batch
    device: str = 'cpu',
    base_shims = None # Optional: if you want to apply shims + orbits together
):
    """
    Runs a comparative experiment to verify the impact of Embedding Orbit Rotations.
    
    1. Generates Target Data (Energy encoding).
    2. Generates Classical Baseline (Gibbs sampling).
    3. Runs QPU WITHOUT Orbits (Control: shifts = 0).
    4. Runs QPU WITH Orbits (Experiment: shifts += orbit_step per batch).
    5. Returns all samples for correlation analysis.
    """
    print(f"--- Starting Orbit Verification Experiment (Energy = {incidence_energy} MeV) ---")
    print(f"--- Beta: {beta} | Step Size: {orbit_step} ---")

    # 1. Prepare Target Data
    target_batch_full = convert_energy_to_binary(
        incidence_energy=incidence_energy, 
        engine=engine, 
        n_cond=n_cond, 
        num_reads=num_reads, 
        device=device
    )
    
    total_samples = target_batch_full.shape[0]

    # Containers
    results = {
        "classical": [],
        "no_orbit": [],
        "with_orbit": [],
        "clean_mask_no_orbit": [],
        "clean_mask_with_orbit": [],
        "num_orbits_used": 0
    }

    # 2. Classical Baseline
    print("Generating Classical Baseline...")
    v_rbm = rbm.sample_v_given_v_clamped(
        clamped_v=target_batch_full, 
        n_clamped=n_cond, 
        gibbs_steps=2000, 
        beta=1.0 
    )
    results["classical"] = v_rbm.cpu()

    # 3. QPU Batched Loop
    print(f"Sampling QPU (Total Reads: {total_samples})...")
    
    # We iterate through the data once, but effectively run two experiments side-by-side
    # (Or sequentially per batch to keep context similar)
    
    unique_orbits = set()

    for batch_idx, i in enumerate(range(0, total_samples, batch_size)):
        current_batch_end = min(i + batch_size, total_samples)
        current_batch = target_batch_full[i : current_batch_end]
        
        # --- A. Control Run: NO ORBITS ---
        # shifts fixed at 0
        res_no_orbit = sample_expanded_flux_conditioned_rigorous_srt(
            rbm=rbm,
            raw_sampler=raw_sampler,
            conditioning_sets=conditioning_sets,
            left_chains=left_chains,
            right_chains=right_chains,
            binary_patterns_batch=current_batch,
            hidden_side='right',
            beta=beta,
            source=f"verify_NO_orbit_E{int(incidence_energy)}",
            use_srt=True,
            additive_flux_offsets=base_shims,
            vis_shift=0,
            hid_shift=0
        )
        
        v_no_orbit, _ = process_analysis_result(res_no_orbit, rbm, conditioning_sets)
        results["no_orbit"].append(v_no_orbit.cpu())
        results["clean_mask_no_orbit"].append(res_no_orbit.clean_mask.cpu())

        # --- B. Experiment Run: WITH ORBITS ---
        # Calculate shift based on batch index
        current_shift = batch_idx * orbit_step
        unique_orbits.add(current_shift)
        
        res_with_orbit = sample_expanded_flux_conditioned_rigorous_srt(
            rbm=rbm,
            raw_sampler=raw_sampler,
            conditioning_sets=conditioning_sets,
            left_chains=left_chains,
            right_chains=right_chains,
            binary_patterns_batch=current_batch,
            hidden_side='right',
            beta=beta,
            source=f"verify_WITH_orbit_E{int(incidence_energy)}_s{current_shift}",
            use_srt=True,
            additive_flux_offsets=base_shims,
            vis_shift=current_shift,
            hid_shift=current_shift
        )

        v_with_orbit, _ = process_analysis_result(res_with_orbit, rbm, conditioning_sets)
        results["with_orbit"].append(v_with_orbit.cpu())
        results["clean_mask_with_orbit"].append(res_with_orbit.clean_mask.cpu())

    # 4. Aggregate
    final_data = {
        "incidence_energy": incidence_energy,
        "num_orbits_used": len(unique_orbits),
        "classical_samples": results["classical"],
        "no_orbit_samples": torch.cat(results["no_orbit"], dim=0),
        "with_orbit_samples": torch.cat(results["with_orbit"], dim=0),
        "no_orbit_mask": torch.cat(results["clean_mask_no_orbit"], dim=0).bool(),
        "with_orbit_mask": torch.cat(results["clean_mask_with_orbit"], dim=0).bool(),
    }
    
    # Simple Yield Stats Output
    ns_clean = final_data["no_orbit_mask"].sum().item()
    s_clean = final_data["with_orbit_mask"].sum().item()
    print(f"Yield Report | No Orbit: {ns_clean}/{total_samples} | With Orbit: {s_clean}/{total_samples}")
    print(f"Distinct Orbits Used: {final_data['num_orbits_used']}")
    
    return final_data


def run_orbit_sweep(
    incidence_energy: float,
    engine,
    rbm,
    raw_sampler,
    conditioning_sets,
    left_chains,
    right_chains,
    n_cond: int = 53,
    beta: float = 3.0,
    num_reads_per_orbit: int = 1024, # Lower per orbit, but we do many orbits
    orbit_steps: list = range(0, 100, 4), # Scan range
    device: str = 'cpu',
    base_shims = None
):
    """
    Runs a "Sweep" experiment to find the optimal orbit rotation.
    
    1. Generates Classical Baseline.
    2. Iterates through a list of 'orbit_steps'.
    3. For each step, collects a batch of samples.
    4. Computes the Correlation Error Norm immediately for that batch.
    """
    print(f"--- Starting Orbit Sweep (Energy = {incidence_energy} MeV) ---")
    print(f"--- Scanning {len(orbit_steps)} orbits. Reads per orbit: {num_reads_per_orbit} ---")

    # 1. Prepare Target Data (Reusable for all batches)
    # We need enough target data for ONE batch, repeated
    target_batch_template = convert_energy_to_binary(
        incidence_energy=incidence_energy, 
        engine=engine, 
        n_cond=n_cond, 
        num_reads=num_reads_per_orbit, 
        device=device
    )
    
    # 2. Classical Baseline (The Gold Standard)
    print("Generating Classical Baseline...")
    v_classical = rbm.sample_v_given_v_clamped(
        clamped_v=target_batch_template, 
        n_clamped=n_cond, 
        gibbs_steps=2000, 
        beta=1.0 
    )
    
    # Helper: Calculate Latent Correlation Matrix
    def get_latent_corr(samples):
        if isinstance(samples, torch.Tensor):
            samples = samples.float().cpu()
        corr = torch.corrcoef(samples.T).numpy()
        # Slice off conditioning units (n_cond onwards)
        latent = corr[n_cond:, n_cond:]
        np.fill_diagonal(latent, 0) # Ignore self-correlation
        return np.nan_to_num(latent, nan=0.0)

    mat_classical = get_latent_corr(v_classical.cpu())

    # Container for detailed metrics
    sweep_results = {
        "classical_matrix": mat_classical,
        "orbit_metrics": [], # List of dicts: {shift, error, samples}
        "best_shift": None,
        "worst_shift": None
    }

    # 3. Sweep Loop
    print(f"Scanning Orbits...")
    
    for shift in orbit_steps:
        # Run the QPU for this specific rotation
        res = sample_expanded_flux_conditioned_rigorous_srt(
            rbm=rbm,
            raw_sampler=raw_sampler,
            conditioning_sets=conditioning_sets,
            left_chains=left_chains,
            right_chains=right_chains,
            binary_patterns_batch=target_batch_template,
            hidden_side='right',
            beta=beta,
            source=f"sweep_E{int(incidence_energy)}_shift{shift}",
            use_srt=True,
            additive_flux_offsets=base_shims,
            vis_shift=shift, # Apply the rotation
            hid_shift=shift
        )
        
        # Process output
        v_sample, _ = process_analysis_result(res, rbm, conditioning_sets)
        
        # --- IMMEDIATE ANALYSIS ---
        # 1. Calculate Matrix for this specific orbit
        mat_orbit = get_latent_corr(v_sample.cpu())
        
        # 2. Calculate Error Norm (Euclidean distance between matrices)
        diff_mat = mat_orbit - mat_classical
        error_norm = np.linalg.norm(diff_mat)
        
        print(f"  -> Shift {shift:3}: Error = {error_norm:.4f}")
        
        sweep_results["orbit_metrics"].append({
            "shift": shift,
            "error_norm": error_norm,
            "matrix": mat_orbit,
            # Optional: Store samples if you want to re-pool them later
            # "samples": v_sample.cpu() 
        })

    # 4. Identify Best/Worst
    sorted_metrics = sorted(sweep_results["orbit_metrics"], key=lambda x: x['error_norm'])
    
    sweep_results["best_orbit"] = sorted_metrics[0]
    sweep_results["worst_orbit"] = sorted_metrics[-1]
    
    print(f"\n--- Sweep Complete ---")
    print(f"BEST Orbit: Shift {sweep_results['best_orbit']['shift']} (Error: {sweep_results['best_orbit']['error_norm']:.4f})")
    print(f"WORST Orbit: Shift {sweep_results['worst_orbit']['shift']} (Error: {sweep_results['worst_orbit']['error_norm']:.4f})")
    
    return sweep_results




def run_monte_carlo_permutation_sweep(
    cond_vec,
    rbm,
    raw_sampler,
    conditioning_sets,
    left_chains,
    right_chains,
    hidden_side: str,
    n_cond: int = 53,
    beta: float = 3.0,
    num_permutations: int = 20,
    num_reads_per_perm: int = 512,
    srt_batches: int = 8,
    device: str = 'cpu',
    base_shims = None,
    anneal_time: int = None,
    default_start: bool = True,
):
    """
    Given a fixed conditioning vector (energy) of shape (1, n_cond),
    Runs a Monte Carlo Sweep over permutation space with SRT Batches.
    """
    
    # 1. Target Data & Classical Baseline
    
    print("Generating Classical Baseline...")
    v_cl = rbm.sample_v_given_v_clamped(
        clamped_v=cond_vec.repeat(10000, 1), n_clamped=n_cond, gibbs_steps=2000, beta=1.0 
    )
    
    def get_corr(samples):
        if isinstance(samples, torch.Tensor): samples = samples.float().cpu()
        corr = torch.corrcoef(samples.T).numpy()
        latent = corr[n_cond:, n_cond:]
        np.fill_diagonal(latent, 0)
        return np.nan_to_num(latent, nan=0.0)

    mat_classical = get_corr(v_cl.cpu())
    mag_classical = v_cl.float().cpu().mean(dim=0)[n_cond:].numpy()

    # 2. Setup Loop
    sweep_results = {
        "classical_matrix": mat_classical,
        "classical_magnetization": mag_classical,
        "perm_metrics": [],
        "default_orbit": None,
        "best_orbit": None,
        "worst_orbit": None,
        "anneal_time": anneal_time,
    }
    
    n_avail_vis = len(left_chains) 
    n_avail_hid = len(right_chains)
    reads_per_batch = num_reads_per_perm // srt_batches

    # 3. Execution Loop
    for i in range(num_permutations):

        # --- A. Determine Mapping ---
        if i == 0 and default_start:
            run_type = "IDENTITY"
            seed = "DEFAULT"
            p_vis = list(range(n_avail_vis))
            p_hid = list(range(n_avail_hid))
            print(f"[{i+1}/{num_permutations}] Running IDENTITY (SRT Averaged)...")
        else:
            run_type = "RANDOM"
            seed = np.random.randint(0, 1000000)
            rng = np.random.default_rng(seed)
            p_vis = rng.permutation(n_avail_vis).tolist()
            p_hid = rng.permutation(n_avail_hid).tolist()
            print(f"[{i+1}/{num_permutations}] Running Seed {seed}...")

        # --- B. Run Sampler in Batches (SRT Ensemble) ---
        batch_samples = []
        batch_breaks = []
        
        for b in range(srt_batches):
            # 1. Run Sampling (Returns raw 629-column result)
            res = sample_expanded_flux_arbitrary(
                rbm=rbm,
                raw_sampler=raw_sampler,
                conditioning_sets=conditioning_sets,
                left_chains=left_chains,
                right_chains=right_chains,
                binary_patterns_batch=cond_vec.repeat(reads_per_batch, 1),
                hidden_side=hidden_side,
                beta=beta,
                source=f"MC_{run_type}_{seed}_b{b}",
                use_srt=True,
                logical_srt=True,
                chain_strength=2.0,
                flux_drift_compensation=True,
                additive_flux_offsets=base_shims,
                vis_mapping=p_vis,
                hid_mapping=p_hid,
                perm_seed=seed,
                annealing_time=anneal_time,
            )
            
            # 2. Extract RELEVANT Samples immediately
            # This is the critical fix: filter the 629 columns down to 75
            # using your original analysis logic.
            v_sample_batch, _ = process_analysis_result(res, rbm, conditioning_sets)
            
            # 3. Accumulate
            batch_samples.append(v_sample_batch.cpu())
            
            # Accumulate raw break matrix (usually we track breaks on all chains or 
            # just active ones; keeping raw is safest for average stats)
            if res.break_matrix is not None:
                batch_breaks.append(res.break_matrix)
        
        # --- C. Aggregation ---
        # Now full_samples will be (Total_Reads, 75), matching mat_classical
        full_samples = torch.cat(batch_samples, dim=0)
        
        # Calculate Chain Break Fraction
        # Note: This averages breaks over ALL allocated chains (629). 
        # If you only want breaks for the 75 active chains, we'd need to mask this matrix.
        # For general stability tracking, global average is usually fine.
        if len(batch_breaks) > 0:
            full_break_matrix = np.vstack(batch_breaks)
            chain_break_frac = np.mean(full_break_matrix) 
        else:
            chain_break_frac = 0.0

        # --- D. Analysis ---
        mat_perm = get_corr(full_samples)

        # Error Norm (Now shapes match: 75x75 - 75x75)
        error_norm = np.linalg.norm(mat_perm - mat_classical)
        print(f"  -> Error: {error_norm:.4f} | Break Frac: {chain_break_frac:.2%}")

        record = {
            "seed": seed,
            "type": run_type,
            "error_norm": error_norm,
            "chain_break_frac": chain_break_frac,
            "matrix": mat_perm,
            "magnetization": full_samples.float().mean(dim=0)[n_cond:].numpy(),
            "samples": full_samples,
            "vis_mapping": p_vis,
            "hid_mapping": p_hid,
        }
        sweep_results["perm_metrics"].append(record)

        if i == 0:
            sweep_results["default_orbit"] = record

    # 4. Sort and Finalize
    sorted_metrics = sorted(sweep_results["perm_metrics"], key=lambda x: x['error_norm'])

    sweep_results["best_orbit"] = sorted_metrics[0]
    sweep_results["worst_orbit"] = sorted_metrics[-1]

    # 5. Aggregate across all orbits
    all_orbit_samples = torch.cat([r["samples"] for r in sweep_results["perm_metrics"]], dim=0)
    mat_agg = get_corr(all_orbit_samples)
    error_agg = float(np.linalg.norm(mat_agg - mat_classical))
    avg_break_frac_agg = float(np.mean([r["chain_break_frac"] for r in sweep_results["perm_metrics"]]))
    sweep_results["aggregated_orbit"] = {
        "matrix": mat_agg,
        "error_norm": error_agg,
        "break_frac": avg_break_frac_agg,
        "magnetization": all_orbit_samples.float().mean(dim=0)[n_cond:].numpy(),
    }

    # 6. Single-orbit SRT aggregation (orthogonal aggregation axis)
    # Run num_permutations * srt_batches total SRT batches on one fixed mapping so
    # the total read count matches the orbit-aggregation case.
    # default_start=True  → identity mapping (current behaviour)
    # default_start=False → best orbit's mapping (the fairer comparison)
    total_srt_batches = num_permutations * srt_batches
    best_orbit = sweep_results["best_orbit"]
    if default_start:
        srt_ref_vis = list(range(n_avail_vis))
        srt_ref_hid = list(range(n_avail_hid))
        srt_ref_seed = "DEFAULT"
        srt_ref_label = "Default"
        print(f"\nRunning Default orbit with {total_srt_batches} SRT batches (orthogonal aggregation)...")
    else:
        srt_ref_vis = best_orbit["vis_mapping"]
        srt_ref_hid = best_orbit["hid_mapping"]
        srt_ref_seed = best_orbit["seed"]
        srt_ref_label = "Best"
        print(f"\nRunning Best orbit (seed={srt_ref_seed}) with {total_srt_batches} SRT batches (orthogonal aggregation)...")

    srt_batch_samples = []
    srt_batch_breaks = []
    for b in range(total_srt_batches):
        res = sample_expanded_flux_arbitrary(
            rbm=rbm,
            raw_sampler=raw_sampler,
            conditioning_sets=conditioning_sets,
            left_chains=left_chains,
            right_chains=right_chains,
            binary_patterns_batch=cond_vec.repeat(reads_per_batch, 1),
            hidden_side=hidden_side,
            beta=beta,
            source=f"MC_{srt_ref_label.upper()}_SRTAGG_b{b}",
            use_srt=True,
            logical_srt=True,
            chain_strength=2.0,
            flux_drift_compensation=True,
            additive_flux_offsets=base_shims,
            vis_mapping=srt_ref_vis,
            hid_mapping=srt_ref_hid,
            perm_seed=srt_ref_seed,
            annealing_time=anneal_time,
        )
        v_sample_batch, _ = process_analysis_result(res, rbm, conditioning_sets)
        srt_batch_samples.append(v_sample_batch.cpu())
        if res.break_matrix is not None:
            srt_batch_breaks.append(res.break_matrix)

    srt_agg_samples = torch.cat(srt_batch_samples, dim=0)
    mat_srt_agg = get_corr(srt_agg_samples)
    error_srt_agg = float(np.linalg.norm(mat_srt_agg - mat_classical))
    break_frac_srt_agg = (
        float(np.mean(np.vstack(srt_batch_breaks))) if srt_batch_breaks else 0.0
    )
    sweep_results["default_srt_aggregated"] = {
        "matrix": mat_srt_agg,
        "error_norm": error_srt_agg,
        "break_frac": break_frac_srt_agg,
        "num_srt_batches": total_srt_batches,
        "label": srt_ref_label,
        "magnetization": srt_agg_samples.float().mean(dim=0)[n_cond:].numpy(),
    }

    print(f"\n--- MC Sweep Complete ---")
    print(f"First Orbit Error:                      {sweep_results['default_orbit']['error_norm']:.4f}")
    print(f"{srt_ref_label} Orbit (SRT-Aggregated):  {error_srt_agg:.4f}")
    print(f"Best Error:                             {best_orbit['error_norm']:.4f}")
    print(f"Aggregated (Orbits) Error:              {error_agg:.4f}")

    return sweep_results


def run_srt_comparison(
    cond_vec,
    rbm,
    raw_sampler,
    conditioning_sets,
    left_chains,
    right_chains,
    hidden_side: str,
    n_cond: int = 53,
    beta: float = 3.0,
    srt_batches: int = 8,
    device: str = 'cpu',
    base_shims=None,
    rbm_factor : int=10
):
    """
    Compares physical (inter-chain) vs logical (intra-chain) Spin Reversal Transform sampling.

    Physical SRT (SpinReversalTransformComposite) flips individual physical qubits randomly,
    which can disrupt intrachain ferromagnetic alignment and effectively scrambles chains.
    Logical SRT flips entire logical variables (whole chains together), preserving intrachain
    coupling while still randomising effective field directions.

    Both modes are run for srt_batches batches and compared against a classical Gibbs baseline
    via latent-variable correlation matrices.

    Returns a dict suitable for plot_srt_comparison.
    """
    # 1. Classical Baseline
    print("Generating Classical Baseline...")
    v_cl = rbm.sample_v_given_v_clamped(
        clamped_v=cond_vec.repeat(rbm_factor, 1), n_clamped=n_cond, gibbs_steps=2000, beta=1.0
    )

    def get_corr(samples):
        if isinstance(samples, torch.Tensor):
            samples = samples.float().cpu()
        corr = torch.corrcoef(samples.T).numpy()
        latent = corr[n_cond:, n_cond:]
        np.fill_diagonal(latent, 0)
        return np.nan_to_num(latent, nan=0.0)

    mat_classical = get_corr(v_cl.cpu())
    mag_classical = v_cl.float().cpu().mean(dim=0)[n_cond:].numpy()

    results = {
        "classical_matrix": mat_classical,
        "classical_magnetization": mag_classical,
        "physical_srt": {},
        "logical_srt": {},
    }

    # 2. Run both SRT modes
    for mode_key, logical_srt_flag in [("physical_srt", False), ("logical_srt", True)]:
        mode_label = "Logical (Inter-Chain) SRT" if logical_srt_flag else "Physical (Intra-Chain) SRT"
        print(f"\n--- Running {mode_label} ({srt_batches} batches) ---")

        batch_samples = []
        batch_breaks = []

        for b in range(srt_batches):
            if logical_srt_flag:
                chain_strength = 2.0
            else:
                chain_strength = 1.0
            res = sample_expanded_flux_arbitrary(
                rbm=rbm,
                raw_sampler=raw_sampler,
                conditioning_sets=conditioning_sets,
                left_chains=left_chains,
                right_chains=right_chains,
                binary_patterns_batch=cond_vec,
                hidden_side='right',
                beta=beta,
                source=f"SRT_{mode_key}_b{b}",
                use_srt=True,
                additive_flux_offsets=base_shims,
                logical_srt=logical_srt_flag,
                chain_strength=chain_strength,
                flux_drift_compensation=True,
            )

            v_sample_batch, _ = process_analysis_result(res, rbm, conditioning_sets)
            batch_samples.append(v_sample_batch.cpu())

            if res.break_matrix is not None:
                batch_breaks.append(res.break_matrix)

            break_frac = np.mean(res.break_matrix) if res.break_matrix is not None else 0.0
            print(f"  Batch {b + 1}/{srt_batches} | Break Frac: {break_frac:.2%}")

        full_samples = torch.cat(batch_samples, dim=0)
        full_break_frac = np.mean(np.vstack(batch_breaks)) if batch_breaks else 0.0
        mat = get_corr(full_samples)
        error = np.linalg.norm(mat - mat_classical)
        mag = full_samples.float().cpu().mean(dim=0)[n_cond:].numpy()

        results[mode_key] = {
            "matrix": mat,
            "error_norm": error,
            "break_frac": full_break_frac,
            "magnetization": mag,
        }
        print(f"  -> Error Norm: {error:.4f} | Avg Break Frac: {full_break_frac:.2%}")

    print("\n--- SRT Comparison Complete ---")
    print(f"Physical SRT: Error={results['physical_srt']['error_norm']:.4f}  Breaks={results['physical_srt']['break_frac']:.2%}")
    print(f"Logical  SRT: Error={results['logical_srt']['error_norm']:.4f}  Breaks={results['logical_srt']['break_frac']:.2%}")

    return results


def run_orbit_sensitivity_experiment(
    incidence_energy: float,
    engine,
    rbm,
    raw_sampler,
    conditioning_sets,
    left_chains,
    right_chains,
    n_cond: int = 52,
    beta: float = 3.0,
    num_reads: int = 2048,
    num_permutations: int = 10, 
    device: str = 'cpu',
    base_shims = None
):
    """
    Runs an orbit permutation sweep analyzing sensitivity to weight signs 
    (Normal vs Ferromagnetic vs Anti-Ferromagnetic).
    
    Logic:
    1. Generates 3 Classical Baselines (Normal, Ferro, Anti).
    2. For each Orbit (Permutation):
       a. Runs QPU Sampling in 3 modes (Normal, Ferro, Anti) with use_srt=False.
       b. Compares QPU results to the corresponding Classical Baseline.
    """
    print(f"--- Starting Orbit Sensitivity Experiment (Energy = {incidence_energy} MeV) ---")
    
    # --- 1. Generate Target Data & Classical Baselines ---
    # We need a shared target batch for clamping
    target_batch = convert_energy_to_binary(
        incidence_energy=incidence_energy, engine=engine, n_cond=n_cond, 
        num_reads=num_reads, device=device
    )
    
    # Helper to calculate correlation matrix for analysis
    def get_corr(samples):
        if isinstance(samples, torch.Tensor): samples = samples.float().cpu()
        corr = torch.corrcoef(samples.T).numpy()
        latent = corr[n_cond:, n_cond:]
        np.fill_diagonal(latent, 0)
        return np.nan_to_num(latent, nan=0.0)

    print("Generating Classical Baselines...")
    baselines = {}
    
    # 1.1 Normal Classical
    v_cl_norm = rbm.sample_v_given_v_clamped(
        clamped_v=target_batch, n_clamped=n_cond, gibbs_steps=2000, beta=1.0 
    )
    baselines['normal'] = get_corr(v_cl_norm)

    # 1.2 Ferromagnetic Classical (Positive Only)
    v_cl_pos = rbm.sample_v_given_v_clamped_positive_only(
        clamped_v=target_batch, n_clamped=n_cond, gibbs_steps=2000, beta=1.0
    )
    baselines['ferro'] = get_corr(v_cl_pos)

    # 1.3 Anti-Ferromagnetic Classical (Negative Only)
    v_cl_neg = rbm.sample_v_given_v_clamped_negative_only(
        clamped_v=target_batch, n_clamped=n_cond, gibbs_steps=2000, beta=1.0
    )
    baselines['anti'] = get_corr(v_cl_neg)
    
    # --- 2. Setup Results Storage ---
    experiment_data = {
        "baselines": baselines,
        "orbits": [],  # List of dicts, one per permutation
        "best_orbit_index": 0
    }

    n_avail_vis = len(left_chains) 
    n_avail_hid = len(right_chains)
    
    # Store original weights to restore after masking
    original_weights = rbm.params["weight_matrix"].clone()
    
    # --- 3. Permutation Loop ---
    for i in range(num_permutations):
        
        # A. Determine Mapping
        if i == 0:
            run_type = "DEFAULT"
            seed = "DEFAULT"
            p_vis = list(range(n_avail_vis))
            p_hid = list(range(n_avail_hid))
            print(f"[{i+1}/{num_permutations}] Running DEFAULT Orbit...")
        else:
            run_type = "RANDOM"
            seed = np.random.randint(0, 1000000)
            rng = np.random.default_rng(seed)
            p_vis = rng.permutation(n_avail_vis).tolist()
            p_hid = rng.permutation(n_avail_hid).tolist()
            print(f"[{i+1}/{num_permutations}] Running Seed {seed}...")

        orbit_result = {
            "seed": seed,
            "modes": {} # Will contain 'normal', 'ferro', 'anti'
        }

        # B. Run 3 Modes (Normal, Ferro, Anti)
        # We iterate through modes, modifying the RBM weights in-place, then running the sampler
        modes_config = [
            ("normal", None), # No mask
            ("ferro", "positive_only"),
            ("anti", "negative_only")
        ]

        try:
            for mode_name, mask_type in modes_config:
                
                # --- Modify Weights for this mode ---
                if mask_type == "positive_only":
                    # Clamp max=0 preserves negatives, so we want the opposite:
                    # We want to KEEP positives (min=0)
                    rbm.params["weight_matrix"] = original_weights.clamp(min=0.0)
                elif mask_type == "negative_only":
                    # We want to KEEP negatives (max=0)
                    rbm.params["weight_matrix"] = original_weights.clamp(max=0.0)
                else:
                    # Restore original
                    rbm.params["weight_matrix"] = original_weights.clone()

                # --- Run QPU Sampler ---
                # use_srt=False as requested
                res = sample_expanded_flux_arbitrary(
                    rbm=rbm,
                    raw_sampler=raw_sampler,
                    conditioning_sets=conditioning_sets,
                    left_chains=left_chains,
                    right_chains=right_chains,
                    binary_patterns_batch=target_batch, 
                    hidden_side='right',
                    beta=beta,
                    source=f"Orb_{i}_{mode_name}",
                    use_srt=True,  
                    additive_flux_offsets=base_shims,
                    vis_mapping=p_vis,
                    hid_mapping=p_hid,
                    perm_seed=seed
                )

                # --- Process Results ---
                v_sample_batch, _ = process_analysis_result(res, rbm, conditioning_sets)
                
                # Metrics
                mat_qpu = get_corr(v_sample_batch)
                
                # Compare against the SPECIFIC classical baseline for this mode
                err = np.linalg.norm(mat_qpu - baselines[mode_name])
                
                # Chain breaks
                breaks = 0.0
                if res.break_matrix is not None:
                    breaks = np.mean(res.break_matrix)
                
                orbit_result["modes"][mode_name] = {
                    "matrix": mat_qpu,
                    "error": err,
                    "breaks": breaks
                }

        finally:
            # SAFETY: Always restore original weights even if loop crashes
            rbm.params["weight_matrix"] = original_weights

        # Store Orbit Data
        experiment_data["orbits"].append(orbit_result)
        
        # Log progress (using Normal mode as the main tracker)
        norm_err = orbit_result["modes"]["normal"]["error"]
        print(f"  -> Normal Err: {norm_err:.4f}")

    # --- 4. Identify Best Orbit ---
    # "The best orbit is defined to be the one with the lowest error norm for the normal sampling"
    errors = [o["modes"]["normal"]["error"] for o in experiment_data["orbits"]]
    best_idx = np.argmin(errors)
    experiment_data["best_orbit_index"] = best_idx
    
    print(f"\n--- Sweep Complete ---")
    print(f"Best Orbit Index: {best_idx} (Seed: {experiment_data['orbits'][best_idx]['seed']})")
    
    return experiment_data



def run_hamming_cliff_classical_only(
    energy_pair: tuple, # e.g. (131071, 131072)
    engine,
    rbm,
    n_cond: int = 52,
    num_reads: int = 10000,
    gibbs_steps: int = 2000,
    device: str = 'cpu',
    use_gray: bool=True
):
    results = {}
    
    print(f"--- Starting Hamming Cliff Experiment (Classical Only): {energy_pair} ---")
    
    for energy_val in energy_pair:
        print(f"\nProcessing Energy: {energy_val} MeV")
        
        # --- 1. Prepare Target Data ---
        # Assuming convert_energy_to_binary is available in your scope
        if use_gray:
            target_batch = convert_energy_to_gray(
                incidence_energy=energy_val, 
                engine=engine, 
                n_cond=n_cond, 
                num_reads=num_reads, 
                device=device
            )
        else:
            target_batch = convert_energy_to_binary(
                incidence_energy=energy_val, 
                engine=engine, 
                n_cond=n_cond, 
                num_reads=num_reads, 
                device=device
            )
        
        # --- 2. Classical RBM Sampling ---
        print(f"  > Sampling Classical RBM ({gibbs_steps} steps)...")
        
        # Sample V given fixed visible nodes (clamped)
        v_rbm = rbm.sample_v_given_v_clamped(
            clamped_v=target_batch, 
            n_clamped=n_cond, 
            gibbs_steps=gibbs_steps, 
            beta=1.0 
        )
        
        # --- 3. Energy Calculation ---
        # Sample H given the final V to calculate joint energy
        h_rbm, _ = rbm._sample_h_given_v(v_rbm, beta=1.0)
        
        with torch.no_grad():
            # joint_energy assumed available in scope
            energies = joint_energy(rbm, v_rbm, h_rbm).cpu().numpy()

        # --- 4. Store Results ---
        results[energy_val] = {
            "samples": v_rbm.cpu(),
            "energies": energies,
            "use_gray": use_gray
        }

    return results


def run_susceptibility_comparison(
    cond_vec,
    rbm,
    raw_sampler,
    conditioning_sets,
    left_chains,
    right_chains,
    hidden_side,
    n_cond: int,
    beta: float = 3.0,
    srt_batches: int = 8,
    num_reads: int = 64,
    device: str = 'cpu',
    base_shims=None
):
    """
    Compares QPU sampling with uniform spreading vs susceptibility-compensated
    logical-J values.

    Susceptibility compensation rescales programmed couplings inversely to the
    pairwise logical susceptibility chi_ab, homogenising effective inter-chain
    coupling strengths that would otherwise be biased by chain topology.

    Both modes use logical SRT and are run for srt_batches batches, then
    compared against a classical Gibbs baseline via latent-variable correlation
    matrices and per-node magnetisations.

    Returns a dict suitable for plot_susceptibility_comparison.
    """
    import math as _math

    # ── 1. Classical Baseline ──
    print("Generating Classical Baseline...")
    v_cl = rbm.sample_v_given_v_clamped(
        clamped_v=cond_vec, n_clamped=n_cond, gibbs_steps=2000, beta=1.0
    )

    def get_corr(samples):
        if isinstance(samples, torch.Tensor):
            samples = samples.float().cpu()
        corr = torch.corrcoef(samples.T).numpy()
        latent = corr[n_cond:, n_cond:]
        np.fill_diagonal(latent, 0)
        return np.nan_to_num(latent, nan=0.0)

    mat_classical = get_corr(v_cl.cpu())
    mag_classical = v_cl.float().cpu().mean(dim=0)[n_cond:].numpy()

    # ── 2. Compute J distributions for diagnostics ──
    n_vis = rbm.params["vbias"].shape[0]
    exp_embedding, fragment_map = build_expanded_embedding_arbitrary(
        conditioning_sets, left_chains, right_chains,
        num_visible=n_vis, hidden_side=hidden_side
    )
    _, J_exp = rbm_to_expanded_ising(
        rbm, fragment_map, exp_embedding, raw_sampler.adjacency, beta
    )
    print("Computing susceptibility compensation factors...")
    node_chi = compute_node_susceptibilities(exp_embedding, raw_sampler.adjacency)
    edge_factors, edge_counts = compute_edge_factors(
        J_exp, exp_embedding, raw_sampler.adjacency, node_chi
    )
    J_compensated = apply_j_scaling(J_exp, edge_factors, edge_counts)

    if edge_factors:
        chi_vals = np.array(list(edge_factors.values()))
        log_sum = sum(_math.log(x) for x in edge_factors.values())
        N_norm = _math.exp(log_sum / len(edge_factors))
        scale_factors = np.array([
            N_norm * edge_counts.get((u, v), 1) / edge_factors.get((u, v), 1.0)
            for (u, v) in J_exp
        ])
        print(f"  chi_ab range: [{chi_vals.min():.4f}, {chi_vals.max():.4f}]")
        print(f"  Scale factor range: [{scale_factors.min():.4f}, {scale_factors.max():.4f}]")

    results = {
        "classical_matrix": mat_classical,
        "classical_magnetization": mag_classical,
        "uniform": {},
        "compensated": {},
        "j_uniform": np.array(list(J_exp.values())),
        "j_compensated": np.array(list(J_compensated.values())),
    }

    # ── 3. Run Both Modes via high-level sampler ──
    modes = [
        ("uniform",      "Uniform Spreading",         False),
        ("compensated",  "Susceptibility Compensated", True),
    ]

    binary_patterns_batch = cond_vec[0:1].expand(num_reads, -1)

    for mode_key, mode_label, susceptibility in modes:
        print(f"\n--- Running {mode_label} ({srt_batches} batches) ---")

        batch_samples = []
        batch_breaks = []

        for b in range(srt_batches):
            res = sample_expanded_flux_arbitrary(
                rbm=rbm,
                raw_sampler=raw_sampler,
                conditioning_sets=conditioning_sets,
                left_chains=left_chains,
                right_chains=right_chains,
                binary_patterns_batch=binary_patterns_batch,
                hidden_side=hidden_side,
                beta=beta,
                chain_strength=2.0,
                additive_flux_offsets=base_shims,
                use_srt=True,
                logical_srt=True,
                use_susceptibility=susceptibility,
            )

            v_sample_batch, _ = process_analysis_result(res, rbm, conditioning_sets)
            batch_samples.append(v_sample_batch.cpu())

            if res.break_matrix is not None:
                batch_breaks.append(res.break_matrix)

            break_frac = np.mean(res.break_matrix) if res.break_matrix is not None else 0.0
            print(f"  Batch {b + 1}/{srt_batches} | Break Frac: {break_frac:.2%}")

        full_samples = torch.cat(batch_samples, dim=0)
        full_break_frac = np.mean(np.vstack(batch_breaks)) if batch_breaks else 0.0
        mat = get_corr(full_samples)
        error = np.linalg.norm(mat - mat_classical)
        mag = full_samples.float().cpu().mean(dim=0)[n_cond:].numpy()

        results[mode_key] = {
            "matrix": mat,
            "error_norm": error,
            "break_frac": full_break_frac,
            "magnetization": mag,
        }
        print(f"  -> Error Norm: {error:.4f} | Avg Break Frac: {full_break_frac:.2%}")

    print("\n--- Susceptibility Comparison Complete ---")
    print(f"Uniform:      Error={results['uniform']['error_norm']:.4f}  Breaks={results['uniform']['break_frac']:.2%}")
    print(f"Compensated:  Error={results['compensated']['error_norm']:.4f}  Breaks={results['compensated']['break_frac']:.2%}")

    return results


def run_flux_drift_comparison(
    cond_vec,
    rbm,
    raw_sampler,
    conditioning_sets,
    left_chains,
    right_chains,
    hidden_side,
    n_cond: int,
    beta: float = 3.0,
    srt_batches: int = 8,
    num_reads: int = 64,
    device: str = 'cpu',
    base_shims=None
):
    """
    Compares QPU sampling with flux_drift_compensation=False vs True.

    Both modes use logical SRT and are run for srt_batches batches.  Results
    are compared against a classical Gibbs baseline via latent-variable
    correlation matrices and per-node magnetisations.

    Returns a dict suitable for plot_flux_drift_comparison.
    """
    import math as _math

    # ── 1. Classical Baseline ──
    print("Generating Classical Baseline...")
    v_cl = rbm.sample_v_given_v_clamped(
        clamped_v=cond_vec.repeat(100, 1), n_clamped=n_cond, gibbs_steps=2000, beta=1.0
    )

    def get_corr(samples):
        if isinstance(samples, torch.Tensor):
            samples = samples.float().cpu()
        corr = torch.corrcoef(samples.T).numpy()
        latent = corr[n_cond:, n_cond:]
        np.fill_diagonal(latent, 0)
        return np.nan_to_num(latent, nan=0.0)

    mat_classical = get_corr(v_cl.cpu())
    mag_classical = v_cl.float().cpu().mean(dim=0)[n_cond:].numpy()

    results = {
        "classical_matrix": mat_classical,
        "classical_magnetization": mag_classical,
        "no_fdc": {},
        "fdc": {},
    }

    # ── 2. Run Both Modes ──
    modes = [
        ("no_fdc", "FDC Off (baseline)", False),
        ("fdc",    "FDC On",             True),
    ]

    binary_patterns_batch = cond_vec[0:1].expand(num_reads, -1)

    for mode_key, mode_label, fdc in modes:
        print(f"\n--- Running {mode_label} ({srt_batches} batches) ---")

        batch_samples = []
        batch_breaks = []

        for b in range(srt_batches):
            res = sample_expanded_flux_arbitrary(
                rbm=rbm,
                raw_sampler=raw_sampler,
                conditioning_sets=conditioning_sets,
                left_chains=left_chains,
                right_chains=right_chains,
                binary_patterns_batch=binary_patterns_batch,
                hidden_side=hidden_side,
                beta=beta,
                chain_strength=2.0,
                additive_flux_offsets=base_shims,
                use_srt=True,
                logical_srt=True,
                use_susceptibility=False,
                flux_drift_compensation=fdc,
            )

            v_sample_batch, _ = process_analysis_result(res, rbm, conditioning_sets)
            batch_samples.append(v_sample_batch.cpu())

            if res.break_matrix is not None:
                batch_breaks.append(res.break_matrix)

            break_frac = np.mean(res.break_matrix) if res.break_matrix is not None else 0.0
            print(f"  Batch {b + 1}/{srt_batches} | Break Frac: {break_frac:.2%}")

        full_samples = torch.cat(batch_samples, dim=0)
        full_break_frac = np.mean(np.vstack(batch_breaks)) if batch_breaks else 0.0
        mat = get_corr(full_samples)
        error = np.linalg.norm(mat - mat_classical)
        mag = full_samples.float().cpu().mean(dim=0)[n_cond:].numpy()

        results[mode_key] = {
            "matrix": mat,
            "error_norm": error,
            "break_frac": full_break_frac,
            "magnetization": mag,
        }
        print(f"  -> Error Norm: {error:.4f} | Avg Break Frac: {full_break_frac:.2%}")

    print("\n--- Flux Drift Compensation Comparison Complete ---")
    print(f"FDC Off:  Error={results['no_fdc']['error_norm']:.4f}  Breaks={results['no_fdc']['break_frac']:.2%}")
    print(f"FDC On:   Error={results['fdc']['error_norm']:.4f}  Breaks={results['fdc']['break_frac']:.2%}")

    return results


def run_mc_permutation_sweep_single(
    cond_vec,
    rbm,
    raw_sampler,
    conditioning_sets,
    left_chains,
    right_chains,
    hidden_side: str,
    beta: float = 3.0,
    num_permutations: int = 20,
    rbm_gibbs_steps: int = 2000,
    rbm_factor: int = 1,
    device: str = 'cpu',
    base_shims=None,
    print_interval: int = 50,
):
    """
    Monte Carlo permutation sweep for heterogeneous conditioning vectors.

    Unlike run_monte_carlo_permutation_sweep, cond_vec has shape (num_reads, n_cond)
    where each row is a distinct conditioning pattern.  Each pattern is sampled
    independently with a single QPU read (no SRT batches), matching the
    find_beta_single single_batch=True strategy.

    Args:
        cond_vec: Tensor of shape (num_reads, n_cond) — one conditioning pattern per row.
        rbm_factor: Repeat the heterogeneous batch this many times for the classical baseline.
        print_interval: Print progress every this many patterns per permutation (0 = off).

    Returns:
        sweep_results dict compatible with plot_permutation_sweep_analysis.
    """
    n_cond = cond_vec.shape[1]
    num_reads = cond_vec.shape[0]

    n_avail_vis = len(left_chains)
    n_avail_hid = len(right_chains)

    # --- 1. Classical Baseline ---
    print("Generating Classical Baseline...")
    rbm_target = cond_vec.repeat(rbm_factor, 1) if rbm_factor > 1 else cond_vec
    v_cl = rbm.sample_v_given_v_clamped(
        clamped_v=rbm_target, n_clamped=n_cond, gibbs_steps=rbm_gibbs_steps, beta=1.0
    )

    def get_corr(samples):
        if isinstance(samples, torch.Tensor):
            samples = samples.float().cpu()
        corr = torch.corrcoef(samples.T).numpy()
        latent = corr[n_cond:, n_cond:]
        np.fill_diagonal(latent, 0)
        return np.nan_to_num(latent, nan=0.0)

    mat_classical = get_corr(v_cl.cpu())
    mag_classical = v_cl.float().cpu().mean(dim=0)[n_cond:].numpy()

    # --- 2. Setup ---
    sweep_results = {
        "classical_matrix": mat_classical,
        "classical_magnetization": mag_classical,
        "perm_metrics": [],
        "default_orbit": None,
        "best_orbit": None,
        "worst_orbit": None,
    }

    # --- 3. Permutation Loop ---
    for i in range(num_permutations):

        # A. Determine mapping
        if i == 0:
            run_type = "IDENTITY"
            seed = "DEFAULT"
            p_vis = list(range(n_avail_vis))
            p_hid = list(range(n_avail_hid))
            print(f"[{i+1}/{num_permutations}] Running IDENTITY (Single-Read Per Pattern)...")
        else:
            run_type = "RANDOM"
            seed = np.random.randint(0, 1000000)
            rng = np.random.default_rng(seed)
            p_vis = rng.permutation(n_avail_vis).tolist()
            p_hid = rng.permutation(n_avail_hid).tolist()
            print(f"[{i+1}/{num_permutations}] Running Seed {seed}...")

        # B. Sample each pattern independently (heterogeneous single-read)
        all_samples = []
        all_break_fracs = []

        for j, pattern in enumerate(cond_vec):
            pattern_batch = pattern.unsqueeze(0)  # (1, n_cond)

            res = sample_expanded_flux_arbitrary(
                rbm=rbm,
                raw_sampler=raw_sampler,
                conditioning_sets=conditioning_sets,
                left_chains=left_chains,
                right_chains=right_chains,
                binary_patterns_batch=pattern_batch,
                hidden_side=hidden_side,
                beta=beta,
                source=f"MC_{run_type}_{seed}_p{j}",
                use_srt=True,
                logical_srt=True,
                chain_strength=2.0,
                flux_drift_compensation=True,
                additive_flux_offsets=base_shims,
                vis_mapping=p_vis,
                hid_mapping=p_hid,
                perm_seed=seed,
            )

            v_sample, _ = process_analysis_result(res, rbm, conditioning_sets)
            all_samples.append(v_sample.cpu())

            if res.break_matrix is not None:
                all_break_fracs.append(np.mean(res.break_matrix))

            if print_interval > 0 and (j + 1) % print_interval == 0:
                print(f"  [{j+1}/{num_reads}] patterns sampled...")

        # C. Aggregate
        full_samples = torch.cat(all_samples, dim=0)
        chain_break_frac = float(np.mean(all_break_fracs)) if all_break_fracs else 0.0

        # D. Analysis
        mat_perm = get_corr(full_samples)
        error_norm = np.linalg.norm(mat_perm - mat_classical)
        print(f"  -> Error: {error_norm:.4f} | Break Frac: {chain_break_frac:.2%}")

        record = {
            "seed": seed,
            "type": run_type,
            "error_norm": error_norm,
            "chain_break_frac": chain_break_frac,
            "matrix": mat_perm,
            "magnetization": full_samples.float().mean(dim=0)[n_cond:].numpy(),
            "samples": full_samples,
        }
        sweep_results["perm_metrics"].append(record)

        if i == 0:
            sweep_results["default_orbit"] = record

    # --- 4. Sort and Finalize ---
    sorted_metrics = sorted(sweep_results["perm_metrics"], key=lambda x: x["error_norm"])
    sweep_results["best_orbit"] = sorted_metrics[0]
    sweep_results["worst_orbit"] = sorted_metrics[-1]

    # --- 5. Aggregate across all orbits ---
    all_orbit_samples = torch.cat([r["samples"] for r in sweep_results["perm_metrics"]], dim=0)
    mat_agg = get_corr(all_orbit_samples)
    error_agg = float(np.linalg.norm(mat_agg - mat_classical))
    avg_break_frac_agg = float(np.mean([r["chain_break_frac"] for r in sweep_results["perm_metrics"]]))
    sweep_results["aggregated_orbit"] = {
        "matrix": mat_agg,
        "error_norm": error_agg,
        "break_frac": avg_break_frac_agg,
        "magnetization": all_orbit_samples.float().mean(dim=0)[n_cond:].numpy(),
    }

    print(f"\n--- MC Sweep Complete ---")
    print(f"Default Error:    {sweep_results['default_orbit']['error_norm']:.4f}")
    print(f"Best Error:       {sweep_results['best_orbit']['error_norm']:.4f}")
    print(f"Aggregated Error: {error_agg:.4f}")

    return sweep_results


def run_srt_aggregation_comparison(
    cond_vec,
    rbm,
    raw_sampler,
    conditioning_sets,
    left_chains,
    right_chains,
    hidden_side: str,
    n_cond: int,
    beta: float = 3.0,
    srt_batches: int = 8,
    num_reads: int = 64,
    base_shims=None,
    rbm_factor: int = 100,
):
    """
    Compare two SRT aggregation strategies using the same total read budget.

    Both strategies run exactly ``srt_batches`` SRT batches of ``num_reads``
    each (total reads = srt_batches * num_reads):

    - **averaged**: all batch samples are pooled together; the correlation
      matrix is computed on the full aggregate.
    - **best_srt**: each batch is scored individually against the classical
      baseline; only the best-scoring batch is reported.

    Results are compared against a classical Gibbs baseline via latent-variable
    correlation matrices and per-node magnetisations.

    Returns a dict suitable for ``plot_srt_aggregation_comparison``.
    """
    # ── 1. Classical Baseline ──
    print("Generating Classical Baseline...")
    v_cl = rbm.sample_v_given_v_clamped(
        clamped_v=cond_vec[0:1].expand(rbm_factor, -1),
        n_clamped=n_cond,
        gibbs_steps=2000,
        beta=1.0,
    )

    def get_corr(samples):
        if isinstance(samples, torch.Tensor):
            samples = samples.float().cpu()
        corr = torch.corrcoef(samples.T).numpy()
        latent = corr[n_cond:, n_cond:]
        np.fill_diagonal(latent, 0)
        return np.nan_to_num(latent, nan=0.0)

    mat_classical = get_corr(v_cl.cpu())
    mag_classical = v_cl.float().cpu().mean(dim=0)[n_cond:].numpy()

    binary_patterns_batch = cond_vec[0:1].expand(num_reads, -1)

    # ── 2. Run srt_batches batches, recording each one individually ──
    print(f"\nRunning {srt_batches} SRT batches ({num_reads} reads each)...")
    batch_samples_list = []
    per_batch = []

    for b in range(srt_batches):
        res = sample_expanded_flux_arbitrary(
            rbm=rbm,
            raw_sampler=raw_sampler,
            conditioning_sets=conditioning_sets,
            left_chains=left_chains,
            right_chains=right_chains,
            binary_patterns_batch=binary_patterns_batch,
            hidden_side=hidden_side,
            beta=beta,
            chain_strength=2.0,
            additive_flux_offsets=base_shims,
            use_srt=True,
            logical_srt=True,
            flux_drift_compensation=True,
            source=f"SRT_agg_b{b}",
        )

        v_sample, _ = process_analysis_result(res, rbm, conditioning_sets)
        v_sample = v_sample.cpu()
        batch_samples_list.append(v_sample)

        break_frac = float(np.mean(res.break_matrix)) if res.break_matrix is not None else 0.0
        mat_b = get_corr(v_sample)
        error_b = float(np.linalg.norm(mat_b - mat_classical))
        per_batch.append({
            "matrix": mat_b,
            "error_norm": error_b,
            "break_frac": break_frac,
            "magnetization": v_sample.float().mean(dim=0)[n_cond:].numpy(),
        })
        print(f"  Batch {b + 1}/{srt_batches} | Error: {error_b:.4f} | Break Frac: {break_frac:.2%}")

    # ── 3. Averaged strategy ──
    full_samples = torch.cat(batch_samples_list, dim=0)
    avg_break_frac = float(np.mean([pb["break_frac"] for pb in per_batch]))
    mat_avg = get_corr(full_samples)
    error_avg = float(np.linalg.norm(mat_avg - mat_classical))
    mag_avg = full_samples.float().mean(dim=0)[n_cond:].numpy()

    # ── 4. Best-SRT strategy ──
    best_idx = int(np.argmin([pb["error_norm"] for pb in per_batch]))
    best = per_batch[best_idx]

    print(f"\n--- SRT Aggregation Comparison ---")
    print(f"Averaged ({srt_batches} batches): Error={error_avg:.4f}  Breaks={avg_break_frac:.2%}")
    print(f"Best SRT (batch {best_idx}):      Error={best['error_norm']:.4f}  Breaks={best['break_frac']:.2%}")
    per_batch_strs = [f"{pb['error_norm']:.4f}" for pb in per_batch]
    print(f"Per-batch errors: {per_batch_strs}")

    return {
        "classical_matrix": mat_classical,
        "classical_magnetization": mag_classical,
        "averaged": {
            "matrix": mat_avg,
            "error_norm": error_avg,
            "break_frac": avg_break_frac,
            "magnetization": mag_avg,
        },
        "best_srt": {
            "matrix": best["matrix"],
            "error_norm": best["error_norm"],
            "break_frac": best["break_frac"],
            "magnetization": best["magnetization"],
            "best_batch_idx": best_idx,
        },
        "per_batch": per_batch,
        "srt_batches": srt_batches,
        "num_reads": num_reads,
    }


def run_annealing_time_sweep(
    cond_vec,
    rbm,
    raw_sampler,
    conditioning_sets,
    left_chains,
    right_chains,
    hidden_side: str,
    n_cond: int,
    annealing_times: list,
    beta_init: float = 3.0,
    lr: float = 0.01,
    num_epochs: int = 15,
    tolerance: float = 0.1,
    num_reads: int = 1024,
    rbm_gibbs_steps: int = 5000,
    vis_mapping: list = None,
    hid_mapping: list = None,
    base_shims=None,
    orbit_seed: int = None,
    flux_drift_compensation: bool = True,
    num_srt_batches: int = 4,
):
    """
    Sweeps over QPU annealing times. For each time:
      1. Calls find_beta_arbitrary until beta converges.
      2. Draws num_srt_batches quality samples at the converged beta, each with a
         freshly shuffled orbit (vis_mapping / hid_mapping), then pools them.
      3. Computes a latent correlation matrix and error norm vs a classical baseline.

    Returns a dict suitable for plot_annealing_time_sweep.
    """
    primary_batch = cond_vec[0:1].expand(num_reads, -1)

    # --- 0. Classical Baseline (computed once) ---
    print("Generating Classical Baseline...")
    v_cl = rbm.sample_v_given_v_clamped(
        clamped_v=primary_batch, n_clamped=n_cond,
        gibbs_steps=rbm_gibbs_steps, beta=1.0,
    )

    def get_corr(samples):
        if isinstance(samples, torch.Tensor):
            samples = samples.float().cpu()
        corr = torch.corrcoef(samples.T).numpy()
        latent = corr[n_cond:, n_cond:]
        np.fill_diagonal(latent, 0)
        return np.nan_to_num(latent, nan=0.0)

    mat_classical = get_corr(v_cl)
    mag_classical = v_cl.float().cpu().mean(dim=0)[n_cond:].numpy()

    # --- 1. Per-time sweep ---
    per_time_results = {}

    for at in annealing_times:
        print(f"\n=== Annealing Time: {at} µs ===")

        beta, beta_hist, rbm_e_hist, qpu_e_hist = find_beta_arbitrary(
            rbm=rbm,
            raw_sampler=raw_sampler,
            conditioning_sets=conditioning_sets,
            left_chains=left_chains,
            right_chains=right_chains,
            binary_patterns_batch=primary_batch,
            vis_mapping=vis_mapping,
            hid_mapping=hid_mapping,
            hidden_side=hidden_side,
            orbit_seed=orbit_seed,
            num_reads=num_reads,
            rbm_gibbs_steps=rbm_gibbs_steps,
            beta_init=beta_init,
            lr=lr,
            num_epochs=num_epochs,
            tolerance=tolerance,
            use_srt=True,
            logical_srt=True,
            flux_drift_compensation=flux_drift_compensation,
            annealing_time=at,
            rbm_factor=10,
        )

        # Final quality sample: num_srt_batches batches, vis/hid_mapping=None lets
        # the sampler apply its default shuffling each call
        print(f"  Final quality sample: {num_srt_batches} batches at beta={beta:.4f}...")
        all_samples = []
        all_breaks = []

        for b in range(num_srt_batches):
            res = sample_expanded_flux_arbitrary(
                rbm=rbm,
                raw_sampler=raw_sampler,
                conditioning_sets=conditioning_sets,
                left_chains=left_chains,
                right_chains=right_chains,
                binary_patterns_batch=primary_batch,
                hidden_side=hidden_side,
                beta=beta,
                vis_mapping=None,
                hid_mapping=None,
                additive_flux_offsets=base_shims,
                use_srt=True,
                logical_srt=True,
                flux_drift_compensation=flux_drift_compensation,
                annealing_time=at,
                source=f"at_sweep_final_t{at}_b{b}",
                chain_strength=2.0,
            )
            v_qpu, _ = process_analysis_result(res, rbm, conditioning_sets)
            all_samples.append(v_qpu.cpu())
            if res.break_matrix is not None:
                all_breaks.append(float(np.mean(res.break_matrix)))
            print(f"    Batch {b+1}/{num_srt_batches} done.")

        pooled = torch.cat(all_samples, dim=0)
        mat = get_corr(pooled)
        error_norm = float(np.linalg.norm(mat - mat_classical))
        break_frac = float(np.mean(all_breaks)) if all_breaks else 0.0
        magnetization = pooled.float().mean(dim=0)[n_cond:].numpy()
        print(f"  Error norm vs classical: {error_norm:.4f}  Break frac: {break_frac:.2%}")

        per_time_results[at] = {
            "effective_beta": beta,
            "beta_hist": beta_hist,
            "rbm_energy_hist": rbm_e_hist,
            "qpu_energy_hist": qpu_e_hist,
            "matrix": mat,
            "error_norm": error_norm,
            "break_frac": break_frac,
            "magnetization": magnetization,
        }

    print("\n--- Annealing Time Sweep Complete ---")
    for at, r in per_time_results.items():
        print(f"  t={at} µs: beta_eff={r['effective_beta']:.4f}  error={r['error_norm']:.4f}  breaks={r['break_frac']:.2%}")

    return {
        "annealing_times": annealing_times,
        "per_time_results": per_time_results,
        "classical_matrix": mat_classical,
        "classical_magnetization": mag_classical,
        "beta_init": beta_init,
        "num_reads": num_reads,
        "num_srt_batches": num_srt_batches,
    }


def run_chain_break_histogram(
    cond_vec,
    rbm,
    raw_sampler,
    conditioning_sets,
    left_chains,
    right_chains,
    hidden_side: str,
    beta: float = 3.0,
    num_reads: int = 1024,
    srt_batches: int = 8,
    base_shims=None,
    cutoffs: list = None,
    rbm_gibbs_steps: int = 2000,
    rbm_factor: int = 10,
):
    """
    Collects per-read chain break fractions across srt_batches QPU calls,
    each using a freshly shuffled orbit (random vis/hid mapping).

    Also computes a classical Gibbs baseline and, for each value in cutoffs,
    evaluates the latent correlation matrix keeping only the cutoff fraction of
    reads with the lowest chain break rates.

    Returns a dict suitable for plot_chain_break_histogram.
    """
    if cutoffs is None:
        cutoffs = [0.25, 0.5, 0.75, 1.0]

    n_cond = len(conditioning_sets)
    reads_per_batch = num_reads // srt_batches
    remainder = num_reads % srt_batches

    if hidden_side == 'right':
        n_vis_orbit, n_hid_orbit = len(left_chains), len(right_chains)
    else:
        n_vis_orbit, n_hid_orbit = len(right_chains), len(left_chains)

    print(f"--- Chain Break Histogram ({num_reads} reads, {srt_batches} batches) ---")

    all_per_read_fracs = []
    all_samples = []
    batch_mean_fracs = []
    batch_seeds = []

    for b in range(srt_batches):
        batch_reads = reads_per_batch + (1 if b < remainder else 0)
        seed = int(np.random.randint(0, 1_000_000))
        batch_seeds.append(seed)
        vis_mapping, hid_mapping = get_orbit_mappings(seed, n_vis_orbit, n_hid_orbit)

        binary_patterns_batch = cond_vec[0:1].expand(batch_reads, -1)

        res = sample_expanded_flux_arbitrary(
            rbm=rbm,
            raw_sampler=raw_sampler,
            conditioning_sets=conditioning_sets,
            left_chains=left_chains,
            right_chains=right_chains,
            binary_patterns_batch=binary_patterns_batch,
            hidden_side=hidden_side,
            beta=beta,
            chain_strength=2.0,
            use_srt=True,
            logical_srt=True,
            flux_drift_compensation=True,
            additive_flux_offsets=base_shims,
            vis_mapping=vis_mapping,
            hid_mapping=hid_mapping,
            perm_seed=seed,
            source=f"break_hist_b{b}",
        )

        if res.break_matrix is not None:
            per_read = np.mean(res.break_matrix, axis=1)
        else:
            per_read = np.zeros(batch_reads)

        v_sample, _ = process_analysis_result(res, rbm, conditioning_sets)
        all_samples.append(v_sample.cpu())
        all_per_read_fracs.append(per_read)
        batch_mean = float(np.mean(per_read))
        batch_mean_fracs.append(batch_mean)
        print(f"  Batch {b+1}/{srt_batches} | seed={seed} | mean break frac: {batch_mean:.2%}")

    all_fracs = np.concatenate(all_per_read_fracs)
    all_v = torch.cat(all_samples, dim=0)

    # --- Classical Baseline ---
    print("Generating Classical Baseline...")
    n_rbm_reads = num_reads * rbm_factor
    v_cl = rbm.sample_v_given_v_clamped(
        clamped_v=cond_vec[0:1].expand(n_rbm_reads, -1),
        n_clamped=n_cond,
        gibbs_steps=rbm_gibbs_steps,
        beta=1.0,
    )

    def get_corr(samples):
        if isinstance(samples, torch.Tensor):
            samples = samples.float().cpu()
        corr = torch.corrcoef(samples.T).numpy()
        latent = corr[n_cond:, n_cond:]
        np.fill_diagonal(latent, 0)
        return np.nan_to_num(latent, nan=0.0)

    mat_classical = get_corr(v_cl)

    # --- Cutoff Analysis ---
    # Sort indices ascending by break fraction (lowest breaks first)
    sorted_idx = np.argsort(all_fracs)
    cutoff_results = {}
    print("\nCutoff Analysis:")
    for c in sorted(cutoffs):
        k = max(1, int(np.ceil(c * len(all_fracs))))
        idx = sorted_idx[:k]
        mat_cut = get_corr(all_v[idx])
        err = float(np.linalg.norm(mat_cut - mat_classical))
        max_break = float(all_fracs[sorted_idx[k - 1]])
        cutoff_results[c] = {
            "matrix": mat_cut,
            "diff_matrix": mat_cut - mat_classical,
            "error_norm": err,
            "n_samples": k,
            "max_break_frac": max_break,
        }
        print(f"  {c:.0%}: {k} samples | max break kept: {max_break:.2%} | error: {err:.4f}")

    overall_mean = float(np.mean(all_fracs))
    pct_clean = float(np.mean(all_fracs == 0.0)) * 100
    print(f"\nOverall mean break frac: {overall_mean:.2%} | Clean reads: {pct_clean:.1f}%")

    return {
        "per_read_break_fracs": all_fracs,
        "batch_mean_fracs": batch_mean_fracs,
        "batch_seeds": batch_seeds,
        "srt_batches": srt_batches,
        "num_reads": len(all_fracs),
        "reads_per_batch": reads_per_batch,
        "overall_mean": overall_mean,
        "pct_clean": pct_clean,
        "classical_matrix": mat_classical,
        "cutoffs": sorted(cutoffs),
        "cutoff_results": cutoff_results,
        "n_cond": n_cond,
    }


def run_energy_histogram(
    cond_vec,
    rbm,
    raw_sampler,
    conditioning_sets,
    left_chains,
    right_chains,
    hidden_side: str,
    beta: float = 3.0,
    num_reads: int = 1024,
    srt_batches: int = 8,
    base_shims=None,
    cutoffs: list = None,
    rbm_gibbs_steps: int = 2000,
    rbm_factor: int = 10,
):
    """
    Collects per-sample joint energies across srt_batches QPU calls,
    each using a freshly shuffled orbit (random vis/hid mapping).

    Also computes a classical Gibbs baseline. For each value in cutoffs,
    evaluates the latent correlation matrix keeping only the lowest-energy
    (most probable) cutoff fraction of QPU samples.

    Returns a dict suitable for plot_energy_histogram.
    """
    if cutoffs is None:
        cutoffs = [0.25, 0.5, 0.75, 1.0]

    n_cond = len(conditioning_sets)
    reads_per_batch = num_reads // srt_batches
    remainder = num_reads % srt_batches

    if hidden_side == 'right':
        n_vis_orbit, n_hid_orbit = len(left_chains), len(right_chains)
    else:
        n_vis_orbit, n_hid_orbit = len(right_chains), len(left_chains)

    print(f"--- Energy Histogram ({num_reads} reads, {srt_batches} batches) ---")

    all_energies = []
    all_samples = []
    batch_seeds = []

    for b in range(srt_batches):
        batch_reads = reads_per_batch + (1 if b < remainder else 0)
        seed = int(np.random.randint(0, 1_000_000))
        batch_seeds.append(seed)
        vis_mapping, hid_mapping = get_orbit_mappings(seed, n_vis_orbit, n_hid_orbit)

        binary_patterns_batch = cond_vec[0:1].expand(batch_reads, -1)

        res = sample_expanded_flux_arbitrary(
            rbm=rbm,
            raw_sampler=raw_sampler,
            conditioning_sets=conditioning_sets,
            left_chains=left_chains,
            right_chains=right_chains,
            binary_patterns_batch=binary_patterns_batch,
            hidden_side=hidden_side,
            beta=beta,
            chain_strength=2.0,
            use_srt=True,
            logical_srt=True,
            flux_drift_compensation=True,
            additive_flux_offsets=base_shims,
            vis_mapping=vis_mapping,
            hid_mapping=hid_mapping,
            perm_seed=seed,
            source=f"energy_hist_b{b}",
        )

        v_sample, h_sample = process_analysis_result(res, rbm, conditioning_sets)
        with torch.no_grad():
            e_batch = joint_energy(rbm, v_sample, h_sample).cpu().numpy()

        all_samples.append(v_sample.cpu())
        all_energies.append(e_batch)
        print(f"  Batch {b+1}/{srt_batches} | seed={seed} | mean energy: {e_batch.mean():.4f}")

    qpu_energies = np.concatenate(all_energies)
    all_v = torch.cat(all_samples, dim=0)

    # --- Classical Baseline ---
    print("Generating Classical Baseline...")
    n_rbm_reads = num_reads * rbm_factor
    v_cl = rbm.sample_v_given_v_clamped(
        clamped_v=cond_vec[0:1].expand(n_rbm_reads, -1),
        n_clamped=n_cond,
        gibbs_steps=rbm_gibbs_steps,
        beta=1.0,
    )
    h_cl, _ = rbm._sample_h_given_v(v_cl, beta=1.0)
    with torch.no_grad():
        classical_energies = joint_energy(rbm, v_cl, h_cl).cpu().numpy()

    def get_corr(samples):
        if isinstance(samples, torch.Tensor):
            samples = samples.float().cpu()
        corr = torch.corrcoef(samples.T).numpy()
        latent = corr[n_cond:, n_cond:]
        np.fill_diagonal(latent, 0)
        return np.nan_to_num(latent, nan=0.0)

    mat_classical = get_corr(v_cl)

    # --- Cutoff Analysis ---
    # Sort ascending by energy (lowest = most probable first)
    sorted_idx = np.argsort(qpu_energies)
    cutoff_results = {}
    print("\nCutoff Analysis:")
    for c in sorted(cutoffs):
        k = max(1, int(np.ceil(c * len(qpu_energies))))
        idx = sorted_idx[:k]
        mat_cut = get_corr(all_v[idx])
        err = float(np.linalg.norm(mat_cut - mat_classical))
        max_energy = float(qpu_energies[sorted_idx[k - 1]])
        cutoff_results[c] = {
            "matrix": mat_cut,
            "diff_matrix": mat_cut - mat_classical,
            "error_norm": err,
            "n_samples": k,
            "max_energy": max_energy,
        }
        print(f"  {c:.0%}: {k} samples | max energy kept: {max_energy:.4f} | error: {err:.4f}")

    print(f"\nQPU mean energy: {qpu_energies.mean():.4f} ± {qpu_energies.std():.4f}")
    print(f"Classical mean energy: {classical_energies.mean():.4f} ± {classical_energies.std():.4f}")

    return {
        "qpu_energies": qpu_energies,
        "classical_energies": classical_energies,
        "batch_seeds": batch_seeds,
        "srt_batches": srt_batches,
        "num_reads": len(qpu_energies),
        "reads_per_batch": reads_per_batch,
        "classical_matrix": mat_classical,
        "cutoffs": sorted(cutoffs),
        "cutoff_results": cutoff_results,
        "n_cond": n_cond,
    }


def run_chain_break_structure_analysis(
    cond_vec,
    rbm,
    raw_sampler,
    conditioning_sets,
    left_chains,
    right_chains,
    hidden_side: str,
    beta: float = 3.0,
    num_reads: int = 1024,
    srt_batches: int = 8,
    base_shims=None,
    rbm_gibbs_steps: int = 2000,
    rbm_factor: int = 10,
):
    """
    Tracks which specific chains break across SRT batches with shuffled orbits,
    distinguishing between physical structure (same hardware qubits break) and
    logical structure (same logical variables break regardless of physical placement).

    Each batch uses a different random orbit (vis/hid mapping), so a logical variable
    lands on a different physical chain each time.  Comparing the coefficient of
    variation (CV = std/mean) of break rates in the logical domain vs the physical
    domain reveals the source of structure:

    - Physical CV >> Logical CV  → breaks cluster on specific hardware qubits
    - Logical  CV >> Physical CV → breaks cluster on specific model variables
    - Both low                   → no consistent structure (random hardware noise)

    Returns a dict suitable for plot_chain_break_structure.
    """
    n_cond = len(conditioning_sets)
    n_vis = rbm.params["vbias"].shape[0]   # total visible = n_cond + n_standard_vis
    n_hid = rbm.params["hbias"].shape[0]
    n_standard_vis = n_vis - n_cond

    reads_per_batch = num_reads // srt_batches
    remainder = num_reads % srt_batches

    if hidden_side == 'right':
        n_vis_orbit, n_hid_orbit = len(left_chains), len(right_chains)
    else:
        n_vis_orbit, n_hid_orbit = len(right_chains), len(left_chains)

    print(f"--- Chain Break Structure Analysis ({num_reads} reads, {srt_batches} batches) ---")
    print(f"    Logical vis: {n_standard_vis}, Logical hid: {n_hid}")
    print(f"    Physical slots: {n_vis_orbit} vis, {n_hid_orbit} hid")

    # Per-batch accumulators shape (srt_batches, n_vars)
    batch_vis_logical  = np.zeros((srt_batches, n_standard_vis))
    batch_hid_logical  = np.zeros((srt_batches, n_hid))
    batch_vis_physical = np.zeros((srt_batches, n_vis_orbit))
    batch_hid_physical = np.zeros((srt_batches, n_hid_orbit))

    batch_seeds       = []
    batch_vis_mappings = []
    batch_hid_mappings = []
    all_per_read_fracs = []

    for b in range(srt_batches):
        batch_reads = reads_per_batch + (1 if b < remainder else 0)
        seed = int(np.random.randint(0, 1_000_000))
        batch_seeds.append(seed)
        vis_mapping, hid_mapping = get_orbit_mappings(seed, n_vis_orbit, n_hid_orbit)
        batch_vis_mappings.append(vis_mapping)
        batch_hid_mappings.append(hid_mapping)

        binary_patterns_batch = cond_vec[0:1].expand(batch_reads, -1)

        res = sample_expanded_flux_arbitrary(
            rbm=rbm,
            raw_sampler=raw_sampler,
            conditioning_sets=conditioning_sets,
            left_chains=left_chains,
            right_chains=right_chains,
            binary_patterns_batch=binary_patterns_batch,
            hidden_side=hidden_side,
            beta=beta,
            chain_strength=2.0,
            use_srt=True,
            logical_srt=True,
            flux_drift_compensation=True,
            additive_flux_offsets=base_shims,
            vis_mapping=vis_mapping,
            hid_mapping=hid_mapping,
            perm_seed=seed,
            source=f"break_struct_b{b}",
        )

        if res.break_matrix is None:
            print(f"  Batch {b+1}/{srt_batches}: no break data")
            continue

        label_to_col = {lbl: i for i, lbl in enumerate(res.variable_labels)}

        vis_logical_rates  = np.zeros(n_standard_vis)
        hid_logical_rates  = np.zeros(n_hid)
        vis_physical_rates = np.zeros(n_vis_orbit)
        hid_physical_rates = np.zeros(n_hid_orbit)

        # Visible logical vars: integer IDs n_cond .. n_vis-1
        for v_idx in range(n_standard_vis):
            logical_id = n_cond + v_idx
            col = label_to_col.get(logical_id)
            if col is None:
                continue
            rate = float(np.mean(res.break_matrix[:, col]))
            vis_logical_rates[v_idx] = rate
            phys_slot = vis_mapping[v_idx]
            vis_physical_rates[phys_slot] = rate

        # Hidden logical vars: integer IDs n_vis .. n_vis+n_hid-1
        for h_idx in range(n_hid):
            logical_id = n_vis + h_idx
            col = label_to_col.get(logical_id)
            if col is None:
                continue
            rate = float(np.mean(res.break_matrix[:, col]))
            hid_logical_rates[h_idx] = rate
            phys_slot = hid_mapping[h_idx]
            hid_physical_rates[phys_slot] = rate

        batch_vis_logical[b]  = vis_logical_rates
        batch_hid_logical[b]  = hid_logical_rates
        batch_vis_physical[b] = vis_physical_rates
        batch_hid_physical[b] = hid_physical_rates

        per_read = np.mean(res.break_matrix, axis=1)
        all_per_read_fracs.append(per_read)

        print(f"  Batch {b+1}/{srt_batches} | seed={seed} | mean break: {float(np.mean(per_read)):.2%}")

    # Aggregate mean and std across batches
    vis_logical_mean  = batch_vis_logical.mean(axis=0)
    hid_logical_mean  = batch_hid_logical.mean(axis=0)
    vis_logical_std   = batch_vis_logical.std(axis=0)
    hid_logical_std   = batch_hid_logical.std(axis=0)

    # Physical: only average over batches where the slot was used (non-zero)
    # Since each slot appears in exactly n_standard_vis/n_vis_orbit * srt_batches batches on average,
    # a simple mean is appropriate; unused-slot zeros pull it down uniformly.
    vis_physical_mean = batch_vis_physical.mean(axis=0)
    hid_physical_mean = batch_hid_physical.mean(axis=0)
    vis_physical_std  = batch_vis_physical.std(axis=0)
    hid_physical_std  = batch_hid_physical.std(axis=0)

    # Coefficient of variation: how heterogeneous are the break rates?
    eps = 1e-8
    vis_log_cv  = float(vis_logical_mean.std()  / (vis_logical_mean.mean()  + eps))
    hid_log_cv  = float(hid_logical_mean.std()  / (hid_logical_mean.mean()  + eps))
    vis_phys_cv = float(vis_physical_mean.std() / (vis_physical_mean.mean() + eps))
    hid_phys_cv = float(hid_physical_mean.std() / (hid_physical_mean.mean() + eps))

    logical_cv  = (vis_log_cv  + hid_log_cv)  / 2
    physical_cv = (vis_phys_cv + hid_phys_cv) / 2

    ratio = physical_cv / (logical_cv + eps)
    if ratio > 1.5:
        interpretation = "physical"
    elif ratio < 0.67:
        interpretation = "logical"
    else:
        interpretation = "mixed/random"

    print(f"\nStructure Analysis:")
    print(f"  Logical  CV: {logical_cv:.3f}")
    print(f"  Physical CV: {physical_cv:.3f}")
    print(f"  Interpretation: {interpretation}")

    # Classical baseline for context
    print("Generating Classical Baseline...")
    v_cl = rbm.sample_v_given_v_clamped(
        clamped_v=cond_vec[0:1].expand(num_reads * rbm_factor, -1),
        n_clamped=n_cond,
        gibbs_steps=rbm_gibbs_steps,
        beta=1.0,
    )

    def get_corr(samples):
        if isinstance(samples, torch.Tensor):
            samples = samples.float().cpu()
        corr = torch.corrcoef(samples.T).numpy()
        latent = corr[n_cond:, n_cond:]
        np.fill_diagonal(latent, 0)
        return np.nan_to_num(latent, nan=0.0)

    mat_classical = get_corr(v_cl)

    all_fracs = np.concatenate(all_per_read_fracs) if all_per_read_fracs else np.array([])

    return {
        "n_cond": n_cond,
        "n_vis": n_vis,
        "n_hid": n_hid,
        "n_standard_vis": n_standard_vis,
        "n_vis_orbit": n_vis_orbit,
        "n_hid_orbit": n_hid_orbit,
        "num_reads": num_reads,
        "srt_batches": srt_batches,

        # Per-batch break rate matrices (srt_batches × n_vars)
        "batch_vis_logical":  batch_vis_logical,
        "batch_hid_logical":  batch_hid_logical,
        "batch_vis_physical": batch_vis_physical,
        "batch_hid_physical": batch_hid_physical,

        # Aggregated means and stds
        "vis_logical_mean":  vis_logical_mean,
        "hid_logical_mean":  hid_logical_mean,
        "vis_logical_std":   vis_logical_std,
        "hid_logical_std":   hid_logical_std,
        "vis_physical_mean": vis_physical_mean,
        "hid_physical_mean": hid_physical_mean,
        "vis_physical_std":  vis_physical_std,
        "hid_physical_std":  hid_physical_std,

        # Structure metrics
        "logical_cv":    logical_cv,
        "physical_cv":   physical_cv,
        "vis_log_cv":    vis_log_cv,
        "hid_log_cv":    hid_log_cv,
        "vis_phys_cv":   vis_phys_cv,
        "hid_phys_cv":   hid_phys_cv,
        "interpretation": interpretation,

        # Overall break stats
        "per_read_break_fracs": all_fracs,
        "overall_mean": float(np.mean(all_fracs)) if len(all_fracs) else 0.0,

        "classical_matrix": mat_classical,
        "batch_seeds": batch_seeds,
        "batch_vis_mappings": batch_vis_mappings,
        "batch_hid_mappings": batch_hid_mappings,
    }


def _build_pause_schedule(pause_point: float, pause_length: float, anneal_time: float):
    """
    Builds a piecewise-linear anneal schedule with a pause at s = pause_point.
    anneal_time is the forward anneal time (µs) excluding the pause; max slope
    1/µs is preserved. Returns None when the pause point is at an endpoint.
    """
    sp = float(pause_point)
    tp = float(pause_length)
    ta = float(anneal_time)
    if sp <= 0.0 or sp >= 1.0 or tp <= 0.0:
        return None
    t1 = ta * sp
    return [(0.0, 0.0), (t1, sp), (t1 + tp, sp), (ta + tp, 1.0)]


def run_pause_sweep(
    cond_vec,
    rbm,
    raw_sampler,
    conditioning_sets,
    left_chains,
    right_chains,
    hidden_side: str,
    n_cond: int,
    pause_points: list,
    pause_length: float = 100.0,
    anneal_time: float = 20.0,
    rbm_scale: float = 2.0,
    num_reads: int = 1024,
    sweep_batches: int = 2,
    base_shims=None,
    flux_drift_compensation: bool = True,
    beta_init: float = 2.0,
    lr: float = 0.01,
    num_epochs: int = 15,
    tolerance: float = 0.5,
    rbm_gibbs_steps: int = 5000,
    rbm_factor: int = 10,
    num_srt_batches: int = 8,
    include_baseline: bool = True,
):
    """
    Pause-anneal sweep (Marshall et al., 2018). Scales the RBM by rbm_scale
    (passed as the Ising beta so h and J are kept in the well-embedded range),
    sweeps QPU anneals with a pause at each s_p in pause_points, and picks the
    pause point that yields the lowest mean joint energy. Then runs
    find_beta_arbitrary at that optimal pause to get a new effective beta, and
    collects a final batched sample for latent correlation.
    """
    primary_batch = cond_vec[0:1].expand(num_reads, -1)

    if hidden_side == 'right':
        n_vis_orbit, n_hid_orbit = len(left_chains), len(right_chains)
    else:
        n_vis_orbit, n_hid_orbit = len(right_chains), len(left_chains)

    # --- 0. Classical Baseline ---
    print("Generating Classical Baseline...")
    n_rbm_reads = num_reads * rbm_factor
    v_cl = rbm.sample_v_given_v_clamped(
        clamped_v=cond_vec[0:1].expand(n_rbm_reads, -1),
        n_clamped=n_cond,
        gibbs_steps=rbm_gibbs_steps,
        beta=1.0,
    )
    h_cl, _ = rbm._sample_h_given_v(v_cl, beta=1.0)
    with torch.no_grad():
        classical_energies = joint_energy(rbm, v_cl, h_cl).cpu().numpy()

    def get_corr(samples):
        if isinstance(samples, torch.Tensor):
            samples = samples.float().cpu()
        corr = torch.corrcoef(samples.T).numpy()
        latent = corr[n_cond:, n_cond:]
        np.fill_diagonal(latent, 0)
        return np.nan_to_num(latent, nan=0.0)

    mat_classical = get_corr(v_cl)

    # --- 1. Sweep pause points at fixed scaled beta ---
    per_pause_results = {}
    sweep_points = list(pause_points)
    if include_baseline:
        sweep_points = [None] + sweep_points

    for sp in sweep_points:
        schedule = _build_pause_schedule(sp, pause_length, anneal_time) if sp is not None else None
        label = "no_pause" if sp is None else f"sp={sp:.3f}"
        print(f"\n=== Pause {label} | beta_scale={rbm_scale} | t_a={anneal_time}µs t_p={pause_length}µs ===")

        batch_energies = []
        batch_samples = []
        batch_break_fracs = []
        for b in range(sweep_batches):
            seed = int(np.random.randint(0, 1_000_000))
            vis_mapping, hid_mapping = get_orbit_mappings(seed, n_vis_orbit, n_hid_orbit)
            res = sample_expanded_flux_arbitrary(
                rbm=rbm,
                raw_sampler=raw_sampler,
                conditioning_sets=conditioning_sets,
                left_chains=left_chains,
                right_chains=right_chains,
                binary_patterns_batch=primary_batch,
                hidden_side=hidden_side,
                beta=rbm_scale,
                chain_strength=2.0,
                use_srt=True,
                logical_srt=True,
                flux_drift_compensation=flux_drift_compensation,
                additive_flux_offsets=base_shims,
                vis_mapping=vis_mapping,
                hid_mapping=hid_mapping,
                perm_seed=seed,
                anneal_schedule=schedule,
                annealing_time=None if schedule is not None else int(max(1, round(anneal_time))),
                source=f"pause_sweep_{label}_b{b}",
            )
            v_s, h_s = process_analysis_result(res, rbm, conditioning_sets)
            with torch.no_grad():
                e_b = joint_energy(rbm, v_s, h_s).cpu().numpy()
            batch_energies.append(e_b)
            batch_samples.append(v_s.cpu())
            batch_break_fracs.append(float(np.mean(res.break_matrix)) if res.break_matrix is not None else 0.0)
            print(f"  Batch {b+1}/{sweep_batches} | mean E={e_b.mean():.4f} | breaks={batch_break_fracs[-1]:.2%}")

        energies = np.concatenate(batch_energies)
        v_pool = torch.cat(batch_samples, dim=0)
        mat = get_corr(v_pool)
        error_norm = float(np.linalg.norm(mat - mat_classical))

        per_pause_results[sp] = {
            "pause_point": sp,
            "anneal_schedule": schedule,
            "energies": energies,
            "mean_energy": float(energies.mean()),
            "std_energy": float(energies.std()),
            "matrix": mat,
            "error_norm": error_norm,
            "mean_break_frac": float(np.mean(batch_break_fracs)),
        }
        print(f"  Pooled mean E={energies.mean():.4f} ± {energies.std():.4f} | corr err={error_norm:.4f}")

    # --- 2. Choose optimal pause point (lowest mean energy among real pauses) ---
    pause_only = {sp: r for sp, r in per_pause_results.items() if sp is not None}
    if not pause_only:
        raise ValueError("pause_points is empty; nothing to optimize.")
    optimal_sp = min(pause_only, key=lambda s: pause_only[s]["mean_energy"])
    optimal_schedule = per_pause_results[optimal_sp]["anneal_schedule"]
    print(f"\n*** Optimal pause point: s_p = {optimal_sp:.3f} "
          f"(mean E = {per_pause_results[optimal_sp]['mean_energy']:.4f}) ***")

    mag_classical = v_cl.float().cpu().mean(dim=0)[n_cond:].numpy()

    def find_beta_and_sample(schedule, annealing_time_val, tag):
        # B. β-finding at this (schedule or anneal time)
        seed = int(np.random.randint(0, 1_000_000))
        vis, hid = get_orbit_mappings(seed, n_vis_orbit, n_hid_orbit)
        print(f"\n[{tag}] Finding beta (orbit seed={seed})...")
        beta, bh, reh, qeh = find_beta_arbitrary(
            rbm=rbm,
            raw_sampler=raw_sampler,
            conditioning_sets=conditioning_sets,
            left_chains=left_chains,
            right_chains=right_chains,
            binary_patterns_batch=primary_batch,
            vis_mapping=vis,
            hid_mapping=hid,
            hidden_side=hidden_side,
            orbit_seed=seed,
            num_reads=num_reads,
            rbm_gibbs_steps=rbm_gibbs_steps,
            beta_init=beta_init,
            lr=lr,
            num_epochs=num_epochs,
            tolerance=tolerance,
            use_srt=True,
            logical_srt=True,
            flux_drift_compensation=flux_drift_compensation,
            anneal_schedule=schedule,
            annealing_time=annealing_time_val,
            rbm_factor=rbm_factor,
        )

        # C. num_srt_batches quality sample, each with a fresh orbit
        print(f"[{tag}] Final quality sample: {num_srt_batches} batches at beta={beta:.4f}...")
        f_samples, f_energies, f_breaks = [], [], []
        for b in range(num_srt_batches):
            res = sample_expanded_flux_arbitrary(
                rbm=rbm,
                raw_sampler=raw_sampler,
                conditioning_sets=conditioning_sets,
                left_chains=left_chains,
                right_chains=right_chains,
                binary_patterns_batch=primary_batch,
                hidden_side=hidden_side,
                beta=beta,
                chain_strength=2.0,
                use_srt=True,
                logical_srt=True,
                flux_drift_compensation=flux_drift_compensation,
                additive_flux_offsets=base_shims,
                vis_mapping=None,
                hid_mapping=None,
                anneal_schedule=schedule,
                annealing_time=annealing_time_val,
                source=f"pause_sweep_final_{tag}_b{b}",
            )
            v_f, h_f = process_analysis_result(res, rbm, conditioning_sets)
            with torch.no_grad():
                f_energies.append(joint_energy(rbm, v_f, h_f).cpu().numpy())
            f_samples.append(v_f.cpu())
            f_breaks.append(float(np.mean(res.break_matrix)) if res.break_matrix is not None else 0.0)
            print(f"  [{tag}] batch {b+1}/{num_srt_batches} done.")

        v_pool = torch.cat(f_samples, dim=0)
        energies = np.concatenate(f_energies)
        mat = get_corr(v_pool)
        err = float(np.linalg.norm(mat - mat_classical))
        mag = v_pool.float().mean(dim=0)[n_cond:].numpy()
        return {
            "beta": beta,
            "beta_hist": bh,
            "rbm_energy_hist": reh,
            "qpu_energy_hist": qeh,
            "matrix": mat,
            "magnetization": mag,
            "energies": energies,
            "error_norm": err,
            "break_frac": float(np.mean(f_breaks)),
            "anneal_schedule": schedule,
            "annealing_time": annealing_time_val,
        }

    # --- 3. Paused vs unpaused comparison at same anneal time ---
    # num_epochs caps convergence so find_beta_arbitrary doesn't run forever.
    # The unpaused run is additionally wrapped in try/except because a diverging
    # beta can plummet to values that make the QPU reject the problem outright.
    at_int = int(max(1, round(anneal_time)))
    paused = find_beta_and_sample(optimal_schedule, None,
                                  f"paused_sp{optimal_sp:.3f}")
    try:
        unpaused = find_beta_and_sample(None, at_int,
                                        f"unpaused_t{at_int}us")
    except Exception as e:
        print(f"\n[WARNING] Unpaused beta-finding failed: {e}")
        print("  Unpaused result will be marked failed in the output.")
        unpaused = {
            "failed": True,
            "reason": str(e),
            "beta": float('nan'),
            "beta_hist": [],
            "rbm_energy_hist": [],
            "qpu_energy_hist": [],
            "matrix": None,
            "magnetization": None,
            "energies": np.array([]),
            "error_norm": float('nan'),
            "break_frac": float('nan'),
            "anneal_schedule": None,
            "annealing_time": at_int,
        }

    print(f"\n--- Pause Sweep Complete ---")
    for sp in sweep_points:
        r = per_pause_results[sp]
        tag = "baseline" if sp is None else f"s_p={sp:.3f}"
        print(f"  {tag:>14s}  meanE={r['mean_energy']:.4f}  err={r['error_norm']:.4f}")
    print(f"  Paused   (s_p={optimal_sp:.3f}, β={paused['beta']:.3f}): "
          f"meanE={paused['energies'].mean():.4f}  err={paused['error_norm']:.4f}")
    if unpaused.get("failed"):
        print(f"  Unpaused (t_a={at_int}µs): FAILED — {unpaused['reason']}")
    else:
        print(f"  Unpaused (t_a={at_int}µs,    β={unpaused['beta']:.3f}): "
              f"meanE={unpaused['energies'].mean():.4f}  err={unpaused['error_norm']:.4f}")
        print(f"  Δerr = unpaused − paused = {unpaused['error_norm'] - paused['error_norm']:+.4f}")

    return {
        "pause_points": list(pause_points),
        "includes_baseline": include_baseline,
        "pause_length": pause_length,
        "anneal_time": anneal_time,
        "rbm_scale": rbm_scale,
        "per_pause_results": per_pause_results,
        "optimal_pause_point": optimal_sp,
        "optimal_schedule": optimal_schedule,
        "paused": paused,
        "unpaused": unpaused,
        "classical_matrix": mat_classical,
        "classical_magnetization": mag_classical,
        "classical_energies": classical_energies,
        "num_reads": num_reads,
        "sweep_batches": sweep_batches,
        "num_srt_batches": num_srt_batches,
        "n_cond": n_cond,
    }


def run_anneal_offset_experiment(
    cond_vec,
    rbm,
    raw_sampler,
    conditioning_sets,
    left_chains,
    right_chains,
    hidden_side: str,
    n_cond: int,
    beta: float = 3.0,
    profile_batches: int = 8,
    sample_batches: int = 8,
    num_reads: int = 512,
    offset_fraction: float = 0.25,
    anneal_offset_value: float = -0.3,
    rbm_gibbs_steps: int = 2000,
    rbm_factor: int = 10,
    base_shims=None,
    rng_seed: int = None,
):
    """
    Identifies persistently breaking logical chains via profiling runs, then applies
    per-qubit anneal offsets (negative = delayed freeze = longer tunnelling) to those
    chains' physical qubits and compares the resulting correlation matrices.

    Both the profile/control phase and the offset phase use shuffled orbits (a fresh
    random orbit per batch).  After profiling identifies which *logical* variables
    break frequently, each offset batch:
      1. Draws a new random orbit.
      2. Builds the embedding for that orbit to map high-break logical vars → physical qubits.
      3. Constructs a per-qubit anneal_offsets array for *this orbit's* physical assignment.
      4. Submits to the QPU with those offsets.

    This tests whether the improvement generalises across different physical placements,
    not just on the fixed-orbit qubit set used during profiling.

    Returns a dict suitable for plot_anneal_offset_experiment.
    """
    n_vis = rbm.params["vbias"].shape[0]
    n_hid = rbm.params["hbias"].shape[0]
    n_standard_vis = n_vis - n_cond
    total_qubits = raw_sampler.properties['num_qubits']

    if hidden_side == 'right':
        n_vis_orbit, n_hid_orbit = len(left_chains), len(right_chains)
    else:
        n_vis_orbit, n_hid_orbit = len(right_chains), len(left_chains)

    rng = np.random.default_rng(rng_seed)
    print(f"--- Anneal Offset Experiment (shuffled orbits) | offset={anneal_offset_value} ---")

    binary_patterns_batch = cond_vec[0:1].expand(num_reads, -1)

    # ── Phase 1: Profile with shuffled orbits (= control samples) ──
    # Using different orbits per batch means the logical-domain break signal
    # is robust: a variable that consistently breaks regardless of physical placement
    # is a genuine model-level problem, not a hardware-qubit artifact.
    print(f"Phase 1 – Profiling break rates ({profile_batches} batches, shuffled orbits)...")
    profile_samples = []
    vis_break_acc = np.zeros(n_standard_vis)
    hid_break_acc = np.zeros(n_hid)
    profile_break_fracs = []
    profile_seeds = []

    for b in range(profile_batches):
        seed = int(rng.integers(0, 1_000_000))
        profile_seeds.append(seed)
        vis_mapping, hid_mapping = get_orbit_mappings(seed, n_vis_orbit, n_hid_orbit)

        res = sample_expanded_flux_arbitrary(
            rbm=rbm,
            raw_sampler=raw_sampler,
            conditioning_sets=conditioning_sets,
            left_chains=left_chains,
            right_chains=right_chains,
            binary_patterns_batch=binary_patterns_batch,
            hidden_side=hidden_side,
            beta=beta,
            chain_strength=2.0,
            use_srt=True,
            logical_srt=True,
            flux_drift_compensation=True,
            additive_flux_offsets=base_shims,
            vis_mapping=vis_mapping,
            hid_mapping=hid_mapping,
            perm_seed=seed,
            source=f"anneal_offset_profile_b{b}",
        )

        v_sample, _ = process_analysis_result(res, rbm, conditioning_sets)
        profile_samples.append(v_sample.cpu())

        if res.break_matrix is not None:
            label_to_col = {lbl: i for i, lbl in enumerate(res.variable_labels)}
            for v_idx in range(n_standard_vis):
                col = label_to_col.get(n_cond + v_idx)
                if col is not None:
                    vis_break_acc[v_idx] += float(np.mean(res.break_matrix[:, col]))
            for h_idx in range(n_hid):
                col = label_to_col.get(n_vis + h_idx)
                if col is not None:
                    hid_break_acc[h_idx] += float(np.mean(res.break_matrix[:, col]))
            batch_frac = float(np.mean(res.break_matrix))
            profile_break_fracs.append(batch_frac)
            print(f"  Batch {b+1}/{profile_batches} | seed={seed} | break frac: {batch_frac:.2%}")

    vis_break_mean = vis_break_acc / profile_batches
    hid_break_mean = hid_break_acc / profile_batches
    all_break_means = np.concatenate([vis_break_mean, hid_break_mean])

    control_samples = torch.cat(profile_samples, dim=0)
    control_break_frac = float(np.mean(profile_break_fracs)) if profile_break_fracs else 0.0

    # ── Identify high-break logical variables ──
    n_total_vars = n_standard_vis + n_hid
    n_to_offset = max(1, int(np.ceil(offset_fraction * n_total_vars)))
    threshold_val = float(np.sort(all_break_means)[::-1][n_to_offset - 1])

    vis_offset_mask = vis_break_mean >= threshold_val
    hid_offset_mask = hid_break_mean >= threshold_val
    n_offset_vis = int(vis_offset_mask.sum())
    n_offset_hid = int(hid_offset_mask.sum())

    print(f"\nBreak threshold: {threshold_val:.3f}")
    print(f"Logical chains to offset: {n_offset_vis} vis + {n_offset_hid} hid = {n_offset_vis + n_offset_hid} total")

    # ── Phase 2: Offset sampling with shuffled orbits ──
    # For each batch we pick a fresh orbit, build the embedding for that orbit to
    # find which physical qubits host the high-break logical variables, build the
    # anneal_offsets array for those qubits, then submit.
    print(f"\nPhase 2 – Sampling with anneal offsets ({sample_batches} batches, shuffled orbits)...")
    offset_samples = []
    offset_break_fracs = []
    offset_vis_break_acc = np.zeros(n_standard_vis)
    offset_hid_break_acc = np.zeros(n_hid)
    offset_seeds = []
    n_offset_qubits_per_batch = []

    for b in range(sample_batches):
        seed = int(rng.integers(0, 1_000_000))
        offset_seeds.append(seed)
        vis_mapping, hid_mapping = get_orbit_mappings(seed, n_vis_orbit, n_hid_orbit)

        # Build embedding for this orbit to resolve logical → physical qubits
        exp_embedding, _ = build_expanded_embedding_arbitrary(
            conditioning_sets, left_chains, right_chains,
            num_visible=n_vis, hidden_side=hidden_side,
            vis_mapping=vis_mapping, hid_mapping=hid_mapping,
        )

        # Compute anneal_offsets for the physical qubits hosting high-break logical vars
        anneal_offsets_arr = np.zeros(total_qubits)
        for v_idx in range(n_standard_vis):
            if vis_offset_mask[v_idx]:
                for qubit in exp_embedding.get(n_cond + v_idx, []):
                    if qubit < total_qubits:
                        anneal_offsets_arr[qubit] = anneal_offset_value
        for h_idx in range(n_hid):
            if hid_offset_mask[h_idx]:
                for qubit in exp_embedding.get(n_vis + h_idx, []):
                    if qubit < total_qubits:
                        anneal_offsets_arr[qubit] = anneal_offset_value
        n_offset_qubits_per_batch.append(int((anneal_offsets_arr != 0).sum()))

        res = sample_expanded_flux_arbitrary(
            rbm=rbm,
            raw_sampler=raw_sampler,
            conditioning_sets=conditioning_sets,
            left_chains=left_chains,
            right_chains=right_chains,
            binary_patterns_batch=binary_patterns_batch,
            hidden_side=hidden_side,
            beta=beta,
            chain_strength=2.0,
            use_srt=True,
            logical_srt=True,
            flux_drift_compensation=True,
            additive_flux_offsets=base_shims,
            vis_mapping=vis_mapping,
            hid_mapping=hid_mapping,
            perm_seed=seed,
            source=f"anneal_offset_run_b{b}",
            anneal_offsets=anneal_offsets_arr,
        )

        v_sample, _ = process_analysis_result(res, rbm, conditioning_sets)
        offset_samples.append(v_sample.cpu())

        if res.break_matrix is not None:
            label_to_col = {lbl: i for i, lbl in enumerate(res.variable_labels)}
            for v_idx in range(n_standard_vis):
                col = label_to_col.get(n_cond + v_idx)
                if col is not None:
                    offset_vis_break_acc[v_idx] += float(np.mean(res.break_matrix[:, col]))
            for h_idx in range(n_hid):
                col = label_to_col.get(n_vis + h_idx)
                if col is not None:
                    offset_hid_break_acc[h_idx] += float(np.mean(res.break_matrix[:, col]))
            batch_frac = float(np.mean(res.break_matrix))
            offset_break_fracs.append(batch_frac)
            print(f"  Batch {b+1}/{sample_batches} | seed={seed} | {n_offset_qubits_per_batch[-1]} qubits offset | break frac: {batch_frac:.2%}")

    offset_v = torch.cat(offset_samples, dim=0)
    offset_break_frac = float(np.mean(offset_break_fracs)) if offset_break_fracs else 0.0
    offset_vis_break_mean = offset_vis_break_acc / sample_batches
    offset_hid_break_mean = offset_hid_break_acc / sample_batches

    # ── Classical Baseline ──
    print("Generating Classical Baseline...")
    v_cl = rbm.sample_v_given_v_clamped(
        clamped_v=cond_vec[0:1].expand(num_reads * rbm_factor, -1),
        n_clamped=n_cond, gibbs_steps=rbm_gibbs_steps, beta=1.0,
    )

    def get_corr(samples):
        if isinstance(samples, torch.Tensor):
            samples = samples.float().cpu()
        corr = torch.corrcoef(samples.T).numpy()
        latent = corr[n_cond:, n_cond:]
        np.fill_diagonal(latent, 0)
        return np.nan_to_num(latent, nan=0.0)

    mat_classical = get_corr(v_cl)
    mag_classical = v_cl.float().cpu().mean(dim=0)[n_cond:].numpy()

    mat_control = get_corr(control_samples)
    mag_control = control_samples.float().cpu().mean(dim=0)[n_cond:].numpy()

    mat_offset = get_corr(offset_v)
    mag_offset = offset_v.float().cpu().mean(dim=0)[n_cond:].numpy()

    error_control = float(np.linalg.norm(mat_control - mat_classical))
    error_offset = float(np.linalg.norm(mat_offset - mat_classical))

    print(f"\n--- Anneal Offset Experiment Complete ---")
    print(f"No Offset:   Error={error_control:.4f}  Breaks={control_break_frac:.2%}")
    print(f"With Offset: Error={error_offset:.4f}  Breaks={offset_break_frac:.2%}")

    return {
        "classical_matrix": mat_classical,
        "classical_magnetization": mag_classical,
        "no_offset": {
            "matrix": mat_control,
            "error_norm": error_control,
            "break_frac": control_break_frac,
            "magnetization": mag_control,
            "break_rates_vis": vis_break_mean,
            "break_rates_hid": hid_break_mean,
        },
        "with_offset": {
            "matrix": mat_offset,
            "error_norm": error_offset,
            "break_frac": offset_break_frac,
            "magnetization": mag_offset,
            "break_rates_vis": offset_vis_break_mean,
            "break_rates_hid": offset_hid_break_mean,
        },
        "offset_mask_vis": vis_offset_mask,
        "offset_mask_hid": hid_offset_mask,
        "offset_value": anneal_offset_value,
        "offset_fraction": offset_fraction,
        "threshold_val": threshold_val,
        "n_offset_vis": n_offset_vis,
        "n_offset_hid": n_offset_hid,
        "n_offset_qubits_per_batch": n_offset_qubits_per_batch,
        "profile_seeds": profile_seeds,
        "offset_seeds": offset_seeds,
        "rng_seed": rng_seed,
        "profile_batches": profile_batches,
        "sample_batches": sample_batches,
        "num_reads": num_reads,
        "n_cond": n_cond,
    }


# ---------------------------------------------------------------------------
# Run-save-plot wrappers
# ---------------------------------------------------------------------------
# Each wrapper runs the corresponding experiment, persists the raw result dict
# with save_result, then calls the matching plot function.  The result is always
# returned so the caller can do further analysis without reloading from disk.
# ---------------------------------------------------------------------------

from .results_io import save_result
from .plots import (
    plot_srt_aggregation_comparison,
    plot_annealing_time_sweep,
    plot_pause_sweep,
    plot_anneal_offset_experiment,
)


def run_and_save_srt_aggregation(
    *args,
    output_dir: str = "results/dwave",
    plot_save_path: str | None = None,
    atlas_label: str = "Simulation",
    **kwargs,
) -> dict:
    """Run :func:`run_srt_aggregation_comparison`, save raw data and plot.

    All positional / keyword arguments are forwarded unchanged to the underlying
    experiment function.  Extra keyword arguments:

    output_dir      : directory where the ``.pt`` result file is written.
    plot_save_path  : if given, the figure is also saved to this path.
    atlas_label     : label passed to the ATLAS stamp on the plot.

    Returns the result dict (same object that was saved to disk).
    """
    result = run_srt_aggregation_comparison(*args, **kwargs)
    save_result(result, "srt_aggregation", output_dir=output_dir)
    plot_srt_aggregation_comparison(result, save_path=plot_save_path, atlas_label=atlas_label)
    return result


def run_and_save_annealing_time_sweep(
    *args,
    output_dir: str = "results/dwave",
    plot_save_path: str | None = None,
    atlas_label: str = "Simulation",
    **kwargs,
) -> dict:
    """Run :func:`run_annealing_time_sweep`, save raw data and plot.

    All positional / keyword arguments are forwarded unchanged to the underlying
    experiment function.  Extra keyword arguments:

    output_dir      : directory where the ``.pt`` result file is written.
    plot_save_path  : if given, the figure is also saved to this path.
    atlas_label     : label passed to the ATLAS stamp on the plot.
    """
    result = run_annealing_time_sweep(*args, **kwargs)
    save_result(result, "annealing_time_sweep", output_dir=output_dir)
    plot_annealing_time_sweep(result, save_path=plot_save_path, atlas_label=atlas_label)
    return result


def run_and_save_pause_sweep(
    *args,
    output_dir: str = "results/dwave",
    plot_save_path: str | None = None,
    atlas_label: str = "Simulation",
    **kwargs,
) -> dict:
    """Run :func:`run_pause_sweep`, save raw data and plot.

    All positional / keyword arguments are forwarded unchanged to the underlying
    experiment function.  Extra keyword arguments:

    output_dir      : directory where the ``.pt`` result file is written.
    plot_save_path  : if given, the figure is also saved to this path.
    atlas_label     : label passed to the ATLAS stamp on the plot.
    """
    result = run_pause_sweep(*args, **kwargs)
    save_result(result, "pause_sweep", output_dir=output_dir)
    plot_pause_sweep(result, save_path=plot_save_path, atlas_label=atlas_label)
    return result


def run_and_save_anneal_offset(
    *args,
    output_dir: str = "results/dwave",
    plot_save_path: str | None = None,
    atlas_label: str = "Simulation",
    **kwargs,
) -> dict:
    """Run :func:`run_anneal_offset_experiment`, save raw data and plot.

    All positional / keyword arguments are forwarded unchanged to the underlying
    experiment function.  Extra keyword arguments:

    output_dir      : directory where the ``.pt`` result file is written.
    plot_save_path  : if given, the figure is also saved to this path.
    atlas_label     : label passed to the ATLAS stamp on the plot.
    """
    result = run_anneal_offset_experiment(*args, **kwargs)
    save_result(result, "anneal_offset", output_dir=output_dir)
    plot_anneal_offset_experiment(result, save_path=plot_save_path, atlas_label=atlas_label)
    return result