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
    calculate_rms_chain_strength
)
from .graphs import (
    get_physical_flux_biases,
    build_logical_to_physical_map,
    build_manual_embedded_ising,
    get_physical_flux_biases_manual,
    build_expanded_embedding,
    get_expanded_flux_biases
)
from .sampling_backend import (
    sample_logical_ising,
    sample_ising_flux_bias,
    sample_physical_with_analysis,
    sample_physical_with_analysis_srt,
    sample_manual_ising
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
    sample_expanded_flux_arbitrary
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
    incidence_energy: float,
    engine,
    rbm,
    raw_sampler,
    conditioning_sets,
    left_chains,
    right_chains,
    n_cond: int = 53,
    beta: float = 3.0,
    num_reads_per_perm: int = 1024,
    num_permutations: int = 20, 
    srt_batches: int = 8,
    device: str = 'cpu',
    base_shims = None
):
    """
    Runs a Monte Carlo Sweep over permutation space with SRT Batches.
    
    Fix: Calls process_analysis_result per batch to extract the correct
    RBM variables before concatenation.
    """
    print(f"--- Starting Monte Carlo Permutation Sweep (Energy = {incidence_energy} MeV) ---")
    
    # 1. Target Data & Classical Baseline
    target_batch = convert_energy_to_binary(
        incidence_energy=incidence_energy, engine=engine, n_cond=n_cond, 
        num_reads=num_reads_per_perm, device=device
    )
    
    print("Generating Classical Baseline...")
    v_cl = rbm.sample_v_given_v_clamped(
        clamped_v=target_batch, n_clamped=n_cond, gibbs_steps=2000, beta=1.0 
    )
    
    def get_corr(samples):
        if isinstance(samples, torch.Tensor): samples = samples.float().cpu()
        corr = torch.corrcoef(samples.T).numpy()
        latent = corr[n_cond:, n_cond:]
        np.fill_diagonal(latent, 0)
        return np.nan_to_num(latent, nan=0.0)

    mat_classical = get_corr(v_cl.cpu())
    
    # 2. Setup Loop
    sweep_results = {
        "classical_matrix": mat_classical,
        "perm_metrics": [],
        "default_orbit": None, 
        "best_orbit": None,
        "worst_orbit": None
    }
    
    n_avail_vis = len(left_chains) 
    n_avail_hid = len(right_chains)
    reads_per_batch = num_reads_per_perm // srt_batches

    # 3. Execution Loop
    for i in range(num_permutations):
        
        # --- A. Determine Mapping ---
        if i == 0:
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
                binary_patterns_batch=target_batch[0:reads_per_batch], 
                hidden_side='right',
                beta=beta,
                source=f"MC_{run_type}_{seed}_b{b}",
                use_srt=True,  
                additive_flux_offsets=base_shims,
                vis_mapping=p_vis,
                hid_mapping=p_hid,
                perm_seed=seed
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
            "matrix": mat_perm
        }
        sweep_results["perm_metrics"].append(record)
        
        if i == 0:
            sweep_results["default_orbit"] = record

    # 4. Sort and Finalize
    sorted_metrics = sorted(sweep_results["perm_metrics"], key=lambda x: x['error_norm'])
    
    sweep_results["best_orbit"] = sorted_metrics[0]
    sweep_results["worst_orbit"] = sorted_metrics[-1]
    
    print(f"\n--- MC Sweep Complete ---")
    print(f"Default Error: {sweep_results['default_orbit']['error_norm']:.4f}")
    print(f"Best Error:    {sweep_results['best_orbit']['error_norm']:.4f}")
    
    return sweep_results




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