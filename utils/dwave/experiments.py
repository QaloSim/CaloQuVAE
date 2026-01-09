import torch
import numpy as np
import copy
from dwave.system.composites import FixedEmbeddingComposite
from datetime import datetime

# --- Explicit Local Imports ---
from .physics import (
    rbm_to_logical_ising,
    joint_energy,
    rbm_to_expanded_ising,
    convert_energy_to_binary,
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