import torch
import numpy as np
import time
import os
import json
from datetime import datetime, timezone, timedelta
import pytz

# --- Your Imports ---
from hydra.utils import instantiate
from hydra import initialize, compose
import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf
from scripts.run import setup_model
from model.rbm.rbm_two_partite import RBM_TwoPartite
from utils.dwave.workflows import sample_expanded_flux_conditioned, find_beta_flux_bias_expanded
from utils.dwave.physics import joint_energy
from utils.dwave.graphs import run_embedding, analyze_target_side, get_sampler_and_biclique_embedding

# --- Configuration ---
OUTPUT_DIR = "./wandb-outputs/dwave_data_campaign"
INCIDENCE_ENERGIES = [1000, 50000, 100000, 200000, 300000]
DEADLINE = datetime(2025, 12, 3, 0, 0, 0, tzinfo=timezone.utc) # Dec 3rd UTC is Dec 2nd 4pm PT

# Sampling Params
CAREFUL_BATCH_SIZE = 256
CAREFUL_TARGET_TOTAL = 1024
FAST_BATCH_TOTAL = 1024
DRIFT_THRESHOLD = 2.0 # If energy diff > 2.0, we consider beta drifted
DRIFT_LR = 0.05 # How much to nudge beta manually before re-estimating

os.makedirs(OUTPUT_DIR, exist_ok=True)

def is_time_remaining():
    now = datetime.now(timezone.utc)
    remaining = DEADLINE - now
    if remaining.total_seconds() > 0:
        print(f"Time remaining: {remaining}")
        return True
    print("Deadline reached. Stopping script.")
    return False

def get_conditioning_batch(runner, energy_val, batch_size, device):
    """
    Creates the binary conditioning pattern for a specific energy.
    Assumes runner.model.encoder logic from your snippet.
    """
    # Create dummy input tensor for the specific energy
    # Assuming input needs to be shape (1, 1) or (1,) depending on your encoder
    # Adjust shape if your encoder expects something else
    e_tensor = torch.tensor([[energy_val]], dtype=torch.float32).to(device)
    
    # Get binary representation
    # Using the num_clamped_bits logic from your snippet
    num_clamped_bits = 53
    with torch.no_grad():
        bin_energy = runner.model.encoder.binary_energy_refactored(e_tensor)[:, :num_clamped_bits]
    
    # Repeat to fill batch
    return bin_energy.repeat(batch_size, 1)

def save_data(energy_val, mode, batch_idx, v_samples, h_samples, beta, beta_history, drift_val=None):
    """
    Saves a unique file for every batch. 
    Added drift_val to metadata for analysis.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Filename includes mode (e.g. "careful", "drifted", "fast_bulk")
    filename = f"samples_E{energy_val}_{mode}_{batch_idx}_{timestamp}.npz"
    path = os.path.join(OUTPUT_DIR, filename)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    v_np = v_samples.detach().cpu().numpy()
    h_np = h_samples.detach().cpu().numpy()
    
    # Save beta history and drift value
    np.savez_compressed(
        path, 
        v_samples=v_np, 
        h_samples=h_np, 
        energy_val=energy_val,
        beta_final=beta,
        beta_history=beta_history,
        drift_value=drift_val, # New metadata
        mode=mode
    )
    print(f"[{mode.upper()}] Saved {v_np.shape[0]} samples to {filename}")


def check_drift(rbm, v_qpu, h_qpu, binary_patterns_batch):
    """
    Compare QPU energy to RBM baseline to detect drift.
    """
    # 1. Calculate QPU Energy
    with torch.no_grad():
        e_qpu = joint_energy(rbm, v_qpu, h_qpu)
        mean_qpu = e_qpu.mean().item()

    # 2. Calculate RBM Baseline Energy
    # We produce a quick RBM sample batch for comparison
    n_clamped = binary_patterns_batch.shape[1]
    v_rbm = rbm.sample_v_given_v_clamped(
        clamped_v=binary_patterns_batch, 
        n_clamped=n_clamped, 
        gibbs_steps=500, # Lower steps for speed in check
        beta=1.0
    )
    h_rbm, _ = rbm._sample_h_given_v(v_rbm, beta=1.0)
    
    with torch.no_grad():
        e_rbm = joint_energy(rbm, v_rbm, h_rbm)
        mean_rbm = e_rbm.mean().item()
        
    diff = mean_qpu - mean_rbm
    return diff, mean_qpu, mean_rbm


def main(cfg=None):
    # --- Initialization ---
    SOLVER_NAME = "Advantage2_system1.8" 
    
    # Load Model
    runner = setup_model(cfg) # Renamed 'self' to 'runner'
    dummy_data = torch.zeros(1, cfg.rbm.latent_nodes_per_p).to(runner.device)
    CHECKPOINT_FILE = "/home/leozhu/CaloQuVAE/wandb-outputs/run_2025-11-15_19-47-10_RBM_TwoPartite/training_checkpoint.h5"
    
    rbm = RBM_TwoPartite(cfg, data=dummy_data)
    try:
        loaded_epoch = rbm.load_checkpoint(CHECKPOINT_FILE, epoch=None) 
        print(f"Loaded checkpoint epoch {loaded_epoch}.")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        exit(1)

    # Setup D-Wave
    raw_sampler, embedding, qpu_sampler = get_sampler_and_biclique_embedding(
        rbm.num_visible, rbm.num_hidden, solver_name=SOLVER_NAME
    )
    
    # Setup Topology
    # Note: Using rbm.params["hbias"].shape[0] for num_hidden
    sampler_emb, working_graph, q_used, left_chains, right_chains = run_embedding(
        rbm.params["hbias"].shape[0], SOLVER_NAME
    )
    target_nodes_right = list(right_chains.values())
    num_nodes_right, conditioning_sets = analyze_target_side(
        "Right Chains", sampler_emb, target_nodes_right, q_used
    )

    # --- Main Execution Loop ---
    
    while is_time_remaining():
        
        for energy in INCIDENCE_ENERGIES:
            if not is_time_remaining(): break
            
            print(f"\n{'='*10} Processing Energy {energy} {'='*10}")
            
            # 1. Prepare Data for this energy
            # We need a batch for estimation and checking
            batch_pattern = get_conditioning_batch(runner, energy, CAREFUL_BATCH_SIZE, rbm.device)
            
            # 2. Initial Beta Estimation
            print("Estimating initial Beta...")
            current_beta, beta_hist, _, _ = find_beta_flux_bias_expanded(
                rbm, qpu_sampler, embedding, conditioning_sets, left_chains, right_chains,
                binary_patterns_batch=batch_pattern,
                num_reads=128,
                num_epochs=10,
                adaptive=False,
                use_fast_sampling=False,
                tolerance=0.5
            )
            print(f"Initial Beta for E={energy}: {current_beta:.4f}")

# 3. The "Careful" Loop (Check Drift)
            collected_careful = 0
            drift_history = []
            
            print(f"--- Starting Careful Sampling Loop (Target: {CAREFUL_TARGET_TOTAL}) ---")
            while collected_careful < CAREFUL_TARGET_TOTAL:
                if not is_time_remaining(): break

                # Sample 256 using Fast Mode
                try:
                    v_s, h_s = sample_expanded_flux_conditioned(
                        rbm, raw_sampler, conditioning_sets, left_chains, right_chains,
                        binary_patterns_batch=batch_pattern,
                        beta=current_beta,
                        clamp_strength_h=50.0,
                        fast_sampling=False 
                    )
                except Exception as e:
                    print(f"D-Wave API Error: {e}. Waiting 10s...")
                    time.sleep(10)
                    continue

                # Check Drift
                diff, q_e, r_e = check_drift(rbm, v_s, h_s, batch_pattern)
                print(f"Batch Check: Beta={current_beta:.3f} | Diff={diff:.3f} (QPU:{q_e:.2f}, RBM:{r_e:.2f})")
                
                drift_history.append({"beta": current_beta, "diff": diff, "timestamp": str(datetime.now())})

                # --- DRIFT HANDLING ---
                if abs(diff) > DRIFT_THRESHOLD:
                    print(f"!!! DRIFT DETECTED (Diff {diff:.3f} > {DRIFT_THRESHOLD}) !!!")
                    
                    # 1. SAVE the drifted batch (marked as 'drifted')
                    # We pass the diff so we know how bad it was later
                    save_data(energy, "drifted", collected_careful, v_s, h_s, current_beta, beta_hist, drift_val=diff)
                    
                    print("Batch saved as 'drifted'. Correcting Beta and retrying...")

                    # 2. Correct Beta
                    # If Diff is positive (QPU > RBM), QPU is too hot/disordered -> Increase Beta (cool it)
                    # If Diff is negative (QPU < RBM), QPU is frozen -> Decrease Beta
                    current_beta = max(0.01, current_beta - DRIFT_LR * diff)
                    
                    print(f"Nudged Beta to {current_beta:.4f}. Running Re-estimation...")
                    
                    # 3. Re-estimate 
                    current_beta, new_hist, _, _ = find_beta_flux_bias_expanded(
                        rbm, qpu_sampler, embedding, conditioning_sets, left_chains, right_chains,
                        binary_patterns_batch=batch_pattern,
                        beta_init=current_beta, # Start from nudged value
                        num_epochs=5, # Short re-estimation
                        adaptive=False,
                        use_fast_sampling=False # Use slow mode for accurate re-estimation
                    )
                    beta_hist.extend(new_hist)
                    
                    # 4. Loop back (Do NOT increment collected_careful)
                    continue 
                
                # --- SUCCESS HANDLING ---
                # If diff is acceptable, save as 'careful' and count it
                save_data(energy, "careful", collected_careful, v_s, h_s, current_beta, beta_hist, drift_val=diff)
                collected_careful += CAREFUL_BATCH_SIZE

            # 4. The "Fast" Loop (Bulk Sampling)
            # Once we trust beta from the careful loop, we grab the rest
            print(f"--- Careful Loop Done. Starting Fast Batch ({FAST_BATCH_TOTAL} samples) ---")
            
            # Construct large batch
            large_batch_pattern = get_conditioning_batch(runner, energy, FAST_BATCH_TOTAL, rbm.device)
            
            try:
                v_fast, h_fast = sample_expanded_flux_conditioned(
                    rbm, raw_sampler, conditioning_sets, left_chains, right_chains,
                    binary_patterns_batch=large_batch_pattern,
                    beta=current_beta,
                    clamp_strength_h=50.0,
                    fast_sampling=True
                )
                save_data(energy, "fast_bulk", 0, v_fast, h_fast, current_beta, beta_hist)
            except Exception as e:
                print(f"Error during fast bulk sampling: {e}")
                # If fast batch fails, we just loop back.
                pass
                
            # Loop continues to next energy...

    print("Script finished successfully.")

if __name__ == "__main__":
    # 1. Dynamically go up one directory from this script's location
    # This finds '.../CaloQuVAE/scripts', then goes up to '.../CaloQuVAE'
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir) 
    
    os.chdir(project_root)
    print(f"Working directory set to: {os.getcwd()}")

    # 2. Initialize Hydra
    # We use config_path="../config" because Hydra looks relative to the 
    # script file location (scripts/), so we point it to the sibling folder (config/).
    GlobalHydra.instance().clear()
    with initialize(version_base=None, config_path="../config"):
        cfg = compose(config_name="config.yaml")
        
        # 3. Run Main
        print(f"Hydra Config loaded. Dataset: {cfg.data.dataset_name}")
        main(cfg)