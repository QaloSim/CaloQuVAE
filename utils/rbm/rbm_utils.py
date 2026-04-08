import torch
import os
import re
from typing import List
from model.rbm.rbm_two_partite import RBM_TwoPartite
from CaloQuVAE import logging
logger = logging.getLogger(__name__)
import math


def decode_binary_energy(x_encoded, lin_bits=23):
    """
    Decodes the original energy value from the linear bits portion 
    of the encoded tensor.
    
    Args:
        x_encoded (torch.Tensor): The full encoded tensor, shape (batch_size, n_latent_nodes).
        lin_bits (int): The number of bits used for the linear encoding (e.g., 23).
        
    Returns:
        torch.Tensor: The decoded energy values, shape (batch_size, 1).
    """
    
    # --- 1. Extract the linear encoding ---
    # We only take the first `lin_bits` columns, which represent the
    # direct binary encoding.
    linear_encoding = x_encoded[:, :lin_bits]
    
    # --- 2. Create the bit-weight mask ---
    # This is the inverse of the mask in the `binary` function.
    # It's a tensor like [1, 2, 4, 8, ..., 2**(lin_bits-1)]
    mask = 2**torch.arange(lin_bits).to(x_encoded.device, x_encoded.dtype)
    
    # --- 3. Multiply and Sum ---
    # We multiply the binary encoding (0s and 1s) by their corresponding
    # bit weights and sum them up.
    # (batch_size, lin_bits) * (lin_bits,) -> (batch_size, lin_bits)
    weighted_bits = linear_encoding * mask
    
    # Sum along dimension 1 to get the final integer value for each item in the batch
    # (batch_size, lin_bits) -> (batch_size,)
    decoded_values = torch.sum(weighted_bits, dim=1)
    
    # --- 4. Reshape and Return ---
    # Add a dimension to match the original input shape (batch_size, 1)
    return decoded_values.unsqueeze(1)


def parallel_trajectory_tempering_clamped(
    checkpoints: List['RBM_TwoPartite'], 
    clamped_v_batch: torch.Tensor, 
    n_clamped: int, 
    num_mcmc_steps: int, 
    k: int = 1, 
    beta: float = 1.0
) -> torch.Tensor:
    """
    Implements PTT for conditional RBMs by clamping the first `n_clamped` nodes.
    
    Args:
        checkpoints: Chronological list of loaded RBM_TwoPartite instances.
        clamped_v_batch: The data tensor containing the conditioning nodes. 
                         Shape: [batch_size, n_clamped] or [batch_size, num_visibles].
        n_clamped: Number of nodes to clamp.
        num_mcmc_steps: Total number of MCMC sampling steps.
        k: Number of Gibbs steps per model before proposing a swap.
        beta: Inverse temperature parameter.
        
    Returns:
        torch.Tensor: Equilibrium visible samples from the final model.
    """
    t_f = len(checkpoints)
    if t_f == 0:
        raise ValueError("Must provide at least one RBM checkpoint.")
        
    final_model = checkpoints[-1]
    device = final_model.device
    num_visibles = final_model.num_visible
    num_unclamped = num_visibles - n_clamped
    batch_size = clamped_v_batch.shape[0]
    
    # Extract strictly the clamped portion and ensure it's on the right device
    clamped_data = clamped_v_batch[:, :n_clamped].to(device)
    
    v_states = []
    
    # --- 1. Initialization Step ---
    # v_0: Clamped nodes + random noise for the rest
    v_0 = torch.zeros((batch_size, num_visibles), device=device, dtype=torch.float32)
    v_0[:, :n_clamped] = clamped_data
    v_0[:, n_clamped:] = torch.randint(0, 2, size=(batch_size, num_unclamped), device=device, dtype=torch.float32)
    v_states.append(v_0)
    
    # Initialize chains by passing them through the sequence of models
    for i in range(1, t_f + 1):
        model = checkpoints[i-1]
        v_i = v_states[i-1].clone()
        
        for _ in range(k):
            h, _ = model._sample_h_given_v(v_i, beta)
            v_i, _ = model._sample_v_given_h(h, beta)
            # Enforce clamp
            v_i[:, :n_clamped] = clamped_data
            
        v_states.append(v_i)
        
    # --- 2. Main Sampling Loop ---
    total_steps = num_mcmc_steps // k
    for step in range(total_steps):
        
        # A) Update Step: k Gibbs steps for each model
        for i in range(1, t_f + 1):
            model = checkpoints[i-1]
            v_i = v_states[i]
            for _ in range(k):
                h, _ = model._sample_h_given_v(v_i, beta)
                v_i, _ = model._sample_v_given_h(h, beta)
                # Enforce clamp
                v_i[:, :n_clamped] = clamped_data
            v_states[i] = v_i
            
        # B) Resample H_0 (noise model)
        v_0_new = torch.zeros((batch_size, num_visibles), device=device, dtype=torch.float32)
        v_0_new[:, :n_clamped] = clamped_data
        v_0_new[:, n_clamped:] = torch.randint(0, 2, size=(batch_size, num_unclamped), device=device, dtype=torch.float32)
        v_states[0] = v_0_new
        
        # C) Swap Step: Propose swaps between adjacent models
        for i in range(1, t_f + 1):
            model_i = checkpoints[i-1]
            model_prev = checkpoints[i-2] if i > 1 else None
            
            v_i = v_states[i]
            v_prev = v_states[i-1]
            
            # Compute Free Energies
            E_i_vi = model_i.compute_energy(v_i)
            E_i_vprev = model_i.compute_energy(v_prev)
            
            if model_prev is not None:
                E_prev_vi = model_prev.compute_energy(v_i)
                E_prev_vprev = model_prev.compute_energy(v_prev)
            else:
                E_prev_vi = torch.zeros_like(E_i_vi)
                E_prev_vprev = torch.zeros_like(E_i_vprev)
            
            # Acceptance probability
            delta_H_vi = E_i_vi - E_prev_vi
            delta_H_vprev = E_i_vprev - E_prev_vprev
            p_acc = torch.exp(delta_H_vi - delta_H_vprev) 
            
            # Roll for acceptance
            accept = torch.rand_like(p_acc) < p_acc
            
            # Execute swaps
            v_i_new = torch.where(accept.unsqueeze(1), v_prev, v_i)
            v_prev_new = torch.where(accept.unsqueeze(1), v_i, v_prev)
            
            v_states[i] = v_i_new
            v_states[i-1] = v_prev_new
            
    return v_states[-1]

def prepare_ptt_checkpoints(checkpoint_dir: str, rbm_cfg) -> List['RBM_TwoPartite']:
    """
    Scans a directory for RBM checkpoints, sorts them chronologically by epoch,
    and returns a list of instantiated RBM_TwoPartite objects ready for PTT.
    """
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Directory not found: {checkpoint_dir}")

    checkpoint_files = []
    # Regex to extract the integer epoch number from your specific filename structure
    pattern = re.compile(r"training_checkpoint_epoch_(\d+)\.h5")
    
    # 1. Scan directory and extract filepaths + epochs
    for filename in os.listdir(checkpoint_dir):
        match = pattern.search(filename)
        if match:
            epoch_num = int(match.group(1))
            filepath = os.path.join(checkpoint_dir, filename)
            checkpoint_files.append((epoch_num, filepath))
            
    if not checkpoint_files:
        raise ValueError(f"No valid checkpoint files found in {checkpoint_dir}")
        
    # 2. Sort chronologically (t=1 to t=t_f) based on the extracted epoch integer
    checkpoint_files.sort(key=lambda x: x[0])
    
    # 3. Determine device for the dummy data
    if rbm_cfg.device == "gpu" and torch.cuda.is_available():
        device = torch.device(f"cuda:{rbm_cfg.gpu_list[0]}")
    else:
        device = torch.device("cpu")
        
    # Create the dummy data tensor needed to bypass your class's init_parameters requirement
    dummy_data = torch.zeros(1, rbm_cfg.rbm.num_visible_nodes, device=device)
    
    checkpoints = []
    print(f"Found {len(checkpoint_files)} checkpoints. Loading chronologically...")
    
    # 4. Instantiate and load each model
    for epoch, filepath in checkpoint_files:
        try:
            # Instantiate a fresh object for each checkpoint
            rbm = RBM_TwoPartite(rbm_cfg, data=dummy_data)
            
            # Load the checkpoint weights and chains directly into the object
            loaded_epoch = rbm.load_checkpoint(filepath, epoch=None)
            checkpoints.append(rbm)
            
            print(f"  -> Loaded epoch {loaded_epoch:05d} from {os.path.basename(filepath)}")
        except Exception as e:
            print(f"  -> Error loading epoch {epoch} from {os.path.basename(filepath)}: {e}")
            # Depending on your preference, you might want to raise the exception here 
            # instead of continuing, as a missing checkpoint breaks the trajectory.
            raise e
            
    print(f"Successfully prepared {len(checkpoints)} models for PTT.")
    return checkpoints



def save_clamped_PTT_samples(
    checkpoints_list: List['RBM_TwoPartite'],
    input_data: torch.Tensor,
    n_clamped: int,
    gibbs_steps: int,
    save_dir: str,
    gen_batch_size: int
):
    """
    Generates conditional samples by clamping the first 'n_clamped'
    nodes of 'input_data' and sampling the rest. Saves them in a
    format loadable by LatentDataset for the VAE.
    
    Args:
        rbm: The trained RBM model.
        input_data (torch.Tensor): The full data tensor to use as the
                                   basis for clamping.
                                   Shape: [n_samples, num_visibles].
        n_clamped (int): The number of nodes (from index 0) to clamp.
        gibbs_steps (int): The number of Gibbs steps for the sampler.
        save_dir (str): Directory to save the files.
        gen_batch_size (int, optional): The batch size for generation.
                                       Defaults to rbm.chains["v"].shape[0].
    """
    n_samples = input_data.shape[0]
    logger.info(f"Attempting to generate {n_samples} clamped samples for VAE...")
    
    n_batches = math.ceil(n_samples / gen_batch_size)
    all_samples = []

    for i in range(n_batches):
        # logger.info(f"Generating clamped batch {i+1}/{n_batches} (gibbs_steps={gibbs_steps})...")
        
        # 1. Get the data slice for this batch
        start_idx = i * gen_batch_size
        end_idx = min((i + 1) * gen_batch_size, n_samples)
        
        # 2. Extract the clamped part from the input data
        # We only need the first n_clamped nodes
        # Shape: [current_batch_size, n_clamped]
        batch_clamped_v = input_data[start_idx:end_idx, :n_clamped]

        # Ensure it's on the correct device (matching the RBM)
        batch_clamped_v = batch_clamped_v.to(checkpoints_list[0].device)
        
        # 3. Generate the completed samples
        # The RBM will return the full [current_batch_size, num_visibles] tensor
        batch_samples = parallel_trajectory_tempering_clamped(
            checkpoints=checkpoints_list,
            clamped_v_batch=batch_clamped_v,
            n_clamped=n_clamped,
            num_mcmc_steps=gibbs_steps
        )
        all_samples.append(batch_samples.cpu()) # Move to CPU for storage

    # Concatenate all batches
    final_samples = torch.cat(all_samples, dim=0)
    
    # No truncation needed as we started from input_data.shape[0]

    # --- Save the samples and dummy labels ---
    
    # 1. Define file paths
    data_filename = f"rbm_clamped_samples_train_data_final.pt"
    label_filename = f"rbm_clamped_samples_train_labels_final.pt"
    
    data_path = os.path.join(save_dir, data_filename)
    label_path = os.path.join(save_dir, label_filename)
    
    # 2. Create dummy labels (to satisfy LatentDataset)
    # Shape is (N_samples, 1) to match incident_energy
    dummy_labels = torch.zeros((final_samples.shape[0], 1), dtype=torch.float32)

    # 3. Save both tensors (final_samples is already on CPU)
    torch.save(final_samples, data_path)
    torch.save(dummy_labels, label_path)
    
    logger.info("="*50)
    logger.info(f"Saved clamped samples for VAE to: {data_path}")
    logger.info(f"Saved dummy labels to: {label_path}")
    logger.info(f"Final data shape: {final_samples.shape}")
    logger.info("="*50)
    
    return data_path
