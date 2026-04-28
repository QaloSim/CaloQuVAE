import torch
import os
import re
import numpy as np
from typing import List, Tuple
from model.rbm.rbm_two_partite import RBM_TwoPartite
from CaloQuVAE import logging
logger = logging.getLogger(__name__)
import math


def run_pls_pipeline(
    data_real: torch.Tensor,
    data_gen: torch.Tensor,
    classical_features: torch.Tensor,
    n_components: int = 4,
    max_iter: int = 500,
    tol: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Partial Least Squares (PLS) via NIPALS.  Finds the latent directions in the
    visible space that maximise covariance with the classical feature matrix.

    Both X (data_real) and Y (classical_features) are centred and Y is
    unit-variance scaled before decomposition.  Projections use the
    PLS rotation matrix W* = W (P^T W)^{-1} so that successive LV scores
    are as orthogonal as possible.

    Parameters
    ----------
    data_real : torch.Tensor  (n_real, n_visible)
        Real / reference samples — used to fit the PLS basis.
    data_gen : torch.Tensor   (n_gen, n_visible)
        Generated samples — projected onto the fitted basis.
    classical_features : torch.Tensor  (n_real, n_features)
        Feature matrix aligned row-wise with data_real.
    n_components : int
        Number of latent variables (LVs) to extract.

    Returns
    -------
    proj_real : np.ndarray  (n_real, n_components)  — LV scores for real data
    proj_gen  : np.ndarray  (n_gen,  n_components)  — LV scores for gen data
    x_weights : np.ndarray  (n_visible,  n_components)  — X weight vectors W
    y_weights : np.ndarray  (n_features, n_components)  — Y weight vectors C
    """
    device = data_real.device
    dtype = torch.float32
    n_visible = data_real.shape[1]

    # --- Centre X, centre + scale Y ---
    x_mean = data_real.mean(0)
    X = (data_real - x_mean).to(dtype)

    Y_raw = classical_features.to(device=device, dtype=dtype)
    if Y_raw.dim() == 1:
        Y_raw = Y_raw.unsqueeze(1)
    y_mean = Y_raw.mean(0)
    y_std = Y_raw.std(0)
    y_std[y_std < 1e-8] = 1.0
    Y = (Y_raw - y_mean) / y_std

    n_features = Y.shape[1]

    X_res = X.clone()
    Y_res = Y.clone()

    W = torch.zeros(n_visible,  n_components, device=device, dtype=dtype)  # X weights
    C = torch.zeros(n_features, n_components, device=device, dtype=dtype)  # Y weights
    P = torch.zeros(n_visible,  n_components, device=device, dtype=dtype)  # X loadings

    print(f"Running NIPALS PLS ({n_components} components)...")
    for k in range(n_components):
        # Initialise with the Y column that has the largest variance
        u = Y_res[:, int(Y_res.var(0).argmax())]

        for _ in range(max_iter):
            # X step
            w = X_res.T @ u
            w = w / w.norm()
            t = X_res @ w

            # Y step
            c = Y_res.T @ t
            c = c / c.norm()
            u_new = Y_res @ c

            if (u_new - u).norm() < tol:
                u = u_new
                break
            u = u_new

        # Store
        W[:, k] = w
        C[:, k] = c

        # X loading and deflation
        p = (X_res.T @ t) / (t @ t)
        P[:, k] = p
        X_res = X_res - t.unsqueeze(1) * p.unsqueeze(0)
        Y_res = Y_res - t.unsqueeze(1) * c.unsqueeze(0)

        print(f"  LV{k} done.")

    # PLS rotation: W* = W (P^T W)^{-1} gives orthogonal scores
    W_star = W @ torch.linalg.inv(P.T @ W)

    scale = n_visible ** 0.5
    proj_real = ((data_real - x_mean) @ W_star / scale).detach().cpu().numpy()
    proj_gen  = ((data_gen  - x_mean) @ W_star / scale).detach().cpu().numpy()

    return proj_real, proj_gen, W.detach().cpu().numpy(), C.detach().cpu().numpy()


def calculate_pls_r2(
    proj_real: np.ndarray,
    classical_features,
) -> np.ndarray:
    """
    Cumulative R² of predicting classical_features from the first k PLS
    latent variables, for k = 1, ..., n_components.

    Uses OLS with an intercept at each k so the result is comparable to
    sklearn's PLSRegression score — it measures how much of the feature
    variance the LV subspace actually captures.

    Parameters
    ----------
    proj_real : np.ndarray  (n_real, n_components)
        LV scores from run_pls_pipeline, aligned with classical_features.
    classical_features : torch.Tensor or np.ndarray  (n_real,) or (n_real, n_features)

    Returns
    -------
    r2 : np.ndarray  (n_components, n_features)
        r2[k, f] is the R² for feature f using the first k+1 LVs.
    """
    if torch.is_tensor(classical_features):
        Y = classical_features.detach().cpu().numpy().astype(np.float64)
    else:
        Y = np.array(classical_features, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]

    n_samples, n_features = Y.shape
    n_components = proj_real.shape[1]

    ss_tot = ((Y - Y.mean(0)) ** 2).sum(0)
    ss_tot = np.where(ss_tot > 0, ss_tot, 1.0)

    r2 = np.zeros((n_components, n_features))
    for k in range(1, n_components + 1):
        T_aug = np.hstack([proj_real[:, :k], np.ones((n_samples, 1))])
        B, _, _, _ = np.linalg.lstsq(T_aug, Y, rcond=None)
        ss_res = ((Y - T_aug @ B) ** 2).sum(0)
        r2[k - 1] = 1.0 - ss_res / ss_tot

    return r2


def run_sir_pipeline(
    data_real: torch.Tensor,
    data_gen: torch.Tensor,
    classical_features,
    n_components: int = 4,
    n_slices: int = 10,
    regularize: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Sliced Inverse Regression (SIR).

    Finds the directions in visible space along which E[X | Y] varies most,
    i.e. the directions most predictive of the classical feature.  Unlike PLS
    (which maximises X-Y covariance), SIR directly targets the conditional
    mean structure, which tends to give much higher R² for a single scalar Y.

    Algorithm (Li 1991):
      1. Whiten X: Z = (X - x̄) Σ_X^{-1/2}
      2. Slice Y into H quantile bins; compute the slice mean z̄_h for each bin.
      3. Form M = Σ_h (n_h/n) z̄_h z̄_hᵀ  (the between-slice variance matrix).
      4. Eigendecompose M; top eigenvectors → SIR directions in whitened space.
      5. Back-transform to original space: η_k = Σ_X^{-1/2} v_k.

    Parameters
    ----------
    data_real : torch.Tensor  (n_real, n_visible)
    data_gen  : torch.Tensor  (n_gen,  n_visible)
    classical_features : Tensor or ndarray  (n_real,) or (n_real, 1)
        Scalar response aligned with data_real.  If 2-D, the first column is used.
    n_components : int
        Number of SIR directions to return.
    n_slices : int
        Number of quantile slices of Y.  Rule of thumb: ~10-20.
    regularize : float
        Floor applied to eigenvalues of Σ_X before inversion (prevents blow-up
        for near-collinear features).

    Returns
    -------
    proj_real  : np.ndarray  (n_real, n_components)
    proj_gen   : np.ndarray  (n_gen,  n_components)
    eta        : np.ndarray  (n_visible, n_components)  — SIR direction matrix
    eigenvalues: np.ndarray  (n_components,)            — eigenvalues of M
    """
    # --- Convert to numpy ---
    X_np = data_real.detach().cpu().numpy().astype(np.float64)
    G_np = data_gen.detach().cpu().numpy().astype(np.float64)

    if torch.is_tensor(classical_features):
        Y_np = classical_features.detach().cpu().numpy().astype(np.float64)
    else:
        Y_np = np.array(classical_features, dtype=np.float64)
    if Y_np.ndim > 1:
        Y_np = Y_np[:, 0]   # SIR is for scalar Y; use first column if multi-dim

    n, p = X_np.shape
    assert len(Y_np) == n, "classical_features must have same length as data_real"

    print("1. Centering and whitening X...")
    x_mean = X_np.mean(0)
    X_c = X_np - x_mean

    Sigma = (X_c.T @ X_c) / n
    evals, evecs = np.linalg.eigh(Sigma)                    # ascending order
    inv_sqrt = np.where(evals > regularize, evals ** -0.5, 0.0)
    Sigma_inv_sqrt = evecs @ np.diag(inv_sqrt) @ evecs.T    # (p, p)
    Z = X_c @ Sigma_inv_sqrt                                 # (n, p) whitened

    print(f"2. Slicing Y into {n_slices} quantile bins...")
    boundaries = np.percentile(Y_np, np.linspace(0, 100, n_slices + 1))
    # Make boundaries unique to avoid empty slices from ties
    boundaries = np.unique(boundaries)
    actual_slices = len(boundaries) - 1

    print("3. Computing between-slice covariance M...")
    M = np.zeros((p, p))
    for h in range(actual_slices):
        lo, hi = boundaries[h], boundaries[h + 1]
        mask = (Y_np >= lo) & (Y_np <= hi if h == actual_slices - 1 else Y_np < hi)
        n_h = mask.sum()
        if n_h == 0:
            continue
        z_bar = Z[mask].mean(0)          # (p,)
        M += (n_h / n) * np.outer(z_bar, z_bar)

    print("4. Eigendecomposing M...")
    evals_M, evecs_M = np.linalg.eigh(M)
    # Descending order
    idx = np.argsort(evals_M)[::-1]
    V = evecs_M[:, idx[:n_components]]          # (p, n_components) in whitened space
    top_evals = evals_M[idx[:n_components]]

    # Back-transform to original space
    eta = Sigma_inv_sqrt @ V                    # (p, n_components)

    print("5. Projecting data...")
    scale = p ** 0.5
    proj_real = (X_c @ eta) / scale
    proj_gen  = ((G_np - x_mean) @ eta) / scale

    return proj_real, proj_gen, eta, top_evals


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
