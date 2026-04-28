from utils.dwave.sampling_backend import ChainAnalysisResult
import torch
import numpy as np
from typing import Tuple, List, Any, Union
import re
from pathlib import Path


def process_rbm_samples(response, num_visible, num_hidden, device):
    total_nodes = num_visible + num_hidden
    num_samples = len(response.record.sample)
    var_to_col_idx = {var: i for i, var in enumerate(response.variables)}
    
    try:
        ordered_cols = [var_to_col_idx[i] for i in range(total_nodes)]
    except KeyError as e:
        print(f"Error: Missing logical node {e}. Returning zeros.")
        return torch.zeros((num_samples, num_visible), device=device), torch.zeros((num_samples, num_hidden), device=device)

    dwave_spin_samples = response.record.sample[:, ordered_cols]
    dwave_spin_samples_t = torch.tensor(dwave_spin_samples, dtype=torch.float, device=device)
    dwave_binary_samples = torch.where(dwave_spin_samples_t == -1.0, torch.tensor(0.0, device=device), dwave_spin_samples_t)

    return dwave_binary_samples[:, :num_visible], dwave_binary_samples[:, num_visible:]

def calculate_manual_chain_breaks(response, embedding_map):
    label_to_idx = {label: i for i, label in enumerate(response.variables)}
    samples = response.record.sample
    num_samples = samples.shape[0]
    total_chains = len(embedding_map)
    total_broken_chains = 0
    
    for logical_id, chain in embedding_map.items():
        if len(chain) < 2: continue
        chain_col_indices = [label_to_idx[q] for q in chain if q in label_to_idx]
        if len(chain_col_indices) < 2: continue
            
        chain_spins = samples[:, chain_col_indices]
        row_mins = np.min(chain_spins, axis=1)
        row_maxs = np.max(chain_spins, axis=1)
        total_broken_chains += np.sum(row_mins != row_maxs)

    return total_broken_chains / (num_samples * total_chains)

def unembed_raw_samples(response, embedding_map, num_visible, num_hidden, device):
    label_to_idx = {label: i for i, label in enumerate(response.variables)}
    phys_samples_np = response.record.sample 
    num_samples = phys_samples_np.shape[0]
    total_logical = num_visible + num_hidden
    logical_samples = np.zeros((num_samples, total_logical), dtype=np.float32)
    
    for logical_id in range(total_logical):
        if logical_id not in embedding_map: continue
        chain = embedding_map[logical_id]
        chain_indices = [label_to_idx[q] for q in chain if q in label_to_idx]
        if not chain_indices: continue
            
        chain_vals = phys_samples_np[:, chain_indices] 
        chain_sum = np.sum(chain_vals, axis=1) 
        logical_samples[:, logical_id] = np.where(chain_sum > 0, 1.0, 0.0)
        
    t_logical = torch.tensor(logical_samples, device=device, dtype=torch.float32)
    return t_logical[:, :num_visible], t_logical[:, num_visible:]

def process_expanded_rbm_samples(response, num_visible, num_hidden, conditioning_sets, device):
    total_nodes = num_visible + num_hidden
    num_cond = len(conditioning_sets)
    var_to_col_idx = {var: i for i, var in enumerate(response.variables)}
    ordered_cols = []
    
    for i in range(total_nodes):
        if i < num_cond:
            phys_q = sorted(list(conditioning_sets[i]))[0]
            target_label = f"C{i}_{phys_q}"
        else:
            target_label = i
            
        if target_label in var_to_col_idx:
            ordered_cols.append(var_to_col_idx[target_label])
        else:
            ordered_cols.append(0)

    dwave_spin_samples = response.record.sample[:, ordered_cols]
    dwave_spin_samples_t = torch.tensor(dwave_spin_samples, dtype=torch.float, device=device)
    dwave_binary_samples = torch.where(dwave_spin_samples_t == -1.0, torch.tensor(0.0, device=device), dwave_spin_samples_t)

    return dwave_binary_samples[:, :num_visible], dwave_binary_samples[:, num_visible:]


def process_analysis_result(analysis_result, rbm, conditioning_sets):
    """
    Extracts and sorts Visible/Hidden samples from a ChainAnalysisResult object.
    
    Ensures columns are ordered: [v_0, ... v_n, h_0, ... h_m].
    Handles the conversion from Spin (-1/+1) to Binary (0/1).
    """
    # 1. Setup Dimensions
    n_vis = rbm.params["vbias"].shape[0]
    n_hid = rbm.params["hbias"].shape[0]
    total_nodes = n_vis + n_hid
    num_cond = len(conditioning_sets)
    
    # 2. Map variable labels to their column index in the sample tensor
    # logical_samples is shape (n_samples, n_logical_vars)
    # variable_labels is a list of names corresponding to columns
    label_to_col_idx = {name: i for i, name in enumerate(analysis_result.variable_labels)}
    
    ordered_col_indices = []
    
    # 3. Build the index list in the strict (Visible + Hidden) order
    for i in range(total_nodes):
        if i < num_cond:
            # Reconstruct the special label used during embedding
            # e.g., "C0_1234" where 1234 is the physical qubit index
            phys_q = sorted(list(conditioning_sets[i]))[0]
            target_label = f"C{i}_{phys_q}"
        else:
            # Standard nodes are just integer labels
            target_label = i
            
        if target_label in label_to_col_idx:
            ordered_col_indices.append(label_to_col_idx[target_label])
        else:
            # Fallback for missing variables (should rarely happen in rigorous mode)
            print(f"Warning: Missing label {target_label} in analysis result.")
            ordered_col_indices.append(0)

    # 4. Extract and Convert
    # Use the indices to reorder the columns
    # analysis_result.logical_samples is already a Tensor on the correct device
    raw_samples = analysis_result.logical_samples[:, ordered_col_indices]
    
    # Convert Spin (-1/+1) to Binary (0/1)
    # If samples are already 0/1, this logic still works if checks are robust, 
    # but usually D-Wave returns -1/+1.
    binary_samples = torch.where(
        raw_samples == -1.0,
        torch.zeros(1, device=raw_samples.device),
        raw_samples
    )
    
    # 5. Split
    v_s = binary_samples[:, :n_vis]
    h_s = binary_samples[:, n_vis:]
    
    return v_s, h_s



def batch_process_visible_samples(
    source_dir: str, 
    out_dir: str,
    rbm: Any, 
    conditioning_sets: List[Any]
):
    """
    Function 1: Reads ChainAnalysisResult objects from source_dir, 
    extracts visible samples, and saves the resulting tensors to out_dir.
    """
    src_path = Path(source_dir)
    dest_path = Path(out_dir)
    
    # Create the separate output directory if it doesn't exist
    dest_path.mkdir(parents=True, exist_ok=True)
    
    files = list(src_path.glob("*.pt"))
    print(f"Found {len(files)} raw files in {src_path}. Processing to {dest_path}...")
    
    count = 0
    skipped = 0
    
    for file_path in files:
        try:
            # 1. Load the heavy ChainAnalysisResult object
            # map_location='cpu' is safer if you are just extracting data and not computing
            result = torch.load(file_path, map_location=rbm.device, weights_only=False)
            
            # 2. Extract Visible Units
            # (Assumes process_analysis_result is defined/imported)
            v_s, _ = process_analysis_result(result, rbm, conditioning_sets)
            
            # 3. Create new filename
            # Input:  prod_E1000__srt_...
            # Output: vis_E1000__srt_...
            new_name = f"vis_{file_path.name}"
            save_path = dest_path / new_name
            
            # 4. Save lightweight tensor
            torch.save(v_s, save_path)
            count += 1
            
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            skipped += 1
            
    print(f"Complete. Processed: {count}, Skipped: {skipped}. Saved to {dest_path}")


def load_and_concatenate_energy(
    source_dir: str, 
    energy: int
) -> torch.Tensor:
    """
    Function 2: Looks into source_dir (which should be the out_dir from Function 1),
    finds all tensors matching the specific energy, and concatenates them.
    """
    search_path = Path(source_dir)
    
    if not search_path.exists():
        raise FileNotFoundError(f"Directory {search_path} does not exist.")

    # Strict pattern matching for energy
    # Looks for files containing "_E{energy}__" to distinguish E1000 from E10000
    file_pattern = f"*_E{energy}__*.pt"
    files = list(search_path.glob(file_pattern))
    
    if not files:
        print(f"No files found for Energy {energy} in {search_path}")
        return torch.empty(0)

    print(f"Found {len(files)} tensor files for Energy {energy}. Concatenating...")
    
    tensor_list = []
    
    for f in files:
        try:
            # Load the tensor (CPU is recommended for simple concatenation)
            t = torch.load(f, map_location='cpu')
            tensor_list.append(t)
        except Exception as e:
            print(f"Error reading {f.name}: {e}")

    if not tensor_list:
        return torch.empty(0)

    # Concatenate along batch dimension (dim 0)
    combined_samples = torch.cat(tensor_list, dim=0)
    
    print(f"Output Shape for Energy {energy}: {combined_samples.shape}")
    return combined_samples


def reprocess_qpu_samples(
    save_dir: str,
    rbm,
    conditioning_sets: List[Any],
) -> None:
    """
    Retroactively creates processed_samples.pt from raw ChainAnalysisResult
    files for runs that were saved before mass_sample_dwave_single started
    doing this automatically.

    Reads every hetero_p*.pt in save_dir, extracts properly-ordered binary
    visible/hidden samples via process_analysis_result, and writes a single
    processed_samples.pt with keys: v, h, clean_mask, pattern_indices.

    Args:
        save_dir: Directory containing hetero_p*.pt ChainAnalysisResult files.
        rbm: The RBM model (needed to reorder logical variables).
        conditioning_sets: The conditioning sets used during sampling.
    """
    save_path = Path(save_dir)
    idx_re = re.compile(r"hetero_p(\d+)_")

    entries = []
    for f in save_path.glob("hetero_p*.pt"):
        m = idx_re.match(f.name)
        if m:
            entries.append((int(m.group(1)), f))

    if not entries:
        raise FileNotFoundError(f"No hetero_p*.pt files found in {save_dir}")

    entries.sort(key=lambda x: x[0])

    v_list, h_list, clean_list, idx_list = [], [], [], []
    skipped = 0

    for idx, fpath in entries:
        try:
            result = torch.load(fpath, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"Error loading {fpath.name}: {e}")
            skipped += 1
            continue

        v_s, h_s = process_analysis_result(result, rbm, conditioning_sets)
        v_list.append(v_s)
        h_list.append(h_s)
        clean_list.append(result.clean_mask.cpu())
        idx_list.extend([idx] * v_s.shape[0])

    if not v_list:
        raise RuntimeError("No valid samples collected from any file.")

    out = {
        'v': torch.cat(v_list, dim=0),
        'h': torch.cat(h_list, dim=0),
        'clean_mask': torch.cat(clean_list, dim=0),
        'pattern_indices': torch.tensor(idx_list, dtype=torch.long),
    }

    out_path = save_path / "processed_samples.pt"
    torch.save(out, out_path)

    n_files = len(entries) - skipped
    print(f"Reprocessed {out['v'].shape[0]} samples from {n_files} files"
          f"{f' (skipped {skipped})' if skipped else ''}")
    print(f"  v: {out['v'].shape}  h: {out['h'].shape}")
    print(f"Saved to {out_path}")


def _load_qpu_samples_single(
    save_dir: str,
    clean_only: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    save_path = Path(save_dir)

    data = torch.load(save_path / "processed_samples.pt", map_location="cpu")
    incidence_energy = torch.load(save_path / "incidence_energy.pt", map_location="cpu")
    u_samples = torch.load(save_path / "u_samples.pt", map_location="cpu")
    E_samples = torch.load(save_path / "E_samples.pt", map_location="cpu")

    v = data['v']
    clean_mask = data['clean_mask']
    pattern_idx = data['pattern_indices']

    if clean_only:
        keep = clean_mask.bool()
        v = v[keep]
        pattern_idx = pattern_idx[keep]

    energies = incidence_energy[pattern_idx]
    u = u_samples[pattern_idx]
    E = E_samples[pattern_idx]

    return v, energies, u, E


def load_qpu_samples(
    save_dir: Union[str, List[str]],
    clean_only: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Loads processed QPU samples aligned with their incidence energies, u values,
    and raw transfusion energy samples.

    Expects each directory to contain (created by dwave_samples.py):
      - processed_samples.pt  (dict with v, h, clean_mask, pattern_indices)
      - incidence_energy.pt   (num_patterns, 1)  — x0 in MeV
      - u_samples.pt          (num_patterns, n_features) — normalised u
      - E_samples.pt          (num_patterns, n_features) — raw transfusion output

    Args:
        save_dir: Directory or list of directories containing saved files.
                  When a list is given, samples from all directories are concatenated.
        clean_only: If True, keep only chain-break-free samples.

    Returns:
        v_samples:  (N, n_vis)       visible unit samples in binary {0,1}
        energies:   (N, 1)           incidence energies (x0, MeV)
        u:          (N, n_features)  normalised u values
        E:          (N, n_features)  raw transfusion layer-energy samples
    """
    dirs = [save_dir] if isinstance(save_dir, str) else list(save_dir)

    all_v, all_energies, all_u, all_E = [], [], [], []
    for d in dirs:
        v, energies, u, E = _load_qpu_samples_single(d, clean_only)
        all_v.append(v)
        all_energies.append(energies)
        all_u.append(u)
        all_E.append(E)
        print(f"  [{d}] {v.shape[0]} samples")

    v_out = torch.cat(all_v, dim=0)
    energies_out = torch.cat(all_energies, dim=0)
    u_out = torch.cat(all_u, dim=0)
    E_out = torch.cat(all_E, dim=0)

    label = f"{len(dirs)} dirs" if len(dirs) > 1 else dirs[0]
    print(f"Loaded {v_out.shape[0]} samples total from {label}{' (clean only)' if clean_only else ''}")
    print(f"  v: {v_out.shape}  energies: {energies_out.shape}  u: {u_out.shape}  E: {E_out.shape}")

    return v_out, energies_out, u_out, E_out