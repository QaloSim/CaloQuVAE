import time
import numpy as np
from typing import Dict, Tuple, List, Optional, Any
from pathlib import Path
from datetime import datetime
import uuid
from dataclasses import dataclass, field
import torch
from dwave.embedding import unembed_sampleset
import dimod
import dwave.embedding
from dwave.preprocessing.composites import SpinReversalTransformComposite

def sample_logical_ising(qpu_sampler, h, J, num_samples=64, measure_time=False, chain_strength=None):
    h_min, h_max = qpu_sampler.child.properties['h_range']
    j_min, j_max = qpu_sampler.child.properties['extended_j_range']

    h_clamped = {k: max(h_min, min(v, h_max)) for k, v in h.items()}
    J_clamped = {k: max(j_min, min(v, j_max)) for k, v in J.items()}
    
    if chain_strength is None: chain_strength = j_max
        
    sample_kwargs = {
        'num_reads': num_samples, 'answer_mode': 'raw', 'auto_scale': False,
        'chain_break_fraction': True, 'chain_strength': chain_strength
    }

    start = time.perf_counter() if measure_time else 0
    response = qpu_sampler.sample_ising(h_clamped, J_clamped, **sample_kwargs)
    sampling_time = time.perf_counter() - start if measure_time else None
    
    if hasattr(response, 'record'):
        print(f"Fraction of broken chains: {np.mean(response.record.chain_break_fraction):.4f}")

    return response, sampling_time

def sample_ising_flux_bias(qpu_sampler, h, J, flux_biases, num_samples=1, measure_time=False, chain_strength=None, print_breaks=True):
    h_min, h_max = qpu_sampler.child.properties['h_range']
    j_min, j_max = qpu_sampler.child.properties['extended_j_range']
    
    h_clamped = {k: max(h_min, min(v, h_max)) for k, v in h.items()}
    J_clamped = {k: max(j_min, min(v, j_max)) for k, v in J.items()}
    
    if chain_strength is None: chain_strength = j_max
        
    sample_kwargs = {
        'num_reads': num_samples, 'answer_mode': 'raw', 'auto_scale': False,
        'chain_break_fraction': True, 'chain_strength': chain_strength,
        'flux_biases': flux_biases, 'flux_drift_compensation': False
    }
    
    start = time.perf_counter() if measure_time else 0
    response = qpu_sampler.sample_ising(h_clamped, J_clamped, **sample_kwargs)
    sampling_time = time.perf_counter() - start if measure_time else None

    if hasattr(response, 'record') and print_breaks:
        print(f"Fraction of broken chains: {np.mean(response.record.chain_break_fraction):.4f}")

    return response, sampling_time

def sample_manual_ising(raw_sampler, h_phys, J_phys, flux_biases, num_reads=100):
    props = raw_sampler.properties
    h_min, h_max = props['h_range']
    j_min, j_max = props['extended_j_range']

    h_clamped = {k: max(h_min, min(v, h_max)) for k, v in h_phys.items()}
    J_clamped = {k: max(j_min, min(v, j_max)) for k, v in J_phys.items()}
    
    response = raw_sampler.sample_ising(
        h_clamped, J_clamped,
        flux_biases=flux_biases, flux_drift_compensation=False,
        num_reads=num_reads, answer_mode='raw', auto_scale=False
    )
    return response

@dataclass
class ChainAnalysisResult:
    # --- 1. The Core Data ---
    logical_samples: torch.Tensor       
    variable_labels: list               
    
    # --- 2. Context ---
    metadata: Dict[str, Any] = field(default_factory=dict)

    # --- 3. Chain Break Stats ---
    clean_mask: torch.Tensor = None     
    breaks_per_sample: np.ndarray = None       
    breaks_per_variable: np.ndarray = None     
    break_matrix: np.ndarray = None            

    # --- 4. Physical Debugging Data ---
    physical_matrix: np.ndarray = None  
    physical_labels: list = None        
    embedding: dict = None              
    physical_response: dimod.SampleSet = None
    
    # --- 5. Spin Reversal Transform Data (New) ---
    srt_active: bool = False
    srt_mask: np.ndarray = None  # The boolean array defining the transform g_i
    
    def __post_init__(self):
        # Auto-calculate masks if missing
        if self.clean_mask is None and self.break_matrix is not None:
            self.breaks_per_sample = np.sum(self.break_matrix, axis=1)
            self.breaks_per_variable = np.sum(self.break_matrix, axis=0)
            self.clean_mask = torch.tensor(
                (self.breaks_per_sample == 0), 
                device=self.logical_samples.device, 
                dtype=torch.bool
            )

    @property
    def total_clean_fraction(self):
        if self.clean_mask is not None:
            return self.clean_mask.float().mean().item()
        return 0.0

    def save(self, directory: str = "./saved_samples", prefix: str = "dwave_run"):
        save_path = Path(directory)
        save_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_pct = int(self.total_clean_fraction * 100)
        run_id = str(uuid.uuid4())[:6]
        
        # Add SRT tag to filename if used? Optional, but useful.
        srt_tag = "_srt" if self.srt_active else ""
        
        filename = f"{prefix}{srt_tag}_{timestamp}_clean{clean_pct}_{run_id}.pt"
        full_path = save_path / filename
        
        torch.save(self, full_path)

def sample_physical_with_analysis(
    raw_sampler, 
    h_logical, 
    J_logical, 
    embedding, 
    flux_biases, 
    num_samples=1, 
    chain_strength=None,
    device='cpu',
    metadata: dict = None,    # <--- NEW: Pass context here
    save_dir: str = None      # <--- NEW: If string provided, auto-saves
):
    if metadata is None: metadata = {}

    # --- 1. Embed and Sample ---
    # (Standard embedding logic...)
    target_adj = raw_sampler.adjacency
    h_phys, J_phys = dwave.embedding.embed_ising(
        h_logical, J_logical, embedding, target_adj, chain_strength=chain_strength
    )
    
    # Clamp values to QPU ranges
    h_min, h_max = raw_sampler.properties['h_range']
    j_min, j_max = raw_sampler.properties['extended_j_range']
    h_phys = {k: max(h_min, min(v, h_max)) for k, v in h_phys.items()}
    J_phys = {k: max(j_min, min(v, j_max)) for k, v in J_phys.items()}

    # Sample
    sample_kwargs = {
        'num_reads': num_samples, 
        'answer_mode': 'raw', 
        'auto_scale': False,
        'flux_biases': flux_biases, 
        'flux_drift_compensation': False
    }
    physical_response = raw_sampler.sample_ising(h_phys, J_phys, **sample_kwargs)
    
    # --- 2. Unembed ---
    source_bqm = dimod.BinaryQuadraticModel.from_ising(h_logical, J_logical)
    logical_response = unembed_sampleset(
        target_sampleset=physical_response,
        embedding=embedding,
        source_bqm=source_bqm,
        chain_break_method=dwave.embedding.chain_breaks.majority_vote,
        chain_break_fraction=True
    )
    
    # --- 3. Analyze Chains ---
    logical_vars_ordered = list(logical_response.variables)
    phys_matrix = physical_response.record.sample
    
    # Calculate break matrix
    n_logical = len(logical_vars_ordered)
    n_samples_actual = phys_matrix.shape[0]
    break_matrix = np.zeros((n_samples_actual, n_logical), dtype=bool)
    
    phys_labels = list(physical_response.variables)
    phys_label_to_col = {lbl: i for i, lbl in enumerate(phys_labels)}

    for col_idx, logical_var in enumerate(logical_vars_ordered):
        chain = embedding.get(logical_var, [])
        if len(chain) <= 1: continue 
        chain_cols = [phys_label_to_col[q] for q in chain]
        chain_vals = phys_matrix[:, chain_cols]
        # Broken if min != max
        break_matrix[:, col_idx] = (np.min(chain_vals, axis=1) != np.max(chain_vals, axis=1))

    # --- 4. Pack & Save ---
    log_tensor = torch.tensor(logical_response.record.sample, dtype=torch.float32, device=device)
    
    total_breaks = np.sum(break_matrix)
        
    result = ChainAnalysisResult(
        logical_samples = log_tensor,
        variable_labels = logical_vars_ordered,
        metadata = metadata,
        break_matrix = break_matrix,
        physical_matrix = phys_matrix,
        physical_labels = phys_labels,
        embedding = embedding,
        physical_response = physical_response
    )
    
    if save_dir:
        # Construct a useful prefix based on metadata
        prefix = "dwave"
        if 'source' in metadata: 
            prefix = f"{metadata['source']}_"
        
        result.save(directory=save_dir, prefix=prefix)

    return result

def sample_physical_with_analysis_srt(
    raw_sampler, 
    h_logical, 
    J_logical, 
    embedding, 
    flux_biases, 
    num_samples=1, 
    chain_strength=None,
    device='cpu',
    metadata: dict = None,
    save_dir: str = None,
    use_srt: bool = True
):
    if metadata is None: metadata = {}

    # --- 1. Embed ---
    target_adj = raw_sampler.adjacency
    h_phys, J_phys = dwave.embedding.embed_ising(
        h_logical, J_logical, embedding, target_adj, chain_strength=chain_strength
    )
    
    # Clamp values
    h_min, h_max = raw_sampler.properties['h_range']
    j_min, j_max = raw_sampler.properties['extended_j_range']
    h_phys = {k: max(h_min, min(v, h_max)) for k, v in h_phys.items()}
    J_phys = {k: max(j_min, min(v, j_max)) for k, v in J_phys.items()}

    # --- 2. Construct BQM ---
    bqm_phys = dimod.BinaryQuadraticModel.from_ising(h_phys, J_phys)

    # --- 3. Prepare Sampler & SRT Mask ---
    sample_kwargs = {
        'num_reads': num_samples, 
        'answer_mode': 'raw', 
        'auto_scale': False,
        'flux_biases': flux_biases, 
        'flux_drift_compensation': False
    }

    final_srt_mask = None # Default if no SRT used

    if use_srt:
        active_sampler = SpinReversalTransformComposite(raw_sampler)
        
        mask_list = []
        for var_label in bqm_phys.variables:
            
            # Check for flux bias protection
            is_protected = False
            if var_label < len(flux_biases):
                if abs(flux_biases[var_label]) > 1e-5:
                    is_protected = True
            
            if is_protected:
                mask_list.append(False) # Protected: DO NOT FLIP
            else:
                mask_list.append(bool(np.random.choice([True, False])))
        
        # Create the array for the sampler
        final_srt_mask = np.array([mask_list], dtype=bool)
        
        # Pass the mask to kwargs
        sample_kwargs['srts'] = final_srt_mask
    else:
        active_sampler = raw_sampler

    # --- 4. Sample ---
    physical_response = active_sampler.sample(bqm_phys, **sample_kwargs)
    physical_response = physical_response.change_vartype(dimod.SPIN, inplace=False)
    
    # --- 5. Unembed ---
    source_bqm = dimod.BinaryQuadraticModel.from_ising(h_logical, J_logical)
    logical_response = dwave.embedding.unembed_sampleset(
        target_sampleset=physical_response,
        embedding=embedding,
        source_bqm=source_bqm,
        chain_break_method=dwave.embedding.chain_breaks.majority_vote,
        chain_break_fraction=True
    )
    
    # --- 6. Analyze Chains ---
    logical_vars_ordered = list(logical_response.variables)
    phys_matrix = physical_response.record.sample
    
    n_logical = len(logical_vars_ordered)
    n_samples_actual = phys_matrix.shape[0]
    break_matrix = np.zeros((n_samples_actual, n_logical), dtype=bool)
    
    phys_labels = list(physical_response.variables)
    phys_label_to_col = {lbl: i for i, lbl in enumerate(phys_labels)}

    for col_idx, logical_var in enumerate(logical_vars_ordered):
        chain = embedding.get(logical_var, [])
        if len(chain) <= 1: continue 
        chain_cols = [phys_label_to_col[q] for q in chain if q in phys_label_to_col]
        if not chain_cols: continue
        chain_vals = phys_matrix[:, chain_cols]
        break_matrix[:, col_idx] = (np.min(chain_vals, axis=1) != np.max(chain_vals, axis=1))

    # --- 7. Pack & Save ---
    log_tensor = torch.tensor(logical_response.record.sample, dtype=torch.float32, device=device)
        
    result = ChainAnalysisResult(
        logical_samples = log_tensor,
        variable_labels = logical_vars_ordered,
        metadata = metadata,
        break_matrix = break_matrix,
        physical_matrix = phys_matrix,
        physical_labels = phys_labels,
        embedding = embedding,
        physical_response = physical_response,
        # --- Store SRT Metadata ---
        srt_active = use_srt,
        srt_mask = final_srt_mask
    )
    
    if save_dir:
        prefix = f"{metadata.get('source', 'dwave')}_"
        result.save(directory=save_dir, prefix=prefix)

    return result