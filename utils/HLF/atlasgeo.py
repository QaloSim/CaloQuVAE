import torch
import torch.nn as nn
import h5py
import numpy as np
import math
from utils.atlas_plots import to_np, make_validation_plots

class AtlasGeometry:
    """
    Handles loading ATLAS calorimeter binning information.
    Fixes the geometric projection to match legacy HighLevelFeatures.
    """
    def __init__(self, filename, relevant_layers=[0, 1, 2, 3, 12]):
        self.filename = filename
        self.relevant_layers = relevant_layers
        self.voxels_per_layer = 14 * 24  # Standard ATLAS dimensions
        
        self._load_h5_data()
        
        self.eta_centers = {}
        self.phi_centers = {}
        self.r_centers = {}  # Store raw r
        
        for layer in self.relevant_layers:
            layer_str = str(layer)
            
            # Calculate r_c
            r_c = self.binstart_radius[layer_str] + self.binsize_radius[layer_str] / 2.0
            alpha_c = self.binstart_alpha[layer_str] + self.binsize_alpha[layer_str] / 2.0
            
            # Save raw r
            self.r_centers[layer] = r_c.float()

            # Project to Eta/Phi
            eta_c = r_c * torch.cos(alpha_c)
            phi_c = r_c * torch.sin(alpha_c)
            
            self.eta_centers[layer] = eta_c.float()
            self.phi_centers[layer] = phi_c.float()


    def _load_h5_data(self):
        self.binsize_alpha = {}
        self.binstart_alpha = {}
        self.binsize_radius = {}
        self.binstart_radius = {}
        
        with h5py.File(self.filename, 'r') as f:
            for key in f.keys():
                if "layer_" not in key: continue
                layer = key.split("_")[-1]
                val = torch.from_numpy(np.array(f[key]))
                
                if "binsize_alpha" in key: self.binsize_alpha[layer] = val
                elif "binstart_alpha" in key: self.binstart_alpha[layer] = val
                elif "binsize_radius" in key: self.binsize_radius[layer] = val
                elif "binstart_radius" in key: self.binstart_radius[layer] = val

class DifferentiableFeatureExtractor(nn.Module):
    def __init__(self, geometry_handler: AtlasGeometry):
        super().__init__()
        self.relevant_layers = geometry_handler.relevant_layers
        self.num_layers = len(self.relevant_layers)
        self.voxels_per_layer = geometry_handler.voxels_per_layer
        
        # Stack grids
        eta_grid_list = [geometry_handler.eta_centers[l] for l in self.relevant_layers]
        phi_grid_list = [geometry_handler.phi_centers[l] for l in self.relevant_layers]
        
        # Buffers
        self.register_buffer('eta_grid', torch.stack(eta_grid_list))   # (L, V)
        self.register_buffer('phi_grid', torch.stack(phi_grid_list))   # (L, V)

        r_grid_list = [geometry_handler.r_centers[l] for l in self.relevant_layers]
        raw_r_stack = torch.stack(r_grid_list) # Shape: (Num_Layers, Voxels_Per_Layer)

        # 1. Compute Min and Max per layer (dim=1)
        # keepdim=True ensures shape remains (Num_Layers, 1) for broadcasting
        r_min = raw_r_stack.min(dim=1, keepdim=True)[0]
        r_max = raw_r_stack.max(dim=1, keepdim=True)[0]

        # 2. Normalize to 0-1 range
        # Added epsilon to prevent div/0 if a layer happens to have constant r (unlikely but safe)
        r_norm = (raw_r_stack - r_min) / (r_max - r_min + 1e-6)

        self.register_buffer('r_grid_norm', r_norm)
        
        self.epsilon = 1e-6

    def forward(self, showers):
        """
        Args:
            showers: (Batch, Total_Voxels) or (Batch, Layers, Voxels)
        """
        if showers.dim() == 2:
            B, Total = showers.shape
            showers = showers.view(B, self.num_layers, self.voxels_per_layer)
        
        # --- Energy Calculation ---
        E_layer_tot = torch.sum(showers, dim=2) # (B, L)
        E_tot = torch.sum(E_layer_tot, dim=1)   # (B, )

        # Safe denominator
        E_layer_denom = torch.clamp(E_layer_tot, min=self.epsilon)

        # --- First Moments (Centers) ---
        # Note: Using 'eta' instead of 'r'
        Eta_weighted = torch.sum(showers * self.eta_grid.unsqueeze(0), dim=2)
        Phi_weighted = torch.sum(showers * self.phi_grid.unsqueeze(0), dim=2)
        
        Eta_center = Eta_weighted / E_layer_denom
        Phi_center = Phi_weighted / E_layer_denom
        
        # Masking zero energy layers
        mask = (E_layer_tot < self.epsilon)
        zeros = torch.zeros_like(Eta_center)
        Eta_center = torch.where(mask, zeros, Eta_center)
        Phi_center = torch.where(mask, zeros, Phi_center)

        # --- Second Moments (Widths) ---
        # (x - mu)
        diff_eta = self.eta_grid.unsqueeze(0) - Eta_center.unsqueeze(2) 
        diff_phi = self.phi_grid.unsqueeze(0) - Phi_center.unsqueeze(2)
        
        # Weighted Variance: sum(w * (x-mu)^2) / sum(w)
        var_eta = torch.sum(showers * (diff_eta ** 2), dim=2) / E_layer_denom
        var_phi = torch.sum(showers * (diff_phi ** 2), dim=2) / E_layer_denom
        
        safe_var_eta = torch.clamp(var_eta, min=1e-8)
        safe_var_phi = torch.clamp(var_phi, min=1e-8)
        
        width_eta = torch.sqrt(safe_var_eta)
        width_phi = torch.sqrt(safe_var_phi)

        # Mask output
        width_eta = torch.where(mask, zeros, width_eta)
        width_phi = torch.where(mask, zeros, width_phi)

        return {
            "E_tot": E_tot,          
            "E_layer": E_layer_tot,  
            "Eta_center": Eta_center,    
            "Phi_center": Phi_center,
            "Eta_width": width_eta,      
            "Phi_width": width_phi   
        }

class FeatureAdapter:
    """
    Adapts DifferentiableFeatureExtractor output to match 
    HighLevelFeatures structure for plotting.
    """
    def __init__(self, features_dict, relevant_layers, e_inc):
        self.relevantLayers = relevant_layers
        self.Einc = to_np(e_inc).flatten()
        self.E_tot = to_np(features_dict['E_tot'])
        
        self.E_layers = {}
        # Changed keys from EC_rs -> EC_etas to match legacy plotting expectations
        self.EC_etas = {} 
        self.EC_phis = {}
        self.width_etas = {}
        self.width_phis = {}

        for i, layer_id in enumerate(self.relevantLayers):
            self.E_layers[layer_id] = to_np(features_dict['E_layer'][:, i])
            # Mapping from 'Eta_center' tensor to 'EC_etas' dict
            self.EC_etas[layer_id]    = to_np(features_dict['Eta_center'][:, i])
            self.EC_phis[layer_id]  = to_np(features_dict['Phi_center'][:, i])
            self.width_etas[layer_id] = to_np(features_dict['Eta_width'][:, i])
            self.width_phis[layer_id] = to_np(features_dict['Phi_width'][:, i])



def compute_radial_overlap_area(min1, max1, min2, max2):
    """
    Computes the 'area' overlap in the radial dimension.
    Corresponds to integral of r*dr -> 0.5 * (r_end^2 - r_start^2).
    """
    intersect_min = torch.max(min1, min2)
    intersect_max = torch.min(max1, max2)
    
    valid_mask = (intersect_max > intersect_min)
    r_sq_diff = (intersect_max ** 2 - intersect_min ** 2)
    
    return torch.where(valid_mask, 0.5 * r_sq_diff, torch.zeros_like(r_sq_diff))

def compute_linear_overlap(min1, max1, min2, max2):
    """
    Computes simple linear overlap (max - min).
    Used for the ANGULAR dimension (integral of d_alpha).
    """
    intersect_min = torch.max(min1, min2)
    intersect_max = torch.min(max1, max2)
    
    overlap = torch.clamp(intersect_max - intersect_min, min=0.0)
    return overlap

def compute_periodic_overlap(min1, max1, min2, max2, period=2*math.pi):
    """
    Computes overlap for cyclic coordinates (phi).
    Checks nominal overlap plus +/- period shifts.
    USES LINEAR OVERLAP.
    """
    # Nominal overlap
    ov_0 = compute_linear_overlap(min1, max1, min2, max2)
    
    # Target shifted +2pi 
    ov_plus = compute_linear_overlap(min1, max1, min2 + period, max2 + period)
    
    # Target shifted -2pi
    ov_minus = compute_linear_overlap(min1, max1, min2 - period, max2 - period)
    
    return ov_0 + ov_plus + ov_minus

# --- 2. The Downsampler Class ---
class CaloDownsampler(torch.nn.Module):
    def __init__(self, source_geo, target_geo):
        super().__init__()
        
        # 1. Flatten Geometries (Same as before)
        src_r_min, src_r_max, src_a_min, src_a_max, src_layer_ids = self._flatten_geometry(source_geo)
        tgt_r_min, tgt_r_max, tgt_a_min, tgt_a_max, tgt_layer_ids = self._flatten_geometry(target_geo)
        
        # ... [Layer matching logic stays the same] ...
        layer_match = (tgt_layer_ids.unsqueeze(1) == src_layer_ids.unsqueeze(0)).float()
        
        # --- CORRECTED AREA CALCULATIONS ---
        
        # 2. Radial Area Overlap (Difference of Squares)
        # Input shapes: (1, N_src) and (N_tgt, 1)
        r_area_overlap = compute_radial_overlap_area(
            src_r_min.unsqueeze(0), src_r_max.unsqueeze(0),
            tgt_r_min.unsqueeze(1), tgt_r_max.unsqueeze(1)
        )
        
        # 3. Angular Overlap (Linear) - Unchanged
        # Angle behaves linearly in the area integral
        a_overlap = compute_periodic_overlap(
            src_a_min.unsqueeze(0), src_a_max.unsqueeze(0),
            tgt_a_min.unsqueeze(1), tgt_a_max.unsqueeze(1)
        )
        
        # 4. Intersection Area
        intersection_area = r_area_overlap * a_overlap * layer_match
        
        # 5. Source Physical Area
        # Area = 0.5 * (r_max^2 - r_min^2) * delta_alpha
        src_r_factor = 0.5 * (src_r_max**2 - src_r_min**2)
        src_a_factor = (src_a_max - src_a_min)
        src_area = src_r_factor * src_a_factor
        
        # 6. Weight Matrix
        W = intersection_area / torch.clamp(src_area.unsqueeze(0), min=1e-8)
        
        self.register_buffer('transfer_matrix', W.to_sparse())


    def _flatten_geometry(self, geo):
        r_mins, r_maxs = [], []
        a_mins, a_maxs = [], []
        layer_ids = []
        
        for layer_idx in geo.relevant_layers:
            l_str = str(layer_idx)
            
            # Use data straight from the geometry class
            r_start = geo.binstart_radius[l_str].float()
            r_size  = geo.binsize_radius[l_str].float()
            a_start = geo.binstart_alpha[l_str].float()
            a_size  = geo.binsize_alpha[l_str].float()
            
            # Store edges
            r_mins.append(r_start)
            r_maxs.append(r_start + r_size)
            a_mins.append(a_start)
            a_maxs.append(a_start + a_size)
            
            # Layer ID
            num_voxels = r_start.shape[0]
            layer_ids.append(torch.full((num_voxels,), layer_idx, dtype=torch.float))
            
        return (
            torch.cat(r_mins), torch.cat(r_maxs),
            torch.cat(a_mins), torch.cat(a_maxs),
            torch.cat(layer_ids)
        )

    def forward(self, source_showers):
        """
        Args:
            source_showers: (Batch, N_source_voxels)
        Returns:
            target_showers: (Batch, N_target_voxels)
        """
        if source_showers.dim() == 3:
             B, L, V = source_showers.shape
             source_showers = source_showers.view(B, -1)
             
        # Sparse Matrix Multiplication
        # (N_t, N_s) @ (N_s, B) -> (N_t, B)
        # Note: PyTorch sparse mm requires (Sparse, Dense)
        downsampled = torch.sparse.mm(self.transfer_matrix, source_showers.t()).t()
        
        return downsampled


class VariableLayerFeatureExtractor(nn.Module):
    """
    A feature extractor that handles layers with different numbers of voxels.
    Replaces DifferentiableFeatureExtractor for 'Optimal' binning schemes.
    """
    def __init__(self, geometry_handler):
        super().__init__()
        self.relevant_layers = geometry_handler.relevant_layers
        self.epsilon = 1e-6
        
        # --- Pre-calculate slices and flatten grids ---
        self.layer_slices = []
        eta_centers_list = []
        phi_centers_list = []
        
        current_idx = 0
        for layer in self.relevant_layers:
            # Get the grid for this specific layer
            l_eta = geometry_handler.eta_centers[layer].float()
            l_phi = geometry_handler.phi_centers[layer].float()
            
            # Record the size and create a slice for indexing the flat input
            num_voxels = l_eta.size(0)
            self.layer_slices.append(slice(current_idx, current_idx + num_voxels))
            current_idx += num_voxels
            
            # Accumulate centers
            eta_centers_list.append(l_eta)
            phi_centers_list.append(l_phi)
            
        # Concatenate everything into 1D Buffers (Total_Voxels, )
        # This avoids the RuntimeError from torch.stack on mismatched sizes
        self.register_buffer('flat_eta_grid', torch.cat(eta_centers_list))
        self.register_buffer('flat_phi_grid', torch.cat(phi_centers_list))

    def forward(self, showers):
        """
        Args:
            showers: (Batch, Total_Voxels) flattened input.
        Returns:
            Dict matching the structure of DifferentiableFeatureExtractor output.
        """
        # Ensure input is flat (Batch, Total_Voxels)
        if showers.dim() == 3:
            B, L, V = showers.shape
            showers = showers.view(B, -1)
            
        B = showers.shape[0]
        
        # Storage for per-layer results
        E_layers = []
        Eta_centers = []
        Phi_centers = []
        Eta_widths = []
        Phi_widths = []

        # Iterate through layers using pre-calculated slices
        # This replaces the vectorized operations over dim=1 from the old class
        for i, slc in enumerate(self.layer_slices):
            # 1. Extract data for this layer
            # shape: (Batch, Voxels_In_This_Layer)
            layer_shower = showers[:, slc]
            layer_eta_grid = self.flat_eta_grid[slc].unsqueeze(0) # (1, V)
            layer_phi_grid = self.flat_phi_grid[slc].unsqueeze(0) # (1, V)
            
            # 2. Energy
            E_L = torch.sum(layer_shower, dim=1) # (B, )
            E_layers.append(E_L)
            
            # Safe denominator for weighting
            E_denom = torch.clamp(E_L, min=self.epsilon).unsqueeze(1) # (B, 1)

            # 3. First Moments (Centers)
            # sum(E_i * eta_i) / sum(E_i)
            eta_weighted = torch.sum(layer_shower * layer_eta_grid, dim=1, keepdim=True)
            phi_weighted = torch.sum(layer_shower * layer_phi_grid, dim=1, keepdim=True)
            
            mu_eta = eta_weighted / E_denom # (B, 1)
            mu_phi = phi_weighted / E_denom # (B, 1)
            
            # Mask zero energy layers to avoid NaNs or junk data
            mask = (E_L < self.epsilon).unsqueeze(1)
            mu_eta = torch.where(mask, torch.zeros_like(mu_eta), mu_eta)
            mu_phi = torch.where(mask, torch.zeros_like(mu_phi), mu_phi)
            
            Eta_centers.append(mu_eta.squeeze(1))
            Phi_centers.append(mu_phi.squeeze(1))

            # 4. Second Moments (Widths)
            # sum(E_i * (eta_i - mu_eta)^2) / sum(E_i)
            # Broadcast subtract: (1, V) - (B, 1) -> (B, V)
            diff_eta = layer_eta_grid - mu_eta
            diff_phi = layer_phi_grid - mu_phi
            
            var_eta = torch.sum(layer_shower * (diff_eta ** 2), dim=1) / E_denom.squeeze(1)
            var_phi = torch.sum(layer_shower * (diff_phi ** 2), dim=1) / E_denom.squeeze(1)
            
            # Clamp for numerical stability (sqrt of negative is bad)
            width_eta = torch.sqrt(torch.clamp(var_eta, min=1e-8))
            width_phi = torch.sqrt(torch.clamp(var_phi, min=1e-8))

            # Mask output
            width_eta = torch.where(mask.squeeze(1), torch.zeros_like(width_eta), width_eta)
            width_phi = torch.where(mask.squeeze(1), torch.zeros_like(width_phi), width_phi)
            
            Eta_widths.append(width_eta)
            Phi_widths.append(width_phi)

        # --- Stack results to match (Batch, Num_Layers) format ---
        # This ensures compatibility with FeatureAdapter
        output = {
            "E_layer": torch.stack(E_layers, dim=1),       # (B, L)
            "Eta_center": torch.stack(Eta_centers, dim=1), # (B, L)
            "Phi_center": torch.stack(Phi_centers, dim=1), # (B, L)
            "Eta_width": torch.stack(Eta_widths, dim=1),   # (B, L)
            "Phi_width": torch.stack(Phi_widths, dim=1)    # (B, L)
        }
        
        # Calculate Total Energy
        output["E_tot"] = torch.sum(output["E_layer"], dim=1) # (B, )
        
        return output
            
def evaluate_and_plot(data_dict, binning_path, output_dir="plots/", device="cpu"):
    """
    Orchestrates the flow: Raw Data -> Fast Extractor -> Adapter -> Existing Plotter
    """
    
    # 1. Setup Geometry & Extractor ONCE
    # (Move to GPU if available)
    
    geo = AtlasGeometry(filename=binning_path)
    extractor = DifferentiableFeatureExtractor(geo).to(device)
    extractor.eval() # Ensure we are in eval mode

    populated_adapters = []
    labels = []

    # 2. Process all datasets
    with torch.no_grad(): # No gradients needed for plotting
        for label, (showers, e_inc) in data_dict.items():
            print(f"Extracting features for: {label}...")
            
            # Ensure data is on the correct device
            if not isinstance(showers, torch.Tensor):
                showers = torch.tensor(showers, dtype=torch.float32)
            showers = showers.to(device)

            # --- THE FAST PART ---
            # One forward pass replaces the nested loops
            features = extractor(showers)
            
            # --- THE ADAPTER ---
            # Wrap results to look like the old class
            adapter = FeatureAdapter(features, geo.relevant_layers, e_inc)
            
            populated_adapters.append(adapter)
            labels.append(label)

    # 3. Separate Reference from Models
    # First item is reference (Data/GEANT), rest are models
    adapter_ref = populated_adapters[0]
    list_adapter_models = populated_adapters[1:]
    model_labels = labels[1:]

    # 4. Call existing plotting code
    # It won't know the difference between 'adapter_ref' and the old 'hlf_ref'
    make_validation_plots(adapter_ref, list_adapter_models, model_labels, output_dir=output_dir)