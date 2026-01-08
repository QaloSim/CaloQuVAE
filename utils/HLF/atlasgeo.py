import torch
import torch.nn as nn
import h5py
import numpy as np

class AtlasGeometry:
    """
    Handles loading ATLAS calorimeter binning information and generating 
    coordinate grids for vectorized operations.
    """
    def __init__(self, filename, relevant_layers=[0, 1, 2, 3, 12]):
        self.filename = filename
        self.relevant_layers = relevant_layers
        self.voxels_per_layer = 14 * 24  # Standard ATLAS dimensions
        
        # Load raw bin edges from file
        self._load_h5_data()
        
        # Compute centers for every voxel in every relevant layer
        # resulting shapes are dictionaries mapping layer_id -> Tensor(shape=[336])
        self.r_centers = {}
        self.phi_centers = {}
        
        for layer in self.relevant_layers:
            layer_str = str(layer)
            
            # r_center = start + size/2
            r_c = self.binstart_radius[layer_str] + self.binsize_radius[layer_str] / 2.0
            
            # alpha_center = start + size/2
            alpha_c = self.binstart_alpha[layer_str] + self.binsize_alpha[layer_str] / 2.0
            
            # phi location definition from your original code: r * sin(alpha)
            # Note: This is an approximation or specific projection used in your physics group
            phi_c = r_c * torch.sin(alpha_c)
            
            self.r_centers[layer] = r_c.float()
            self.phi_centers[layer] = phi_c.float()

    def _load_h5_data(self):
        self.binsize_alpha = {}
        self.binstart_alpha = {}
        self.binsize_radius = {}
        self.binstart_radius = {}
        
        with h5py.File(self.filename, 'r') as f:
            for key in f.keys():
                # Extract layer number from key string (e.g. "binsize_alpha_layer_0")
                if "layer_" not in key: continue
                layer = key.split("_")[-1]
                
                # Convert immediately to torch tensors
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
        
        # We need to stack the geometry grids into a single tensor for batch processing.
        # Shape: (Num_Layers, Voxels_Per_Layer)
        r_grid_list = [geometry_handler.r_centers[l] for l in self.relevant_layers]
        phi_grid_list = [geometry_handler.phi_centers[l] for l in self.relevant_layers]
        
        # Register as buffers so they are part of the model state (move to GPU, save/load)
        self.register_buffer('r_grid', torch.stack(r_grid_list))       # (L, V)
        self.register_buffer('phi_grid', torch.stack(phi_grid_list))   # (L, V)
        
        # Epsilon for numerical stability
        self.epsilon = 1e-6

    def forward(self, showers):
        """
        Args:
            showers: (Batch, Total_Voxels) or (Batch, Layers, Voxels)
        """
        # 1. Standardize Input Shape
        if showers.dim() == 2:
            B, Total = showers.shape
            showers = showers.view(B, self.num_layers, self.voxels_per_layer)
        
        # 2. Total Energy Calculation
        E_layer_tot = torch.sum(showers, dim=2) # (B, L)
        E_tot = torch.sum(E_layer_tot, dim=1)   # (B, )

        # Avoid division by zero for empty layers
        E_layer_denom = E_layer_tot.clone()
        # Use torch.where to replace safely, though clone() separates the graph enough for this specific op usually.
        # But specifically for the denom, the inplace clamp is "okay" if E_layer_denom isn't used for gradients,
        # but pure torch operations are safer:
        E_layer_denom = torch.clamp(E_layer_denom, min=self.epsilon)

        # 3. First Moments (Centers)
        # (B, L, V) * (1, L, V) -> sum(dim=2) -> (B, L)
        R_weighted = torch.sum(showers * self.r_grid.unsqueeze(0), dim=2)
        Phi_weighted = torch.sum(showers * self.phi_grid.unsqueeze(0), dim=2)
        
        R_center = R_weighted / E_layer_denom
        Phi_center = Phi_weighted / E_layer_denom
        
        # Mask definition
        mask = (E_layer_tot < self.epsilon) # (B, L) boolean tensor
        
        # --- FIX 1: Replace In-Place Center Masking ---
        # OLD (Error): R_center[mask] = 0.0
        # NEW: Use torch.where to create a new tensor
        zeros = torch.zeros_like(R_center)
        R_center = torch.where(mask, zeros, R_center)
        Phi_center = torch.where(mask, zeros, Phi_center)

        # 4. Second Moments (Widths)
        # Broadcast centers: (B, L, 1)
        diff_r = self.r_grid.unsqueeze(0) - R_center.unsqueeze(2)       # (B, L, V)
        diff_phi = self.phi_grid.unsqueeze(0) - Phi_center.unsqueeze(2) # (B, L, V)
        
        # Weighted Variance
        var_r = torch.sum(showers * (diff_r ** 2), dim=2) / E_layer_denom
        var_phi = torch.sum(showers * (diff_phi ** 2), dim=2) / E_layer_denom
        
        # --- FIX: Avoid infinite gradient at 0 ---
        # OLD: width_r = torch.sqrt(torch.clamp(var_r, min=0.0))
        # NEW: Clamp min to epsilon (e.g. 1e-8) so we never take sqrt(0)
        safe_var_r = torch.clamp(var_r, min=1e-8)
        safe_var_phi = torch.clamp(var_phi, min=1e-8)
        
        width_r = torch.sqrt(safe_var_r)
        width_phi = torch.sqrt(safe_var_phi)

        # Mask empty layers (restore true 0.0 for output, but gradients remain stable)
        width_r = torch.where(mask, zeros, width_r)
        width_phi = torch.where(mask, zeros, width_phi)
        return {
            "E_tot": E_tot,          
            "E_layer": E_layer_tot,  
            "R_center": R_center,    
            "Phi_center": Phi_center,
            "R_width": width_r,      
            "Phi_width": width_phi   
        }