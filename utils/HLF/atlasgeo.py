import torch
import torch.nn as nn
import h5py
import numpy as np
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
        
        # Changed from r_centers to eta_centers to match legacy logic
        self.eta_centers = {}
        self.phi_centers = {}
        
        for layer in self.relevant_layers:
            layer_str = str(layer)
            
            # 1. Calculate Midpoints
            r_c = self.binstart_radius[layer_str] + self.binsize_radius[layer_str] / 2.0
            alpha_c = self.binstart_alpha[layer_str] + self.binsize_alpha[layer_str] / 2.0
            
            # 2. Project to Eta/Phi (This matches the legacy HighLevelFeatures)
            # Legacy: eta = R * cos(alpha)
            # Legacy: phi = R * sin(alpha)
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