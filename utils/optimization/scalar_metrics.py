import torch
import numpy as np
from scipy.stats import wasserstein_distance
from utils.HLF.atlasgeo import AtlasGeometry
from utils.HLF.atlasgeo import DifferentiableFeatureExtractor
from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class MetricResult:
    """
    Holds the artifacts for the metric.
    'score' is used for optimization (Normalized Wasserstein).
    'counts' and 'bins' are preserved purely for visualization.
    """
    name: str
    score: float          # Normalized Wasserstein Distance (for objective)
    ws_dist: float        # Raw Wasserstein Distance (physical units)
    bins: np.ndarray      # Bin edges (for plotting)
    counts_ref: np.ndarray 
    counts_gen: np.ndarray 
    is_log_y: bool = False


def get_bins_given_edges(low_edge: float, high_edge: float, nbins: int, decimals: int = 8, logscale=False):
    """Calculates bin edges linearly or logarithmically."""
    if logscale:
        bins = np.around(np.geomspace(low_edge, high_edge, num=nbins), decimals)
    else:
        bin_width = (high_edge - low_edge) / nbins
        low_bin_center = low_edge + bin_width / 2
        high_bin_center = high_edge - bin_width / 2
        bins = np.around(np.linspace(low_bin_center, high_bin_center, nbins), decimals)
    return bins


def get_adaptive_bins(data_ref, n_bins=100, plot_range_factor=(2, 10), max_range=False):
    """Calculates range based on median and quantiles of the REFERENCE data."""
    if len(data_ref) == 0:
        return np.linspace(0, 1, n_bins)
    if max_range:
        return get_bins_given_edges(np.min(data_ref), np.max(data_ref), n_bins, decimals=8, logscale=False)

    median = np.median(data_ref)
    q05 = np.quantile(data_ref, 0.05)
    q_high_check = np.quantile(data_ref, 1 - 0.16) 
    
    high = median + min([
        np.absolute(np.max(data_ref) - median),
        np.absolute(np.quantile(data_ref, 1 - 0.05) - median) * plot_range_factor[0],
        np.absolute(q_high_check - median) * plot_range_factor[1]
    ])
    
    low = median - min([
        np.absolute(np.min(data_ref) - median),
        np.absolute(q05 - median) * plot_range_factor[0],
        np.absolute(q_high_check - median) * plot_range_factor[1]
    ])
    
    if high <= low:
        low = np.min(data_ref)
        high = np.max(data_ref) + 1e-5

    if high - low < 1e-9:
        high = low + 1.0

    return get_bins_given_edges(low, high, n_bins, decimals=8, logscale=False)


def calculate_wasserstein(data_ref, data_gen, name="metric", n_bins=100, max_range=False) -> MetricResult:
    """
    Calculates the Wasserstein Distance on UNBINNED data.
    Also computes histograms strictly for plotting purposes.
    """
    # Filter NaNs/Infs
    data_ref = data_ref[np.isfinite(data_ref)]
    data_gen = data_gen[np.isfinite(data_gen)]
    
    if len(data_ref) == 0 or len(data_gen) == 0:
        # Soft penalty for empty layers
        return MetricResult(name, 100.0, 100.0, np.array([0,1]), np.array([0]), np.array([0]))

    # --- 1. Calculate Score (Unbinned) ---
    # Raw Wasserstein distance (has units of the input, e.g., MeV or Radians)
    ws_raw = wasserstein_distance(data_ref, data_gen)
    
    # Normalize by std dev of reference to make it scale-invariant
    # This allows summing Energy scores (scale ~1000) with Angle scores (scale ~0.1)
    ref_std = np.std(data_ref)
    if ref_std < 1e-9: ref_std = 1.0 # Prevent div/0 for delta functions
    
    score = ws_raw / ref_std

    # --- 2. Calculate Plotting Artifacts (Binned) ---
    # We maintain the adaptive binning logic so the plots look exactly the same
    bins = get_adaptive_bins(data_ref, n_bins=n_bins, max_range=max_range)
    
    # Clip data to bin range for visualization consistency
    data_ref_clipped = np.clip(data_ref, bins[0], bins[-1])
    data_gen_clipped = np.clip(data_gen, bins[0], bins[-1])
    
    y_ref, _ = np.histogram(data_ref_clipped, bins=bins)
    y_gen, _ = np.histogram(data_gen_clipped, bins=bins)

    return MetricResult(
        name=name,
        score=score,
        ws_dist=ws_raw,
        bins=bins,
        counts_ref=y_ref,
        counts_gen=y_gen
    )


class ScalarMetricCalculator:
    def __init__(self, binning_path, device='cuda'):
        self.device = device
        self.geo = AtlasGeometry(filename=binning_path)
        self.extractor = DifferentiableFeatureExtractor(self.geo).to(device)
        self.extractor.eval()

    def _get_features(self, showers):
        if not isinstance(showers, torch.Tensor):
            showers = torch.tensor(showers, dtype=torch.float32)
        showers = showers.to(self.device)
        with torch.no_grad():
            features = self.extractor(showers)
        return {k: v.cpu().numpy() for k, v in features.items()}
    
    # NOTE: _log_transform removed. Wasserstein is linear and stable.

    def calculate_metrics(self, ref_showers, gen_showers) -> Tuple[float, Dict[str, MetricResult]]:
        feats_ref = self._get_features(ref_showers)
        feats_gen = self._get_features(gen_showers)

        results = {}
        total_loss = 0.0
        
        # 1. Global Energy
        res_etot = calculate_wasserstein(feats_ref['E_tot'], feats_gen['E_tot'], name="Global Energy Sum")
        res_etot.is_log_y = True
        results["global_Etot"] = res_etot
        # Weight Global Energy higher as it is the most critical physics feature
        total_loss += res_etot.score * 5.0 

        # 2. Layer-wise Evaluation
        for i, layer_id in enumerate(self.geo.relevant_layers):
            layer_key = f"L{layer_id}"
            
            # -- Energy --
            e_ref = feats_ref['E_layer'][:, i]
            e_gen = feats_gen['E_layer'][:, i]
            
            res_e = calculate_wasserstein(e_ref, e_gen, name=f"Layer {layer_id} Energy")
            res_e.is_log_y = True
            results[f"{layer_key}_E"] = res_e
            total_loss += res_e.score

            # Spatial properties (only on hits with energy)
            mask_ref = e_ref > 1e-6
            mask_gen = e_gen > 1e-6

            for prop in ['Eta_center', 'Phi_center', 'Eta_width', 'Phi_width']:
                readable_name = prop.replace('Eta_', 'eta_').replace('Phi_', 'phi_')
                full_name = f"{layer_key}_{readable_name}"
                
                val_ref = feats_ref[prop][:, i][mask_ref]
                val_gen = feats_gen[prop][:, i][mask_gen]
                
                if len(val_ref) > 0 and len(val_gen) > 0:
                    res_spatial = calculate_wasserstein(val_ref, val_gen, name=f"Layer {layer_id} {readable_name}")
                else:
                    # Penalty for empty intersection
                    res_spatial = MetricResult(full_name, 10.0, 10.0, np.array([0,1]), np.array([0]), np.array([0]))
                
                res_spatial.is_log_y = True
                results[full_name] = res_spatial
                total_loss += res_spatial.score

        return total_loss, results

    def calculate_correlations(self, gt_showers, recon_showers, bins=50) -> Dict[str, dict]:
        """
        Calculates 2D histograms for paired GT and reconstructed showers.
        Requires gt_showers and recon_showers to be ordered exactly the same.
        """
        feats_gt = self._get_features(gt_showers)
        feats_recon = self._get_features(recon_showers)

        results = {}
        
        # 1. Global Energy
        H, xedges, yedges = np.histogram2d(feats_gt['E_tot'], feats_recon['E_tot'], bins=bins)
        results["global_Etot"] = {
            "name": "Global Energy", "hist": H, "xedges": xedges, "yedges": yedges
        }

        # 2. Layer-wise Evaluation
        for i, layer_id in enumerate(self.geo.relevant_layers):
            layer_key = f"L{layer_id}"
            
            # -- Energy --
            e_gt = feats_gt['E_layer'][:, i]
            e_recon = feats_recon['E_layer'][:, i]
            
            # Use dynamic binning based on the global min/max of the layer
            min_e = min(e_gt.min(), e_recon.min())
            max_e = max(e_gt.max(), e_recon.max())
            bin_edges = np.linspace(min_e, max_e, bins + 1)
            
            H, xedges, yedges = np.histogram2d(e_gt, e_recon, bins=bin_edges)
            results[f"{layer_key}_E"] = {
                "name": f"Layer {layer_id} Energy", "hist": H, "xedges": xedges, "yedges": yedges
            }

            # -- Spatial properties --
            # Mask out cases where GT had no energy to avoid skewing spatial correlations with zeros
            mask_gt = e_gt > 1e-6
            
            for prop in ['Eta_center', 'Phi_center', 'Eta_width', 'Phi_width']:
                readable_name = prop.replace('Eta_', 'eta_').replace('Phi_', 'phi_')
                full_name = f"{layer_key}_{readable_name}"
                
                val_gt = feats_gt[prop][:, i][mask_gt]
                val_recon = feats_recon[prop][:, i][mask_gt]
                
                if len(val_gt) > 0 and len(val_recon) > 0:
                    min_val = min(val_gt.min(), val_recon.min())
                    max_val = max(val_gt.max(), val_recon.max())
                    spatial_bins = np.linspace(min_val, max_val, bins + 1)
                    
                    H, xedges, yedges = np.histogram2d(val_gt, val_recon, bins=spatial_bins)
                else:
                    H, xedges, yedges = np.zeros((bins, bins)), np.zeros(bins+1), np.zeros(bins+1)
                
                results[full_name] = {
                    "name": f"Layer {layer_id} {readable_name}", "hist": H, "xedges": xedges, "yedges": yedges
                }

        return results