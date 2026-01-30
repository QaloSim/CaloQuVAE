import matplotlib.pyplot as plt
import numpy as np
import torch
import math
import os
from typing import Dict, Tuple
from utils.optimization.scalar_metrics import MetricResult

class ShowerPlotter:
    """
    Purely visual class. 
    It does NOT recalculate bins or data. It visualizes the MetricResult objects.
    """
    def __init__(self, save_dir="plots"):
        self.save_dir = save_dir
        self.ref_color = 'black'
        self.gen_color = '#e41a1c'

    def plot_from_results(self, results: Dict[str, MetricResult]):
        os.makedirs(self.save_dir, exist_ok=True)
        
        # 1. Plot Global Energy
        if "global_Etot" in results:
            self._plot_single(results["global_Etot"], "global_energy.png")

        # 2. Group Layer Metrics for Grid Plotting
        # We assume keys format "L{id}_{feature}"
        grouped_features = {} 
        for key, res in results.items():
            if key == "global_Etot": continue
            
            # Extract feature name (e.g. "E", "eta_center")
            parts = key.split('_')
            feature_type = "_".join(parts[1:]) # "E", "eta_center", etc.
            
            if feature_type not in grouped_features:
                grouped_features[feature_type] = []
            grouped_features[feature_type].append(res)

        # 3. Plot Grids
        for feature_name, res_list in grouped_features.items():
            self._plot_grid(res_list, feature_name)

    def _plot_single(self, res: MetricResult, filename: str):
        fig, ax = plt.subplots(figsize=(6, 5))
        self._draw_hist_on_axis(ax, res)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, filename), dpi=150)
        plt.close()

    def _plot_grid(self, res_list, feature_name):
        n_plots = len(res_list)
        cols = 3
        rows = int(np.ceil(n_plots / cols))
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols*5, rows*4))
        axes = axes.flatten()

        for i, res in enumerate(res_list):
            self._draw_hist_on_axis(axes[i], res)

        # Hide empty axes
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.suptitle(f"Feature: {feature_name}", fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig(os.path.join(self.save_dir, f"{feature_name}_grid.png"), dpi=150)
        plt.close()

    def _draw_hist_on_axis(self, ax, res: MetricResult):
        """
        Draws the pre-computed histogram counts onto the axis.
        """
        # Centers for step plotting
        centers = 0.5 * (res.bins[1:] + res.bins[:-1])
        widths = res.bins[1:] - res.bins[:-1]
        
        # Normalize to density for visual comparison (optional, but standard)
        # We use a safe division
        ref_area = np.sum(res.counts_ref * widths)
        gen_area = np.sum(res.counts_gen * widths)
        
        y_ref = res.counts_ref / (ref_area if ref_area > 0 else 1.0)
        y_gen = res.counts_gen / (gen_area if gen_area > 0 else 1.0)

        # Plot Reference
        ax.hist(centers, bins=res.bins, weights=y_ref, 
                histtype='step', color=self.ref_color, label='Geant4 (Ref)', linewidth=1.5)
        
        # Plot Generated
        ax.hist(centers, bins=res.bins, weights=y_gen, 
                histtype='step', color=self.gen_color, label='CaloQVAE (Gen)', 
                linewidth=1.5, linestyle='--')

        # Formatting
        ax.set_title(f"{res.name}\n$WS_{{norm}} = {res.score:.4f}$", fontsize=10)
        
        if res.is_log_y:
            ax.set_yscale('log')
        ax.legend(prop={'size': 8})