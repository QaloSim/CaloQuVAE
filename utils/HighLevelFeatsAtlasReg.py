import numpy as np
import torch
from matplotlib.patches import Wedge
from matplotlib.collections import PatchCollection
from matplotlib.colors import LogNorm
from matplotlib.cm import ScalarMappable
import matplotlib.pyplot as plt
import h5py

# from io import BytesIO
# from PIL import Image

class HighLevelFeatures_ATLAS_regular:
    def __init__(self, particle, filename, relevantLayers=[0,1,2,3,12,13,14]):
        """
        Initialize the PLT_ATLAS object.

        Parameters:
        - filename: Path to the raw HDF5 data file.
        - event_n: Event number to process.
        - relevantLayers: List of layer numbers to plot. Defaults to [0, 1, 2, 3, 12].
        """
        self.relevantLayers = relevantLayers
        self.ATLAS_raw_dir = filename
        # self.data = h5py.File(self.ATLAS_raw_dir, 'r')
        self.bin_info()
        #print(self.ATLAS_raw_dir)

        self.r_centers = {}
        self.alpha_centers = {}
        self.phi_locations = {} # For phi = r * sin(alpha)

        for layer in self.binsize_alpha.keys():
            # r_a = r_start + r_size / 2
            self.r_centers[layer] = self.binstart_radius[layer] + self.binsize_radius[layer] / 2.0
            # alpha_a = alpha_start + alpha_size / 2
            self.alpha_centers[layer] = self.binstart_alpha[layer] + self.binsize_alpha[layer] / 2.0
            # phi_a location = r_a * sin(alpha_a)
            self.phi_locations[layer] = self.r_centers[layer] * torch.sin(self.alpha_centers[layer])

        
    def bin_info(self):
        with h5py.File(self.ATLAS_raw_dir, 'r') as file:
            # List all groups
            # self.data = {}
            self.binsize_alpha = {}
            self.binstart_alpha = {}
            self.binsize_radius = {}
            self.binstart_radius = {}
            #print("Keys: %s" % list(file.keys()))
            for key in file.keys():
                # self.data[key] = torch.tensor(np.array(file[key]))
                if "binsize_alpha_layer_" in key:
                    layer = key.split("_")[-1]
                    self.binsize_alpha[layer] = torch.tensor(np.array(file[key])) # self.data[f"binsize_alpha_layer_{layer}"]
                if "binstart_alpha_layer_" in key:
                    layer = key.split("_")[-1]
                    self.binstart_alpha[layer] = torch.tensor(np.array(file[key])) #self.data[f"binstart_alpha_layer_{layer}"]
                if "binsize_radius_layer_" in key:
                    layer = key.split("_")[-1]
                    self.binsize_radius[layer] = torch.tensor(np.array(file[key])) #self.data[f"binsize_radius_layer_{layer}"]
                if "binstart_radius_layer_" in key:
                    layer = key.split("_")[-1]
                    self.binstart_radius[layer] = torch.tensor(np.array(file[key])) #self.data[f"binstart_radius_layer_{layer}"]

    def get_sector_arrays(self, layer):
        # All in torch, then to numpy once
        r0 = self.binstart_radius[layer]
        r1 = r0 + self.binsize_radius[layer]
        a0 = torch.rad2deg(self.binstart_alpha[layer]).round()
        a1 = a0 + torch.rad2deg(self.binsize_alpha[layer]).round()
        e  = self.single_event_energy

        # Convert once
        return (
            r0.cpu().numpy(),
            r1.cpu().numpy(),
            a0.cpu().numpy(),
            a1.cpu().numpy(),
            e
        )

    def _make_equal_bin_transform(self, r0, r1):
        # Build sorted unique boundaries
        bounds = np.unique(np.concatenate([r0, r1]))
        N = len(bounds) - 1
        plot_bounds = np.linspace(0, 1, N+1)

        # For any radius array R, get indices of lower boundary
        def transform(R):
            idx = np.searchsorted(bounds, R, side="right") - 1
            # clip to valid range
            idx = np.clip(idx, 0, N-1)
            # fraction within each bin
            frac = (R - bounds[idx]) / (bounds[idx+1] - bounds[idx])
            return plot_bounds[idx] + frac*(plot_bounds[idx+1] - plot_bounds[idx])

        return transform

    def plot_calorimeter(self, ax, scale='equal_bin',
                         cmap="rainbow", norm=None, title=None):
        # Collect arrays
        r0, r1, a0, a1, e = self.get_sector_arrays(self.current_layer)

        # Setup normalization
        if norm is None:
            norm = LogNorm(vmin=max(e.min(),1e-4), vmax=max(e.max(),1e-4))

        # Precompute transform
        if scale=='equal_bin':
            transform = self._make_equal_bin_transform(r0, r1)
            r0p, r1p = transform(r0), transform(r1)
        else:
            transform = lambda R: R
            r0p, r1p = transform(r0), transform(r1)

        # Build all wedges and colors
        patches = []
        for inner, outer, start, end in zip(r0p, r1p, a0, a1):
            width = outer - inner
            patches.append(Wedge((0,0), outer, start, end, width=width))

        # Create and add a single PatchCollection
        pc = PatchCollection(patches, cmap=cmap, norm=norm, edgecolor="grey", linewidths=0.1)
        pc.set_array(e)
        ax.add_collection(pc)
        ax.grid(False)

        # Adjust limits
        Rmax = r1p.max()
        ax.set_xlim(-Rmax-0.1, Rmax+0.1)
        ax.set_ylim(-Rmax-0.1, Rmax+0.1)
        ax.set_aspect('equal')
        ax.axis('off')
        if title:
            ax.set_title(title, fontsize=15)

    def DrawSingleShower(self, data, filename=None, title=None, scale='equal_bin',
                         vmin=1e-4, vmax=1e4, cmap='rainbow'):
        """
        Plot all specified layers of the calorimeter for the given event in a composite figure.

        Parameters:
        - scale: 'linear' or 'equal_bin', determines the type of radial scale for all subplots.
        """
        num = len(self.relevantLayers)
        fig, axes = plt.subplots(1, num, figsize=(15,15), dpi=200)
        norm = LogNorm(vmin=vmin, vmax=vmax)

        vox = 14*24
        for ax, layer, i in zip(axes, self.relevantLayers, range(num)):
            self.single_event_energy = data[i*vox:(i+1)*vox]
            self.current_layer = str(layer)
            self.plot_calorimeter(ax, scale=scale, cmap=cmap, norm=norm,
                                  title=f"Layer {layer}")
        # Add a single horizontal colorbar below the subplots
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        fig.colorbar(sm, ax=axes.ravel().tolist(), orientation='horizontal',
                     fraction=0.05, pad=0.1, label='Energy')
        return fig

    def plot_single_layer_with_highlights(self, data, layer, r, phi, highlight_coords, scale='equal_bin', cmap='rainbow', vmin=1e-4, vmax=1e4, title=None):
        """
        Plot one calorimeter layer slice and highlight specific (r_idx, phi_idx) patches.
        Adds (r, phi) labels next to each highlighted voxel to specify the r and phi values
        """
        fig, ax = plt.subplots(figsize=(6, 6))
        norm = LogNorm(vmin=vmin, vmax=vmax)
        
        layer_idx = self.relevantLayers.index(layer)
        self.single_event_energy = data[layer_idx * r * phi : (layer_idx + 1) * r * phi]
        self.current_layer = str(layer)

        # plot base calorimeter
        self.plot_calorimeter(ax, scale=scale, cmap=cmap, norm=norm, title=title or f"Layer {layer}")

        # overlay highlighted patches
        r0, r1, a0, a1, _ = self.get_sector_arrays(self.current_layer)

        if scale == "equal_bin":
            transform = self._make_equal_bin_transform(r0, r1)
            r0p, r1p = transform(r0), transform(r1)
        else:
            r0p, r1p = r0, r1
        
        # go through coordinates given:
        for (r_idx, phi_idx) in highlight_coords:
            flat_idx = r_idx * phi + phi_idx
            inner, outer = r0p[flat_idx], r1p[flat_idx]
            theta_start, theta_end = a0[flat_idx], a1[flat_idx]

            # highlight the voxel selected
            wedge = Wedge(
                center=(0, 0),
                r=outer,
                theta1=theta_start,
                theta2=theta_end,
                width=outer - inner,
                facecolor='none',
                edgecolor='black',
                linewidth=1.5)
            ax.add_patch(wedge)

            # compute label position to add labels
            theta = 0.5 * (theta_start + theta_end)
            r_text = 0.5 * (inner + outer)
            x_text = r_text * np.cos(np.deg2rad(theta))
            y_text = r_text * np.sin(np.deg2rad(theta))
            
            # add the labels
            ax.text(
                x_text,
                y_text,
                f"({r_idx},{phi_idx})",
                color="black",
                fontsize=7,
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="black", lw=0.3, alpha=0.8))

        plt.show()
        return fig

    def CalculateFeatures(self, data):
        """
        Calculates the layer-wise statistics of a given event, including the center and width of energy distribution in r and phi.

        Also calculates the total energy in each layer and total energy in the event.

        Parameters:
        - event_data: A torch tensor or numpy array of energy depositions for N showers.

        Updates:
        - self.E_tot: Total energy in the event.
        - self.E_layers: A dictionary where keys are layer numbers and values are lists of total energy in each layer.
        - self.r: A dictionary for the center of energy in r.
        - self.phi: A dictionary for the center of energy in phi.
        - self.width_r: A dictionary for the width of energy in r.
        - self.width_phi: A dictionary for the width of energy in phi.
        """
        self.E_tot = []
        self.E_layers = {layer: [] for layer in self.relevantLayers}
        self.EC_rs = {layer: [] for layer in self.relevantLayers}
        self.EC_phis = {layer: [] for layer in self.relevantLayers}
        self.width_rs = {layer: [] for layer in self.relevantLayers}
        self.width_phis = {layer: [] for layer in self.relevantLayers}

        for event in data:
            stats = self.calculate_energy_centers(event)
            self.E_tot.append(stats['total_energy'])
            for layer in self.relevantLayers:
                self.E_layers[layer].append(stats[layer]['total_layer_energy'])
                self.EC_rs[layer].append(stats[layer]['r_center'])
                self.EC_phis[layer].append(stats[layer]['phi_center'])
                self.width_rs[layer].append(stats[layer]['r_width'])
                self.width_phis[layer].append(stats[layer]['phi_width'])
        self.E_tot = np.array(self.E_tot)
        for layer in self.relevantLayers:
            self.E_layers[layer] = np.array(self.E_layers[layer])
            self.EC_rs[layer] = np.array(self.EC_rs[layer])
            self.EC_phis[layer] = np.array(self.EC_phis[layer])
            self.width_rs[layer] = np.array(self.width_rs[layer])
            self.width_phis[layer] = np.array(self.width_phis[layer])


    def calculate_energy_centers(self, event_data):
        """
        Calculates the layer-wise statistics of a given event, including the center and width of energy distribution in r and phi.

        Also calculates the total energy in each layer and total energy in the event.

        Parameters:
        - event_data: A 1D torch tensor or numpy array of energy depositions for one shower.

        Returns:
        - A dictionary where keys are layer numbers and values are dicts
          containing the calculated {'r_center', 'phi_center'}.
        """
        if isinstance(event_data, np.ndarray):
            event_data = torch.from_numpy(event_data)

        layer_stats = {}

        for i, layer_num in enumerate(self.relevantLayers):
            layer_str = str(layer_num)

            # 1. Slice the 1D vector to get energies for the current layer
            voxels_per_layer = 14*24
            start_index = i * voxels_per_layer
            end_index = (i + 1) * voxels_per_layer
            layer_energies = event_data[start_index:end_index] # This is I_ia in the formula

            # 2. Get the total energy in the layer (the denominator)
            total_layer_energy = torch.sum(layer_energies)

            # Avoid division by zero if a layer has no energy
            if total_layer_energy <= 0:
                layer_stats[layer_num] = {
                    'total_layer_energy': 0,
                    'r_center': 0, 
                    'phi_center': 0,
                    'r_width': 0,
                    'phi_width': 0
                }
                continue

            # 3. Get the pre-calculated voxel locations (l_a in the formula)
            r_locs = self.r_centers[layer_str]
            phi_locs = self.phi_locations[layer_str]

            # 4. Calculate the weighted sum for r (the numerator)
            weighted_sum_r = torch.sum(r_locs * layer_energies)

            # 5. Calculate the weighted sum for phi
            weighted_sum_phi = torch.sum(phi_locs * layer_energies)
            
            # 6. Compute the final center of energy values
            center_of_energy_r = weighted_sum_r / total_layer_energy
            center_of_energy_phi = weighted_sum_phi / total_layer_energy

            # 7. Calculate widths
            width_energy_r = torch.sqrt(torch.sum((r_locs - center_of_energy_r) ** 2 * layer_energies) / total_layer_energy)
            width_energy_phi = torch.sqrt(torch.sum((phi_locs - center_of_energy_phi) ** 2 * layer_energies) / total_layer_energy)

            layer_stats[layer_num] = {
                'total_layer_energy': total_layer_energy.item(),
                'r_center': center_of_energy_r.item(),
                'phi_center': center_of_energy_phi.item(),
                'r_width': width_energy_r.item(),
                'phi_width': width_energy_phi.item()
            }
        total_energy = sum(stat['total_layer_energy'] for stat in layer_stats.values())
        layer_stats['total_energy'] = total_energy

        return layer_stats
    
#     def energy_center_histograms(self, sample_sets, num_bins=50, filename_prefix=None):
#         """
#         Processes a batch of shower samples, calculates energy centers, and plots histograms.

#         Parameters:
#         - shower_samples: A 2D tensor or numpy array of shower events (num_events, num_voxels).
#         - num_bins: The number of bins to use for the histograms.
#         - filename_prefix: (Optional) If provided, saves plots to files instead of displaying.
#                            e.g., 'my_analysis' will create 'my_analysis_r_centers.png'.
#         """
#         # Prepare data structures to store the results
#         r_centers = {layer: {label: [] for _, label in sample_sets} for layer in self.relevantLayers}
#         phi_centers = {layer: {label: [] for _, label in sample_sets} for layer in self.relevantLayers}

#         # Loop through all events and calculate features
#         for samples, label in sample_sets:
#             print(f"Processing {len(samples)} events for {label}...")
#             for event_data in samples:
#                 centers = self.calculate_energy_centers(event_data)
#                 for layer_num, values in centers.items():
#                     r_centers[layer_num][label].append(values['r_center'])
#                     phi_centers[layer_num][label].append(values['phi_center'])
#             print(f"Processing complete for {label}.")
        
#         print("Processing complete.")

#         # Plot histograms for r_center
#         fig_r, axes_r = plt.subplots(2, 4, figsize=(20, 10), dpi=100, sharey=True)
#         axes_r = axes_r.flatten()[:7]
#         fig_r.suptitle('Distribution of Energy Center in r-direction ($<r>$)', fontsize=16)
#         for ax, layer in zip(axes_r, self.relevantLayers):
#             for label in r_centers[layer]:
#                 ax.hist(r_centers[layer][label], bins=num_bins, histtype='step', lw=1.5, label=label, log=True, density=True)
#             ax.set_title(f'Layer {layer}')
#             ax.set_xlabel('$<r>$ (mm)')
#             ax.legend()
#         axes_r[0].set_ylabel('Probability')

#         if filename_prefix:
#             r_filename = f"{filename_prefix}_r_centers.png"
#             plt.savefig(r_filename)
#             print(f"Saved r-center histogram to {r_filename}")
#         else:
#             plt.show()
#         plt.close(fig_r)

#         # Plot histograms for phi_center
#         fig_phi, axes_phi = plt.subplots(2, 4, figsize=(20, 10), dpi=100, sharey=True)
#         axes_phi = axes_phi.flatten()[:7]
#         fig_phi.suptitle('Distribution of Energy Center in $\phi$-direction ($<r \\sin(\\alpha)>$)', fontsize=16)
#         for ax, layer in zip(axes_phi, self.relevantLayers):
#             for label in phi_centers[layer]:
#                 ax.hist(phi_centers[layer][label], bins=num_bins, histtype='step', lw=1.5, label=label, log=True, density=True)
#             ax.set_title(f'Layer {layer}')
#             ax.set_xlabel('$\phi$ (mm)')
#             ax.legend()
#         axes_phi[0].set_ylabel('Number of Events')

#         if filename_prefix:
#             phi_filename = f"{filename_prefix}_phi_centers.png"
#             plt.savefig(phi_filename)
#             print(f"Saved phi-center histogram to {phi_filename}")
#         else:
#             plt.show()
#         plt.close(fig_phi)


    def GetEtot(self):
        return self.E_tot
    def GetElayers(self):
        return self.E_layers
    def GetECrs(self):
        return self.EC_rs
    def GetECphis(self):
        return self.EC_phis
    def GetWidthrs(self):
        return self.width_rs
    def GetWidthphis(self):
        return self.width_phis
