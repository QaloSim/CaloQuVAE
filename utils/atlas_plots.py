import json
import wandb
import torch
import importlib
import utils.HighLevelFeatsAtlasReg
importlib.reload(utils.HighLevelFeatsAtlasReg)
from utils.HighLevelFeatsAtlasReg import HighLevelFeatures_ATLAS_regular

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import mplhep as hep
from scipy.stats import ks_2samp, entropy, wasserstein_distance
import os
from matplotlib.backends.backend_pdf import PdfPages 


from utils.HighLevelFeatures import HighLevelFeatures

def plot_calorimeter_shower(cfg, showers, showers_recon, showers_sampled, epoch, save_dir=None, incidence_energy_choice=None, incidence_energy_gt=None, incidence_energy_generated=None):
    """
    Creates calorimeter slice plots. 
    Returns the figure handles and the specific incidence energy value found for the generated sample.
    """
    # hlf set up:
    dataset_name = cfg.data.dataset_name.lower()
    
    if "atlas" in dataset_name:
        HLF = HighLevelFeatures_ATLAS_regular(
            particle=cfg.data.particle,
            filename=cfg.data.binning_path,
            relevantLayers=cfg.data.relevantLayers
        )
    else:
        HLF = HighLevelFeatures(
            particle=cfg.data.particle,
            filename=cfg.data.binning_path,
            relevantLayers=cfg.data.relevantLayers
        )

    # Variable to store the specific energy found
    found_energy_val = None

    if incidence_energy_choice is not None and incidence_energy_gt is not None and incidence_energy_generated is not None:
            
        # 1. Find index for Ground Truth (and Recon)
        diff_gt = torch.abs(incidence_energy_gt - incidence_energy_choice)
        idx_gt = torch.argmin(diff_gt).item()
        
        # 2. Find index for Sampled (independent search)
        diff_sampled = torch.abs(incidence_energy_generated - incidence_energy_choice)
        idx_sampled = torch.argmin(diff_sampled).item()
        
        # Select the data
        real = showers[idx_gt]
        recon = showers_recon[idx_gt]
        sampled = showers_sampled[idx_sampled]
        
        # Extract values
        e_real_val = incidence_energy_gt[idx_gt].item() if incidence_energy_gt.dim() > 0 else incidence_energy_gt.item()
        e_sample_val = incidence_energy_generated[idx_sampled].item() if incidence_energy_generated.dim() > 0 else incidence_energy_generated.item()
        
        # Save the found energy to return later
        found_energy_val = e_sample_val
        
        title_suffix_real = f" (Incidence Energy: {e_real_val:.1f} MeV)"
        title_suffix_sample = f" (Incidence Energy: {e_sample_val:.1f} MeV)"

    else:
        # Fallback to original logic
        idx = showers.sum(dim=1).argsort()[-2]
        real = showers[idx]
        recon = showers_recon[idx]
        sampled = showers_sampled[idx]
        
        title_suffix_real = ""
        title_suffix_sample = ""
        
        # If no specific energy was searched, we return None or the energy at this fallback index if available
        found_energy_val = None 

    real_avg = showers.mean(dim=0)
    recon_avg = showers_recon.mean(dim=0)
    sampled_avg = showers_sampled.mean(dim=0)
        
    input_path = f"{save_dir}/val_input_epoch{epoch}.png" if save_dir else None
    recon_path = f"{save_dir}/val_recon_epoch{epoch}.png" if save_dir else None
    sample_path = f"{save_dir}/val_sampled_epoch{epoch}.png" if save_dir else None

    # images
    image_input = HLF.DrawSingleShower(real, title=f"Val Input (Epoch {epoch}){title_suffix_real}", filename=input_path, cmap='rainbow')
    image_recon = HLF.DrawSingleShower(recon, title=f"Val Recon (Epoch {epoch}){title_suffix_real}", filename=recon_path, cmap='rainbow')
    image_sample = HLF.DrawSingleShower(sampled, title=f"Val Sampled (Epoch {epoch}){title_suffix_sample}", filename=sample_path, cmap='rainbow')
    
    image_input_avg = HLF.DrawSingleShower(real_avg, title=f"Val Input (Epoch {epoch})", filename=input_path, cmap='rainbow')
    image_recon_avg = HLF.DrawSingleShower(recon_avg, title=f"Val Recon (Epoch {epoch})", filename=recon_path, cmap='rainbow')
    image_sample_avg = HLF.DrawSingleShower(sampled_avg, title=f"Val Sampled (Epoch {epoch})", filename=sample_path, cmap='rainbow')
    
    # # Single-layer with highlighted patches (ATLAS only)
    # if "atlas" in dataset_name:
    #     highlight_coords = [(0, 0)] + [(r_, phi_) for r_ in [4, 10, 15] for phi_ in [0, 3, 6, 9]]
    #     HLF.plot_single_layer_with_highlights(
    #         data=real,
    #         layer=0,
    #         r=cfg.data.r,
    #         phi=cfg.data.phi,
    #         highlight_coords=highlight_coords,
    #         title=f"(Epoch {epoch}) Highlighted Voxels")

    return image_input, image_recon, image_sample, image_input_avg, image_recon_avg, image_sample_avg

def plot_calorimeter_shower_simplified(
    gt_showers, 
    showers_recon, 
    incident_energies, 
    choice, 
    num_events, 
    showers_sampled=None, 
    cfg=None
):
    """
    Plots the GT, Recon, and Sampled events for the 'num_events' closest to the 'choice' energy.
    
    Args:
        gt_showers: Tensor of ground truth showers.
        showers_recon: Tensor of reconstructed showers.
        incident_energies: Tensor of incident energies corresponding to the showers.
        choice (float): The target energy to search for.
        num_events (int): Number of events to plot.
        showers_sampled (Tensor, optional): Tensor of sampled showers. Defaults to None.
        cfg: Configuration object required for HLF initialization.
    """
    
    # 1. Initialize HighLevelFeatures (HLF) 
    # We need this object to handle the specific binning/geometry of the plots
    dataset_name = cfg.data.dataset_name.lower()
    if "atlas" in dataset_name:
        HLF = HighLevelFeatures_ATLAS_regular(
            particle=cfg.data.particle,
            filename=cfg.data.binning_path,
            relevantLayers=cfg.data.relevantLayers
        )
    else:
        HLF = HighLevelFeatures(
            particle=cfg.data.particle,
            filename=cfg.data.binning_path,
            relevantLayers=cfg.data.relevantLayers
        )

    # 2. Find the indices of events closest to the chosen energy
    # We calculate the absolute difference and sort by smallest difference
    diff = torch.abs(incident_energies.flatten() - choice)
    
    # Get the indices of the 'num_events' smallest differences
    # largest=False means we get the smallest values (closest energies)
    closest_indices = torch.topk(diff, k=num_events, largest=False).indices.tolist()

    # 3. Iterate and Plot
    figures = []
    
    for i, idx in enumerate(closest_indices):
        # Extract the specific energy for this event
        current_energy = incident_energies[idx].item()
        
        # Select Data
        real = gt_showers[idx]
        recon = showers_recon[idx]
        
        # Prepare Title Suffix
        title_suffix = f" (E_true: {current_energy:.1f} MeV)"

        # Plot GT
        fig_real = HLF.DrawSingleShower(
            real, 
            title=f"Event {i+1} GT{title_suffix}", 
            filename=None, 
            cmap='rainbow'
        )
        figures.append(fig_real)
        
        # Plot Recon
        fig_recon = HLF.DrawSingleShower(
            recon, 
            title=f"Event {i+1} Recon{title_suffix}", 
            filename=None, 
            cmap='rainbow'
        )
        figures.append(fig_recon)

        # Plot Sampled (only if provided)
        if showers_sampled is not None:
            # Assuming showers_sampled aligns with gt_showers index-wise
            sampled = showers_sampled[idx]
            fig_sampled = HLF.DrawSingleShower(
                sampled, 
                title=f"Event {i+1} Sampled{title_suffix}", 
                filename=None, 
                cmap='rainbow'
            )
            figures.append(fig_sampled)

    return figures

class AtlasEvaluator:
    def __init__(self):
        self.metrics = {}

    def calculate(self, data_true, data_gen, counts_true, counts_gen):
        # Basic Statistics
        mu_true, mu_gen = np.mean(data_true), np.mean(data_gen)
        std_true, std_gen = np.std(data_true), np.std(data_gen)
        
        # KS and Wasserstein
        ks_stat, _ = ks_2samp(data_true, data_gen)
        wd = wasserstein_distance(data_true, data_gen)

        # Approximate Chi2 (Shape only)
        safe_true = counts_true + 1e-10
        safe_gen = counts_gen + 1e-10
        chi_sq = np.sum((safe_true - safe_gen)**2 / (safe_true + safe_gen))

        return {
            "mu_ratio": mu_gen / mu_true if mu_true != 0 else 0,
            "std_ratio": std_gen / std_true if std_true != 0 else 0,
            "ks": ks_stat,
            "wd": wd,
            "chi2": chi_sq
        }

    def get_text(self, results, labels):
        lines = []
        for label, res in zip(labels, results):
            lines.append(f"{label}")
            lines.append(f"KS: {res['ks']:.2f} | WD: {res['wd']:.2f}")
            lines.append(f"$\chi^2$: {res['chi2']:.1f}")
            lines.append(f"$\mu$: {res['mu_ratio']:.2f} | $\sigma$: {res['std_ratio']:.2f}")
            lines.append("") 
        return "\n".join(lines[:-1])

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def dup_last(a):
    return np.append(a, a[-1])

def to_np(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    return np.array(data)

def get_bins(all_data, xscale='linear'):
    if not all_data or all(len(d) == 0 for d in all_data):
        return None

    vmin = min(np.min(d) for d in all_data if len(d) > 0)
    vmax = max(np.max(d) for d in all_data if len(d) > 0)

    if xscale == 'log':
        pos_vals = np.concatenate([d[d > 0] for d in all_data if len(d[d > 0]) > 0] or [np.array([1e-5])])
        vmin = max(vmin, pos_vals.min() if len(pos_vals) > 0 else 1e-5)
        if vmin <= 0: vmin = 1e-5
        if vmax <= vmin: vmax = vmin * 10 
        bins = np.logspace(np.log10(vmin), np.log10(vmax), 100)
    else:
        bins = np.linspace(vmin, vmax, 100)
    return bins

# -----------------------------------------------------------------------------
# Individual Plotting Function
# -----------------------------------------------------------------------------
def plot_atlas_style_multi(data_ref, data_list, labels, xlabel, output_path, 
                           yscale='log', xscale='linear', 
                           colors=None, linestyles=None, pdf=None, fixed_bins=None):
    
    if colors is None: colors = ['red', 'green', 'orange', 'purple', 'cyan']
    if linestyles is None: linestyles = ['-', '--', '-.', ':', '-']

    data_ref = data_ref[np.isfinite(data_ref)]
    clean_data_list = [d[np.isfinite(d)] for d in data_list]
    all_data = [data_ref] + clean_data_list

    # Override get_bins if fixed_bins are provided
    if fixed_bins is not None:
        bins = fixed_bins
    else:
        bins = get_bins(all_data, xscale)
        
    if bins is None:
        print(f"Warning: No valid data for {output_path}")
        return

    fig = plt.figure(figsize=(8, 7))
    gs = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)

    # Reference
    counts_ref, _ = np.histogram(data_ref, bins=bins)
    ns_ref, _ = np.histogram(data_ref, bins=bins, density=True)
    
    mask = counts_ref > 0
    ref_err = np.zeros_like(ns_ref)
    ref_err[mask] = ns_ref[mask] / np.sqrt(counts_ref[mask])
    
    ax0.step(bins, dup_last(ns_ref), color='black', alpha=0.8, 
             linewidth=1.5, where='post', label='Data')
    ax0.fill_between(bins, dup_last(ns_ref - ref_err), dup_last(ns_ref + ref_err),
                     facecolor='blue', alpha=0.2, step='post')

    # Models
    evaluator = AtlasEvaluator()
    results = []
    
    for i, (data_mod, label) in enumerate(zip(clean_data_list, labels)):
        col = colors[i % len(colors)]
        ls = linestyles[i % len(linestyles)]
        
        counts_mod, _ = np.histogram(data_mod, bins=bins)
        ns_mod, _ = np.histogram(data_mod, bins=bins, density=True)
        
        ax0.step(bins, dup_last(ns_mod), color=col, linestyle=ls, 
                 linewidth=1.5, where='post', label=label)
        
        res = evaluator.calculate(data_ref, data_mod, counts_ref, counts_mod)
        results.append(res)
        
        # Ratio
        ratio = np.divide(ns_mod, ns_ref, out=np.zeros_like(ns_mod), where=ns_ref!=0)
        ax1.step(bins, dup_last(ratio), color=col, linestyle=ls, 
                 linewidth=1.5, where='post')

    if len(results) <= 3:
        text = evaluator.get_text(results, labels)
        ax0.text(0.96, 0.96, text, transform=ax0.transAxes,
                 fontsize=9, verticalalignment='top', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    ax0.set_yscale(yscale)
    ax0.set_xscale(xscale)
    ax0.set_ylabel("Normalized Counts", fontsize=14)
    hep.atlas.label("Preliminary", data=False, rlabel="", ax=ax0, loc=0)
    ax0.legend(fontsize=8, loc='upper left', frameon=False)
    ax0.tick_params(labelbottom=False)
    
    ax1.set_ylabel("Ratio", fontsize=12)
    ax1.set_xlabel(xlabel, fontsize=14)
    ax1.set_xscale(xscale)
    ax1.axhline(1, color='gray', linestyle='--', alpha=0.7)
    ax1.set_ylim(0.5, 1.5) 
    ax1.grid(True, which='both', linestyle=':', alpha=0.5)

    # Save intermediate data for replotting
    npz_path = os.path.splitext(output_path)[0] + '.npz'
    npz_data = {'bins': bins, 'data_ref': data_ref}
    for lbl, dat in zip(labels, clean_data_list):
        npz_data[lbl] = dat
    np.savez(npz_path, **npz_data)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    if pdf is not None:
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return {lbl: {k: float(v) for k, v in res.items()} for lbl, res in zip(labels, results)}
# -----------------------------------------------------------------------------
# 3. Combined Grid Plotter
# -----------------------------------------------------------------------------
def plot_layer_grid(layer_data_dict, property_name, labels, output_dir, 
                    yscale='log', xscale='linear', pdf=None,
                    colors=None, linestyles=None):
    """
    Creates a single figure with subplots for each layer.
    """
    if colors is None: colors = ['red', 'green', 'orange', 'purple', 'cyan']
    if linestyles is None: linestyles = ['-', '--', '-.', ':', '-']

    layers = sorted(layer_data_dict.keys())
    n_layers = len(layers)
    if n_layers == 0: return

    # Grid Dimensions
    n_cols = (n_layers + 1) // 2 
    n_rows = 2
    
    fig = plt.figure(figsize=(5 * n_cols, 10))
    outer_grid = GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.3, top=0.87)

    for idx, layer in enumerate(layers):
        row = idx // n_cols
        col = idx % n_cols

        inner_grid = GridSpecFromSubplotSpec(2, 1,
                        subplot_spec=outer_grid[row, col],
                        height_ratios=[3, 1], hspace=0.05)
        
        ax_main = fig.add_subplot(inner_grid[0])
        ax_ratio = fig.add_subplot(inner_grid[1], sharex=ax_main)
        
        data_ref = layer_data_dict[layer]['ref']
        data_models = layer_data_dict[layer]['models']
        
        data_ref = data_ref[np.isfinite(data_ref)]
        clean_models = [d[np.isfinite(d)] for d in data_models]
        all_d = [data_ref] + clean_models
        
        bins = get_bins(all_d, xscale)
        if bins is None: continue

        # Reference
        ns_ref, _ = np.histogram(data_ref, bins=bins, density=True)
        counts_ref, _ = np.histogram(data_ref, bins=bins)
        mask = counts_ref > 0
        ref_err = np.zeros_like(ns_ref)
        ref_err[mask] = ns_ref[mask] / np.sqrt(counts_ref[mask])

        ax_main.step(bins, dup_last(ns_ref), color='black', alpha=0.8, lw=1.5, 
                     where='post', label='Data' if idx==0 else "")
        ax_main.fill_between(bins, dup_last(ns_ref - ref_err), dup_last(ns_ref + ref_err),
                             facecolor='blue', alpha=0.2, step='post')

        # Models
        for i, (d_mod, lbl) in enumerate(zip(clean_models, labels)):
            ns_mod, _ = np.histogram(d_mod, bins=bins, density=True)
            ax_main.step(bins, dup_last(ns_mod), color=colors[i], ls=linestyles[i], lw=1.5, 
                         where='post', label=lbl if idx==0 else "")
            
            ratio = np.divide(ns_mod, ns_ref, out=np.zeros_like(ns_mod), where=ns_ref!=0)
            ax_ratio.step(bins, dup_last(ratio), color=colors[i], ls=linestyles[i], lw=1.5, where='post')

        # Styling
        ax_main.set_yscale(yscale)
        ax_main.set_xscale(xscale)
        ax_main.set_title(f"Layer {layer}", fontsize=12, fontweight='bold')
        ax_main.tick_params(labelbottom=False)

        if col == 0:
            ax_main.set_ylabel("Norm. Counts", fontsize=10)
            ax_ratio.set_ylabel("Ratio", fontsize=9)

        ax_ratio.set_xscale(xscale)
        ax_ratio.set_xlabel(property_name, fontsize=10)
        ax_ratio.axhline(1, color='gray', linestyle='--', alpha=0.7)
        ax_ratio.set_ylim(0.5, 1.5)
        ax_ratio.grid(True, which='both', linestyle=':', alpha=0.5)

    handles, legends = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, legends, loc='upper center', bbox_to_anchor=(0.5, 0.97), ncol=len(labels)+1, frameon=False)
    plt.suptitle(f"Combined {property_name} across Layers", y=1.01, fontsize=16)

    # 1. Grab the exact left edge of the first subplot
    x0 = fig.axes[0].get_position().x0
    
    # 2. Create an invisible, flat axis near the top of the figure
    # [left, bottom, width, height]
    ax_dummy = fig.add_axes([x0, 0.96, 0.5, 0.01])
    ax_dummy.axis('off')
    
    # 3. Attach the official mplhep label to the invisible axis
    hep.atlas.label("Preliminary", data=False, rlabel="", ax=ax_dummy, loc=0)

    sanitized_name = property_name.replace(' ', '_').replace('$','').replace('\\','').replace('{','').replace('}','')
    out_name = f"Grid_{sanitized_name}.png"
    plt.savefig(f"{output_dir}/{out_name}", dpi=300, bbox_inches='tight')
    
    if pdf:
        pdf.savefig(fig, dpi=300, bbox_inches='tight')
    plt.close(fig)

# -----------------------------------------------------------------------------
# 4. Main Driver (UPDATED: Fixed keys from 'r' to 'eta')
# -----------------------------------------------------------------------------
def make_validation_plots(hlf_ref, list_hlf_models, labels, output_dir="plots/"):
    print(f"Generating plots in {output_dir}...")
    pdf_path = os.path.join(output_dir, "all_plots.pdf")
    os.makedirs(output_dir, exist_ok=True)
    
    ref_etot = to_np(hlf_ref.E_tot)
    ref_einc = to_np(hlf_ref.Einc)

    # Collections for Grid Plots
    grid_energy = {}
    grid_mean_eta = {}  # Changed from grid_mean_r
    grid_width_eta = {} # Changed from grid_width_r
    grid_mean_phi = {}
    grid_width_phi = {}

    all_stats = {}
    with PdfPages(pdf_path) as pdf:

        # 1. Total Energy Ratio
        try:
            ratio = ref_etot / ref_einc.flatten()
            ref_etot_einc = to_np(ratio)
            models_etot_einc = [to_np(hlf.E_tot)/to_np(hlf.Einc).flatten() for hlf in list_hlf_models]

            s = plot_atlas_style_multi(
                ref_etot_einc, models_etot_einc, labels,
                xlabel=r'$E_{tot} / E_{inc}$',
                output_path=f"{output_dir}/Etot_over_Einc.png",
                yscale='log', pdf=pdf
            )
            if s: all_stats['Etot_over_Einc'] = s
        except Exception as e: print(f"FAILED Etot/Einc: {e}")

        # 2. Total Energy
        try:
            models_etot = [to_np(hlf.E_tot) for hlf in list_hlf_models]
            s = plot_atlas_style_multi(
                ref_etot, models_etot, labels,
                xlabel=r'$E_{tot}$ [MeV]',
                output_path=f"{output_dir}/Etot.png",
                yscale='log', pdf=pdf
            )
            if s: all_stats['Etot'] = s
        except Exception as e: print(f"FAILED Etot: {e}")

        # 3. Layer Loop
        for layer in hlf_ref.relevantLayers:
            try:
                # Energy
                ref_dat = to_np(hlf_ref.E_layers[layer])
                mod_dat = [to_np(hlf.E_layers[layer]) for hlf in list_hlf_models]
                grid_energy[layer] = {'ref': ref_dat, 'models': mod_dat}

                s = plot_atlas_style_multi(
                    ref_dat, mod_dat, labels,
                    xlabel=f'$E_{{layer {layer}}}$ [MeV]',
                    output_path=f"{output_dir}/Layer{layer}_Energy.png",
                    yscale='log', pdf=pdf
                )
                if s: all_stats[f'Layer{layer}_Energy'] = s

                # Mean Eta
                ref_dat = to_np(hlf_ref.EC_etas[layer])
                mod_dat = [to_np(hlf.EC_etas[layer]) for hlf in list_hlf_models]
                grid_mean_eta[layer] = {'ref': ref_dat, 'models': mod_dat}

                s = plot_atlas_style_multi(
                    ref_dat, mod_dat, labels,
                    xlabel=f'$\langle \eta \\rangle_{{layer {layer}}}$',
                    output_path=f"{output_dir}/Layer{layer}_MeanEta.png",
                    yscale='log', pdf=pdf
                )
                if s: all_stats[f'Layer{layer}_MeanEta'] = s

                # Width Eta
                ref_dat = to_np(hlf_ref.width_etas[layer])
                mod_dat = [to_np(hlf.width_etas[layer]) for hlf in list_hlf_models]
                grid_width_eta[layer] = {'ref': ref_dat, 'models': mod_dat}

                s = plot_atlas_style_multi(
                    ref_dat, mod_dat, labels,
                    xlabel=f'$\sigma_{{\eta, layer {layer}}}$',
                    output_path=f"{output_dir}/Layer{layer}_WidthEta.png",
                    yscale='log', pdf=pdf
                )
                if s: all_stats[f'Layer{layer}_WidthEta'] = s

                # Mean Phi
                ref_dat = to_np(hlf_ref.EC_phis[layer])
                mod_dat = [to_np(hlf.EC_phis[layer]) for hlf in list_hlf_models]
                grid_mean_phi[layer] = {'ref': ref_dat, 'models': mod_dat}

                s = plot_atlas_style_multi(
                    ref_dat, mod_dat, labels,
                    xlabel=f'$\langle \phi \\rangle_{{layer {layer}}}$',
                    output_path=f"{output_dir}/Layer{layer}_MeanPhi.png",
                    yscale='log', pdf=pdf
                )
                if s: all_stats[f'Layer{layer}_MeanPhi'] = s

                # Width Phi
                ref_dat = to_np(hlf_ref.width_phis[layer])
                mod_dat = [to_np(hlf.width_phis[layer]) for hlf in list_hlf_models]
                grid_width_phi[layer] = {'ref': ref_dat, 'models': mod_dat}

                s = plot_atlas_style_multi(
                    ref_dat, mod_dat, labels,
                    xlabel=f'$\sigma_{{\phi, layer {layer}}}$',
                    output_path=f"{output_dir}/Layer{layer}_WidthPhi.png",
                    yscale='log', pdf=pdf
                )
                if s: all_stats[f'Layer{layer}_WidthPhi'] = s

            except Exception as e:
                print(f"!! CRASH on Layer {layer}: {e}")
                continue

        # 4. Generate Grid Plots
        print("  Generating Grid Plots...")
        plot_layer_grid(grid_energy, 'Layer Energy [MeV]', labels, output_dir, yscale='log', pdf=pdf)
        
        # Updated calls for Eta grids
        plot_layer_grid(grid_mean_eta, 'Mean Eta', labels, output_dir, yscale='log', pdf=pdf)
        plot_layer_grid(grid_width_eta, 'Width Eta', labels, output_dir, yscale='log', pdf=pdf)
        
        plot_layer_grid(grid_mean_phi, 'Mean Phi', labels, output_dir, yscale='log', pdf=pdf)
        plot_layer_grid(grid_width_phi, 'Width Phi', labels, output_dir, yscale='log', pdf=pdf)

    stats_path = os.path.join(output_dir, 'stats.json')
    with open(stats_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    print(f"Stats saved to {stats_path}")
    print("Done! PDF saved to", pdf_path)




def create_grid_figure(layer_data_dict, property_name, labels, yscale='log', xscale='linear'):
    """
    Creates a matplotlib Figure for WandB logging from layer-wise data.
    """
    # Define colors/styles for consistency
    colors = ['red', 'green', 'orange', 'purple', 'cyan']
    linestyles = ['-', '--', '-.', ':', '-']

    layers = sorted(layer_data_dict.keys())
    n_layers = len(layers)
    if n_layers == 0: return None

    # Calculate grid dimensions
    n_cols = (n_layers + 1) // 2
    n_rows = 2

    fig = plt.figure(figsize=(5 * n_cols, 10))
    outer_grid = GridSpec(n_rows, n_cols, figure=fig, hspace=0.3, wspace=0.3, top=0.87)

    for idx, layer in enumerate(layers):
        row = idx // n_cols
        col = idx % n_cols

        inner_grid = GridSpecFromSubplotSpec(2, 1,
                        subplot_spec=outer_grid[row, col],
                        height_ratios=[3, 1], hspace=0.05)

        ax_main = fig.add_subplot(inner_grid[0])
        ax_ratio = fig.add_subplot(inner_grid[1], sharex=ax_main)

        data_ref = layer_data_dict[layer]['ref']
        data_models = layer_data_dict[layer]['models']

        # Filter NaNs/Infs
        data_ref = data_ref[np.isfinite(data_ref)]
        clean_models = [d[np.isfinite(d)] for d in data_models]
        all_d = [data_ref] + clean_models

        # Get bins (assumes get_bins is in scope)
        bins = get_bins(all_d, xscale)
        if bins is None: continue

        # Reference Plotting
        ns_ref, _ = np.histogram(data_ref, bins=bins, density=True)
        counts_ref, _ = np.histogram(data_ref, bins=bins)
        mask = counts_ref > 0
        ref_err = np.zeros_like(ns_ref)
        ref_err[mask] = ns_ref[mask] / np.sqrt(counts_ref[mask])

        # dup_last helper used for step plots
        ax_main.step(bins, dup_last(ns_ref), color='black', alpha=0.8, lw=1.5, 
                     where='post', label='Data' if idx==0 else "")
        ax_main.fill_between(bins, dup_last(ns_ref - ref_err), dup_last(ns_ref + ref_err),
                             facecolor='blue', alpha=0.2, step='post')

        # Model Plotting
        for i, (d_mod, lbl) in enumerate(zip(clean_models, labels)):
            ns_mod, _ = np.histogram(d_mod, bins=bins, density=True)
            ax_main.step(bins, dup_last(ns_mod), color=colors[i], ls=linestyles[i], lw=1.5, 
                         where='post', label=lbl if idx==0 else "")
            
            # Ratio Plotting
            ratio = np.divide(ns_mod, ns_ref, out=np.zeros_like(ns_mod), where=ns_ref!=0)
            ax_ratio.step(bins, dup_last(ratio), color=colors[i], ls=linestyles[i], lw=1.5, where='post')

        # Styling
        ax_main.set_yscale(yscale)
        ax_main.set_xscale(xscale)
        ax_main.set_title(f"Layer {layer}", fontsize=12, fontweight='bold')
        ax_main.tick_params(labelbottom=False)
        
        if col == 0:
            ax_main.set_ylabel("Norm. Counts", fontsize=10)
            ax_ratio.set_ylabel("Ratio", fontsize=9)

        ax_ratio.set_xscale(xscale)
        ax_ratio.set_xlabel(property_name, fontsize=10)
        ax_ratio.axhline(1, color='gray', linestyle='--', alpha=0.7)
        ax_ratio.set_ylim(0.5, 1.5)
        ax_ratio.grid(True, which='both', linestyle=':', alpha=0.5)

    # Legend and Layout
    handles, legends = fig.axes[0].get_legend_handles_labels()
    fig.legend(handles, legends, loc='upper center', bbox_to_anchor=(0.5, 0.97), ncol=len(labels)+1, frameon=False)
    plt.suptitle(f"Combined {property_name} across Layers", y=1.01, fontsize=16)

    return fig




def make_validation_plots_fixed(hlf_ref, list_hlf_models, labels, bin_ranges, num_bins=100, output_dir="plots/"):
    """
    Creates validation plots using explicit fixed ranges. Grid plots are skipped.
    
    Args:
        bin_ranges (dict): A dictionary mapping property names to (min, max) tuples. 
                           Keys can be 'Etot_over_Einc', 'Etot', 'Energy', 'MeanEta', 'WidthEta', 'MeanPhi', 'WidthPhi'.
        num_bins (int): The number of bins to divide the range into.
    """
    print(f"Generating fixed-bin plots in {output_dir}...")
    pdf_path = os.path.join(output_dir, "all_plots_fixed.pdf")
    os.makedirs(output_dir, exist_ok=True)
    
    ref_etot = to_np(hlf_ref.E_tot)
    ref_einc = to_np(hlf_ref.Einc)

    def create_bins(prop_name):
        """Helper to safely generate the bin array if the range was provided."""
        if prop_name not in bin_ranges:
            return None
        vmin, vmax = bin_ranges[prop_name]
        return np.linspace(vmin, vmax, num_bins + 1)

    all_stats = {}
    with PdfPages(pdf_path) as pdf:

        # 1. Total Energy Ratio
        if 'Etot_over_Einc' in bin_ranges:
            try:
                ratio = ref_etot / ref_einc.flatten()
                ref_etot_einc = to_np(ratio)
                models_etot_einc = [to_np(hlf.E_tot)/to_np(hlf.Einc).flatten() for hlf in list_hlf_models]

                s = plot_atlas_style_multi(
                    ref_etot_einc, models_etot_einc, labels,
                    xlabel=r'$E_{tot} / E_{inc}$',
                    output_path=f"{output_dir}/Etot_over_Einc.png",
                    yscale='log', pdf=pdf,
                    fixed_bins=create_bins('Etot_over_Einc')
                )
                if s: all_stats['Etot_over_Einc'] = s
            except Exception as e: print(f"FAILED Etot/Einc: {e}")

        # 2. Total Energy
        if 'Etot' in bin_ranges:
            try:
                models_etot = [to_np(hlf.E_tot) for hlf in list_hlf_models]
                s = plot_atlas_style_multi(
                    ref_etot, models_etot, labels,
                    xlabel=r'$E_{tot}$ [MeV]',
                    output_path=f"{output_dir}/Etot.png",
                    yscale='log', pdf=pdf,
                    fixed_bins=create_bins('Etot')
                )
                if s: all_stats['Etot'] = s
            except Exception as e: print(f"FAILED Etot: {e}")

        # 3. Layer Loop
        for layer in hlf_ref.relevantLayers:
            try:
                # Energy
                if 'Energy' in bin_ranges:
                    ref_dat = to_np(hlf_ref.E_layers[layer])
                    mod_dat = [to_np(hlf.E_layers[layer]) for hlf in list_hlf_models]
                    s = plot_atlas_style_multi(
                        ref_dat, mod_dat, labels,
                        xlabel=f'$E_{{layer {layer}}}$ [MeV]',
                        output_path=f"{output_dir}/Layer{layer}_Energy.png",
                        yscale='log', pdf=pdf,
                        fixed_bins=create_bins('Energy')
                    )
                    if s: all_stats[f'Layer{layer}_Energy'] = s

                # Mean Eta
                if 'MeanEta' in bin_ranges:
                    ref_dat = to_np(hlf_ref.EC_etas[layer])
                    mod_dat = [to_np(hlf.EC_etas[layer]) for hlf in list_hlf_models]
                    s = plot_atlas_style_multi(
                        ref_dat, mod_dat, labels,
                        xlabel=f'$\langle \eta \\rangle_{{layer {layer}}}$',
                        output_path=f"{output_dir}/Layer{layer}_MeanEta.png",
                        yscale='log', pdf=pdf,
                        fixed_bins=create_bins('MeanEta')
                    )
                    if s: all_stats[f'Layer{layer}_MeanEta'] = s

                # Width Eta
                if 'WidthEta' in bin_ranges:
                    ref_dat = to_np(hlf_ref.width_etas[layer])
                    mod_dat = [to_np(hlf.width_etas[layer]) for hlf in list_hlf_models]
                    s = plot_atlas_style_multi(
                        ref_dat, mod_dat, labels,
                        xlabel=f'$\sigma_{{\eta, layer {layer}}}$',
                        output_path=f"{output_dir}/Layer{layer}_WidthEta.png",
                        yscale='log', pdf=pdf,
                        fixed_bins=create_bins('WidthEta')
                    )
                    if s: all_stats[f'Layer{layer}_WidthEta'] = s

                # Mean Phi
                if 'MeanPhi' in bin_ranges:
                    ref_dat = to_np(hlf_ref.EC_phis[layer])
                    mod_dat = [to_np(hlf.EC_phis[layer]) for hlf in list_hlf_models]
                    s = plot_atlas_style_multi(
                        ref_dat, mod_dat, labels,
                        xlabel=f'$\langle \phi \\rangle_{{layer {layer}}}$',
                        output_path=f"{output_dir}/Layer{layer}_MeanPhi.png",
                        yscale='log', pdf=pdf,
                        fixed_bins=create_bins('MeanPhi')
                    )
                    if s: all_stats[f'Layer{layer}_MeanPhi'] = s

                # Width Phi
                if 'WidthPhi' in bin_ranges:
                    ref_dat = to_np(hlf_ref.width_phis[layer])
                    mod_dat = [to_np(hlf.width_phis[layer]) for hlf in list_hlf_models]
                    s = plot_atlas_style_multi(
                        ref_dat, mod_dat, labels,
                        xlabel=f'$\sigma_{{\phi, layer {layer}}}$',
                        output_path=f"{output_dir}/Layer{layer}_WidthPhi.png",
                        yscale='log', pdf=pdf,
                        fixed_bins=create_bins('WidthPhi')
                    )
                    if s: all_stats[f'Layer{layer}_WidthPhi'] = s

            except Exception as e:
                print(f"!! CRASH on Layer {layer}: {e}")
                continue

    stats_path = os.path.join(output_dir, 'stats.json')
    with open(stats_path, 'w') as f:
        json.dump(all_stats, f, indent=2)
    print(f"Stats saved to {stats_path}")
    print("Done! PDF saved to", pdf_path)


# -----------------------------------------------------------------------------
# Filename → axis label mapping for replot_from_npz
# -----------------------------------------------------------------------------
import re as _re

_STEM_TO_XLABEL = {
    'Etot_over_Einc': r'$E_{tot} / E_{inc}$',
    'Etot':           r'$E_{tot}$ [MeV]',
}

_LAYER_SUFFIX_TO_XLABEL = {
    'Energy':   r'$E_{{layer {layer}}}$ [MeV]',
    'MeanEta':  r'$\langle \eta \rangle_{{layer {layer}}}$',
    'WidthEta': r'$\sigma_{{\eta, layer {layer}}}$',
    'MeanPhi':  r'$\langle \phi \rangle_{{layer {layer}}}$',
    'WidthPhi': r'$\sigma_{{\phi, layer {layer}}}$',
}

_LAYER_RE = _re.compile(r'^Layer(\d+)_(\w+)$')


def _infer_xlabel(stem):
    if stem in _STEM_TO_XLABEL:
        return _STEM_TO_XLABEL[stem]
    m = _LAYER_RE.match(stem)
    if m:
        layer, suffix = m.group(1), m.group(2)
        template = _LAYER_SUFFIX_TO_XLABEL.get(suffix)
        if template:
            return template.format(layer=layer)
    return stem  # fallback: use filename stem as-is


def _infer_xscale(stem):
    return 'linear'


def _infer_yscale(stem):
    return 'log'


# -----------------------------------------------------------------------------
# Replot from saved .npz files
# -----------------------------------------------------------------------------
def replot_from_npz(save_dir, output_dir=None, yscale=None, xscale=None,
                    colors=None, linestyles=None, make_pdf=True, glob_pattern='*.npz'):
    """
    Regenerates all validation plots (individual + grid) from .npz files saved
    by plot_atlas_style_multi, matching the full output of evaluate_and_plot.

    Each .npz file must contain:
        bins      – bin edges used for the original histogram
        data_ref  – reference (ground truth) raw samples
        <label>   – one array per model, keyed by its label string

    Grid plots (Layer Energy, Mean/Width Eta/Phi) are reconstructed automatically
    from the per-layer .npz files — no shower tensors needed.

    Args:
        save_dir:      Directory that contains the .npz files.
        output_dir:    Where to write the new PNGs and PDF.
                       Defaults to ``save_dir/replot/``.
        yscale:        Y-axis scale override for every plot ('log' or 'linear').
                       When None, inferred per-file from filename.
        xscale:        X-axis scale override. When None, inferred per-file.
        colors:        List of colour strings for the model lines.
        linestyles:    List of linestyle strings for the model lines.
        make_pdf:      If True, compile all plots into a single PDF.
        glob_pattern:  Glob pattern used to find .npz files inside save_dir.
    """
    import glob as _glob
    from collections import defaultdict

    if output_dir is None:
        output_dir = os.path.join(save_dir, 'replot')
    os.makedirs(output_dir, exist_ok=True)

    if colors is None:
        colors = ['red', 'green', 'orange', 'purple', 'cyan']
    if linestyles is None:
        linestyles = ['-', '--', '-.', ':', '-']

    npz_files = sorted(_glob.glob(os.path.join(save_dir, glob_pattern)))
    if not npz_files:
        print(f"No .npz files found in {save_dir}")
        return

    pdf_path = os.path.join(output_dir, 'all_plots_replot.pdf')
    pdf_ctx = PdfPages(pdf_path) if make_pdf else None

    # Accumulators for grid reconstruction: {suffix -> {layer_int -> {'ref': arr, 'models': [arr,...]}}}
    _GRID_SUFFIXES = {
        'Energy':   'Layer Energy [MeV]',
        'MeanEta':  'Mean Eta',
        'WidthEta': 'Width Eta',
        'MeanPhi':  'Mean Phi',
        'WidthPhi': 'Width Phi',
    }
    grid_data  = {s: {} for s in _GRID_SUFFIXES}
    grid_labels = None  # set from the first layer file processed

    all_stats = {}
    try:
        # --- Pass 1: individual plots ---
        for npz_path in npz_files:
            stem = os.path.splitext(os.path.basename(npz_path))[0]
            try:
                npz = np.load(npz_path, allow_pickle=False)
            except Exception as e:
                print(f"  Skipping {npz_path}: {e}")
                continue

            keys = list(npz.files)
            if 'bins' not in keys or 'data_ref' not in keys:
                print(f"  Skipping {stem}: missing 'bins' or 'data_ref'")
                continue

            bins      = npz['bins']
            data_ref  = npz['data_ref']
            labels    = [k for k in keys if k not in ('bins', 'data_ref')]
            data_list = [npz[lbl] for lbl in labels]

            # Collect raw arrays for grid reconstruction if this is a per-layer file
            m = _LAYER_RE.match(stem)
            if m:
                layer_int = int(m.group(1))
                suffix    = m.group(2)
                if suffix in _GRID_SUFFIXES:
                    grid_data[suffix][layer_int] = {
                        'ref':    data_ref.copy(),
                        'models': [d.copy() for d in data_list],
                    }
                    if grid_labels is None:
                        grid_labels = labels

            xlabel  = _infer_xlabel(stem)
            _yscale = yscale if yscale is not None else _infer_yscale(stem)
            _xscale = xscale if xscale is not None else _infer_xscale(stem)

            output_png = os.path.join(output_dir, f"{stem}.png")

            # Build the figure (bins already fixed — no re-binning needed)
            data_ref_clean = data_ref[np.isfinite(data_ref)]
            clean_list     = [d[np.isfinite(d)] for d in data_list]

            fig = plt.figure(figsize=(8, 7))
            gs  = GridSpec(2, 1, height_ratios=[3, 1], hspace=0.05)
            ax0 = fig.add_subplot(gs[0])
            ax1 = fig.add_subplot(gs[1], sharex=ax0)

            counts_ref, _ = np.histogram(data_ref_clean, bins=bins)
            ns_ref, _     = np.histogram(data_ref_clean, bins=bins, density=True)
            mask    = counts_ref > 0
            ref_err = np.zeros_like(ns_ref)
            ref_err[mask] = ns_ref[mask] / np.sqrt(counts_ref[mask])

            ax0.step(bins, dup_last(ns_ref), color='black', alpha=0.8,
                     linewidth=1.5, where='post', label='Data')
            ax0.fill_between(bins, dup_last(ns_ref - ref_err), dup_last(ns_ref + ref_err),
                             facecolor='blue', alpha=0.2, step='post')

            evaluator = AtlasEvaluator()
            results   = []

            for i, (d_mod, lbl) in enumerate(zip(clean_list, labels)):
                col = colors[i % len(colors)]
                ls  = linestyles[i % len(linestyles)]

                counts_mod, _ = np.histogram(d_mod, bins=bins)
                ns_mod, _     = np.histogram(d_mod, bins=bins, density=True)

                ax0.step(bins, dup_last(ns_mod), color=col, linestyle=ls,
                         linewidth=1.5, where='post', label=lbl)

                res = evaluator.calculate(data_ref_clean, d_mod, counts_ref, counts_mod)
                results.append(res)

                ratio = np.divide(ns_mod, ns_ref, out=np.zeros_like(ns_mod), where=ns_ref != 0)
                ax1.step(bins, dup_last(ratio), color=col, linestyle=ls,
                         linewidth=1.5, where='post')

            all_stats[stem] = {
                lbl: {k: float(v) for k, v in res.items()}
                for lbl, res in zip(labels, results)
            }

            if len(results) <= 3:
                text = evaluator.get_text(results, labels)
                ax0.text(0.96, 0.96, text, transform=ax0.transAxes,
                         fontsize=9, verticalalignment='top', horizontalalignment='right',
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

            ax0.set_yscale(_yscale)
            ax0.set_xscale(_xscale)
            ax0.set_ylabel("Normalized Counts", fontsize=14)
            hep.atlas.label("Preliminary", data=False, rlabel="", ax=ax0, loc=0)
            ax0.legend(fontsize=8, loc='upper left', frameon=False)
            ax0.tick_params(labelbottom=False)

            ax1.set_ylabel("Ratio", fontsize=12)
            ax1.set_xlabel(xlabel, fontsize=14)
            ax1.set_xscale(_xscale)
            ax1.axhline(1, color='gray', linestyle='--', alpha=0.7)
            ax1.set_ylim(0.5, 1.5)
            ax1.grid(True, which='both', linestyle=':', alpha=0.5)

            plt.savefig(output_png, dpi=300, bbox_inches='tight')
            if pdf_ctx is not None:
                pdf_ctx.savefig(fig, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved {output_png}")

        # Write stats JSON after all individual plots are processed
        stats_path = os.path.join(output_dir, 'stats.json')
        with open(stats_path, 'w') as f:
            json.dump(all_stats, f, indent=2)
        print(f"  Stats saved to {stats_path}")

        # --- Pass 2: grid plots (reconstructed from accumulated layer data) ---
        if grid_labels is not None:
            print("  Generating grid plots...")
            for suffix, prop_label in _GRID_SUFFIXES.items():
                layer_dict = grid_data[suffix]
                if not layer_dict:
                    continue
                plot_layer_grid(
                    layer_dict, prop_label, grid_labels,
                    output_dir=output_dir,
                    yscale=yscale if yscale is not None else 'log',
                    xscale=xscale if xscale is not None else 'linear',
                    colors=colors,
                    linestyles=linestyles,
                    pdf=pdf_ctx,
                )
        else:
            print("  No per-layer .npz files found — grid plots skipped.")

    finally:
        if pdf_ctx is not None:
            pdf_ctx.close()

    print(f"Done! Replot complete. Output in {output_dir}"
          + (f"\nPDF: {pdf_path}" if make_pdf else ""))