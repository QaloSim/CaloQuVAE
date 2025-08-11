import numpy as np
import matplotlib.pyplot as plt
import torch
import h5py
def overall_plots(incident_energies, incident_energies_pos, incident_energies_fine, showers, showers_pos, showers_fine):
    fig, ax = plt.subplots(3, 2, figsize=(10, 15))

    # Plot total deposited energy
    showers_sum = torch.sum(showers, dim=1).numpy()
    showers_sum_pos = torch.sum(showers_pos, dim=1).numpy()
    showers_sum_fine = torch.sum(showers_fine, dim=1).numpy()
    n_bins = 1000

    sum_hist, bin_edges = np.histogram(showers_sum, bins=n_bins, range=(min(showers_sum), max(showers_sum)))
    sum_hist_pos, _ = np.histogram(showers_sum_pos, bins=bin_edges)
    sum_hist_fine, _ = np.histogram(showers_sum_fine, bins=bin_edges)
    ax[0, 0].stairs(sum_hist, bin_edges, label='Combined', color='blue', fill=True, alpha=0.5)
    ax[0, 0].stairs(sum_hist_pos, bin_edges, label='Positive', color='orange', fill=False, alpha=0.5)
    ax[0, 0].stairs(sum_hist_fine, bin_edges, label='Fine', color='green', fill=False, alpha=0.5)
    ax[0, 0].set_title('Total Deposited Energy')
    ax[0, 0].set_xlabel('Energy (MeV)')
    ax[0, 0].set_ylabel('Counts')
    ax[0, 0].set_yscale('log')
    ax[0, 0].legend()
    #plot differences on ax to the right
    ax[0, 1].set_title('Difference in Total Deposited Energy')
    ax[0, 1].scatter(bin_edges[:-1], sum_hist - sum_hist_pos, label='Combined - Positive', color='blue')
    ax[0, 1].scatter(bin_edges[:-1], sum_hist - sum_hist_fine, label='Combined - Fine', color='green')
    ax[0, 1].set_xlabel('Energy (MeV)')
    ax[0, 1].set_ylabel('Difference in Counts')
    ax[0, 1].legend()

    # Plot sparsity
    sparsity = (torch.sum(showers==0, dim=1).numpy() / showers.shape[1])
    sparsity_pos = (torch.sum(showers_pos==0, dim=1).numpy() / showers_pos.shape[1])
    sparsity_fine = (torch.sum(showers_fine==0, dim=1).numpy() / showers_fine.shape[1])

    sparsity_hist, bin_edges = np.histogram(sparsity, bins=n_bins, range=(0, 1))
    sparsity_hist_pos, _ = np.histogram(sparsity_pos, bins=bin_edges)
    sparsity_hist_fine, _ = np.histogram(sparsity_fine, bins=bin_edges)
    ax[1, 0].stairs(sparsity_hist, bin_edges, label='Combined', color='blue', fill=True, alpha=0.5)
    ax[1, 0].stairs(sparsity_hist_pos, bin_edges, label='Positive', color='orange', fill=False, alpha=0.5)
    ax[1, 0].stairs(sparsity_hist_fine, bin_edges, label='Fine', color='green', fill=False, alpha=0.5)
    ax[1, 0].set_title('Sparsity of Showers')
    ax[1, 0].set_xlabel('Sparsity')
    ax[1, 0].set_ylabel('Counts')
    ax[1, 0].set_yscale('log')
    ax[1, 0].legend()
    #plot differences on ax to the right
    ax[1, 1].set_title('Difference in Sparsity')
    ax[1, 1].scatter(bin_edges[:-1], sparsity_hist - sparsity_hist_pos, label='Combined - Positive', color='blue')
    ax[1, 1].scatter(bin_edges[:-1], sparsity_hist - sparsity_hist_fine, label='Combined - Fine', color='green')
    ax[1, 1].set_xlabel('Sparsity')
    ax[1, 1].set_ylabel('Difference in Counts')
    ax[1, 1].legend()

    # Plot ratio of sum over incident energy
    ratio_sum = showers_sum / incident_energies.numpy()
    ratio_sum_pos = showers_sum_pos / incident_energies_pos.numpy()
    ratio_sum_fine = showers_sum_fine / incident_energies_fine.numpy()
    ratio_hist, bin_edges = np.histogram(ratio_sum, bins=n_bins, range=(0, max(ratio_sum)))
    ratio_hist_pos, _ = np.histogram(ratio_sum_pos, bins=bin_edges)
    ratio_hist_fine, _ = np.histogram(ratio_sum_fine, bins= bin_edges)
    ax[2, 0].stairs(ratio_hist, bin_edges, label='Combined', color='blue', fill=True, alpha=0.5)
    ax[2, 0].stairs(ratio_hist_pos, bin_edges, label='Positive', color='orange', fill=False, alpha=0.5)
    ax[2, 0].stairs(ratio_hist_fine, bin_edges, label='Fine', color='green', fill=False, alpha=0.5)
    ax[2, 0].set_title('Ratio of Total Deposited Energy to Incident Energy')
    ax[2, 0].set_xlabel('Ratio')
    ax[2, 0].set_ylabel('Counts')
    ax[2, 0].set_yscale('log')
    ax[2, 0].legend()
    #plot differences on ax to the right
    ax[2, 1].set_title('Difference in Ratio')
    ax[2, 1].scatter(bin_edges[:-1], ratio_hist - ratio_hist_pos, label='Combined - Positive', color='blue')
    ax[2, 1].scatter(bin_edges[:-1], ratio_hist - ratio_hist_fine, label='Combined - Fine', color='green')
    ax[2, 1].set_xlabel('Ratio')
    ax[2, 1].set_ylabel('Difference in Counts')
    ax[2, 1].legend()

    fig.tight_layout()
    plt.legend()
    plt.show()

def load_showers_and_incident_energy(path, valid_layers):
    if valid_layers:
        with h5py.File(path, 'r') as file:
            data = {key: torch.tensor(file[key][:]) for key in file.keys()}
            incident_energy = data["incident_energy"]

            combined = torch.cat([data[f"energy_layer_{l}"] for l in valid_layers], dim=1)
            showers = combined * incident_energy.unsqueeze(1)
            return showers.numpy(), incident_energy.numpy()
    else:
        with h5py.File(path, 'r') as file:
            showers = torch.tensor(file['showers'][:])
            incident_energy = torch.tensor(file['incident_energy'][:])
            return showers.numpy(), incident_energy.numpy()

def get_global_valid_layers(split_paths):
    global_valid_layers = set()
    for path in split_paths:
        with h5py.File(path, 'r') as file:
            data = {key: torch.tensor(file[key][:]) for key in file.keys()}
            for l in range(24):
                key = f"energy_layer_{l}"
                if key in data and (data[key].sum(dim=1) != 0).any():
                    global_valid_layers.add(l)
    return sorted(global_valid_layers)

def compare_datasets(combined_incident_energies, combined_showers, rebuilt_incident_energies, rebuilt_showers):
    fig, ax = plt.subplots(4, 2, figsize=(10, 20))
    n_bins = 1000
    # Compare incident energies
    inc_hist, bin_edges = np.histogram(combined_incident_energies, bins=n_bins, range=(min(combined_incident_energies), max(combined_incident_energies)))
    inc_hist_rebuilt, _ = np.histogram(rebuilt_incident_energies, bins=bin_edges)
    ax[0, 0].stairs(inc_hist, bin_edges, label='Combined', color='blue', fill=True, alpha=0.5)
    ax[0, 0].stairs(inc_hist_rebuilt, bin_edges, label='Rebuilt', color='orange', fill=False, alpha=0.5)
    ax[0, 0].set_title('Incident Energy Distribution')
    ax[0, 0].set_xlabel('Incident Energy (MeV)')
    ax[0, 0].set_ylabel('Counts')
    ax[0, 0].set_yscale('log')
    ax[0, 0].legend()
    #plot differences on ax to the right
    ax[0, 1].set_title('Difference in Incident Energy Distribution')
    ax[0, 1].scatter(bin_edges[:-1], inc_hist - inc_hist_rebuilt, label='Combined - Rebuilt', color='blue')
    ax[0, 1].set_xlabel('Incident Energy (MeV)')
    ax[0, 1].set_ylabel('Difference in Counts')
    ax[0, 1].legend()

    # Compare total deposited energy
    combined_showers_sum = torch.sum(torch.tensor(combined_showers), dim=1).numpy()
    rebuilt_showers_sum = torch.sum(torch.tensor(rebuilt_showers), dim=1).numpy()
    sum_hist, bin_edges = np.histogram(combined_showers_sum, bins=n_bins, range=(min(combined_showers_sum), max(combined_showers_sum)))
    sum_hist_rebuilt, _ = np.histogram(rebuilt_showers_sum, bins=bin_edges)
    ax[1, 0].stairs(sum_hist, bin_edges, label='Combined', color='blue', fill=True, alpha=0.5)
    ax[1, 0].stairs(sum_hist_rebuilt, bin_edges, label='Rebuilt', color='orange', fill=False, alpha=0.5)
    ax[1, 0].set_title('Total Deposited Energy Distribution')
    ax[1, 0].set_xlabel('Total Deposited Energy (MeV)')
    ax[1, 0].set_ylabel('Counts')
    ax[1, 0].set_yscale('log')
    ax[1, 0].legend()
    #plot differences on ax to the right
    ax[1, 1].set_title('Difference in Total Deposited Energy Distribution')
    ax[1, 1].scatter(bin_edges[:-1], sum_hist - sum_hist_rebuilt, label='Combined - Rebuilt', color='blue')
    ax[1, 1].set_xlabel('Total Deposited Energy (MeV)')
    ax[1, 1].set_ylabel('Difference in Counts')
    ax[1, 1].legend()

    # Compare sparsity
    combined_sparsity = (torch.sum(torch.tensor(combined_showers)==0, dim=1).numpy() / combined_showers.shape[1])
    rebuilt_sparsity = (torch.sum(torch.tensor(rebuilt_showers)==0, dim=1).numpy() / rebuilt_showers.shape[1])
    sparsity_hist, bin_edges = np.histogram(combined_sparsity, bins=n_bins, range=(0, 1))
    sparsity_hist_rebuilt, _ = np.histogram(rebuilt_sparsity, bins=bin_edges)
    ax[2, 0].stairs(sparsity_hist, bin_edges, label='Combined', color='blue', fill=True, alpha=0.5)
    ax[2, 0].stairs(sparsity_hist_rebuilt, bin_edges, label='Rebuilt', color='orange', fill=False, alpha=0.5)
    ax[2, 0].set_title('Sparsity Distribution')
    ax[2, 0].set_xlabel('Sparsity')
    ax[2, 0].set_ylabel('Counts')
    ax[2, 0].set_yscale('log')
    ax[2, 0].legend()
    #plot differences on ax to the right
    ax[2, 1].set_title('Difference in Sparsity Distribution')
    ax[2, 1].scatter(bin_edges[:-1], sparsity_hist - sparsity_hist_rebuilt, label='Combined - Rebuilt', color='blue')
    ax[2, 1].set_xlabel('Sparsity')
    ax[2, 1].set_ylabel('Difference in Counts')
    ax[2, 1].legend()  

    # Compare ratio of total deposited energy to incident energy
    ratio_combined = combined_showers_sum / combined_incident_energies
    ratio_rebuilt = rebuilt_showers_sum / rebuilt_incident_energies
    ratio_hist, bin_edges = np.histogram(ratio_combined, bins=n_bins, range=(0, max(ratio_combined)))
    ratio_hist_rebuilt, _ = np.histogram(ratio_rebuilt, bins=bin_edges)
    ax[3, 0].stairs(ratio_hist, bin_edges, label='Combined', color='blue', fill=True, alpha=0.5)
    ax[3, 0].stairs(ratio_hist_rebuilt, bin_edges, label='Rebuilt', color='orange', fill=False, alpha=0.5)
    ax[3, 0].set_title('Ratio of Total Deposited Energy to Incident Energy Distribution')
    ax[3, 0].set_xlabel('Ratio')
    ax[3, 0].set_ylabel('Counts')
    ax[3, 0].set_yscale('log')
    ax[3, 0].legend()
    #plot differences on ax to the right
    ax[3, 1].set_title('Difference in Ratio Distribution')  
    ax[3, 1].scatter(bin_edges[:-1], ratio_hist - ratio_hist_rebuilt, label='Combined - Rebuilt', color='blue')
    ax[3, 1].set_xlabel('Ratio')
    ax[3, 1].set_ylabel('Difference in Counts')
    ax[3, 1].legend()
    fig.tight_layout()
    plt.legend()
    plt.show()
