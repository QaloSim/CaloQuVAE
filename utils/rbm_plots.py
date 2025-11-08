import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from model.rbm.rbm_two_partite import RBM_TwoPartite

def plot_rbm_histogram(rbm_post, rbm_prior, rbm_prior_qpu=None):
    # Assuming rbm_post and rbm_prior are tensors with energy values
    energy_post = rbm_post.numpy()
    energy_prior = rbm_prior.numpy()

    if rbm_prior_qpu is not None:
        energy_qpu = rbm_prior_qpu.numpy()
        minVal, maxVal = min(energy_post.min(), energy_prior.min(), energy_qpu.min()), max(energy_post.max(), energy_prior.max(), energy_qpu.max())
    else:
        minVal, maxVal = min(energy_post.min(), energy_prior.min()), max(energy_post.max(), energy_prior.max())

    binwidth = (maxVal - minVal) / 30

    fig = plt.figure(figsize=(8, 6))
    plt.hist(energy_post, bins=np.arange(minVal, maxVal + binwidth, binwidth), linewidth=2.5, color="b", density=True, log=True, label="RBM Post", alpha=0.7)
    plt.hist(energy_prior, bins=np.arange(minVal - 10, maxVal + binwidth, binwidth), color="orange", density=True, fc=(1, 0, 1, 0.5), log=True, histtype='step', linewidth=2.5, label="RBM Prior")
    
    if rbm_prior_qpu is not None:
        plt.hist(energy_qpu, bins=np.arange(minVal - 10, maxVal + binwidth, binwidth), color="m", density=True, fc=(1, 0, 1, 0.5), log=True, histtype='step', linewidth=2.5, label="RBM QPU")

    plt.xlabel("RBM Energy", fontsize=17)
    plt.ylabel("Probability density function", fontsize=17)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=15)
    plt.legend(fontsize=18)
    plt.grid("True")
    return fig

def plot_rbm_params(engine):
    engine.model.eval()
    F = {}
    bias_svd_u = {}
    bias_svd_v = {}
    for key in engine.model.prior.weight_dict.keys():
        F[key] = torch.svd(engine.model.prior.weight_dict[key].detach().cpu())
        # bias_svd_u[key] = {}
        # bias_svd_v[key] = {}
        # for key_2 in engine.model.prior.bias_dict.keys():
        #     bias_svd_u[key][key_2] = F[key].U.transpose(0,1) @ engine.model.prior.bias_dict[key_2].detach().cpu()
        #     bias_svd_v[key][key_2] = F[key].V.transpose(0,1) @ engine.model.prior.bias_dict[key_2].detach().cpu()

    # Assume engine, F, bias_svd are already defined and engine.model.prior.weight_dict has exactly 8 keys.
    keys = list(engine.model.prior.weight_dict.keys())  # length = 8
    bias_keys = list(engine.model.prior.bias_dict.keys())

    # Create a 3×8 grid of subplots
    fig, axes = plt.subplots(nrows=5, ncols=6, figsize=(32, 20), constrained_layout=True)
    fig.tight_layout(pad=4.0)

    # Row 0: Histograms of F[key].S for each of the 8 keys
    for col, key in enumerate(keys):
        ax = axes[0, col]
        values = F[key].S.numpy().ravel()
        ax.hist(values, bins=30, log=True, label=key, alpha=0.75, density=True)
        ax.set_title(f"{key}", fontsize=20)
        ax.xaxis.set_tick_params(labelsize=18)
        ax.yaxis.set_tick_params(labelsize=18)
        

    # Row 1: Histograms of prior.weight_dict[key] for each of the 8 keys
    for col, key in enumerate(keys):
        ax = axes[1, col]
        w_vals = engine.model.prior.weight_dict[key].view(-1).detach().cpu().numpy()
        ax.hist(w_vals, bins=30, log=True, label=key, alpha=0.75, density=False)
        ax.xaxis.set_tick_params(labelsize=18)
        ax.yaxis.set_tick_params(labelsize=18)
        # ax.set_title(f"weights[{key}]")
        # ax.legend(fontsize="small")
        
    # Row 1: Histograms of prior.weight_dict[key] for each of the 8 keys
    for col, key in enumerate(keys):
        ax = axes[1, col]
        # w_vals = engine.model.prior.weight_dict[key].view(-1).detach().cpu().numpy()
        tmp = engine.model.prior.weight_dict[key].abs() > 0.0 
        w_vals = engine.model.prior.weight_dict[key][tmp].view(-1).detach().cpu().numpy()
        ax.hist(w_vals, bins=30, log=True, alpha=0.5, label=key, density=False)
        ax.xaxis.set_tick_params(labelsize=18)
        ax.yaxis.set_tick_params(labelsize=18)
        # ax.set_title(f"weights[{key}]")
        
    # Row 2: Histograms of prior.weight_dict[key] for each of the 8 keys
    for col, key in enumerate(keys):
        ax = axes[2, col]
        num = (engine.model.prior._weight_mask_dict[key].sum(dim=1) * engine.model.prior.weight_dict[key].pow(2)).mean(dim=1).sqrt().detach().cpu()
        denom = engine.model.prior.bias_dict[key[0]].abs().detach().cpu()
        w_vals = torch.where(denom> 1e-8, num / denom, torch.zeros_like(num)).numpy()
        # w_vals = (engine.model.prior._weight_mask_dict[key].sum(dim=1) * engine.model.prior.weight_dict[key].pow(2).mean(dim=1).sqrt() / engine.model.prior.bias_dict[key[0]]).abs().detach().cpu().numpy()
        ratio = np.around((w_vals >= 1).sum()/w_vals.shape[0],6)
        med = np.around(np.median(w_vals),4)
        # print(med)
        ax.hist(w_vals, bins=30, log=True, color='blue', label=f'median:{"{:0.4f}".format(med)}, \n ratio:{ratio}', alpha=0.75, density=True)
        ax.xaxis.set_tick_params(labelsize=18)
        ax.yaxis.set_tick_params(labelsize=18)
        ax.legend(fontsize=20)

        
    # Row 3: Histograms of prior.weight_dict[key] for each of the 8 keys
    for col, key in enumerate(bias_keys):
        ax = axes[3, col]
        w_vals = engine.model.prior.bias_dict[key].view(-1).detach().cpu().numpy()
        ax.hist(w_vals, bins=30, log=True, label=key, alpha=0.75, density=True)
        ax.xaxis.set_tick_params(labelsize=18)
        ax.yaxis.set_tick_params(labelsize=18)

    # Row 4: Histograms of prior.weight_dict[key] for each of the 8 keys
    for col, key in enumerate(bias_keys):
        ax = axes[4, col]
        if col==0:
            w_vals = (engine.model.prior.bias_dict['1'].abs() \
            + engine.model.prior.weight_dict['01'].pow(2).sum(dim=0).sqrt() \
            + engine.model.prior.weight_dict['12'].pow(2).sum(dim=1).sqrt() \
            + engine.model.prior.weight_dict['13'].pow(2).sum(dim=0).sqrt()).detach().cpu().numpy()
            ax.hist(w_vals, bins=30, log=True, label=key, color='C1', alpha=0.75, density=True)
        elif col==1:
            w_vals = (engine.model.prior.bias_dict['2'].abs() \
            + engine.model.prior.weight_dict['02'].pow(2).sum(dim=0).sqrt() \
            + engine.model.prior.weight_dict['12'].pow(2).sum(dim=0).sqrt() \
            + engine.model.prior.weight_dict['23'].pow(2).sum(dim=1).sqrt()).detach().cpu().numpy()
            ax.hist(w_vals, bins=30, log=True, label=key, color='C1', alpha=0.75, density=True)
        elif col==2:
            w_vals = (engine.model.prior.bias_dict['3'].abs() \
            + engine.model.prior.weight_dict['03'].pow(2).sum(dim=0).sqrt() \
            + engine.model.prior.weight_dict['13'].pow(2).sum(dim=0).sqrt() \
            + engine.model.prior.weight_dict['23'].pow(2).sum(dim=0).sqrt()).detach().cpu().numpy()
            ax.hist(w_vals, bins=30, log=True, label=key, color='C1', alpha=0.75, density=True)
        ax.xaxis.set_tick_params(labelsize=18)
        ax.yaxis.set_tick_params(labelsize=18)

        axes[4, 3].axis('off')
        axes[4, 4].axis('off')
        axes[4, 5].axis('off')


    # Add an overall title and display
    fig.suptitle("Grid of Histograms: Singular values, Weights, and Bias SVD", fontsize=20, y=1.02)
    return fig

def plot_forward_output_v2(self, i=0):

    with torch.inference_mode():
        fwd_output = self.model((self.showers_reduce[i,:].repeat(1000,1).to(self.device), self.incident_energy[i,0].repeat(1000,1).to(self.device)))
        post_logits, post_samples = [fwd_output[1][j].detach().cpu() for j in range(len(fwd_output[1]))], [fwd_output[2][j].detach().cpu() for j in range(len(fwd_output[2]))]

    hist_colors   = ['C0', 'C1', 'C2']
    line_colors   = ['C3', 'C4', 'C5', 'C6']
    combo_colors  = ['C7', 'C8']  # for the third‐row overlay plots

    # Create a 3×4 grid of subplots
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    fig.tight_layout(pad=4.0)

    # -----------------------------
    # ROW 0: Three Histograms + 1 Empty
    # -----------------------------
    for idx in range(3):
        ax = axes[0, idx]
        data = nn.Sigmoid()(post_logits[idx]).view(-1).numpy()
        ax.hist(data, bins=50, log=True, color=hist_colors[idx], alpha=0.75)
        ax.set_title(f"Histogram of Sigmoid(post_logits[{idx}])")
        ax.grid(True, linestyle=':', alpha=0.5)

    # Leave the last cell in row 0 empty
    axes[0, 3].axis('off')
    # -----------------------------
    # ROW 1: Four Std‐Dev Line Plots (post_samples[idx].std)
    # -----------------------------
    for idx in range(4):
        ax = axes[1, idx]
        std_vals = (
            post_samples[idx].std(dim=0).numpy()
        )
        ax.plot(std_vals, color=line_colors[idx], linewidth=1.5)
        ax.set_title(f"Std‐dev of post_samples[{idx}]")
        ax.set_xlabel("Feature Index")
        ax.set_ylabel("Std Dev")
        ax.grid(True, linestyle=':', alpha=0.5)
    # -----------------------------
    # ROW 2: Overlay of Mean√[p(1−p)] & Next Sample’s Std (for idx=0,1,2)
    # -----------------------------
    for idx in range(3):
        ax = axes[2, idx]

        # Compute: √[p * (1 − p)] averaged over batch for post_logits[idx]
        p = nn.Sigmoid()(post_logits[idx])
        mean_sqrt = (
            (p * (1 - p)).sqrt().mean(dim=0).numpy()
        )

        # Compute: std‐dev of post_samples[idx+1] across axis=0
        std_next = (
            post_samples[idx + 1].std(dim=0).numpy()
        )

        ax.plot(mean_sqrt, color=combo_colors[0], linestyle='-', linewidth=1.5,
                label='Mean √[p(1−p)]')
        ax.plot(std_next, color=combo_colors[1], linestyle='--', linewidth=1.5,
                label=f'Std of post_samples[{idx+1}]')
        ax.set_title(f"Idx={idx} → Mean√p(1−p) & Std sample {idx+1}")
        ax.set_xlabel("Feature Index")
        ax.set_ylabel("Value")
        ax.legend(loc='upper right', fontsize='small')
        ax.grid(True, linestyle=':', alpha=0.5)

    # Leave the last cell in row 2 empty
    axes[2, 3].axis('off')

    # -----------------------------
    # ROW 3: Three Histograms + 1 Empty
    # -----------------------------
    for idx in range(3):
        ax = axes[3, idx]
        data = post_logits[idx].view(-1).numpy()
        ax.hist(data, bins=50, log=True, color=hist_colors[idx], alpha=0.75)
        ax.set_title(f"Histogram of raw post_logits[{idx}]")
        ax.grid(True, linestyle=':', alpha=0.5)

    # Leave the last cell in row 0 empty
    axes[3, 3].axis('off')

    return fig

def plot_forward_output_hidden(self, i=0):

    with torch.inference_mode():
        fwd_output = self.model((self.showers_reduce[i,:].repeat(1000,1).to(self.device), self.incident_energy[i,0].repeat(1000,1).to(self.device)))
        post_logits, post_samples = [fwd_output[1][j].detach().cpu() for j in range(len(fwd_output[1]))], [fwd_output[2][j].detach().cpu() for j in range(len(fwd_output[2]))]

    hist_colors   = ['C0', 'C1', 'C2']
    line_colors   = ['C3', 'C4', 'C5', 'C6']
    combo_colors  = ['C7', 'C8']  # for the third‐row overlay plots

    # Create a 3×4 grid of subplots
    fig, axes = plt.subplots(4, 3, figsize=(16, 12))
    fig.tight_layout(pad=4.0)

    # -----------------------------
    # ROW 0: Three Histograms + 1 Empty
    # -----------------------------
    for idx in range(2):
        ax = axes[0, idx]
        data = nn.Sigmoid()(post_logits[idx]).view(-1).numpy()
        ax.hist(data, bins=50, log=True, color=hist_colors[idx], alpha=0.75)
        ax.set_title(f"Histogram of Sigmoid(post_logits[{idx}])")
        ax.grid(True, linestyle=':', alpha=0.5)

    # Leave the last cell in row 0 empty
    axes[0, 2].axis('off')
    # -----------------------------
    # ROW 1: Four Std‐Dev Line Plots (post_samples[idx].std)
    # -----------------------------
    for idx in range(3):
        ax = axes[1, idx]
        std_vals = (
            post_samples[idx].std(dim=0).numpy()
        )
        ax.plot(std_vals, color=line_colors[idx], linewidth=1.5)
        ax.set_title(f"Std‐dev of post_samples[{idx}]")
        ax.set_xlabel("Feature Index")
        ax.set_ylabel("Std Dev")
        ax.grid(True, linestyle=':', alpha=0.5)
    # -----------------------------
    # ROW 2: Overlay of Mean√[p(1−p)] & Next Sample’s Std (for idx=0,1,2)
    # -----------------------------
    for idx in range(2):
        ax = axes[2, idx]

        # Compute: √[p * (1 − p)] averaged over batch for post_logits[idx]
        p = nn.Sigmoid()(post_logits[idx])
        mean_sqrt = (
            (p * (1 - p)).sqrt().mean(dim=0).numpy()
        )

        # Compute: std‐dev of post_samples[idx+1] across axis=0
        std_next = (
            post_samples[idx + 1].std(dim=0).numpy()
        )

        ax.plot(mean_sqrt, color=combo_colors[0], linestyle='-', linewidth=1.5,
                label='Mean √[p(1−p)]')
        ax.plot(std_next, color=combo_colors[1], linestyle='--', linewidth=1.5,
                label=f'Std of post_samples[{idx+1}]')
        ax.set_title(f"Idx={idx} → Mean√p(1−p) & Std sample {idx+1}")
        ax.set_xlabel("Feature Index")
        ax.set_ylabel("Value")
        ax.legend(loc='upper right', fontsize='small')
        ax.grid(True, linestyle=':', alpha=0.5)

    # Leave the last cell in row 2 empty
    axes[2, 2].axis('off')

    # -----------------------------
    # ROW 3: Three Histograms + 1 Empty
    # -----------------------------
    for idx in range(2):
        ax = axes[3, idx]
        data = post_logits[idx].view(-1).numpy()
        ax.hist(data, bins=50, log=True, color=hist_colors[idx], alpha=0.75)
        ax.set_title(f"Histogram of raw post_logits[{idx}]")
        ax.grid(True, linestyle=':', alpha=0.5)

    # Leave the last cell in row 0 empty
    axes[3, 2].axis('off')

    return fig

def plot_forward_output(self, i=0):

    with torch.inference_mode():
        x, x0 = next(iter(self.data_mgr.val_loader))
        x = x.to(self.device)
        x0 = x0.to(self.device)
        x = self._reduce(x,x0)
        fwd_output = self.model((x[i,:].repeat(1000,1), x0[i,0].repeat(1000,1)))
        post_logits, post_samples = [fwd_output[1][j].detach().cpu() for j in range(len(fwd_output[1]))], [fwd_output[2][j].detach().cpu() for j in range(len(fwd_output[2]))]

    hist_colors   = ['C0', 'C1', 'C2']
    line_colors   = ['C3', 'C4', 'C5', 'C6']
    combo_colors  = ['C7', 'C8']  # for the third‐row overlay plots

    # Create a 3×4 grid of subplots
    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    fig.tight_layout(pad=4.0)

    # -----------------------------
    # ROW 0: Three Histograms + 1 Empty
    # -----------------------------
    for idx in range(3):
        ax = axes[0, idx]
        data = nn.Sigmoid()(post_logits[idx]).view(-1).numpy()
        ax.hist(data, bins=50, log=True, color=hist_colors[idx], alpha=0.75)
        ax.set_title(f"Histogram of Sigmoid(post_logits[{idx}])")
        ax.grid(True, linestyle=':', alpha=0.5)

    # Leave the last cell in row 0 empty
    axes[0, 3].axis('off')
    # -----------------------------
    # ROW 1: Four Std‐Dev Line Plots (post_samples[idx].std)
    # -----------------------------
    for idx in range(4):
        ax = axes[1, idx]
        std_vals = (
            post_samples[idx].std(dim=0).numpy()
        )
        ax.plot(std_vals, color=line_colors[idx], linewidth=1.5)
        ax.set_title(f"Std‐dev of post_samples[{idx}]")
        ax.set_xlabel("Feature Index")
        ax.set_ylabel("Std Dev")
        ax.grid(True, linestyle=':', alpha=0.5)
    # -----------------------------
    # ROW 2: Overlay of Mean√[p(1−p)] & Next Sample’s Std (for idx=0,1,2)
    # -----------------------------
    for idx in range(3):
        ax = axes[2, idx]

        # Compute: √[p * (1 − p)] averaged over batch for post_logits[idx]
        p = nn.Sigmoid()(post_logits[idx])
        mean_sqrt = (
            (p * (1 - p)).sqrt().mean(dim=0).numpy()
        )

        # Compute: std‐dev of post_samples[idx+1] across axis=0
        std_next = (
            post_samples[idx + 1].std(dim=0).numpy()
        )

        ax.plot(mean_sqrt, color=combo_colors[0], linestyle='-', linewidth=1.5,
                label='Mean √[p(1−p)]')
        ax.plot(std_next, color=combo_colors[1], linestyle='--', linewidth=1.5,
                label=f'Std of post_samples[{idx+1}]')
        ax.set_title(f"Idx={idx} → Mean√p(1−p) & Std sample {idx+1}")
        ax.set_xlabel("Feature Index")
        ax.set_ylabel("Value")
        ax.legend(loc='upper right', fontsize='small')
        ax.grid(True, linestyle=':', alpha=0.5)

    # Leave the last cell in row 2 empty
    axes[2, 3].axis('off')

    # -----------------------------
    # ROW 3: Three Histograms + 1 Empty
    # -----------------------------
    for idx in range(3):
        ax = axes[3, idx]
        data = post_logits[idx].view(-1).numpy()
        ax.hist(data, bins=50, log=True, color=hist_colors[idx], alpha=0.75)
        ax.set_title(f"Histogram of raw post_logits[{idx}]")
        ax.grid(True, linestyle=':', alpha=0.5)

    # Leave the last cell in row 0 empty
    axes[3, 3].axis('off')

    return fig


def plot_weight_distribution(rbm: RBM_TwoPartite, bins: int = 50, title: str = None):
    """
    Plots a histogram of the RBM's weight_matrix.

    Args:
        rbm (RBM_TwoPartite): An instance of the RBM model.
        bins (int, optional): Number of bins for the histogram. Defaults to 50.
        title (str, optional): Title for the plot. If None, a default is used.
    """
    
    # 1. Get the weights tensor
    weights = rbm.params["weight_matrix"]
    
    # 2. Move to CPU (if on GPU) and convert to NumPy
    # .detach() is important to remove it from the computation graph
    weights_np = weights.detach().cpu().numpy()
    
    # 3. Flatten the 2D matrix into a 1D array for the histogram
    weights_flat = weights_np.flatten()
    
    # 4. Create the plot
    plt.figure(figsize=(10, 6))

    plt.hist(weights_flat, bins=bins, density=False, alpha=0.75, color='blue', edgecolor='black', label='Weight Distribution', log=True)

    # Add a title
    if title is None:
        title = 'Distribution of RBM Weights'
    plt.title(title, fontsize=16)
    
    # Add labels
    plt.xlabel('Weight Value', fontsize=12)
    plt.ylabel('Counts', fontsize=12)
    
    # Add a legend and grid
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Add mean and std dev text
    mean_val = np.mean(weights_flat)
    std_val = np.std(weights_flat)
    plt.text(0.05, 0.95, f'Mean: {mean_val:.4f}\nStd: {std_val:.4f}',
             transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8))
    
    # 5. Show the plot
    plt.tight_layout()
    plt.show()
