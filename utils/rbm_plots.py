import matplotlib.pyplot as plt
import numpy as np
import torch

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
        # w_vals = (engine.model.prior.weight_dict[key].sum(dim=1) / engine.model.prior.bias_dict[key[0]]).detach().cpu().numpy()
        w_vals = (engine.model.prior._weight_mask_dict[key].sum(dim=1) * engine.model.prior.weight_dict[key].pow(2).mean(dim=1).sqrt() / engine.model.prior.bias_dict[key[0]]).abs().detach().cpu().numpy()
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