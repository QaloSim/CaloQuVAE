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

    plt.figure(figsize=(8, 6))
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
    plt.show()