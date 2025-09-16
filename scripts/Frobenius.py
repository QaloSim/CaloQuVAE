import os
import re
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import wandb
from scripts.run import get_project_id
import matplotlib.figure as mf

from scripts.run import setup_model
from hydra import initialize, compose

from CaloQuVAE import logging
from utils.correlation_plotting import correlation_plots

logger = logging.getLogger(__name__)

def main(cfg=None):
    """
    Main function to run the correlation and Frobenius script.
    """
    initialize(version_base=None, config_path="../config")
    config1 = compose(config_name="config.yaml")
    path = config1.config_path
    config = OmegaConf.load(path)
    iden = get_project_id(config.run_path)
    wandb.init(tags=[config.data.dataset_name], project=config.wandb.project,
               entity=config.wandb.entity, config=OmegaConf.to_container(config, resolve=True),
               mode='disabled', resume='allow', id=iden)

    filenames = create_filenames_dict(config)
    for i, idx in enumerate(np.sort(list(filenames.keys()))):
        engine = load_engine(filenames[idx], config1)
        engine.evaluate_vae(engine.data_mgr.val_loader, 0)

        if i == 0:
            logger.info("First instance model")
            corrMet = CorrelationMetrics(engine)

        corrMet.run(engine, idx)

    corrMet.flush(config.run_path)

class CorrelationMetrics:
    """
    Class to handle correlation and Frobenius metrics
    """
    def __init__(self, engine):
            print(engine._config.data.dataset_name)
            self.epoch_list = []
            self.frobenius_scores = []
            self.config = engine._config

            # figures to save
            self.plots = {
                # voxel energy and sparsity correlation plots:
                'fig_target_corr': [], 'fig_sampled_corr': [],
                'fig_gt_grid': [], 'fig_prior_grid': [],
                'fig_frob_layerwise': [], 'fig_gt': [], 'fig_prior': [],
                'gt_spars_corr': [], 'prior_spars_corr': [],
                'fig_gt_sparsity': [], 'fig_prior_sparsity': [],
                'fig_gt_sparsity_corr': [], 'fig_prior_sparsity_corr': [],
                'fig_gt_patch': [], 'fig_prior_patch': [],
                # latent correlation plots:
                'fig_latent_post_corr': [], 'fig_latent_prior_corr': [],
                'fig_latent_frob_groups': [], 'fig_lat_partitions': [],
                'node_post': [], 'node_prior': [],
                'fig_grid_prior_latent': [], 'fig_grid_post_latent': [],
            }

    def run(self, engine, idx):
        results = correlation_plots(
            cfg=engine._config,
            incident_energy=engine.incident_energy,
            showers=engine.showers,
            showers_prior=engine.showers_prior,
            epoch=idx,
            post_samples=engine.post_samples,
            prior_samples=engine.prior_samples,
        )

        # unpack: metrics dict + latent figs dict
        main_figs   = results[:-2]
        metrics     = results[-2]
        latent_figs = results[-1]

        # base metrics
        frobs = {
            "voxel_corr": metrics["frob_dist_voxel"],
            "layer_corr": metrics["frob_dist_energy_corr_layer"],
            "sparsity_voxel": metrics["sparsity_frob_distance"],
            "sparsity_layer": metrics["frob_sparsity_dists_layer"], 
            "layer_sparsity": metrics["frob_sparsity_combined_layer"],
            "patch_layer": metrics["frob_patch_layer"],
        }
        # latent
        if "latent_frob_all" in metrics:
            frobs["latent_frob_all"] = metrics["latent_frob_all"]
        if "latent_frob_per_group" in metrics:
            for g, val in metrics["latent_frob_per_group"].items():
                frobs[f"latent_frob_per_group_g{g}"] = val
        if "pair_frobs" in metrics:
            # expand per pair (p1,p2) -> float
            for (p1, p2), val in metrics["pair_frobs"].items():
                frobs[f"latent_crosspair_frob_p{p1}_{p2}"] = val

        if not hasattr(self, "frob_all"):
            self.frob_all = {key: [] for key in frobs}

        # metrics aligned by epoch
        for key, val in frobs.items():
            self.frob_all.setdefault(key, []).append((idx, val))

        self.epoch_list.append(idx)
        self.frobenius_scores.append(frobs['voxel_corr'])
        logger.info(f"Frobenius score at epoch {idx}: {frobs['voxel_corr']:.4f}")

        # maps returned figs to keys in order
        keys = [
            'fig_target_corr','fig_sampled_corr','fig_gt_grid','fig_prior_grid',
            'fig_frob_layerwise','fig_gt','fig_prior',
            'gt_spars_corr','prior_spars_corr',
            'fig_gt_sparsity','fig_prior_sparsity',
            'fig_gt_sparsity_corr','fig_prior_sparsity_corr',
            'fig_gt_patch','fig_prior_patch',
            'fig_lat_partitions',    
        ]
        for fig_key, fig in zip(keys, main_figs):
            self.plots[fig_key].append((idx, fig))

        # added latent figs
        for k in [
            'fig_latent_post_corr','fig_latent_prior_corr','fig_latent_frob_groups',
            'node_post','node_prior','fig_grid_prior_latent','fig_grid_post_latent',
        ]:
            if k in latent_figs and latent_figs[k] is not None:
                self.plots[k].append((idx, latent_figs[k]))


    def flush(self, run_path):
        """
        Save figures to the folder CorrPlots/ and all frobenius metrics to Frobenius_All.npz
        """
        base = run_path.split('files')[0] + 'files/'
        os.makedirs(base, exist_ok=True)
        plots_dir = os.path.join(base, 'CorrPlots')
        os.makedirs(plots_dir, exist_ok=True)

        # save figures
        for key, fig_list in self.plots.items():
            for idx, fig in fig_list:
                if isinstance(fig, mf.Figure):
                    fig.savefig(os.path.join(plots_dir, f"{key}_epoch{idx}.png"),
                                dpi=300, bbox_inches="tight")

        # saving metrics
        if hasattr(self, "frob_all") and self.frob_all:
            epochs_sorted = np.array(sorted(set(self.epoch_list)))
            pack = {"epochs": epochs_sorted}

            for key, items in self.frob_all.items():
                # items is structured like list[(epoch, value)]
                by_epoch = dict(items)
                vals = [by_epoch[e] for e in epochs_sorted if e in by_epoch]
                # first try to stack
                try:
                    arr = np.stack(vals, axis=0)
                except Exception:
                    arr = np.array(vals, dtype=object)
                pack[key] = arr

            out_path = os.path.join(base, "Frobenius_All.npz")
            np.savez(out_path, **pack)

        logger.info("Frobenius and correlation metrics flushed and saved.")

def stack_or_object(vals):
    """
    Will try to stack a list of arrays/scalars and if shapes differ it will return an object array
    """
    try:
        return np.stack(vals, axis=0)  # (n_epochs, ...)
    except Exception:
        return np.array(vals, dtype=object)

def create_filenames_dict(config):
    pattern = r'\d+.pth$'
    prefix = config.run_path.split('files')[0] + 'files/'
    _fn = list(np.sort(os.listdir(prefix)))
    modelnames = [prefix + word for word in _fn if re.search(pattern, word)]
    confignames = [re.sub(r'\.pth$', '_config.yaml', name) for name in modelnames]

    idx = [int(name.split('_')[-1].split('.')[0]) for name in modelnames]
    filenames = {idx[i]: [modelnames[i], confignames[i]] for i in range(len(modelnames))}
    return filenames

def load_engine(filename, config1):
    """
    Load the engine instance from the config file
    """
    model_name, config_name = filename
    logger.info(f"Processing model: {model_name} \n config: {config_name}")
    config_loaded = OmegaConf.load(config_name)
    config_loaded.gpu_list = config1.gpu_list
    config_loaded.load_state = 1
    self = setup_model(config_loaded)
    self._model_creator.load_state(config_loaded.run_path, self.device)
    return self

# if __name__ == "__main__":
#     logger.info("Starting main executable.")
#     main()
#     logger.info("Finished running script")
