import os
import re
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import wandb
from scripts.run import get_project_id

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
        self.plots = {
            'fig_target_corr': [], 'fig_sampled_corr': [],
            'fig_gt_grid': [], 'fig_prior_grid': [],
            'fig_frob_layerwise': [], 'fig_gt': [], 'fig_prior': [],
            'fig_gt_sparsity': [], 'fig_prior_sparsity': [],
            'fig_gt_sparsity_corr': [], 'fig_prior_sparsity_corr': [],
            'fig_gt_patch': [], 'fig_prior_patch': []
        }

    def run(self, engine, idx):
        results = correlation_plots(
            cfg=engine._config,
            incident_energy=engine.incident_energy,
            showers=engine.showers,
            showers_prior=engine.showers_prior,
            epoch=idx
        )
        
        frobs = {
            "voxel_corr": results[-1]["frob_dist_voxel"],
            "layer_corr": results[-1]["frob_dist_energy_corr_layer"],
            "sparsity_voxel": results[-1]["sparsity_frob_distance"],
            "sparsity_layer": results[-1]["frob_sparsity_dists_layer"],
            "patch_layer": results[-1]["frob_patch_layer"]
        }
        if not hasattr(self, "frob_all"):
            self.frob_all = {key: [] for key in frobs}
        for key, val in frobs.items():
            self.frob_all[key].append((idx, val))

        self.epoch_list.append(idx)
        frob_val = frobs['voxel_corr']
        self.frobenius_scores.append(frob_val)
        logger.info(f"Frobenius score at epoch {idx}: {frob_val:.4f}")

        keys = list(self.plots.keys())
        for i, key in enumerate(keys):
            self.plots[key].append((idx, results[i]))

    def flush(self, run_path):
        """
        Save the metrics to a file
        """
        path = run_path.split('files')[0] + 'files/'
        os.makedirs(path, exist_ok=True)

        # saving figures
        for key, fig_list in self.plots.items():
            for idx, fig in fig_list:
                fig.savefig(os.path.join(path, f"{key}_epoch{idx}.png"))
                
        if hasattr(self, "frob_all"):
            for key, values in self.frob_all.items():
                epochs, vals = zip(*values)
                np.savez(os.path.join(path, f"{key}_FrobeniusScores.npz"),
                         epochs=np.array(epochs),
                         frobenius=np.array(vals))
                
        logger.info("Frobenius and correlation metrics flushed and saved.")

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

if __name__ == "__main__":
    logger.info("Starting main executable.")
    main()
    logger.info("Finished running script")
