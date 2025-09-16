import os
import re
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import wandb
from scripts.run import get_project_id, setup_model
from utils.HighLevelFeatures import HighLevelFeatures as HLF
from utils.HighLevelFeatsAtlasReg import HighLevelFeatures_ATLAS_regular as HLF2
import jetnet
import hydra
from hydra.utils import instantiate
from hydra import initialize, compose

from CaloQuVAE import logging
logger = logging.getLogger(__name__)

class HepMetrics:
    """
    Class to handle HEP metrics calculations and plotting.
    """
    def __init__(self, engine):
        dataset_name = engine._config.data.dataset_name
        logger.info(f"Initializing HepMetrics for dataset: {dataset_name}")
        self.if_Atlas = "Atlas" in dataset_name
        binning_path = engine._config.data.binning_path
        if self.if_Atlas:
            relevantLayers = engine._config.data.relevantLayers
            self.hlf = HLF2('electron', filename=binning_path, relevantLayers=relevantLayers)
            self.ref_hlf = HLF2('electron', filename=binning_path, relevantLayers=relevantLayers)
        else:
            self.hlf = HLF('electron', filename=binning_path)
            self.ref_hlf = HLF('electron', filename=binning_path)

        self.hlf.Einc = engine.incident_energy
        self.en_list = []
        self.fpd_recon, self.fpd_sample = [], []
        self.fpd_recon_err, self.fpd_sample_err = [], []
        self.kpd_recon, self.kpd_sample = [], []
        self.kpd_recon_err, self.kpd_sample_err = [], []

    def run(self, engine, idx):
        recon_HEPMetrics = get_fpd_kpd_metrics(
            np.array(engine.showers), 
            np.array(engine.showers_recon), 
            False, self.hlf, self.ref_hlf, if_Atlas=self.if_Atlas
        )
        sample_HEPMetrics = get_fpd_kpd_metrics(
            np.array(engine.showers), 
            np.array(engine.showers_prior), 
            False, self.hlf, self.ref_hlf, if_Atlas=self.if_Atlas
        )
        logger.info(f"Sample HEP Metrics for idx {idx}: {sample_HEPMetrics}")

        self.en_list.append(idx)
        self.fpd_recon.append(recon_HEPMetrics[0])
        self.fpd_recon_err.append(recon_HEPMetrics[1])
        self.kpd_recon.append(recon_HEPMetrics[2])
        self.kpd_recon_err.append(recon_HEPMetrics[3])

        self.fpd_sample.append(sample_HEPMetrics[0])
        self.fpd_sample_err.append(sample_HEPMetrics[1])
        self.kpd_sample.append(sample_HEPMetrics[2])
        self.kpd_sample_err.append(sample_HEPMetrics[3])
        logger.info(f"Finished generating HEP Metrics for epoch {idx}.")

    def flush(self, run_path):
        """
        Save the metrics to a file and reset internal lists.
        """
        save_plot(
            (
                self.en_list, self.fpd_recon, self.fpd_recon_err, 
                self.kpd_recon, self.kpd_recon_err, self.fpd_sample, 
                self.fpd_sample_err, self.kpd_sample, self.kpd_sample_err
            ), 
            run_path
        )
        self.en_list = []
        self.fpd_recon, self.fpd_sample = [], []
        self.fpd_recon_err, self.fpd_sample_err = [], []
        self.kpd_recon, self.kpd_sample = [], []
        self.kpd_recon_err, self.kpd_sample_err = [], []
        logger.info("HEP Metrics flushed and saved successfully.")

def save_plot(HEPMetric_output, run_path):
    """
    Save FPD and KPD metrics as plots and arrays.
    """
    path = run_path.split('files')[0] + 'files/'
    (
        en_list, fpd_recon, fpd_recon_err, kpd_recon, kpd_recon_err,
        fpd_sample, fpd_sample_err, kpd_sample, kpd_sample_err
    ) = HEPMetric_output
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot fpd_recon and fpd_sample on ax1
    ax1.errorbar(en_list, fpd_recon, yerr=fpd_recon_err, color='blue', label='FPD Recon')
    ax1.errorbar(en_list, fpd_sample, yerr=fpd_sample_err, color='green', label='FPD Sample')
    ax1.axhline(0.2, color='red', linestyle=':', linewidth=2, label='y=0.2')
    ax1.set_xlabel('Epoch Number')
    ax1.set_ylabel('FPD Values')
    ax1.set_title('FPD Recon vs FPD Sample')
    ax1.legend()

    # Plot kpd_recon and kpd_sample on ax2
    ax2.errorbar(en_list, kpd_recon, yerr=kpd_recon_err, color='blue', label='KPD Recon')
    ax2.errorbar(en_list, kpd_sample, yerr=kpd_sample_err, color='green', label='KPD Sample')
    # ax2.axhline(0.2, color='red', linestyle=':', linewidth=2, label='y=0.2')
    ax2.set_xlabel('Epoch Number')
    ax2.set_ylabel('KPD Values')
    ax2.set_title('KPD Recon vs KPD Sample')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(path, 'KPD_and_FPD.png'))
    np.savez(
        os.path.join(path, 'JetData.npz'),
        array1=en_list, array2=fpd_recon, array3=fpd_recon_err,
        array4=kpd_recon, array5=kpd_recon_err, array6=fpd_sample, 
        array7=fpd_sample_err, array8=kpd_sample, array9=kpd_sample_err
    )

def prepare_high_data_for_classifier(test, e_inc, hlf_class, label):
    """
    Extract high-level features for classifier and append label.
    """
    voxel, E_inc = test, e_inc
    E_tot = hlf_class.GetEtot()
    E_layer = [hlf_class.GetElayers()[layer_id].reshape(-1, 1) for layer_id in hlf_class.GetElayers()]
    EC_etas = [hlf_class.GetECEtas()[layer_id].reshape(-1, 1) for layer_id in hlf_class.layersBinnedInAlpha]
    EC_phis = [hlf_class.GetECPhis()[layer_id].reshape(-1, 1) for layer_id in hlf_class.layersBinnedInAlpha]
    Width_etas = [hlf_class.GetWidthEtas()[layer_id].reshape(-1, 1) for layer_id in hlf_class.layersBinnedInAlpha]
    Width_phis = [hlf_class.GetWidthPhis()[layer_id].reshape(-1, 1) for layer_id in hlf_class.layersBinnedInAlpha]
    E_layer = np.concatenate(E_layer, axis=1)
    EC_etas = np.concatenate(EC_etas, axis=1)
    EC_phis = np.concatenate(EC_phis, axis=1)
    Width_etas = np.concatenate(Width_etas, axis=1)
    Width_phis = np.concatenate(Width_phis, axis=1)
    ret = np.concatenate([
        np.log10(E_inc), np.log10(E_layer+1e-8), EC_etas/1e2, EC_phis/1e2,
        Width_etas/1e2, Width_phis/1e2, label*np.ones_like(E_inc)
    ], axis=1)
    return ret

def prepare_high_data_for_classifier_atlas(test, e_inc, hlf_class, label):
    """
    Prepare high-level features for classifier evaluation (ATLAS version).
    """
    voxel, E_inc = test, e_inc
    E_tot = hlf_class.GetEtot()
    E_layers = [hlf_class.GetElayers()[layer_id].reshape(-1, 1) for layer_id in hlf_class.GetElayers()]
    EC_rs = [hlf_class.GetECrs()[layer_id].reshape(-1, 1) for layer_id in hlf_class.GetElayers()]
    EC_phis = [hlf_class.GetECphis()[layer_id].reshape(-1, 1) for layer_id in hlf_class.GetElayers()]
    width_rs = [hlf_class.GetWidthrs()[layer_id].reshape(-1, 1) for layer_id in hlf_class.GetElayers()]
    width_phis = [hlf_class.GetWidthphis()[layer_id].reshape(-1, 1) for layer_id in hlf_class.GetElayers()]
    E_layers = np.concatenate(E_layers, axis=1)
    EC_rs = np.concatenate(EC_rs, axis=1)
    EC_phis = np.concatenate(EC_phis, axis=1)
    width_rs = np.concatenate(width_rs, axis=1)
    width_phis = np.concatenate(width_phis, axis=1)
    ret = np.concatenate((
        np.log10(E_inc), np.log10(E_layers + 1e-8), 
        EC_rs / 1e2, EC_phis / 1e2, width_rs / 1e2, width_phis / 1e2, 
        E_tot.reshape(-1, 1), label * np.ones_like(E_inc)
    ), axis=1)
    return ret

def check_and_replace_nans_infs(data):
    """
    Replace NaNs and Infs in data with zeros.
    """
    if np.isnan(data).any() or np.isinf(data).any():
        logger.info("Data contains NaNs or Infs. Handling them...")
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return data

def get_fpd_kpd_metrics(test_data, gen_data, syn_bool, hlf, ref_hlf, if_Atlas=False):
    """
    Compute FPD and KPD metrics between test and generated data.
    """
    if syn_bool:
        data_showers = np.array(test_data['showers'])
        energy = np.array(test_data['incident_energies'])
        gen_showers = np.array(gen_data['showers'], dtype=float)
        hlf.Einc = energy
    else:
        data_showers = test_data
        gen_showers = gen_data
    hlf.CalculateFeatures(data_showers)
    ref_hlf.CalculateFeatures(gen_showers)
    if if_Atlas:
        hlf_test_data = prepare_high_data_for_classifier_atlas(test_data, hlf.Einc, hlf, 0.)[:, :-1]
        hlf_gen_data = prepare_high_data_for_classifier_atlas(gen_data, hlf.Einc, ref_hlf, 1.)[:, :-1]
    else:
        hlf_test_data = prepare_high_data_for_classifier(test_data, hlf.Einc, hlf, 0.)[:, :-1]
        hlf_gen_data = prepare_high_data_for_classifier(gen_data, hlf.Einc, ref_hlf, 1.)[:, :-1]
    hlf_test_data = check_and_replace_nans_infs(hlf_test_data)
    hlf_gen_data = check_and_replace_nans_infs(hlf_gen_data)
    fpd_val, fpd_err = jetnet.evaluation.fpd(hlf_test_data, hlf_gen_data)
    kpd_val, kpd_err = jetnet.evaluation.kpd(hlf_test_data, hlf_gen_data)
    logger.info(
        f"FPD (x10^3): {fpd_val*1e3:.4f} ± {fpd_err*1e3:.4f}\n"
        f"KPD (x10^3): {kpd_val*1e3:.4f} ± {kpd_err*1e3:.4f}"
    )
    return fpd_val, fpd_err, kpd_val, kpd_err

def get_reference_point():
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize(version_base=None, config_path="../config")
    cfg = compose(config_name="config.yaml")
    wandb.init(tags = [cfg.data.dataset_name], project=cfg.wandb.project, entity=cfg.wandb.entity, config=OmegaConf.to_container(cfg, resolve=True), mode='disabled')
    engine1 = setup_model(cfg)
    engine2 = setup_model(cfg)
    engine1.evaluate_vae(engine1.data_mgr.val_loader, 0)
    engine2.evaluate_vae(engine2.data_mgr.test_loader, 0)
    
    # Get the lengths of both shower arrays
    len1 = len(engine1.showers)
    len2 = len(engine2.showers)
    
    # Determine the minimum length and truncate if necessary
    min_length = min(len1, len2)
    logger.info(f"Truncating showers to minimum length: {min_length} (original lengths: {len1}, {len2})")
    
    hlf = HLF2('electron', filename=engine1._config.data.binning_path)
    hlf.Einc = engine1.incident_energy
    ref_hlf = HLF2('electron', filename=engine1._config.data.binning_path)
    
    # Use truncated arrays for metric calculation
    ref_metrics = get_fpd_kpd_metrics(
        np.array(engine2.showers[:min_length]), 
        np.array(engine1.showers[:min_length]), 
        False, hlf, ref_hlf, if_Atlas=True
    )
    return ref_metrics

if __name__=="__main__":
    logger.info("Starting main executable.")
    main()
    logger.info("Finished running script")
