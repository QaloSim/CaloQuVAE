import os
import re
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import wandb
from scripts.run import get_project_id

from utils.HighLevelFeatures import HighLevelFeatures as HLF
from utils.HighLevelFeatsAtlasReg import HighLevelFeatures_ATLAS_regular as HLF2

from scripts.run import setup_model
from hydra import initialize, compose
import hydra

import jetnet

from CaloQuVAE import logging
logger = logging.getLogger(__name__)

def main(cfg=None):
    """
    Main function to run the Jet metrics script.
    """
    initialize(version_base=None, config_path="../config")
    config1=compose(config_name="config.yaml")
    path = config1.config_path
    config = OmegaConf.load(path)
    iden = get_project_id(config.run_path)
    wandb.init(tags = [config.data.dataset_name], project=config.wandb.project,
                entity=config.wandb.entity, config=OmegaConf.to_container(config, resolve=True), 
                mode='disabled', resume='allow', id=iden)
    
    filenames = create_filenames_dict(config)
    for i,idx in enumerate(np.sort(list(filenames.keys()))):
        engine = load_engine(filenames[idx], config1)
        engine.evaluate_vae(engine.data_mgr.val_loader, 0)
        if i == 0:
            logger.info("First instance model")
            hepMet = HepMetrics(engine)
            
        hepMet.run(engine, idx)

        # if i == 1:
        #     break
    
    hepMet.flush(config.run_path)
    
class HepMetrics:
    """
    Class to handle HEP metrics calculations and plotting.
    """
    def __init__(self, engine):
        print(engine._config.data.dataset_name)
        self.if_Atlas = "Atlas" in engine._config.data.dataset_name
        if self.if_Atlas:
            self.hlf = HLF2('electron', filename=engine._config.data.binning_path)
            self.ref_hlf = HLF2('electron', filename=engine._config.data.binning_path)
        else:
            self.hlf = HLF('electron', filename=engine._config.data.binning_path)
            self.ref_hlf = HLF('electron', filename=engine._config.data.binning_path)

        self.hlf.Einc = engine.incident_energy
        self.en_list = []
        self.fpd_recon, self.fpd_sample = [], []
        self.fpd_recon_err, self.fpd_sample_err = [], []
        self.kpd_recon, self.kpd_sample = [], []
        self.kpd_recon_err, self.kpd_sample_err = [], []

    def run(self, engine, idx):
        recon_HEPMetrics = get_fpd_kpd_metrics(np.array(engine.showers), 
                    np.array(engine.showers_recon), False, self.hlf, self.ref_hlf, if_Atlas=self.if_Atlas)
        sample_HEPMetrics = get_fpd_kpd_metrics(np.array(engine.showers), 
                    np.array(engine.showers_prior), False, self.hlf, self.ref_hlf, if_Atlas=self.if_Atlas)
        print(sample_HEPMetrics)
        
        self.en_list.append(idx)
        self.fpd_recon.append(recon_HEPMetrics[0])
        self.fpd_recon_err.append(recon_HEPMetrics[1])
        self.kpd_recon.append(recon_HEPMetrics[2])
        self.kpd_recon_err.append(recon_HEPMetrics[3])

        self.fpd_sample.append(sample_HEPMetrics[0])
        self.fpd_sample_err.append(sample_HEPMetrics[1])
        self.kpd_sample.append(sample_HEPMetrics[2])
        self.kpd_sample_err.append(sample_HEPMetrics[3])
        logger.info("Finished generating HEP Metrics for epoch " + str(idx) + " ...")

    def flush(self, run_path):
        """
        Save the metrics to a file.
        """
        save_plot((self.en_list, self.fpd_recon, self.fpd_recon_err, 
                   self.kpd_recon, self.kpd_recon_err, self.fpd_sample, 
                   self.fpd_sample_err, self.kpd_sample, self.kpd_sample_err), 
                  run_path)
        self.en_list = []
        self.fpd_recon, self.fpd_sample = [], []
        self.fpd_recon_err, self.fpd_sample_err = [], []
        self.kpd_recon, self.kpd_sample = [], []
        self.kpd_recon_err, self.kpd_sample_err = [], []
        logger.info("HEP Metrics flushed and saved successfully.")

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
    Load the engine instance from the config file.
    """
    model_name, config_name = filename
    logger.info(f"Processing model: {model_name} \n config: {config_name}")
    config_loaded = OmegaConf.load(config_name)
    config_loaded.gpu_list = config1.gpu_list
    config_loaded.load_state = 1
    self = setup_model(config_loaded)
    self._model_creator.load_state(config_loaded.run_path, self.device)
    return self

def save_plot(HEPMetric_output, run_path):
    path = run_path.split('files')[0] + 'files/'
    en_list, fpd_recon, fpd_recon_err, kpd_recon, kpd_recon_err, fpd_sample, fpd_sample_err, kpd_sample, kpd_sample_err = HEPMetric_output
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot fpd_recon and fpd_sample on ax1
    ax1.errorbar(en_list, fpd_recon, yerr=fpd_recon_err, color='blue', label='FPD Recon')
    ax1.errorbar(en_list, fpd_sample, yerr=fpd_sample_err, color='green', label='FPD Sample')
    ax1.set_xlabel('Epoch Number')
    ax1.set_ylabel('FPD Values')
    ax1.set_title('FPD Recon vs FPD Sample')
    ax1.legend()

    # Plot kpd_recon and kpd_sample on ax2
    ax2.errorbar(en_list, kpd_recon, yerr=kpd_recon_err, color='blue', label='KPD Recon')
    ax2.errorbar(en_list, kpd_sample, yerr=kpd_sample_err, color='green', label='KPD Sample')
    ax2.set_xlabel('Epoch Number')
    ax2.set_ylabel('KPD Values')
    ax2.set_title('KPD Recon vs KPD Sample')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(path + f'KPD_and_FPD.png')
    # plt.show()
    np.savez(path + 'JetData.npz', array1=en_list, array2=fpd_recon, array3=fpd_recon_err,
             array4=kpd_recon, array5=kpd_recon_err, array6=fpd_sample, 
             array7=fpd_sample_err, array8=kpd_sample, array9=kpd_sample_err)


def prepare_high_data_for_classifier(test, e_inc, hlf_class, label):
    """ takes hdf5_file, extracts high-level features, appends label, returns array """
    # voxel, E_inc = extract_shower_and_energy(hdf5_file, label)
    voxel, E_inc = test, e_inc
    E_tot = hlf_class.GetEtot()
    E_layer = []
    for layer_id in hlf_class.GetElayers():
        E_layer.append(hlf_class.GetElayers()[layer_id].reshape(-1, 1))
    EC_etas = []
    EC_phis = []
    Width_etas = []
    Width_phis = []
    for layer_id in hlf_class.layersBinnedInAlpha:
        EC_etas.append(hlf_class.GetECEtas()[layer_id].reshape(-1, 1))
        EC_phis.append(hlf_class.GetECPhis()[layer_id].reshape(-1, 1))
        Width_etas.append(hlf_class.GetWidthEtas()[layer_id].reshape(-1, 1))
        Width_phis.append(hlf_class.GetWidthPhis()[layer_id].reshape(-1, 1))
    E_layer = np.concatenate(E_layer, axis=1)
    EC_etas = np.concatenate(EC_etas, axis=1)
    EC_phis = np.concatenate(EC_phis, axis=1)
    Width_etas = np.concatenate(Width_etas, axis=1)
    Width_phis = np.concatenate(Width_phis, axis=1)
    ret = np.concatenate([np.log10(E_inc), np.log10(E_layer+1e-8), EC_etas/1e2, EC_phis/1e2,
                          Width_etas/1e2, Width_phis/1e2, label*np.ones_like(E_inc)], axis=1)
    return ret
def prepare_high_data_for_classifier_atlas(test, e_inc, hlf_class, label):
    """
    Prepare high-level features for classifier evaluation.
    """
    voxel, E_inc = test, e_inc
    E_tot = hlf_class.GetEtot()
    E_layers = []
    EC_rs = []
    EC_phis = []
    width_rs = []
    width_phis = []
    for layer_id in hlf_class.GetElayers():
        E_layers.append(hlf_class.GetElayers()[layer_id].reshape(-1, 1))
        EC_rs.append(hlf_class.GetECrs()[layer_id].reshape(-1, 1))
        EC_phis.append(hlf_class.GetECphis()[layer_id].reshape(-1, 1))
        width_rs.append(hlf_class.GetWidthrs()[layer_id].reshape(-1, 1))
        width_phis.append(hlf_class.GetWidthphis()[layer_id].reshape(-1, 1))
    E_layers = np.concatenate(E_layers, axis=1)
    EC_rs = np.concatenate(EC_rs, axis=1)
    EC_phis = np.concatenate(EC_phis, axis=1)
    width_rs = np.concatenate(width_rs, axis=1)
    width_phis = np.concatenate(width_phis, axis=1)
    ret = np.concatenate((np.log10(E_inc), np.log10(E_layers + 1e-8), 
    EC_rs / 1e2, EC_phis / 1e2, width_rs / 1e2, width_phis / 1e2, E_tot.reshape(-1, 1), label * np.ones_like(E_inc)), axis=1)
    return ret


def check_and_replace_nans_infs(data):
    if np.isnan(data).any() or np.isinf(data).any():
        logger.info("Data contains NaNs or Infs. Handling them...")
        # Replace NaNs and Infs with zeros (or you can choose a different strategy)
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return data

def get_fpd_kpd_metrics(test_data, gen_data, syn_bool, hlf, ref_hlf, if_Atlas=False):
    # print("TESTING HELLO")
    if syn_bool == True:
        data_showers = (np.array(test_data['showers']))
        energy = (np.array(test_data['incident_energies']))
        gen_showers = (np.array(gen_data['showers'], dtype=float))
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
    
    result_str = (
        f"FPD (x10^3): {fpd_val*1e3:.4f} ± {fpd_err*1e3:.4f}\n" 
        f"KPD (x10^3): {kpd_val*1e3:.4f} ± {kpd_err*1e3:.4f}"
    )
    
    logger.info(result_str)
    return fpd_val, fpd_err, kpd_val, kpd_err

def get_reference_point():
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    initialize(version_base=None, config_path="../config")
    cfg1 = compose(config_name="config.yaml")
    cfg2 = compose(config_name="test_config.yaml")
    wandb.init(tags = [cfg1.data.dataset_name], project=cfg1.wandb.project, entity=cfg1.wandb.entity, config=OmegaConf.to_container(cfg1, resolve=True), mode='disabled')
    wandb.init(tags = [cfg2.data.dataset_name], project=cfg2.wandb.project, entity=cfg2.wandb.entity, config=OmegaConf.to_container(cfg2, resolve=True), mode='disabled')
    engine1 = setup_model(cfg1)
    engine2 = setup_model(cfg2)
    engine1.evaluate_vae(engine1.data_mgr.test_loader, 0)
    engine2.evaluate_vae(engine2.data_mgr.test_loader, 0)
    hlf = HLF2('electron', filename=engine1._config.data.binning_path)
    hlf.Einc = engine1.incident_energy
    ref_hlf = HLF2('electron', filename=engine1._config.data.binning_path)
    ref_metrics = get_fpd_kpd_metrics(np.array(engine2.showers), np.array(engine1.showers), False, hlf, ref_hlf, if_Atlas=True)
    return ref_metrics

if __name__=="__main__":
    logger.info("Starting main executable.")
    main()
    logger.info("Finished running script")
