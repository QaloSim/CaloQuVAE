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
from scripts.Jet_metrics import *

from CaloQuVAE import logging
logger = logging.getLogger(__name__)

class HepMetricsAtlas(HepMetrics):
    """
    similar to HepMetrics in jet_metrics.py, but adapted for Atlas data and models
    """

    def __init__(self, engine):
        self.hlf = HLF2('electron', filename=engine._config.data.binning_path)
        self.ref_hlf = HLF2('electron', filename=engine._config.data.binning_path)
        self.hlf.Einc = engine.incident_energy
        self.en_list = []
        self.fpd_recon, self.fpd_sample = [], []
        self.fpd_recon_err, self.fpd_sample_err = [], []
        self.kpd_recon, self.kpd_sample = [], []
        self.kpd_recon_err, self.kpd_sample_err = [], []
    
    def run(self, engine, idx):
        recon_HEPMetrics = get_fpd_kpd_metrics(np.array(engine.showers), 
                    np.array(engine.showers_recon), False, self.hlf, self.ref_hlf, if_Atlas=True)
        sample_HEPMetrics = get_fpd_kpd_metrics(np.array(engine.showers), 
                    np.array(engine.showers_prior), False, self.hlf, self.ref_hlf, if_Atlas=True)
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



def main_run():
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
            hepMet = HepMetricsAtlas(engine)
            
        hepMet.run(engine, idx)
    
    hepMet.flush(config.run_path)


