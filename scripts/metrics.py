import os
import re
import numpy as np
from omegaconf import OmegaConf

from scripts.Frobenius import CorrelationMetrics
from scripts.Jet_metrics import HepMetrics
import wandb
from scripts.run import get_project_id, setup_model
from hydra import initialize, compose

from CaloQuVAE import logging
logger = logging.getLogger(__name__)

def main(cfg=None):
    """
    Main function to run the metrics evaluation script.
    Loads models and configs, evaluates metrics, and flushes results.
    """
    initialize(version_base=None, config_path="../config")
    config1 = compose(config_name="config.yaml")
    path = config1.config_path
    config = OmegaConf.load(path)
    iden = get_project_id(config.run_path)
    wandb.init(
        tags=[config.data.dataset_name],
        project=config.wandb.project,
        entity=config.wandb.entity,
        config=OmegaConf.to_container(config, resolve=True),
        mode='disabled',
        resume='allow',
        id=iden
    )

    filenames = create_filenames_dict(config)
    hepMet, corrMet = None, None

    for i, idx in enumerate(np.sort(list(filenames.keys()))):
        engine = load_engine(filenames[idx], config1)
        engine.evaluate_ae(engine.data_mgr.val_loader, 0)
        if i == 0:
            logger.info("First instance model")
            hepMet = HepMetrics(engine)
            corrMet = CorrelationMetrics(engine)

        hepMet.run(engine, idx)
        corrMet.run(engine, idx)

    if hepMet:
        hepMet.flush(config.run_path)
    if corrMet:
        corrMet.flush(config.run_path)

def load_engine(filename, config1):
    """
    Load the engine instance from the config file.
    Args:
        filename (list): [model_path, config_path]
        config1: Base config object
    Returns:
        engine: Loaded engine object
    """
    model_name, config_name = filename
    logger.info(f"Processing model: {model_name} \n config: {config_name}")
    config_loaded = OmegaConf.load(config_name)
    config_loaded.gpu_list = config1.gpu_list
    config_loaded.load_state = 1
    engine = setup_model(config_loaded)
    engine._model_creator.load_state(config_loaded.run_path, engine.device)
    return engine


def create_filenames_dict(config):
    """
    Create a dictionary mapping checkpoint indices to model and config file paths.
    
    Handles two formats:
    1. '..._{idx}.pth' (e.g., 'ae_separate_109.pth')
    2. '..._best_epoch{idx}.pth' (e.g., 'ae_separate_best_epoch99.pth')
    
    Args:
        config: Config object with run_path
    Returns:
        dict: {idx: [model_path, config_path]}
    """
    # Regex for '..._best_epochXX.pth'
    pattern_best = re.compile(r'_best_epoch(\d+)\.pth$')
    
    # Regex for '..._XX.pth'
    pattern_simple = re.compile(r'_(\d+)\.pth$')
    
    prefix = config.run_path.split('files')[0] + 'files/'
    
    try:
        # Get all files in the directory
        _fn = list(np.sort(os.listdir(prefix)))
    except FileNotFoundError:
        print(f"Warning: Directory not found: {prefix}")
        return {}
        
    filenames = {}
    
    # Iterate over all files, not just a pre-filtered list
    for filename in _fn:
        # We only care about .pth files
        if not filename.endswith('.pth'):
            continue

        model_path = prefix + filename
        idx = None
                
        match_best = pattern_best.search(filename)
        
        if match_best:
            # Found '..._best_epoch99.pth'
            # .group(1) captures just the digits (\d+)
            idx = int(match_best.group(1)) 
        else:
            # check for the simple format.
            match_simple = pattern_simple.search(filename)
            if match_simple:
                # Found '..._109.pth'
                idx = int(match_simple.group(1))
        
        # If we successfully found an index (idx) from either pattern
        if idx is not None:
            # Use the original (and correct) logic to create the config name
            config_path = re.sub(r'\.pth$', '_config.yaml', model_path)
            
            # Add the entry to our dictionary
            filenames[idx] = [model_path, config_path]
            
    return filenames
if __name__ == "__main__":
    logger.info("Starting main executable.")
    main()
