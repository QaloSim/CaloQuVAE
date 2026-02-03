import torch
import wandb
import logging
from omegaconf import OmegaConf, open_dict
from scripts.run import setup_model
from utils.optimization.orchestrator import EvaluationOrchestrator
import os
import gc
from hydra import initialize_config_dir, compose
from hydra.core.global_hydra import GlobalHydra

logger = logging.getLogger(__name__)

def update_recursive(config, key_path, value):
    """Updates nested config dictionary using dot notation."""
    keys = key_path.split('.')
    current = config
    for k in keys[:-1]:
        current = current[k] # Accessing as dict now
    current[keys[-1]] = value

def worker_task(gpu_id, trial_index, parameters, base_cfg_dict, training_settings, result_queue, save_dir, hydra_config_dir, hydra_config_name):
    try:
        # --- FIX: INITIALIZE HYDRA IN WORKER ---
        # We use the absolute path passed from main to avoid relative path issues in the worker
        GlobalHydra.instance().clear()
        with initialize_config_dir(version_base=None, config_dir=hydra_config_dir):            
            # 1. Rehydrate Config
            trial_cfg = OmegaConf.create(base_cfg_dict)
            
            # 2. Apply Trial-Specific Settings
            trial_cfg.gpu_list = [gpu_id]

            # 3. Apply Search Space Parameters
            for param_name, param_value in parameters.items():
                OmegaConf.update(trial_cfg, param_name, param_value)

            eval_window = training_settings.eval_window

            # Pass config_name to train_and_evaluate
            score = train_and_evaluate(trial_cfg, trial_index, gpu_id, save_dir, eval_window, hydra_config_name)
            
            if torch.isnan(torch.tensor(score)):
                 raise ValueError("Returned score is NaN")
                 
            result_queue.put((trial_index, score, gpu_id))

    except Exception as e:
        logger.error(f"Worker process failed on GPU {gpu_id}: {e}", exc_info=True)
        result_queue.put((trial_index, float('inf'), gpu_id))


def train_and_evaluate(cfg, trial_index, gpu_id, save_dir, eval_window, config_name):
    run = wandb.init(
        project=cfg.wandb.project,
        entity=cfg.wandb.entity,
        group="Hyperopt_Campaign",
        name=f"trial_{trial_index}",
        config=OmegaConf.to_container(cfg, resolve=True),
        reinit=True,
        mode="online"
    )
    
    engine = None
    orchestrator = None
    best_objective = 1e9
    run_failed = False 

    try:
        engine = setup_model(cfg)
        start_eval_epoch = max(0, cfg.n_epochs - eval_window)
        logger.info(f"Training Trial {trial_index} on GPU {gpu_id} for {cfg.n_epochs} epochs.")

        for epoch in range(cfg.epoch_start, start_eval_epoch):
            engine.fit_ae(epoch)

        # PASS CONFIG NAME TO ORCHESTRATOR
        orchestrator = EvaluationOrchestrator(
            base_cfg=cfg, 
            model=engine.model, 
            reduce_fn=engine._reduce, 
            inv_reduce_fn=engine._reduceinv, 
            device=engine.device,
            config_name=config_name  # <--- PASSED HERE
        )

        for epoch in range(start_eval_epoch, cfg.n_epochs):
            engine.fit_ae(epoch) 
            engine.evaluate_ae(engine.data_mgr.val_loader, epoch) 
            
            # The orchestrator can now safely call compose()
            current_obj, results_map = orchestrator.evaluate_objective()
            logger.info(f"Trial {trial_index}, Epoch {epoch}: Objective = {current_obj}")
            
            if current_obj < best_objective:
                best_objective = current_obj
                orchestrator.save_plots(results_map, save_dir, trial_index)
                orchestrator.save_model(engine, os.path.join(save_dir, f"trial_{trial_index}_best"))

        return best_objective

    except Exception as e:
        run_failed = True
        logger.error(f"Trial {trial_index} failed: {e}")
        raise e 

    finally:
        wandb.finish(exit_code=1 if run_failed else 0)
        if engine: del engine
        gc.collect()
        torch.cuda.empty_cache()