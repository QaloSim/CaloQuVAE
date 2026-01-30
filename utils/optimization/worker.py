import torch
import wandb
import logging
from omegaconf import OmegaConf, open_dict
from scripts.run import setup_model
from utils.optimization.orchestrator import EvaluationOrchestrator
import os
import gc

logger = logging.getLogger(__name__)

def update_recursive(config, key_path, value):
    """Updates nested config dictionary using dot notation."""
    keys = key_path.split('.')
    current = config
    for k in keys[:-1]:
        current = current[k] # Accessing as dict now
    current[keys[-1]] = value

def worker_task(gpu_id, trial_index, parameters, base_cfg_dict, training_settings, result_queue, save_dir):
    try:
        # 1. Rehydrate Config
        trial_cfg = OmegaConf.create(base_cfg_dict)
        
        # 2. Apply Trial-Specific Settings
        trial_cfg.gpu_list = [gpu_id]

        # 3. Apply Search Space Parameters (The only thing that changes per trial)
        for param_name, param_value in parameters.items():
            OmegaConf.update(trial_cfg, param_name, param_value)

        # 4. Extract Logic Control Vars (Not model config)
        eval_window = training_settings.eval_window

        # --- EXECUTION ---
        score = train_and_evaluate(trial_cfg, trial_index, gpu_id, save_dir, eval_window)
        result_queue.put((trial_index, score, gpu_id))

    except Exception as e:
        logger.error(f"Worker process failed on GPU {gpu_id}: {e}", exc_info=True)
        result_queue.put((trial_index, 1e9, gpu_id))


def train_and_evaluate(cfg, trial_index, gpu_id, save_dir, eval_window):
    # Initialize WandB with specific group settings for aggregation
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

    try:
        engine = setup_model(cfg)
        start_eval_epoch = max(0, cfg.n_epochs - eval_window)

        # Phase 1: Burn-in
        for epoch in range(cfg.epoch_start, start_eval_epoch):
            engine.fit_ae(epoch)

        # Phase 2: Evaluation Window
        orchestrator = EvaluationOrchestrator(
            base_cfg=cfg, model=engine.model, 
            reduce_fn=engine._reduce, inv_reduce_fn=engine._reduceinv, 
            device=engine.device
        )

        for epoch in range(start_eval_epoch, cfg.n_epochs):
            engine.fit_ae(epoch)
            engine.evaluate_ae(engine.data_mgr.val_loader, epoch) 
            engine.generate_plots(epoch, "ae")

            
            # Custom Objective Evaluation
            current_obj, results_map = orchestrator.evaluate_objective()
            
            if current_obj < best_objective:
                best_objective = current_obj
                orchestrator.save_plots(results_map, save_dir, trial_index)
                orchestrator.save_model(engine, os.path.join(save_dir, f"trial_{trial_index}_best"))

        return best_objective

    finally:
        wandb.finish()
        if engine: del engine
        gc.collect()
        torch.cuda.empty_cache()