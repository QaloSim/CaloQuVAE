import hydra
import logging
import os
import torch.multiprocessing as mp
from omegaconf import OmegaConf
from datetime import datetime
from ax.service.ax_client import AxClient, ObjectiveProperties
import queue
import sys

# Local imports
from utils.optimization.worker import worker_task

logger = logging.getLogger(__name__)
from hydra import compose, initialize
from ax.exceptions.generation_strategy import MaxParallelismReachedException
from hydra.core.global_hydra import GlobalHydra 

def build_ax_client(cfg):
    """Parses Hydra config to create AxClient."""
    ax_client = AxClient()
    
    # Dynamic Parameter Construction
    parameters = []
    for param in cfg.search_space:
        p_dict = OmegaConf.to_container(param, resolve=True)
        parameters.append(p_dict)

    ax_client.create_experiment(
        name=cfg.experiment.name,
        parameters=parameters,
        objectives={
            k: ObjectiveProperties(minimize=v.minimize) 
            for k, v in cfg.objectives.items()
        },
    )
    return ax_client

@hydra.main(config_path="../config/optimization", config_name="hyperopt", version_base=None)
def main(cfg):
    mp.set_start_method('spawn', force=True)

    # 1. Setup Directories
    # Use generic timestamps, avoiding user-specific hardcoded paths
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_dir = os.path.join(cfg.experiment.save_dir_root, f"BO_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)
    json_path = os.path.join(save_dir, "experiment_snapshot.json")
    
    logger.info(f"Artifacts: {save_dir}")

    # 2. Load Base Model Config (The Template)
    # We use Hydra to compose the base config manually
    hydra_overrides = []
    if "fixed_data_settings" in cfg and "data_config_name" in cfg.fixed_data_settings:
        hydra_overrides.append(f"data={cfg.fixed_data_settings.data_config_name}")
        logger.info(f"Forcing Data Config: {cfg.fixed_data_settings.data_config_name}")

    GlobalHydra.instance().clear()
    abs_config_dir = os.path.abspath(os.path.join(os.getcwd(), "../config"))
    
    with initialize(version_base=None, config_path="../config"):
        base_cfg = compose(config_name=cfg.base_model_config, overrides=hydra_overrides)

    base_cfg_dict = OmegaConf.to_container(base_cfg, resolve=True)
    
    if "fixed_data_settings" in cfg and "validation_datasets" in cfg.fixed_data_settings:
        base_cfg_dict['validation_datasets'] = OmegaConf.to_container(
            cfg.fixed_data_settings.validation_datasets, resolve=True
        )
        logger.info(f"Forcing Validation Datasets: {base_cfg_dict['validation_datasets']}")

    # B. Inject Training Duration (Consistency)
    # Since these don't change between trials, bake them in now.
    base_cfg_dict['n_epochs'] = cfg.training.n_epochs
    base_cfg_dict['epoch_start'] = cfg.training.epoch_start
    
    # 3. Setup Ax Client
    if cfg.experiment.resume and os.path.exists(json_path):
        ax_client = AxClient.load_from_json_file(filepath=json_path)
    else:
        ax_client = build_ax_client(cfg)
        # Attach seeds from YAML
        if cfg.get("manual_seeds"):
            for seed in cfg.manual_seeds:
                try:
                    _, idx = ax_client.attach_trial(parameters=OmegaConf.to_container(seed))
                    logger.info(f"Attached manual seed as Trial {idx}")
                except Exception as e:
                    logger.warning(f"Failed to attach seed: {e}")

    # 4. Resource Management
    manager = mp.Manager()
    gpu_queue = manager.Queue()
    result_queue = manager.Queue()
    
    # Populate GPUs
    gpu_list = cfg.resources.gpus
    for g in gpu_list: 
        gpu_queue.put(g)

    trials_to_run = cfg.experiment.total_trials
    active_processes = []
    
    # Extract optimization settings to pass to worker
    opt_settings = OmegaConf.to_container(cfg.training, resolve=True)

# Initialize with trials that are already terminal (completed/failed) so we don't rerun them
    launched_trial_indices = {
        t.index for t in ax_client.experiment.trials.values()
        if t.status.is_terminal
    }
    
    logger.info(f"Already completed trials: {launched_trial_indices}")

    # --- MAIN LOOP ---
    try:
        while True:
            # Check global completion status
            trials_completed = len([t for t in ax_client.experiment.trials.values() if t.status.is_completed])
            if trials_completed >= trials_to_run:
                logger.info("Target trial count reached. Exiting loop.")
                break

            # --- LAUNCHER ---
            # While we have GPUs and haven't hit the max parallelism...
            while not gpu_queue.empty():
                
                # Check Parallelism Limit
                if len(active_processes) >= cfg.resources.max_parallelism:
                    break
                
                # Stop if we have generated enough trials (including running ones)
                if len(ax_client.experiment.trials) >= trials_to_run:
                    # Only break if we don't have pending manual seeds (rare edge case, but safe)
                    pass 

                parameters = None
                trial_index = None

                # STRATEGY: ADOPT OR GENERATE
                # 1. Check for "Orphaned" Trials (Manual Seeds or Resumed Runners)
                # These are trials Ax knows about, but we haven't launched locally yet.
                existing_trials = [
                    t for t in ax_client.experiment.trials.values()
                    if t.index not in launched_trial_indices 
                    and not t.status.is_terminal
                ]
                # Sort to ensure we run manual seeds (0, 1, 2) in order
                existing_trials.sort(key=lambda t: t.index)

                if existing_trials:
                    # Adopt the existing trial
                    trial = existing_trials[0]
                    trial_index = trial.index
                    parameters = trial.arm.parameters
                    logger.info(f"Adopting existing/manual Trial {trial_index} (Status: {trial.status.name})")
                else:
                    # 2. Generate New Trial
                    try:
                        parameters, trial_index = ax_client.get_next_trial()
                    except MaxParallelismReachedException:
                        break
                    except Exception as e:
                        logger.warning(f"Ax failed to generate trial: {e}")
                        break

                # If we successfully got a trial to run
                if parameters is not None:
                    gpu_id = gpu_queue.get()
                    
                    # Mark as launched immediately so we don't double-launch
                    launched_trial_indices.add(trial_index)
                    
                    p = mp.Process(
                        target=worker_task,
                        args=(
                            gpu_id, 
                            trial_index, 
                            parameters, 
                            base_cfg_dict, 
                            cfg.training, 
                            result_queue, 
                            save_dir,
                            abs_config_dir,        # <--- Pass Absolute Path
                            cfg.base_model_config  # <--- Pass Config Name
                        )
                    )
                    p.start()
                    active_processes.append(p)
                    logger.info(f"Launched Trial {trial_index} on GPU {gpu_id}")
                else:
                    # No existing trials and could not generate new one
                    break

            # --- COLLECTOR ---
            # Wait for results if busy, else quick check
            should_block = len(active_processes) > 0
            timeout = 1.0 if should_block else 0.1
            
            try:
                idx, score, freed_gpu = result_queue.get(timeout=timeout)
                
                logger.info(f"Trial {idx} returned score: {score}")

                if score >= 1e9:
                    ax_client.log_trial_failure(trial_index=idx)
                else:
                    ax_client.complete_trial(trial_index=idx, raw_data=score)
                
                # Return GPU to pool
                gpu_queue.put(freed_gpu)
                
                # Save Snapshot
                try:
                    ax_client.save_to_json_file(filepath=json_path)
                except Exception as e:
                    logger.warning(f"Snapshot failed: {e}")
                
                # Cleanup process list
                active_processes = [p for p in active_processes if p.is_alive()]
                
            except queue.Empty:
                pass

    except KeyboardInterrupt:
        logger.warning("Shutting down...")
        for p in active_processes: p.terminate()
        sys.exit(1)

    # --- FINAL REPORTING ---
    best_parameters, _ = ax_client.get_best_parameters()
    logger.info(f"Best Params: {best_parameters}")
    # Add your visualization code here...

if __name__ == "__main__":
    main()