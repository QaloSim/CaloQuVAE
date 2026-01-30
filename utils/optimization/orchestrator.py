import copy
from datetime import datetime
import torch
from utils.optimization.scalar_metrics import ScalarMetricCalculator
from data.dataManager import DataManager
from hydra import compose
from hydra.core.hydra_config import HydraConfig
from utils.optimization.plots import ShowerPlotter
import os

class EvaluationOrchestrator:
    def __init__(self, base_cfg, model, reduce_fn, inv_reduce_fn, device='cuda'):
        """
        Args:
            base_cfg: Config object.
            model: The CaloQVAE model.
            reduce_fn: Function (x, x0) -> x_reduce
            inv_reduce_fn: Function (x_reduce, x0) -> x_physical
            device: 'cuda' or 'cpu'
        """
        self.cfg = base_cfg
        self.model = model.to(device)
        self.reduce_fn = reduce_fn
        self.inv_reduce_fn = inv_reduce_fn
        self.device = device
        
        self.calculator = ScalarMetricCalculator(
            binning_path=base_cfg.data.binning_path, 
            device=device
        )

    def extract_showers(self, dataloader):
        """
        Replicates evaluate_ae logic:
        1. Reduce Input -> 2. Forward Pass -> 3. Inverse Reduce Output
        """
        self.model.eval()
        
        ref_list = []
        gen_list = []
        
        with torch.no_grad():
            for i, (x, x0) in enumerate(dataloader):
                x = x.to(self.device)
                x0 = x0.to(self.device)

                # 1. Pre-process (Reduce)
                x_reduce = self.reduce_fn(x, x0)

                # 2. Forward Pass
                # Model expects tuple (x_reduce, x0)
                output = self.model((x_reduce, x0))
                
                # Output structure matches your snippet: output[3] is recon
                x_reduce_recon = output[3]

                # 3. Post-process (Inverse Reduce) to get back to physical space
                x_recon = self.inv_reduce_fn(x_reduce_recon, x0)

                # Store Physical Space Showers
                ref_list.append(x.cpu())
                gen_list.append(x_recon.cpu())

        return torch.cat(ref_list), torch.cat(gen_list)

    def evaluate_objective(self):
        """
        Calculates the objective and returns the score AND the results 
        needed for plotting (to avoid re-inference).
        
        Returns:
            aggregate_score (float): The mean loss across datasets.
            results_map (dict): {dataset_name: results_dict}
        """
        target_datasets = self.cfg.get("validation_datasets", [])
        aggregate_score = 0.0
        results_map = {} # Store results for plotting later
        
        try:
            root_config_name = HydraConfig.get().job.config_name
        except Exception:
            root_config_name = "config" 

        for dataset_name in target_datasets:
            try:
                fresh_cfg = compose(config_name=root_config_name, overrides=[f"data={dataset_name}"])
            except Exception as e:
                print(f"Failed to compose config for {dataset_name}. Error: {e}")
                continue

            temp_cfg = copy.deepcopy(self.cfg)
            temp_cfg.data = fresh_cfg.data

            val_data_manager = DataManager(temp_cfg) 
            val_loader = val_data_manager.val_loader
            
            ref_showers, gen_showers = self.extract_showers(val_loader)
            loss, results = self.calculator.calculate_metrics(ref_showers, gen_showers)
            
            aggregate_score += loss
            results_map[dataset_name] = results # Save for plotter

        final_score = aggregate_score / max(len(target_datasets), 1)
        return final_score, results_map

    def save_plots(self, results_map, save_dir, trial_index):
        """
        Iterates through the results map and calls the ShowerPlotter.
        """
        # Create a specific folder for this trial's best plots to avoid overwriting other trials
        trial_plot_dir = os.path.join(save_dir, f"trial_{trial_index}_plots")
        os.makedirs(trial_plot_dir, exist_ok=True)
        plotter = ShowerPlotter(save_dir=trial_plot_dir)
        
        for dataset_name, results in results_map.items():
            #make subfolder for each dataset
            plotter.save_dir = os.path.join(trial_plot_dir, dataset_name)
            plotter.plot_from_results(results)

    def save_model(self, engine, save_dir):
        """
        Saves model weights and optimizer states
        """
        engine._save_model(name="best", override_path=save_dir)
