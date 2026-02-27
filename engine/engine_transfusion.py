"""
Engine Class for Training and Evaluating TransfusionModel
"""
import torch
import numpy as np
import wandb
from utils.evaluate_transfusion import evaluate_transfusion_distributions
from data.layers import reduce_inverse

from CaloQuVAE import logging
logger = logging.getLogger(__name__)


class EngineTransfusion():

    def __init__(self, cfg):
        self._config = cfg
        self._model = None
        self._optimizer = None
        self._data_mgr = None
        self._device = None
        self._model_creator = None
        self._scheduler = None

        self.feature_mean, self.feature_std = self.load_feature_stats()

    @property
    def device(self):
        return self._device
    
    @device.setter
    def device(self, device):
        self._device = device
    
    @property
    def model(self):
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
    
    @property
    def data_mgr(self):
        return self._data_mgr
    
    @data_mgr.setter   
    def data_mgr(self,data_mgr):
        assert data_mgr is not None, "Empty Data Manager"
        self._data_mgr=data_mgr

    
    @property
    def scheduler(self):
        return getattr(self, '_scheduler', None)

    @scheduler.setter
    def scheduler(self, scheduler):
        self._scheduler = scheduler

    @property
    def model_creator(self):
        return self._model_creator
    
    @model_creator.setter
    def model_creator(self, model_creator):
        assert model_creator is not None
        self._model_creator = model_creator

    def _save_model(self, name="blank", override_path=None):
        config_string = "_".join(str(i) for i in [self._config.model.model_name,f'{name}'])
        config_path = self._model_creator.save_state_tfusion(config_string, opt=self.optimizer, sched=self.scheduler, override_path=override_path)
        return config_path

    def load_feature_stats(self):
        stats_path = getattr(self._config, 'feature_stats_path', None)
        if stats_path:
            stats = torch.load(stats_path)
            feature_mean = stats['mean']
            feature_std = stats['std']
            logger.info(f"Loaded feature statistics from {stats_path}")
            return feature_mean.to(self._device), feature_std.to(self._device)
        else:
            raise ValueError("No feature stats path found in config. Cannot proceed without loading or computing feature statistics.")
    


    

    def fit_tfusion(self, epoch):
        """
        Train TransfusionModel for one epoch.
        """
        log_batch_idx = max(len(self.data_mgr.train_loader)//self._config.engine.n_batches_log_train, 1)
        self.model.train()
        for i, (x, x0) in enumerate(self.data_mgr.train_loader):
            x, x0 = x.to(self._device), x0.to(self._device)
            loss = self.model.batch_loss(x, x0)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()


            if i % log_batch_idx == 0:
                current_lr = self.optimizer.param_groups[0]['lr']
                logger.info('Epoch: {} [{}/{} ({:.0f}%)]\t Batch Loss: {:.4f}\t LR: {:.6f}'.format(
                    epoch, i, len(self.data_mgr.train_loader), 
                    100. * i / len(self.data_mgr.train_loader), loss.item(), current_lr))
                wandb.log({"train_loss": loss.item(), "learning_rate": current_lr})

    def evaluate_tfusion(self, data_loader, epoch, close_plots=True):
        """
        Evaluate TransfusionModel on a validation set.
        Every self._config.engine.n_epochs_sample epochs, also generates samples and logs them to WandB.
        """
        log_batch_idx = max(len(data_loader)//self._config.engine.n_batches_log_val, 1)
        self.model.eval()
        avg_loss = 0
        with torch.no_grad():
            for i, (x, x0) in enumerate(data_loader):
                x, x0 = x.to(self._device), x0.to(self._device)
                loss = self.model.batch_loss(x, x0)
                
                if i % log_batch_idx == 0:
                    logger.info('Epoch: {} [{}/{} ({:.0f}%)]\t Batch Loss: {:.4f}'.format(epoch,
                        i, len(data_loader), 100. * i / len(data_loader), loss.item()))

                wandb.log({"val_loss": loss.item()})
                avg_loss += loss.item()
        loss = avg_loss / len(data_loader)
        logger.info(f"Validation Loss: {loss:.4f}")

        if epoch % self._config.engine.n_epochs_sample == 0:
            all_samples, all_x, all_x0 = self.sample_tfusion(epoch, data_loader)
            statistics, plots = evaluate_transfusion_distributions(all_x, all_samples, all_x0, close_plots=close_plots)
            wandb.log(plots)
            wandb.log(statistics)

        return loss

    def sample_tfusion(self, epoch, data_loader):
        """
        Generate samples from TransfusionModel and return them alongside GT and conditions.
        """
        self.model.eval()
        all_samples, all_x, all_x0 = [], [], []
        with torch.no_grad():
            for i, (x, x0) in enumerate(data_loader):
                x, x0 = x.to(self._device), x0.to(self._device)
                x0_cond = x0.unsqueeze(-1) # (batch_size, 1, 1)
                samples = self.model.net(incidence_energy=x0_cond, rev=True)
                self.feature_mean = self.feature_mean.to(samples.device)
                self.feature_std = self.feature_std.to(samples.device)
                samples, x0_inv = reduce_inverse(samples, x0, self.feature_mean, self.feature_std)
                x, x0_inv = reduce_inverse(x, x0, self.feature_mean, self.feature_std)

                
                all_samples.append(samples.cpu())
                all_x.append(x.cpu())
                all_x0.append(x0_inv.cpu())

                
        return torch.cat(all_samples), torch.cat(all_x), torch.cat(all_x0)     

