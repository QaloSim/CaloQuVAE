import torch
from model.rbm.rbm import RBM
from torch.optim import Adam

class RBMtorch(RBM):
    def __init__(self, cfg=None):
        super(RBMtorch, self).__init__(cfg)

    @property
    def weight_dict(self):
        """Return masked weights as a dict.

        :return: dict with partition combinations as str keys ('01', '02', ...)
                and masked partition weight matrices as values.
        """
        masked_weight_dict = {}
        for key in self._weight_dict.keys():
            masked_weight_dict[key] = self._weight_dict[key] * self._weight_mask_dict[key]
        return masked_weight_dict


    def initOpt(self):
        params_to_optimize = list(self._weight_dict.values()) + list(self.bias_dict.values())

        self.opt = Adam(
            params_to_optimize, 
            lr=self._config.rbm.lr, 
        )
    
    def update_params(self):
        self.opt.zero_grad()

        with torch.no_grad():
            for key, param in self._weight_dict.items():
                if key in self.grad["weight"]:
                    param.grad = -1 * self.grad["weight"][key].detach()
            for key, param in self.bias_dict.items():
                if key in self.grad["bias"]:
                    param.grad = -1 * self.grad["bias"][key].detach()
        self.opt.step()
