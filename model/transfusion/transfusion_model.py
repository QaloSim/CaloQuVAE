import torch
import torch.nn as nn
from model.transfusion.transfusion_net import ARTransformer

def linear_trajectory(x_0, x_1, t):
    x_t = (1 - t) * x_0 + t * x_1
    x_t_dot = x_1 - x_0
    return x_t, x_t_dot

class TransfusionModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.t_min = 0
        self.t_max = 1
        self.distribution = torch.distributions.Uniform(low=self.t_min, high=self.t_max) # distribution for sampling time steps during training
        self.trajectory = linear_trajectory # trajectory function for computing the interpolated layer energy and its time derivative
        self.dim_embedding = cfg.model.dim_embedding
        self.net = self.build_net()


    def build_net(self):
        return ARTransformer(self.cfg)

    def batch_loss(self, x:torch.Tensor, incidence_energy:torch.Tensor):
        """
        Computes the loss for a batch of data.
        x: (batch_size, dim_in) - input layer energy sequence
        incidence_energy: (batch_size, 1) - incidence energy
        """
        incidence_energy = incidence_energy.unsqueeze(-1) # (batch_size, 1, 1)
        x = x.unsqueeze(-1) # (batch_size, dim_in, 1)
        
        # sample time steps
        t = self.distribution.sample(list(x.shape[:2]) + [1]*(x.ndim-2)).to(dtype=x.dtype, device=x.device)

        # sample noise
        x_0 = torch.randn(x.shape, device=x.device, dtype=x.dtype)
        x_t, x_t_dot = self.trajectory(x_0, x, t)
        v_pred = self.net(incidence_energy, x_t=x_t, t=t, x=x,rev=False)

        loss = nn.MSELoss()(v_pred, x_t_dot)
        return loss

