"""
Gumbel reparameterization trick Module

Author: Abhi (abhishek@myumanitoba.ca)
"""
import torch

class GumbelMod(torch.nn.Module):
    
    def __init__(self):
        super(GumbelMod, self).__init__()
        self.activation_fct = torch.nn.Sigmoid()
        
    def forward(self, logits, beta=100.0):
        """
        Gumbel reparameterization trick
        """
        rho = torch.rand(logits.size(), device=logits.device)
        logits_gumbel = logits + torch.log(rho) - torch.log(1 - rho)
        if self.training:
            out = self.activation_fct(logits_gumbel * beta)
        else:
            out = torch.heaviside(logits_gumbel, torch.tensor([0.], device=logits.device))
        return out



class GumbelNoNoise(torch.nn.Module):
    def __init__(self):
        super(GumbelNoNoise, self).__init__()
        self.activation_fct = torch.nn.Sigmoid()
        
    def forward(self, logits, beta=100.0):
        """
        Deterministic annealing. 
        As beta -> inf, this approaches a step function, 
        but strictly stays in the continuous domain during training.
        """
        if self.training:
            # Pure temperature-scaled sigmoid (No noise injection)
            out = self.activation_fct(logits * beta)
        else:
            # Hard threshold at inference
            out = torch.heaviside(logits, torch.tensor([0.], device=logits.device))
        return out

class GumbelTemperature(torch.nn.Module):
    def __init__(self, T=1.0):
        super(GumbelTemperature, self).__init__()
        self.activation_fct = torch.nn.Sigmoid()
        self.T = T
        
    def forward(self, logits, beta=100.0):
        """
        Gumbel reparameterization trick with temperature scaling
        """
        rho = torch.rand(logits.size(), device=logits.device)
        logits_gumbel = (logits / self.T) + torch.log(rho) - torch.log(1 - rho)
            
        if self.training:
            out = self.activation_fct(logits_gumbel * beta)
        else:
            out = torch.heaviside(logits_gumbel, torch.tensor([0.], device=logits.device))
        return out

class STEActivation(torch.nn.Module):
    def __init__(self):
        super(STEActivation, self).__init__()
        
    def forward(self, logits, beta=100.0):
        """
        Straight-Through Estimator with Annealed Gradient
        Forward: Hard Threshold (0 or 1)
        Backward: Gradient of Sigmoid(logits * beta)
        """
        # 1. The "Soft" proxy (used only for gradient calculation)
        # We use sigmoid here to mimic the gradients you were getting from Gumbel
        soft_proxy = torch.sigmoid(logits * beta)
        
        # 2. The "Hard" actual value (used for the forward pass)
        # We use soft_proxy > 0.5 to keep it consistent with the sigmoid logic
        hard_binary = (soft_proxy > 0.5).float()
        
        # 3. The STE Trick
        # forward: returns hard_binary (since soft_proxy cancels out)
        # backward: returns grad(soft_proxy) (since hard_binary is detached)
        if self.training:
            out = hard_binary - soft_proxy.detach() + soft_proxy
        else:
            out = hard_binary
            
        return out