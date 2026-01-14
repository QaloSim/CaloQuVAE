import torch
import torch.nn as nn


def gaussian_kernel_matrix(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    """
    Computes a multi-scale Gaussian Kernel Matrix.
    
    Returns:
        kernel_matrix: (N, N) tensor where N = batch_size * 2
    """
    # 1. Concatenate inputs
    total = torch.cat([source, target], dim=0)
    n_samples = total.size(0)
    
    # 2. Compute Pairwise Squared L2 Distances (Optimized)
    # Using cdist is faster and uses less memory than the unsqueeze/expand method.
    L2_distance = torch.cdist(total, total, p=2) ** 2
    
    # 3. Compute Bandwidth (Sigma)
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        # Calculate mean distance of off-diagonal elements
        bandwidth = torch.sum(L2_distance.detach()) / (n_samples**2 - n_samples)
        
        # FIX: Clamp to avoid division by zero if all samples are identical
        if bandwidth.item() == 0:
            bandwidth = torch.tensor(1.0, device=L2_distance.device)

    # 4. Multi-scale Bandwidth List
    # Centers the bandwidths around the calculated or fixed value
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    
    # 5. Compute Kernel Matrix
    # Sum of exp(-dist / bandwidth) for each bandwidth
    kernel_val = [torch.exp(-L2_distance / (b + 1e-8)) for b in bandwidth_list]
    
    # Sum the kernels to get the final matrix
    return sum(kernel_val)



class ConditionNormalizer(nn.Module):
    """
    Normalizes incidence energy
    """
    def __init__(self, method='log_minmax', min_val=0.0, max_val=300000.0):
        super().__init__()
        self.method = method
        
        # Register constants as buffers so they move to device with the model
        # but don't get updated by the optimizer.
        self.register_buffer('min_val', torch.tensor(float(min_val)))
        self.register_buffer('max_val', torch.tensor(float(max_val)))
        
        # Avoid log(0)
        self.epsilon = 1e-5

    def forward(self, conditions):
        """
        Args:
            conditions: Tensor of shape (Batch, 1) or (Batch,)
        Returns:
            normalized_conditions: Tensor in range [0, 1] (roughly)
        """
        # Ensure correct shape (Batch, 1) for kernel broadcasting later
        if conditions.dim() == 1:
            conditions = conditions.unsqueeze(1)
            
        if self.method == 'linear':
            # Simple Min-Max: (x - min) / (max - min)
            # WARNING: Bad for power-law distributions (like particle energies)
            denom = self.max_val - self.min_val
            return (conditions - self.min_val) / (denom + 1e-8)
            
        elif self.method == 'log_minmax':
            # Log-space scaling: (log(x) - log(min)) / (log(max) - log(min))
            # Preserves resolution at low energies (1 vs 10 GeV)
            
            # Clamp inputs to min_val to avoid issues below min
            c_clamped = torch.clamp(conditions, min=self.min_val)
            
            log_c = torch.log(c_clamped + self.epsilon)
            log_min = torch.log(self.min_val + self.epsilon)
            log_max = torch.log(self.max_val + self.epsilon)
            
            return (log_c - log_min) / (log_max - log_min)
            
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")


    
def compute_cmmd(source_x, source_c, target_x, target_c):
    """
    Conditional MMD. 
    source_c and target_c should be normalized to [0,1].
    """
    batch_size = int(source_x.size()[0])
    
    # 1. Compute Kernel for Features (X)
    # FIX: Pass source and target directly. Do not concat manually.
    k_x = gaussian_kernel_matrix(source_x, target_x)
    
    # 2. Compute Kernel for Conditions (C)
    # FIX: Pass source and target directly.
    k_c = gaussian_kernel_matrix(source_c, target_c, fix_sigma=0.1) 
    
    # 3. Joint Kernel
    joint_kernel = k_x * k_c
    
    # Now joint_kernel is (2B, 2B) and these slices will work
    XX = joint_kernel[:batch_size, :batch_size]
    YY = joint_kernel[batch_size:, batch_size:]
    XY = joint_kernel[:batch_size, batch_size:]
    YX = joint_kernel[batch_size:, :batch_size]
    
    loss = torch.mean(XX + YY - XY - YX)
    return loss