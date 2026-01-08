import torch

def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)
    
    total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
    L2_distance = ((total0-total1)**2).sum(2) 
    
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        # --- FIX: Prevent bandwidth from being 0 ---
        bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        # If all samples are identical, bandwidth is 0. Clamp it to 1.0 (or any small float)
        if bandwidth.item() == 0:
            bandwidth = torch.tensor(1.0).to(L2_distance.device)
            
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    
    # Add epsilon to denominator just in case
    kernel_val = [torch.exp(-L2_distance / (bandwidth_temp + 1e-8)) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)


    
def compute_mmd(source, target):
    """
    Computes Maximum Mean Discrepancy (MMD) distance.
    """
    batch_size = int(source.size()[0])
    kernels = gaussian_kernel(source, target)
    
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    
    loss = torch.mean(XX + YY - XY - YX)
    return loss