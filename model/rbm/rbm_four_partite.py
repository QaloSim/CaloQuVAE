import torch
from typing import Dict, Tuple, Optional
from model.rbm.zephyr import ZephyrRBM
from CaloQuVAE import logging

logger = logging.getLogger(__name__)

class RBM_FourPartite:
    def __init__(self, config, data):
        """Initialize a 4-partite RBM.
        
        Args:
            config: Configuration object
            data: Initial data for parameter initialization [batch, 4*p_size]
        """
        self.config = config
        self.p_size = self.config.rbm.latent_nodes_per_p
        
        if self.config.device == "gpu" and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.config.gpu_list[0]}")
        else:
            self.device = torch.device("cpu")
        
        self.init_parameters(data, self.p_size, self.device)
        self.init_chains(self.config.rbm.num_chains, self.p_size, self.device)
    
    def init_parameters(
        self,
        data: torch.Tensor,
        p_size: int,
        device: torch.device
    ) -> None:
        """Initialize biases and weight matrices for 4 partitions."""
        eps = 1e-4
        init_std = self.config.rbm.w_std[0]
        logger.info(f"Initializing 4-partite RBM with std: {init_std}")
        
        data = data.to(device)
        self.params = {}
        
        # Initialize biases for each partition
        for i in range(4):
            partition_data = data[:, i*p_size:(i+1)*p_size]
            frequencies = partition_data.mean(0)
            frequencies = torch.clamp(frequencies, min=eps, max=(1. - eps))
            self.params[f"bias_{i}"] = (
                torch.log(frequencies) - torch.log(1. - frequencies)
            )
        
        # Initialize weight matrices for pairs (i,j) where j > i
        for i in range(4):
            for j in range(i+1, 4):
                self.params[f"weight_{i}{j}"] = torch.randn(
                    p_size, p_size, device=device, dtype=torch.float32
                ) * init_std
    
    def init_chains(
        self,
        num_chains: int,
        p_size: int,
        device: torch.device
    ) -> None:
        """Initialize Markov chains for sampling."""
        self.chains = {}
        for i in range(4):
            self.chains[f"p{i}"] = torch.randint(
                0, 2, (num_chains, p_size),
                device=device, dtype=torch.float32
            )
            self.chains[f"mp{i}"] = torch.zeros(
                num_chains, p_size, device=device, dtype=torch.float32
            )
    
    def sample_partition(
        self,
        partition_idx: int,
        beta: float = 1.0
    ) -> None:
        """Sample one partition given all others."""
        bias = self.params[f"bias_{partition_idx}"]
        
        # Start with zeros (will be [batch_size, p_size])
        activation = torch.zeros_like(self.chains[f"p{partition_idx}"])
        
        # Add contributions from connected partitions
        for other_idx in range(4):
            if other_idx == partition_idx:
                continue
            
            i, j = min(partition_idx, other_idx), max(partition_idx, other_idx)
            weight = self.params[f"weight_{i}{j}"]
            other_state = self.chains[f"p{other_idx}"]
            
            activation += (
                other_state @ weight if partition_idx < other_idx
                else other_state @ weight.T
            )
        
        # Add bias (broadcasts automatically)
        activation += bias
        
        self.chains[f"mp{partition_idx}"] = torch.sigmoid(beta * activation)
        self.chains[f"p{partition_idx}"] = torch.bernoulli(
            self.chains[f"mp{partition_idx}"]
        )    
    def sample_state(self, beta: float = 1.0) -> None:
        """Perform block Gibbs sampling."""
        for _ in range(self.config.rbm.bgs_steps):
            for i in range(4):
                self.sample_partition(i, beta)
    
    def sample_conditional(
        self,
        initial_state: torch.Tensor,
        clamped_mask: Tuple[bool, bool, bool, bool],
        beta: float = 1.0
    ) -> Tuple[torch.Tensor, ...]:
        """Sample with some partitions clamped.
        
        Args:
            initial_state: Initial states [batch, 4*p_size]
            clamped_mask: Which partitions to clamp
            beta: Inverse temperature
            
        Returns:
            Tuple of 4 partition states
        """
        batch_size = initial_state.shape[0]
        p_size = self.p_size
        
        # Initialize states
        states = []
        for i in range(4):
            if clamped_mask[i]:
                states.append(initial_state[:, i*p_size:(i+1)*p_size].clone())
            else:
                states.append(
                    torch.rand(batch_size, p_size, device=self.device).bernoulli()
                )
        
        # Cache weights
        weights = {}
        for i in range(4):
            for j in range(i+1, 4):
                weights[f"{i}{j}"] = self.params[f"weight_{i}{j}"]
                weights[f"{j}{i}"] = weights[f"{i}{j}"].T
        
        biases = [self.params[f"bias_{i}"] for i in range(4)]
        
        # Block Gibbs sampling
        for _ in range(self.config.rbm.bgs_steps):
            for idx in range(4):
                if clamped_mask[idx]:
                    continue
                
                # Start with zeros (will have correct batch shape)
                activation = torch.zeros(batch_size, p_size, device=self.device)
                
                # Add contributions from other partitions
                for other_idx in range(4):
                    if other_idx != idx:
                        activation += states[other_idx] @ weights[f"{other_idx}{idx}"]
                
                # Add bias (broadcasts automatically)
                activation += biases[idx]
                
                states[idx] = torch.bernoulli(torch.sigmoid(beta * activation))
        
        return tuple(s.detach() for s in states)    

        
    def compute_gradient(
        self,
        data: Dict[str, torch.Tensor],
        centered: bool = True
    ) -> Dict[str, torch.Tensor]:
        """Compute gradient of log-likelihood.
        
        Args:
            data: Dict with "p0"-"p3" and optional "weights"
            centered: Whether to use centered gradients
        """
        chain_states = [self.chains[f"p{i}"] for i in range(4)]
        num_chains = chain_states[0].shape[0]
        
        # Handle sample weights
        if "weights" in data and data["weights"] is not None:
            weights = data["weights"].view(-1, 1)
            total_weight = data["weights"].sum()
        else:
            batch_size = data["p0"].shape[0]
            weights = torch.ones(batch_size, 1, device=self.device)
            total_weight = batch_size
        
        # Compute means
        data_means = [
            torch.clamp(
                (data[f"p{i}"] * weights).sum(0) / total_weight,
                min=1e-4, max=(1. - 1e-4)
            )
            for i in range(4)
        ]
        
        chain_means = [
            torch.clamp(chain_states[i].mean(0), min=1e-4, max=(1. - 1e-4))
            for i in range(4)
        ]
        
        grad = {}
        
        if centered:
            # Weight gradients - center using data means
            for i in range(4):
                for j in range(i+1, 4):
                    data_i_c = data[f"p{i}"] - data_means[i]
                    data_j_c = data[f"p{j}"] - data_means[j]
                    data_corr = (data_i_c * weights).T @ data_j_c / total_weight
                    
                    # Center chain states using data means (not chain means!)
                    chain_i_c = chain_states[i] - data_means[i]
                    chain_j_c = chain_states[j] - data_means[j]
                    chain_corr = chain_i_c.T @ chain_j_c / num_chains
                    
                    grad[f"weight_{i}{j}"] = data_corr - chain_corr
            
            # Bias gradients with correction term
            for i in range(4):
                # Base gradient
                grad[f"bias_{i}"] = data_means[i] - chain_means[i]
                
                # Correction term: subtract weight_grad @ data_mean_j for each neighbor
                for j in range(4):
                    if j == i:
                        continue
                    
                    if j > i:
                        weight_grad = grad[f"weight_{i}{j}"]
                        # Use ONLY data_means[j], not averaged with chain_means
                        grad[f"bias_{i}"] -= weight_grad @ data_means[j]
                    else:
                        weight_grad = grad[f"weight_{j}{i}"]
                        # Use ONLY data_means[j], not averaged with chain_means
                        grad[f"bias_{i}"] -= weight_grad.T @ data_means[j]
        else:
            # Non-centered gradients
            for i in range(4):
                for j in range(i+1, 4):
                    data_corr = (
                        (data[f"p{i}"] * weights).T @ data[f"p{j}"] / total_weight
                    )
                    chain_corr = chain_states[i].T @ chain_states[j] / num_chains
                    grad[f"weight_{i}{j}"] = data_corr - chain_corr
            
            for i in range(4):
                grad[f"bias_{i}"] = data_means[i] - chain_means[i]
        
        return grad    
    def update_parameters(
        self,
        data: Dict[str, torch.Tensor],
        centered: bool = True
    ) -> None:
        """Update RBM parameters."""
        grad = self.compute_gradient(data, centered)
        
        lr = self.config.rbm.lr
        gamma = getattr(self.config.rbm, 'gamma', 0.0)
        
        # Update parameters
        for i in range(4):
            self.params[f"bias_{i}"] += lr * grad[f"bias_{i}"]
        
        for i in range(4):
            for j in range(i+1, 4):
                self.params[f"weight_{i}{j}"] += (
                    lr * grad[f"weight_{i}{j}"]
                    - gamma * self.params[f"weight_{i}{j}"]
                )
    
    def fit_batch(
        self,
        data: Dict[str, torch.Tensor],
        centered: bool = True
    ) -> None:
        """Fit model to a batch."""
        self.sample_state()
        self.update_parameters(data, centered)
    
    def reset_chains(self) -> None:
        """Reinitialize chains."""
        num_chains = self.chains["p0"].shape[0]
        
        for i in range(4):
            self.chains[f"p{i}"] = torch.randint(
                0, 2, (num_chains, self.p_size),
                device=self.device, dtype=torch.float32
            )
            self.chains[f"mp{i}"].zero_()
    
    def energy(
        self,
        p0: torch.Tensor,
        p1: torch.Tensor,
        p2: torch.Tensor,
        p3: torch.Tensor
    ) -> torch.Tensor:
        """Compute energy [batch]."""
        states = [p0, p1, p2, p3]
        energy = torch.zeros(p0.shape[0], device=self.device)
        
        # Bias terms
        for i in range(4):
            energy -= states[i] @ self.params[f"bias_{i}"]
        
        # Pairwise terms
        for i in range(4):
            for j in range(i+1, 4):
                weight = self.params[f"weight_{i}{j}"]
                energy -= torch.einsum("bi,ij,bj->b", states[i], weight, states[j])
        
        return energy