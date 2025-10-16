import torch
from typing import Dict
from model.rbm.zephyr import ZephyrRBM
from CaloQuVAE import logging
logger = logging.getLogger(__name__)

class RBM_TwoPartite:
    def __init__(self, config, data):
        self.config = config
        self.p_size = self.config.rbm.latent_nodes_per_p
        if self.config.device == "gpu" and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.config.gpu_list[0]}")
        else:
            self.device = torch.device("cpu")

        self.init_parameters(data, self.p_size, self.p_size, self.device)
        self.init_chains(self.config.rbm.num_chains, self.p_size, self.p_size, self.device)

    def init_parameters(self,
        data : torch.Tensor,
        num_visibles : int,
        num_hiddens : int,
        device : torch.device
    ) -> None:
        
        eps = 1e-4
        init_std = self.config.rbm.w_std[0]
        logger.info(f"Initializing RBM parameters with std: {init_std}")
        num_visibles = data.shape[1]
        frequencies = data.to(device).mean(0)
        frequencies = torch.clamp(frequencies, min=eps, max=(1. - eps))

        self.params = {}
        self.params["vbias"] = torch.log(frequencies) - torch.log(1. - frequencies)
        self.params["hbias"] = torch.zeros(num_hiddens, device=device, dtype=torch.float32)
        self.params["weight_matrix"] = torch.randn(size=(num_visibles, num_hiddens), device=device) * init_std
    
    def init_chains(
        self,
        num_chains : int,
        num_visibles : int,
        num_hiddens : int,
        device : torch.device
    ) -> None:
        
        self.chains = {}
        self.chains["v"] = torch.randint(0, 2, size=(num_chains, num_visibles), device=device, dtype=torch.float32)
        self.chains["h"] = torch.randint(0, 2, size=(num_chains, num_hiddens), device=device, dtype=torch.float32)
        self.chains["mv"] = torch.zeros(size=(num_chains, num_visibles), device=device, dtype=torch.float32)
        self.chains["mh"] = torch.zeros(size=(num_chains, num_hiddens), device=device, dtype=torch.float32)

    
    def compute_gradient(
        self,
        data: Dict[str, torch.Tensor],
        centered: bool = True,
    ) -> Dict[str, torch.Tensor]:
        v = self.chains["v"]
        h = self.chains["h"]
        num_chains = len(v)

        data_new = data["v"].clone()
        
        # Reshape weights for broadcasting: (B,) -> (B, 1)
        weights = data["weights"].view(-1, 1)
        
        # Averages over data and generated samples
        v_data_mean = (data_new * weights).sum(0) / data["weights"].sum()
        torch.clamp_(v_data_mean, min=1e-4, max=(1. - 1e-4))
        h_data_mean = (data["mh"] * weights).sum(0) / data["weights"].sum()
        v_gen_mean = v.mean(0)
        torch.clamp_(v_gen_mean, min=1e-4, max=(1. - 1e-4))
        h_gen_mean = h.mean(0)
        
        grad = {}
        
        if centered:
            # Centered variables
            v_data_centered = data_new - v_data_mean
            h_data_centered = data["mh"] - h_data_mean
            v_gen_centered = v - v_data_mean
            h_gen_centered = h - h_data_mean

            # Gradient
            grad["weight_matrix"] = (
                (v_data_centered * weights).T @ h_data_centered
            ) / data["weights"].sum() - (
                v_gen_centered.T @ h_gen_centered
            ) / num_chains
            
            grad["vbias"] = (
                v_data_mean - v_gen_mean - (grad["weight_matrix"] @ h_data_mean)
            )
            grad["hbias"] = (
                h_data_mean - h_gen_mean - (v_data_mean @ grad["weight_matrix"])
            )
            
        else:
            # Gradient
            grad["weight_matrix"] = (
                (data["v"] * weights).T @ data["mh"]
            ) / data["weights"].sum() - (v.T @ h) / num_chains
            
            grad["vbias"] = v_data_mean - v_gen_mean
            grad["hbias"] = h_data_mean - h_gen_mean
        
        return grad
    def sample_hidden(self,beta:float = 1.0) -> None:
        """Samples the hidden units given the visible units.

        Args:
            v (torch.Tensor): Visible units.
            params (Dict[str, torch.Tensor]): Parameters of the model.
            beta (float, optional): Inverse temperature. Defaults to 1.0.

        Updates:
            self.chains["h"]: Samples of the hidden units.
        """
        self.chains["mh"] = torch.sigmoid(beta * (self.params["hbias"] + self.chains["v"] @ self.params["weight_matrix"]))
        self.chains["h"] = torch.bernoulli(self.chains["mh"])

    def sample_visibles(self, beta: float = 1.0) -> None:
        """Update v/mv given h."""
        self.chains["mv"] = torch.sigmoid(beta * (self.params["vbias"] + self.chains["h"] @ self.params["weight_matrix"].T))
        self.chains["v"] = torch.bernoulli(self.chains["mv"])
    
    def sample_state(self, beta: float = 1.0) -> None:
        """Update h/mh and v/mv."""
        for _ in range(self.config.rbm.bgs_steps):
            self.sample_hidden(beta)
            self.sample_visibles(beta)

    def reset_chains(self) -> None:
        """Reinitialize chains to random binary values."""
        num_chains = self.chains["v"].shape[0]
        num_visibles = self.chains["v"].shape[1]
        num_hiddens = self.chains["h"].shape[1]
	    
        self.chains["v"] = torch.randint(
            0, 2, (num_chains, num_visibles), device=self.device, dtype=torch.float32
        )
        self.chains["h"] = torch.randint(
            0, 2, (num_chains, num_hiddens), device=self.device, dtype=torch.float32
        )
        self.chains["mv"].zero_()
        self.chains["mh"].zero_()

    def update_parameters(
        self,
        data : dict[str, torch.Tensor],
        centered : bool=True,
        ) -> None:
        """Computes the gradient of the log-likelihood and updates the parameters of the model.

        Args:
            data (Dict[str, torch.Tensor]): Observed data.
            chains (Dict[str, torch.Tensor]): Monte Carlo chains.
            params (Dict[str, torch.Tensor]): Parameters of the model.
            lr (float): Learning rate.
            centered (bool, optional): Whether to center the gradient or not. Defaults to True.

        Updates:
            self.params: Parameters of the model.
        """
        # Compute the gradient of the log-likelihood
        grad = self.compute_gradient(data=data, centered=centered)

        lr = self.config.rbm.lr
        gamma = self.config.rbm.gamma
        # Update the parameters
        self.params["vbias"] += lr * grad["vbias"]
        self.params["hbias"] += lr * grad["hbias"]
        self.params["weight_matrix"] += lr * grad["weight_matrix"] - gamma * self.params["weight_matrix"]
    
    def fit_batch(self,
        data : Dict[str, torch.Tensor],
        centered : bool=True) -> None:
        """Fits the model to a batch of data.
        Args:
            data (Dict[str, torch.Tensor]): Observed data.
            chains (Dict[str, torch.Tensor]): Monte Carlo chains.
            params (Dict[str, torch.Tensor]): Parameters of the model.
            lr (float): Learning rate.
            centered (bool, optional): Whether to center the gradient or not. Defaults to True.
        Updates:
            self.params: Parameters of the model.
            self.chains: Monte Carlo chains.
        """

        self.sample_state()
        self.update_parameters(data, centered)



class ZephyrRBM_TwoPartite(RBM_TwoPartite):
    """
    An RBM that extends the base RBM_TwoPartite to incorporate a weight mask
    derived from the D-Wave Zephyr hardware topology.

    This class uses the ZephyrRBM helper to generate the connectivity for a
    4-partite graph and then extracts the connections between two of those
    partitions to form a bipartite (visible-hidden) graph.
    """
    def __init__(self, config, data):
        # 1. Instantiate the ZephyrRBM helper to generate topology masks
        # This object's sole purpose is to give us the weight masks.
        zephyr_helper = ZephyrRBM(config)

        # 2. Select two partitions to act as visible and hidden layers.
        # Here, we arbitrarily choose partitions '0' and '1'.
        # The key for the mask dictionary is the sorted pair, e.g., '01'.
        visible_partition_key = '0'
        hidden_partition_key = '1'
        mask_key = ''.join(sorted((visible_partition_key, hidden_partition_key)))
        
        # 3. Store the extracted weight mask for our bipartite RBM
        self.weight_mask = zephyr_helper._weight_mask_dict[mask_key]
        print(f"Extracted weight mask for partitions {visible_partition_key} and {hidden_partition_key}.")

        # 4. Call the parent class's __init__ to perform the standard setup.
        super().__init__(config, data)

    def init_parameters(self,
        data : torch.Tensor,
        num_visibles : int,
        num_hiddens : int,
        device : torch.device
    ) -> None:
        """
        Initializes parameters and applies the Zephyr weight mask.
        """
        # First, run the standard parameter initialization from the parent class.
        super().init_parameters(data, num_visibles, num_hiddens, device)
        
        # Store the mask in the parameters dictionary and move it to the correct device.
        self.params["weight_mask"] = self.weight_mask.to(device)

        # **Crucially, apply the mask to the initial random weights.**
        # This ensures the model starts with the correct sparse topology.
        # We use .data to modify the tensor in-place without affecting the computation graph.
        self.params["weight_matrix"].data *= self.params["weight_mask"]
        logger.info("Initial weight matrix has been masked with Zephyr topology.")

    def update_parameters(
        self,
        data : dict[str, torch.Tensor],
        centered : bool=True,
    ) -> None:
        """
        Computes the gradient, masks it, and then updates the parameters.
        """
        # Compute the gradient of the log-likelihood using the parent's method.
        grad = self.compute_gradient(data=data, centered=centered)
        
        # **Apply the mask to the gradient of the weight matrix.**
        # This is the most important step. It ensures that only weights
        # corresponding to physical connections receive updates.
        grad["weight_matrix"] *= self.params["weight_mask"]

        # Proceed with the standard parameter update rule.
        lr = self.config.rbm.lr
        gamma = self.config.rbm.gamma
        
        self.params["vbias"] += lr * grad["vbias"]
        self.params["hbias"] += lr * grad["hbias"]
        self.params["weight_matrix"] += lr * grad["weight_matrix"] - gamma * self.params["weight_matrix"]




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
        init_std = self.config.rbm.w_std[0] if hasattr(self.config.rbm, 'w_std') else 0.01
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