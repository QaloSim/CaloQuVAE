import torch
from typing import Dict
from model.rbm.zephyr import ZephyrRBM
from CaloQuVAE import logging
import h5py
import numpy as np
import os
from omegaconf import OmegaConf
logger = logging.getLogger(__name__)

class RBM_TwoPartite:
    def __init__(self, config, data):
        self.config = config
        self.num_visible = self.config.rbm.num_visible_nodes
        self.num_hidden = self.config.rbm.num_hidden_nodes
        if self.config.device == "gpu" and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.config.gpu_list[0]}")
        else:
            self.device = torch.device("cpu")

        self.init_parameters(data, self.num_visible, self.num_hidden, self.device)
        self.init_chains(self.config.rbm.num_chains, self.num_visible, self.num_hidden, self.device)

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

    def compute_energy(self, v_nodes: torch.Tensor) -> torch.Tensor:
        """
        Computes the Free Energy of the model given a batch of visible states.

        Args:
            v_nodes (torch.Tensor): A batch of visible states,
                                    shape [batch_size, num_visibles].
            
        Returns:
            torch.Tensor: The free energy for each visible state in the batch,
                          shape [batch_size].
        """
        # Ensure the input tensor is on the same device as the model parameters
        v_nodes = v_nodes.to(self.device)
        
        # Access parameters from self
        vbias = self.params["vbias"]
        hbias = self.params["hbias"]
        weight_matrix = self.params["weight_matrix"]

        field = v_nodes @ vbias
        exponent = hbias + (v_nodes @ weight_matrix)
        
        log_term = torch.where(
            exponent < 10, 
            torch.log(1. + torch.exp(exponent)), 
            exponent
        )
        
        # Sum over the hidden dimension (axis 1)
        return -field - log_term.sum(1)


    def _sample_h_given_v(self, v: torch.Tensor, beta: float = 1.0) -> (torch.Tensor, torch.Tensor):
        """
        Helper: Samples hiddens given an arbitrary visible tensor 'v'.
        Does NOT modify self.chains.
        """
        mh = torch.sigmoid(beta * (self.params["hbias"] + v @ self.params["weight_matrix"]))
        h = torch.bernoulli(mh)
        return h, mh

    def _sample_v_given_h(self, h: torch.Tensor, beta: float = 1.0) -> (torch.Tensor, torch.Tensor):
        """
        Helper: Samples visibles given an arbitrary hidden tensor 'h'.
        Does NOT modify self.chains.
        """
        mv = torch.sigmoid(beta * (self.params["vbias"] + h @ self.params["weight_matrix"].T))
        v = torch.bernoulli(mv)
        return v, mv

    def sample_v_given_v_clamped(
        self,
        clamped_v: torch.Tensor,
        n_clamped: int,
        gibbs_steps: int,
        beta: float = 1.0
    ) -> torch.Tensor:
        """
        Performs conditional sampling (clamping) of the visible units.
        Given the state of the first 'n_clamped' visible units, this function
        samples the state of the remaining visible units.

        This method creates its own sampling chains and does NOT
        modify the persistent 'self.chains' used for training.

        Returns:
            torch.Tensor:
            - v_samples (torch.Tensor): The final sampled visible states,
                                        shape [num_samples, num_visibles].
        """
        clamped_v = clamped_v.to(self.device)
        
        num_samples = clamped_v.shape[0]
        num_visibles = self.params["vbias"].shape[0]
        num_unclamped = num_visibles - n_clamped

        if n_clamped >= num_visibles or n_clamped <= 0:
            raise ValueError(f"n_clamped ({n_clamped}) must be > 0 and < num_visibles ({num_visibles})")
        if clamped_v.shape[1] != n_clamped:
             raise ValueError(f"clamped_v shape[1] ({clamped_v.shape[1]}) does not match n_clamped ({n_clamped})")

        # 1. Initialize the full visible state for sampling
        v_sample = torch.zeros(num_samples, num_visibles, device=self.device, dtype=torch.float32)
        
        # Set the clamped part
        v_sample[:, :n_clamped] = clamped_v
        
        # Initialize the unclamped part (randomly)
        v_sample[:, n_clamped:] = torch.randint(
            0, 2, (num_samples, num_unclamped), device=self.device, dtype=torch.float32
        )

        # Run the Gibbs sampling loop
        h_sample = None
        
        for _ in range(gibbs_steps):
            # Sample hidden given full visible
            h_sample, _ = self._sample_h_given_v(v_sample, beta)
            
            # Sample visibles given hidden
            v_sample_new, _ = self._sample_v_given_h(h_sample, beta)
            
            # Enforce the clamp
            v_sample_new[:, :n_clamped] = clamped_v
            
            # Update v_sample for the next iteration
            v_sample = v_sample_new
            

        # v_sample is the final state after the last loop
        return v_sample

    
    
    def save_checkpoint(self, filepath: str, epoch: int, config: OmegaConf):
        """
        Saves the RBM's state to an HDF5 file.

        This method creates the file if it doesn't exist and manages
        hyperparameters, model parameters, persistent chains, and RNG states.
        
        Args:
            filepath (str): Path to the HDF5 checkpoint file.
            epoch (int): The current epoch number (to be saved).
            config (OmegaConf): The Hydra config object to save hyperparameters.
        """
        # 'a' mode: read/write if file exists, create otherwise
        with h5py.File(filepath, 'a') as f:
            
            # --- 1. Save Hyperparameters (only on first save) ---
            if 'hyperparameters' not in f:
                logger.info(f"Creating new checkpoint file: {filepath}")
                h_group = f.create_group('hyperparameters')
                
                # Save the full config as a YAML string
                h_group.attrs['config_yaml'] = OmegaConf.to_yaml(config)
                
                # Save key hyperparameters for quick access (like in the example)
                h_group['num_visibles'] = self.num_visible 
                h_group['num_hiddens'] = self.num_hidden 
                h_group['num_chains'] = config.rbm.num_chains
                h_group['learning_rate'] = config.rbm.lr
                # ... add any other key hyperparameters ...
            
            # --- 2. Create Group for this specific checkpoint ---
            group_name = f"epoch_{epoch}"
            if group_name in f:
                del f[group_name]  # Overwrite old checkpoint for this epoch
            cp_group = f.create_group(group_name)

            # --- 3. Save Model Parameters into the group ---
            for key, tensor in self.params.items():
                cp_group.create_dataset(key, data=tensor.cpu().numpy())

            # --- 4. Save RNG States into the group (for reproducibility) ---
            cp_group.create_dataset('torch_rng_state', data=torch.get_rng_state())
            
            # Save numpy RNG state (using the same format as the source)
            np_rng_state = np.random.get_state()
            cp_group.create_dataset('numpy_rng_arg0', data=np.string_(np_rng_state[0]))
            cp_group.create_dataset('numpy_rng_arg1', data=np_rng_state[1])
            cp_group.create_dataset('numpy_rng_arg2', data=np_rng_state[2])
            cp_group.create_dataset('numpy_rng_arg3', data=np_rng_state[3])
            cp_group.create_dataset('numpy_rng_arg4', data=np_rng_state[4])

            # --- 5. Save Persistent Chains (at root level) ---
            # This stores the *latest* chain state, as in the source.
            if 'parallel_chains_v' in f:
                del f['parallel_chains_v']
            f.create_dataset('parallel_chains_v', data=self.chains['v'].cpu().numpy())
            
            # --- 6. Update the 'last_epoch' pointer ---
            f.attrs['last_epoch'] = epoch

    def load_checkpoint(self, filepath: str, epoch: int = None) -> int:
        """
        Loads the RBM's state from an HDF5 file.

        This method loads parameters, persistent chains, and RNG states
        into the *current* RBM instance.
        
        Args:
            filepath (str): Path to the HDF5 checkpoint file.
            epoch (int, optional): Specific epoch to load. 
                                   If None, loads the latest epoch.
        
        Returns:
            int: The epoch number that was loaded.
        
        Raises:
            FileNotFoundError: If the checkpoint file doesn't exist.
            KeyError: If the specified epoch or data is not in the file.
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Checkpoint file not found: {filepath}")

        with h5py.File(filepath, 'r') as f:
            
            # --- 1. Determine which epoch to load ---
            if epoch is None:
                if 'last_epoch' not in f.attrs:
                    raise KeyError("Cannot find 'last_epoch' in checkpoint file. File may be corrupt or empty.")
                loaded_epoch = f.attrs['last_epoch']
            else:
                loaded_epoch = epoch
            
            group_name = f"epoch_{loaded_epoch}"
            if group_name not in f:
                raise KeyError(f"Checkpoint group '{group_name}' not found in file.")
            
            cp_group = f[group_name]
            
            # --- 2. Load Model Parameters ---
            logger.info(f"Loading parameters from {group_name}...")
            for key in self.params.keys():
                if key not in cp_group:
                    raise KeyError(f"Parameter '{key}' not found in checkpoint group '{group_name}'.")
                self.params[key] = torch.tensor(cp_group[key][()], device=self.device)

            # --- 3. Load Persistent Chains (from root) ---
            if 'parallel_chains_v' not in f:
                raise KeyError("'parallel_chains_v' not found in checkpoint file.")
            
            self.chains['v'] = torch.tensor(f['parallel_chains_v'][()], device=self.device)
            # You might need to re-sample hiddens based on loaded chains
            self.sample_hidden() 
            logger.info("Loaded persistent chains.")

            # --- 4. Load and Set RNG States ---
            logger.info("Restoring RNG states...")
            torch.set_rng_state(torch.tensor(cp_group['torch_rng_state'][()]))
            
            np_rng_state = (
                cp_group['numpy_rng_arg0'][()].decode('utf-8'),
                cp_group['numpy_rng_arg1'][()],
                cp_group['numpy_rng_arg2'][()],
                cp_group['numpy_rng_arg3'][()],
                cp_group['numpy_rng_arg4'][()]
            )
            np.random.set_state(np_rng_state)
            
            return loaded_epoch



    def prune_weights(self, tolerance: float) -> int:
            """
            Sets weights in the weight_matrix to 0 if their absolute
            value is less than the given tolerance.

            This operation is in-place and permanent for this instance.

            Args:
                tolerance (float): The threshold. Weights with |w| < tolerance
                                will be zeroed.
            
            Returns:
                int: The number of weights that were pruned (set to 0).
            """
            if tolerance < 0:
                logger.error("Tolerance must be non-negative.")
                raise ValueError("Tolerance must be non-negative.")
                
            weights = self.params["weight_matrix"]
            
            # Find weights where |w| < tolerance
            mask = torch.abs(weights) < tolerance
            
            # Count how many we are about to prune
            # We count the non-zero elements that are now matching the mask
            original_nonzero_to_be_pruned = torch.count_nonzero(weights[mask]).item()
            
            # Apply the mask in-place
            weights[mask] = 0.0
            
            # Log the results
            total_weights = weights.numel()
            remaining_nonzero = torch.count_nonzero(weights).item()
            
            logger.info(f"Pruned {original_nonzero_to_be_pruned} weights (Tolerance={tolerance}).")
            logger.info(f"Total weights: {total_weights}. Remaining: {remaining_nonzero} "
                        f"({(remaining_nonzero/total_weights)*100:.2f}%).")
            
            return original_nonzero_to_be_pruned




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