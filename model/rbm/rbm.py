import torch
from model.rbm.zephyr import ZephyrRBM
from torch.optim import Optimizer
import torch.nn as nn

class RBM(ZephyrRBM):
    def __init__(self, cfg=None):
        super(RBM, self).__init__(cfg)
        self._model_name = "RBM"
        self._chain = False
        self.p_zetas_pcd_chains = torch.tensor([])
        self._n_batches = 1
        # self.initOpt()

    def type(self):
        """String identifier for current model.

        Returns:
            model_type: "RBM"
        """
        return self._model_name
    
    def initOpt(self):
        """Initialize optimizer for RBM parameters."""
        self.opt = {"bias": {}, "weight":{}}
        
        for i in range(3):
            for j in [0,1,2,3]:
                if j > i:
                    self.opt["weight"][str(i)+str(j)] = AdamOpt(self.weight_dict[str(i)+str(j)], self._config.rbm.lr, self._config.rbm.gamma)
        for i in range(4):
            self.opt["bias"][str(i)] = AdamOpt(self.bias_dict[str(i)], self._config.rbm.lr, self._config.rbm.gamma)

    def _p_state(self, weights_ax, weights_bx, weights_cx,
                 pa_state, pb_state, pc_state, bias_x) -> torch.Tensor:
        """partition_state()

        :param weights_a (torch.Tensor) : (n_nodes_a, n_nodes_x)
        :param weights_b (torch.Tensor) : (n_nodes_b, n_nodes_x)
        :param weights_c (torch.Tensor) : (n_nodes_c, n_nodes_x)
        :param pa_state (torch.Tensor) : (batch_size, n_nodes_a)
        :param pb_state (torch.Tensor) : (batch_size, n_nodes_b)
        :param pc_state (torch.Tensor) : (batch_size, n_nodes_c)
        :param bias_x (torch.Tensor) : (n_nodes_x)
        """
        p_activations = (torch.matmul(pa_state, weights_ax) +
                         torch.matmul(pb_state, weights_bx) +
                         torch.matmul(pc_state, weights_cx) + bias_x)
        return torch.bernoulli(torch.sigmoid(p_activations))
    
    def block_gibbs_sampling_cond(self, p0, p1=None, p2=None, p3=None):
        """Run block‐Gibbs sampling conditioned on p0.

        Returns:
        p0_state, p1_state, p2_state, p3_state: each (batch_size, latent)
        """
        batch_size, device = p0.shape[0], p0.device
        latent = self._config.rbm.latent_nodes_per_p

        # --- 1) Initialize missing chains in-place to avoid extra tensors ---
        if p1 is None:
            p1 = torch.rand(batch_size, latent, device=device).bernoulli_()
        if p2 is None:
            p2 = torch.rand(batch_size, latent, device=device).bernoulli_()
        if p3 is None:
            p3 = torch.rand(batch_size, latent, device=device).bernoulli_()

        # --- 2) Cache weights, transposes, and biases outside the loop ---
        W01 = self.weight_dict['01']
        W02 = self.weight_dict['02']
        W03 = self.weight_dict['03']

        W12 = self.weight_dict['12']
        W13 = self.weight_dict['13']
        W23 = self.weight_dict['23']

        # precompute the needed transposes only once
        W12_T = W12.T
        W13_T = W13.T
        W23_T = W23.T

        b1 = self.bias_dict['1']
        b2 = self.bias_dict['2']
        b3 = self.bias_dict['3']

        # --- 3) Gibbs loop ---
        for _ in range(self._config.rbm.bgs_steps):
            p1 = self._p_state(W01,   W12_T, W13_T, p0, p2, p3, b1)
            
            p2 = self._p_state(W02,   W12,   W23_T, p0, p1, p3, b2)
            
            p3 = self._p_state(W03,   W13,   W23,   p0, p1, p2, b3)

        # --- 4) Detach once at the end ---
        return p0.detach(), p1.detach(), p2.detach(), p3.detach()


    def block_gibbs_sampling(self, batch_size=64):
        """
        Runs block-Gibbs sampling starting from a random state with no clamps.
        This is a special case of conditional Gibbs sampling.
        """
        latent = self._config.rbm.latent_nodes_per_p
        # A robust way to get the model's device
        device = next(self.parameters()).device

        # 1. Initialize all four partitions randomly
        initial_partitions = tuple(
            torch.rand(batch_size, latent, device=device).bernoulli_()
            for _ in range(4)
        )

        # 2. Define a mask indicating that nothing is clamped
        clamped_mask = [False, False, False, False]

        # 3. Call the general conditional sampler
        return self.conditional_gibbs_sampling(initial_partitions, clamped_mask)


    def conditional_gibbs_sampling(self, initial_partitions, clamped_mask):
        """
        Runs block-Gibbs sampling, clamping partitions specified by the mask.
        """
        p = list(initial_partitions)
        
        # --- Cache weights, transposes, and biases outside the loop ---
        W01, W02, W03 = self.weight_dict['01'], self.weight_dict['02'], self.weight_dict['03']
        W12, W13, W23 = self.weight_dict['12'], self.weight_dict['13'], self.weight_dict['23']
        
        # Precompute needed transposes
        W01_T, W02_T, W03_T = W01.T, W02.T, W03.T
        W12_T, W13_T, W23_T = W12.T, W13.T, W23.T

        b0, b1, b2, b3 = self.bias_dict['0'], self.bias_dict['1'], self.bias_dict['2'], self.bias_dict['3']
        
        # --- Gibbs loop ---
        for _ in range(self._config.rbm.bgs_steps):
            # Sample each partition that is NOT clamped
            if not clamped_mask[0]:
                p[0] = self._p_state(W01_T, W02_T, W03_T, p[1], p[2], p[3], b0)
                
            if not clamped_mask[1]:
                p[1] = self._p_state(W01,   W12_T, W13_T, p[0], p[2], p[3], b1)
            
            if not clamped_mask[2]:
                p[2] = self._p_state(W02,   W12,   W23_T, p[0], p[1], p[3], b2)
            
            if not clamped_mask[3]:
                p[3] = self._p_state(W03,   W13,   W23,   p[0], p[1], p[2], b3)

        return tuple(pi.detach() for pi in p)    
    def gradient_rbm(self, post_samples):
        n_nodes_p = self._config.rbm.latent_nodes_per_p

        post_zetas = torch.cat(post_samples, 1)
        data_mean = post_zetas.mean(dim=0)
        torch.clamp_(data_mean, min=1e-4, max=(1. - 1e-4))
        vh_data_mean = (post_zetas.transpose(0,1) @ post_zetas) / post_zetas.size(0)

        if self._config.rbm.method == "PCD":
            if self._chain==False:
                pass
            else:
                #generate a random index between 0 and self._n_batches
                idx = torch.randint(self._n_batches,(1,)).item()
                post_zetas = self.p_zetas_pcd_chains[idx,:,:].to(post_zetas.device)

        p0_state, p1_state, p2_state, p3_state = self.block_gibbs_sampling_cond(post_zetas[:, :n_nodes_p],
                                             post_zetas[:, n_nodes_p:2*n_nodes_p],
                                             post_zetas[:, 2*n_nodes_p:3*n_nodes_p],
                                             post_zetas[:, 3*n_nodes_p:])
        
        if self._config.rbm.method == "PCD":
            if self._chain==False:
                ps = torch.cat([p0_state, p1_state, p2_state, p3_state], dim=1).to('cpu')
                self.p_zetas_pcd_chains = torch.cat([self.p_zetas_pcd_chains, ps.unsqueeze(0)],dim=0)
                if self.p_zetas_pcd_chains.shape[0] >= self._n_batches:
                    self._chain = True
            else:
                self.p_zetas_pcd_chains[idx,:,:] = torch.cat([p0_state, p1_state, p2_state, p3_state], dim=1).to('cpu') #post_zetas.to('cpu')

        post_zetas_gen = torch.cat([p0_state,p1_state,p2_state,p3_state], dim=1)
        data_gen = post_zetas_gen.mean(dim=0)
        torch.clamp_(data_gen, min=1e-4, max=(1. - 1e-4));
        vh_gen_mean = (post_zetas_gen.transpose(0,1) @ post_zetas_gen) / post_zetas_gen.size(0)
        
        # compute gradient
        self.grad = {"bias": {}, "weight":{}}
        for i in range(4):
            self.grad["bias"][str(i)] = data_mean[n_nodes_p*i:n_nodes_p*(i+1)] - data_gen[n_nodes_p*i:n_nodes_p*(i+1)]
            
        for i in range(3):
            for j in [0,1,2,3]:
                if j > i:
                    self.grad["weight"][str(i)+str(j)] = (vh_data_mean[n_nodes_p*i:n_nodes_p*(i+1),n_nodes_p*j:n_nodes_p*(j+1)] - vh_gen_mean[n_nodes_p*i:n_nodes_p*(i+1),n_nodes_p*j:n_nodes_p*(j+1)]) * self._weight_mask_dict[str(i)+str(j)]

    def gradient_rbm_centered(self, post_samples):
        n_nodes_p = self._config.rbm.latent_nodes_per_p
        
        post_zetas = torch.cat(post_samples, 1)
        data_mean = post_zetas.mean(dim=0)
        torch.clamp_(data_mean, min=1e-4, max=(1. - 1e-4))
        vh_data_cov = torch.cov(post_zetas.T)
        
        if self._config.rbm.method == "PCD":
            if self._chain==False:
                pass
            else:
                #generate a random index between 0 and self._n_batches
                idx = torch.randint(self._n_batches,(1,)).item()
                post_zetas = self.p_zetas_pcd_chains[idx,:,:].to(post_zetas.device)

        if hasattr(self._config.rbm, "clamped") and not self._config.rbm.clamped:
            p0_state, p1_state, p2_state, p3_state = self.block_gibbs_sampling(post_zetas.shape[0])
        else:

            p0_state, p1_state, p2_state, p3_state = self.block_gibbs_sampling_cond(post_zetas[:, :n_nodes_p],
                                                post_zetas[:, n_nodes_p:2*n_nodes_p],
                                                post_zetas[:, 2*n_nodes_p:3*n_nodes_p],
                                                post_zetas[:, 3*n_nodes_p:])
        
        if self._config.rbm.method == "PCD":
            if self._chain==False:
                ps = torch.cat([p0_state, p1_state, p2_state, p3_state], dim=1).to('cpu')
                self.p_zetas_pcd_chains = torch.cat([self.p_zetas_pcd_chains, ps.unsqueeze(0)],dim=0)
                if self.p_zetas_pcd_chains.shape[0] >= self._n_batches:
                    self._chain = True
            else:
                self.p_zetas_pcd_chains[idx,:,:] = torch.cat([p0_state, p1_state, p2_state, p3_state], dim=1).to('cpu') #post_zetas.to('cpu')

        post_zetas_gen = torch.cat([p0_state,p1_state,p2_state,p3_state], dim=1)
        data_gen = post_zetas_gen.mean(dim=0)
        torch.clamp_(data_gen, min=1e-4, max=(1. - 1e-4));
        vh_gen_cov = torch.cov(post_zetas_gen.T)

        # compute gradient
        self.grad = {"bias": {}, "weight":{}}
        for i in range(3):
            for j in [0,1,2,3]:
                if j > i:
                    self.grad["weight"][str(i)+str(j)] = (vh_data_cov[n_nodes_p*i:n_nodes_p*(i+1),n_nodes_p*j:n_nodes_p*(j+1)] - vh_gen_cov[n_nodes_p*i:n_nodes_p*(i+1),n_nodes_p*j:n_nodes_p*(j+1)]) * self._weight_mask_dict[str(i)+str(j)]
                    
        
        for i in range(4):
            self.grad["bias"][str(i)] = data_mean[n_nodes_p*i:n_nodes_p*(i+1)] - data_gen[n_nodes_p*i:n_nodes_p*(i+1)]
            for j in range(4):
                if j > i:
                    self.grad["bias"][str(i)] = self.grad["bias"][str(i)] - 0.5 * torch.matmul(self.grad["weight"][str(i)+str(j)], (data_mean[n_nodes_p*j:n_nodes_p*(j+1)] + data_gen[n_nodes_p*j:n_nodes_p*(j+1)]))
                elif j < i:
                    self.grad["bias"][str(i)] = self.grad["bias"][str(i)] - 0.5 * torch.matmul(self.grad["weight"][str(j)+str(i)].T , (data_mean[n_nodes_p*j:n_nodes_p*(j+1)] + data_gen[n_nodes_p*j:n_nodes_p*(j+1)]))

    def gradient_rbm_standardized(self, post_samples):
        n_nodes_p = self._config.rbm.latent_nodes_per_p
        
        post_zetas = torch.cat(post_samples, 1)
        data_mean = ((post_zetas - post_zetas.mean(dim=0)) / (post_zetas.std(dim=0) + 1e-8)).mean(dim=0)
        torch.clamp_(data_mean, min=1e-4, max=(1. - 1e-4))
        vh_data_cov = torch.cov(post_zetas.T) / (post_zetas.std(dim=0) * post_zetas.std(dim=0).T + 1e-8)
        
        if self._config.rbm.method == "PCD":
            if self._chain==False:
                pass
            else:
                #generate a random index between 0 and self._n_batches
                idx = torch.randint(self._n_batches,(1,)).item()
                post_zetas = self.p_zetas_pcd_chains[idx,:,:].to(post_zetas.device)
                

        p0_state, p1_state, p2_state, p3_state = self.block_gibbs_sampling_cond(post_zetas[:, :n_nodes_p],
                                             post_zetas[:, n_nodes_p:2*n_nodes_p],
                                             post_zetas[:, 2*n_nodes_p:3*n_nodes_p],
                                             post_zetas[:, 3*n_nodes_p:])
        
        if self._config.rbm.method == "PCD":
            if self._chain==False:
                ps = torch.cat([p0_state, p1_state, p2_state, p3_state], dim=1).to('cpu')
                self.p_zetas_pcd_chains = torch.cat([self.p_zetas_pcd_chains, ps.unsqueeze(0)],dim=0)
                if self.p_zetas_pcd_chains.shape[0] >= self._n_batches:
                    self._chain = True
            else:
                self.p_zetas_pcd_chains[idx,:,:] = torch.cat([p0_state, p1_state, p2_state, p3_state], dim=1).to('cpu') #post_zetas.to('cpu')

        post_zetas_gen = torch.cat([p0_state,p1_state,p2_state,p3_state], dim=1)
        data_gen = ((post_zetas_gen - post_zetas_gen.mean(dim=0)) / (post_zetas_gen.std(dim=0) + 1e-8)).mean(dim=0)
        torch.clamp_(data_gen, min=1e-4, max=(1. - 1e-4));
        vh_gen_cov = torch.cov(post_zetas_gen.T) / (post_zetas_gen.std(dim=0) * post_zetas_gen.std(dim=0).T + 1e-8)

        # compute gradient
        self.grad = {"bias": {}, "weight":{}}
        for i in range(3):
            for j in [0,1,2,3]:
                if j > i:
                    self.grad["weight"][str(i)+str(j)] = (vh_data_cov[n_nodes_p*i:n_nodes_p*(i+1),n_nodes_p*j:n_nodes_p*(j+1)] - vh_gen_cov[n_nodes_p*i:n_nodes_p*(i+1),n_nodes_p*j:n_nodes_p*(j+1)]) * self._weight_mask_dict[str(i)+str(j)]
                    
        
        for i in range(4):
            self.grad["bias"][str(i)] = data_mean[n_nodes_p*i:n_nodes_p*(i+1)] - data_gen[n_nodes_p*i:n_nodes_p*(i+1)]
            for j in range(4):
                if j > i:
                    self.grad["bias"][str(i)] = self.grad["bias"][str(i)] - 0.5 * torch.matmul(self.grad["weight"][str(i)+str(j)], (data_mean[n_nodes_p*j:n_nodes_p*(j+1)] + data_gen[n_nodes_p*j:n_nodes_p*(j+1)]))
                elif j < i:
                    self.grad["bias"][str(i)] = self.grad["bias"][str(i)] - 0.5 * torch.matmul(self.grad["weight"][str(j)+str(i)].T , (data_mean[n_nodes_p*j:n_nodes_p*(j+1)] + data_gen[n_nodes_p*j:n_nodes_p*(j+1)]))
    
    def gradient_rbm_stan(self, post_samples):
        n_nodes_p = self._config.rbm.latent_nodes_per_p

        post_zetas = torch.cat(post_samples, 1)
        data_mean = ((post_zetas - post_zetas.mean(dim=0)) / (post_zetas.std(dim=0) + 1e-8)).mean(dim=0)
        torch.clamp_(data_mean, min=1e-4, max=(1. - 1e-4))
        vh_data_cov = torch.cov(post_zetas.T) / (post_zetas.std(dim=0) * post_zetas.std(dim=0).T + 1e-8)

        if self._config.rbm.method == "PCD":
            if self._chain==False:
                pass
            else:
                #generate a random index between 0 and self._n_batches
                idx = torch.randint(self._n_batches,(1,)).item()
                post_zetas = self.p_zetas_pcd_chains[idx,:,:].to(post_zetas.device)

        p0_state, p1_state, p2_state, p3_state = self.block_gibbs_sampling_cond(post_zetas[:, :n_nodes_p],
                                             post_zetas[:, n_nodes_p:2*n_nodes_p],
                                             post_zetas[:, 2*n_nodes_p:3*n_nodes_p],
                                             post_zetas[:, 3*n_nodes_p:])
        
        if self._config.rbm.method == "PCD":
            if self._chain==False:
                ps = torch.cat([p0_state, p1_state, p2_state, p3_state], dim=1).to('cpu')
                self.p_zetas_pcd_chains = torch.cat([self.p_zetas_pcd_chains, ps.unsqueeze(0)],dim=0)
                if self.p_zetas_pcd_chains.shape[0] >= self._n_batches:
                    self._chain = True
            else:
                self.p_zetas_pcd_chains[idx,:,:] = torch.cat([p0_state, p1_state, p2_state, p3_state], dim=1).to('cpu') #post_zetas.to('cpu')

        post_zetas_gen = torch.cat([p0_state,p1_state,p2_state,p3_state], dim=1)
        data_gen = ((post_zetas_gen - post_zetas_gen.mean(dim=0)) / (post_zetas_gen.std(dim=0) + 1e-8)).mean(dim=0)
        torch.clamp_(data_gen, min=1e-4, max=(1. - 1e-4));
        vh_gen_cov = torch.cov(post_zetas_gen.T) / (post_zetas_gen.std(dim=0) * post_zetas_gen.std(dim=0).T + 1e-8)
        
        # compute gradient
        self.grad = {"bias": {}, "weight":{}}
        for i in range(4):
            self.grad["bias"][str(i)] = data_mean[n_nodes_p*i:n_nodes_p*(i+1)] - data_gen[n_nodes_p*i:n_nodes_p*(i+1)]
            
        for i in range(3):
            for j in [0,1,2,3]:
                if j > i:
                    self.grad["weight"][str(i)+str(j)] = (vh_data_cov[n_nodes_p*i:n_nodes_p*(i+1),n_nodes_p*j:n_nodes_p*(j+1)] - vh_gen_cov[n_nodes_p*i:n_nodes_p*(i+1),n_nodes_p*j:n_nodes_p*(j+1)]) * self._weight_mask_dict[str(i)+str(j)]


    def calculate_mock_gradient(self):
        """Creates a deterministic mock gradient for testing."""
        self.grad = {"bias": {}, "weight": {}}
        for key, param in self.bias_dict.items():
            self.grad["bias"][key] = torch.ones_like(param) * 0.1
        for key, param in self.weight_dict.items():
            self.grad["weight"][key] = torch.ones_like(param) * -0.05


    def update_params_SGD(self):
        for i in range(4):
            self.bias_dict[str(i)] = self.bias_dict[str(i)] + self._config.rbm.lr * self.grad["bias"][str(i)]

        for i in range(3):
            for j in [0,1,2,3]:
                if j > i:
                    self.weight_dict[str(i)+str(j)] = self.weight_dict[str(i)+str(j)] + self._config.rbm.lr * self.grad["weight"][str(i)+str(j)]

    def update_params(self):
        for i in range(4):
            self.bias_dict[str(i)] = self.opt["bias"][str(i)].step(self.grad["bias"][str(i)].detach())

        for i in range(3):
            for j in [0,1,2,3]:
                if j > i:
                    self.weight_dict[str(i)+str(j)] = self.opt["weight"][str(i)+str(j)].step(self.grad["weight"][str(i)+str(j)].detach())

    def energy_exp_cond(self, p0, p1, p2, p3):
        # pull biases and weights into locals for fewer dict lookups
        b0, b1, b2, b3 = (self.bias_dict[k] for k in ("0","1","2","3"))
        W01, W02, W03 = (self.weight_dict[k] for k in ("01","02","03"))
        W12, W13, W23 = (self.weight_dict[k] for k in ("12","13","23"))

        # start with unary terms
        energy = -( p1 @ b1 + p2 @ b2 + p3 @ b3)

        # subtract out each pairwise interaction via einsum
        energy -= torch.einsum("bi,ij,bj->b", p0, W01, p1)
        energy -= torch.einsum("bi,ij,bj->b", p0, W02, p2)
        energy -= torch.einsum("bi,ij,bj->b", p0, W03, p3)
        energy -= torch.einsum("bi,ij,bj->b", p1, W12, p2)
        energy -= torch.einsum("bi,ij,bj->b", p1, W13, p3)
        energy -= torch.einsum("bi,ij,bj->b", p2, W23, p3)

        batch_energy = energy

        return batch_energy
    
    def energy_exp(self, p0, p1, p2, p3):
        # pull biases and weights into locals for fewer dict lookups
        b0, b1, b2, b3 = (self.bias_dict[k] for k in ("0","1","2","3"))
        W01, W02, W03 = (self.weight_dict[k] for k in ("01","02","03"))
        W12, W13, W23 = (self.weight_dict[k] for k in ("12","13","23"))

        # start with unary terms
        energy = -(p0 @ b0 + p1 @ b1 + p2 @ b2 + p3 @ b3)

        # subtract out each pairwise interaction via einsum
        energy -= torch.einsum("bi,ij,bj->b", p0, W01, p1)
        energy -= torch.einsum("bi,ij,bj->b", p0, W02, p2)
        energy -= torch.einsum("bi,ij,bj->b", p0, W03, p3)
        energy -= torch.einsum("bi,ij,bj->b", p1, W12, p2)
        energy -= torch.einsum("bi,ij,bj->b", p1, W13, p3)
        energy -= torch.einsum("bi,ij,bj->b", p2, W23, p3)

        batch_energy = energy

        return batch_energy

class RBM_Hidden(RBM):
    def __init__(self, cfg=None):
        super(RBM_Hidden, self).__init__(cfg)
        self._model_name = "RBM_Hidden"
        self._chain = False
        self.p_zetas_pcd_chains = torch.tensor([])
        self._n_batches = 1
        self.initOpt()

    def sigmoid_C_k(self, weights_ax, weights_bx, weights_cx,
                 pa_state, pb_state, pc_state, bias_x) -> torch.Tensor:
        """partition_state()

        :param weights_a (torch.Tensor) : (n_nodes_a, n_nodes_x)
        :param weights_b (torch.Tensor) : (n_nodes_b, n_nodes_x)
        :param weights_c (torch.Tensor) : (n_nodes_c, n_nodes_x)
        :param pa_state (torch.Tensor) : (batch_size, n_nodes_a)
        :param pb_state (torch.Tensor) : (batch_size, n_nodes_b)
        :param pc_state (torch.Tensor) : (batch_size, n_nodes_c)
        :param bias_x (torch.Tensor) : (n_nodes_x)
        """
        p_activations = (torch.matmul(pa_state, weights_ax) +
                         torch.matmul(pb_state, weights_bx) +
                         torch.matmul(pc_state, weights_cx) + bias_x)
        return torch.sigmoid(p_activations).detach()

    def block_gibbs_sampling_cond(self, p0, p1=None, p2=None):
        """Run block‐Gibbs sampling conditioned on p0.

        Returns:
        p0_state, p1_state, p2_state, p3_state: each (batch_size, latent)
        """
        batch_size, device = p0.shape[0], p0.device
        latent = self._config.rbm.latent_nodes_per_p

        # --- 1) Initialize missing chains in-place to avoid extra tensors ---
        if p1 is None:
            p1 = torch.rand(batch_size, latent, device=device).bernoulli_()
        if p2 is None:
            p2 = torch.rand(batch_size, latent, device=device).bernoulli_()

        # --- 2) Cache weights, transposes, and biases outside the loop ---
        W01 = self.weight_dict['01']
        W02 = self.weight_dict['02']
        W03 = self.weight_dict['03']

        W12 = self.weight_dict['12']
        W13 = self.weight_dict['13']
        W23 = self.weight_dict['23']

        # precompute the needed transposes only once
        W12_T = W12.T
        W13_T = W13.T
        W23_T = W23.T

        b1 = self.bias_dict['1']
        b2 = self.bias_dict['2']
        b3 = self.bias_dict['3']

        # --- 3) Gibbs loop ---
        for _ in range(self._config.rbm.bgs_steps):
            p3 = self._p_state(W03,   W13,   W23,   p0, p1, p2, b3)

            p2 = self._p_state(W02,   W12,   W23_T, p0, p1, p3, b2)

            p1 = self._p_state(W01,   W12_T, W13_T, p0, p2, p3, b1)

        # --- 4) Detach once at the end ---
        return p0.detach(), p1.detach(), p2.detach(), p3.detach()


    def block_gibbs_sampling(self, batch_size=64):
        """
        Runs block-Gibbs sampling starting from a random state with no clamps.
        This is a special case of conditional Gibbs sampling.
        """
        latent = self._config.rbm.latent_nodes_per_p
        # A robust way to get the model's device
        device = next(self.parameters()).device

        # 1. Initialize all four partitions randomly
        initial_partitions = tuple(
                torch.rand(batch_size, latent, device=device).bernoulli_()
                for _ in range(3)
            )

        # 2. Define a mask indicating that nothing is clamped
        clamped_mask = [False, False, False, False]

        # 3. Call the general conditional sampler
        return self.conditional_gibbs_sampling(initial_partitions, clamped_mask)


    def conditional_gibbs_sampling(self, initial_partitions, clamped_mask):
        """
        Runs block-Gibbs sampling, clamping partitions specified by the mask.
        """
        p = list(initial_partitions)
        p3 = self.sigmoid_C_k(self.weight_dict['03'],   self.weight_dict['13'],   self.weight_dict['23'], 
                            p[0],p[1],p[2], self.bias_dict['3'])
        p.append(p3)

        
        # --- Cache weights, transposes, and biases outside the loop ---
        W01, W02, W03 = self.weight_dict['01'], self.weight_dict['02'], self.weight_dict['03']
        W12, W13, W23 = self.weight_dict['12'], self.weight_dict['13'], self.weight_dict['23']
        
        # Precompute needed transposes
        W01_T, W02_T, W03_T = W01.T, W02.T, W03.T
        W12_T, W13_T, W23_T = W12.T, W13.T, W23.T

        b0, b1, b2, b3 = self.bias_dict['0'], self.bias_dict['1'], self.bias_dict['2'], self.bias_dict['3']
        
        # --- Gibbs loop ---
        for _ in range(self._config.rbm.bgs_steps):
            # Sample each partition that is NOT clamped
            if not clamped_mask[0]:
                p[0] = self._p_state(W01_T, W02_T, W03_T, p[1], p[2], p[3], b0)
                
            if not clamped_mask[1]:
                p[1] = self._p_state(W01,   W12_T, W13_T, p[0], p[2], p[3], b1)
            
            if not clamped_mask[2]:
                p[2] = self._p_state(W02,   W12,   W23_T, p[0], p[1], p[3], b2)
            
            p[3] = self._p_state(W03,   W13,   W23,   p[0], p[1], p[2], b3)

        return tuple(pi.detach() for pi in p)    

        
    def gradient_rbm(self, post_samples):
        n_nodes_p = self._config.rbm.latent_nodes_per_p

        p3 = self.sigmoid_C_k(self.weight_dict['03'],   self.weight_dict['13'],   self.weight_dict['23'], 
                              post_samples[0],post_samples[1],post_samples[2], self.bias_dict['3'])
        post_samples.append(p3)

        post_zetas = torch.cat(post_samples, 1)
        data_mean = post_zetas.mean(dim=0)
        torch.clamp_(data_mean, min=1e-4, max=(1. - 1e-4))
        vh_data_mean = (post_zetas.transpose(0,1) @ post_zetas) / post_zetas.size(0)

        if self._config.rbm.method == "PCD":
            if self._chain==False:
                pass
            else:
                #generate a random index between 0 and self._n_batches
                idx = torch.randint(self._n_batches,(1,)).item()
                post_zetas = self.p_zetas_pcd_chains[idx,:,:].to(post_zetas.device)

        p0_state, p1_state, p2_state, p3_state = self.block_gibbs_sampling_cond(post_zetas[:, :n_nodes_p],
                                             post_zetas[:, n_nodes_p:2*n_nodes_p],
                                             post_zetas[:, 2*n_nodes_p:3*n_nodes_p],
                                             post_zetas[:, 3*n_nodes_p:])
        
        if self._config.rbm.method == "PCD":
            if self._chain==False:
                ps = torch.cat([p0_state, p1_state, p2_state, p3_state], dim=1).to('cpu')
                self.p_zetas_pcd_chains = torch.cat([self.p_zetas_pcd_chains, ps.unsqueeze(0)],dim=0)
                if self.p_zetas_pcd_chains.shape[0] >= self._n_batches:
                    self._chain = True
            else:
                self.p_zetas_pcd_chains[idx,:,:] = torch.cat([p0_state, p1_state, p2_state, p3_state], dim=1).to('cpu') #post_zetas.to('cpu')

        post_zetas_gen = torch.cat([p0_state,p1_state,p2_state,p3_state], dim=1)
        data_gen = post_zetas_gen.mean(dim=0)
        torch.clamp_(data_gen, min=1e-4, max=(1. - 1e-4));
        vh_gen_mean = (post_zetas_gen.transpose(0,1) @ post_zetas_gen) / post_zetas_gen.size(0)
        
        # compute gradient
        self.grad = {"bias": {}, "weight":{}}
        for i in range(4):
            self.grad["bias"][str(i)] = data_mean[n_nodes_p*i:n_nodes_p*(i+1)] - data_gen[n_nodes_p*i:n_nodes_p*(i+1)]
            
        for i in range(3):
            for j in [0,1,2,3]:
                if j > i:
                    self.grad["weight"][str(i)+str(j)] = (vh_data_mean[n_nodes_p*i:n_nodes_p*(i+1),n_nodes_p*j:n_nodes_p*(j+1)] - vh_gen_mean[n_nodes_p*i:n_nodes_p*(i+1),n_nodes_p*j:n_nodes_p*(j+1)]) * self._weight_mask_dict[str(i)+str(j)]

    def gradient_rbm_centered(self, post_samples):
        n_nodes_p = self._config.rbm.latent_nodes_per_p

        p3 = self.sigmoid_C_k(self.weight_dict['03'],   self.weight_dict['13'],   self.weight_dict['23'], 
                              post_samples[0],post_samples[1],post_samples[2], self.bias_dict['3'])
        post_samples.append(p3)

        post_zetas = torch.cat(post_samples, 1)
        data_mean = post_zetas.mean(dim=0)
        torch.clamp_(data_mean, min=1e-4, max=(1. - 1e-4))
        vh_data_cov = torch.cov(post_zetas.T)
        
        if self._config.rbm.method == "PCD":
            if self._chain==False:
                pass
            else:
                #generate a random index between 0 and self._n_batches
                idx = torch.randint(self._n_batches,(1,)).item()
                post_zetas = self.p_zetas_pcd_chains[idx,:,:].to(post_zetas.device)
                
        if hasattr(self._config.rbm, "clamped") and not self._config.rbm.clamped:
            p0_state, p1_state, p2_state, p3_state = self.block_gibbs_sampling(post_zetas.shape[0])
        else:
            p0_state, p1_state, p2_state, p3_state = self.block_gibbs_sampling_cond(post_zetas[:, :n_nodes_p],
                                                post_zetas[:, n_nodes_p:2*n_nodes_p],
                                                post_zetas[:, 2*n_nodes_p:3*n_nodes_p])
        
        if self._config.rbm.method == "PCD":
            if self._chain==False:
                ps = torch.cat([p0_state, p1_state, p2_state, p3_state], dim=1).to('cpu')
                self.p_zetas_pcd_chains = torch.cat([self.p_zetas_pcd_chains, ps.unsqueeze(0)],dim=0)
                if self.p_zetas_pcd_chains.shape[0] >= self._n_batches:
                    self._chain = True
            else:
                self.p_zetas_pcd_chains[idx,:,:] = torch.cat([p0_state, p1_state, p2_state, p3_state], dim=1).to('cpu') #post_zetas.to('cpu')

        post_zetas_gen = torch.cat([p0_state,p1_state,p2_state,p3_state], dim=1)
        data_gen = post_zetas_gen.mean(dim=0)
        torch.clamp_(data_gen, min=1e-4, max=(1. - 1e-4));
        vh_gen_cov = torch.cov(post_zetas_gen.T)

        # compute gradient
        self.grad = {"bias": {}, "weight":{}}
        for i in range(3):
            for j in [0,1,2,3]:
                if j > i:
                    self.grad["weight"][str(i)+str(j)] = (vh_data_cov[n_nodes_p*i:n_nodes_p*(i+1),n_nodes_p*j:n_nodes_p*(j+1)] - vh_gen_cov[n_nodes_p*i:n_nodes_p*(i+1),n_nodes_p*j:n_nodes_p*(j+1)]) * self._weight_mask_dict[str(i)+str(j)]
                    
        
        for i in range(4):
            self.grad["bias"][str(i)] = data_mean[n_nodes_p*i:n_nodes_p*(i+1)] - data_gen[n_nodes_p*i:n_nodes_p*(i+1)]
            for j in range(4):
                if j > i:
                    self.grad["bias"][str(i)] = self.grad["bias"][str(i)] - 0.5 * torch.matmul(self.grad["weight"][str(i)+str(j)], (data_mean[n_nodes_p*j:n_nodes_p*(j+1)] + data_gen[n_nodes_p*j:n_nodes_p*(j+1)]))
                elif j < i:
                    self.grad["bias"][str(i)] = self.grad["bias"][str(i)] - 0.5 * torch.matmul(self.grad["weight"][str(j)+str(i)].T , (data_mean[n_nodes_p*j:n_nodes_p*(j+1)] + data_gen[n_nodes_p*j:n_nodes_p*(j+1)]))

    def energy_exp_cond(self, p0, p1, p2, p3):
        # pull biases and weights into locals for fewer dict lookups
        b0, b1, b2, b3 = (self.bias_dict[k] for k in ("0","1","2","3"))
        W01, W02, W03 = (self.weight_dict[k] for k in ("01","02","03"))
        W12, W13, W23 = (self.weight_dict[k] for k in ("12","13","23"))

        # start with unary terms
        energy = -( p1 @ b1 + p2 @ b2 + p3 @ b3)

        # subtract out each pairwise interaction via einsum
        energy -= torch.einsum("bi,ij,bj->b", p0, W01, p1)
        energy -= torch.einsum("bi,ij,bj->b", p0, W02, p2)
        energy -= torch.einsum("bi,ij,bj->b", p0, W03, p3)
        energy -= torch.einsum("bi,ij,bj->b", p1, W12, p2)
        energy -= torch.einsum("bi,ij,bj->b", p1, W13, p3)
        energy -= torch.einsum("bi,ij,bj->b", p2, W23, p3)

        batch_energy = energy

        return batch_energy

###################Optimizer
############################

class AdamOpt():
    def __init__(self, theta, a=1e-3, gamma=0.0):
        self.theta = theta.detach()
        self.m = torch.zeros_like(theta)
        self.v = torch.zeros_like(theta)
        self.b1 = 0.9
        self.b2 = 0.999
        self.a = a
        self.eps = 1e-8
        self.t = 0
        self.gamma = gamma

    def step(self, grad):
        with torch.no_grad():
            self.t += 1
            self.m = self.b1 * self.m + (1 - self.b1) * grad
            self.v = self.b2 * self.v + (1 - self.b2) * grad ** 2

            m_hat = self.m / (1 - self.b1 ** self.t)
            v_hat = self.v / (1 - self.b2 ** self.t)

            delta = m_hat / (torch.sqrt(v_hat) + self.eps)
            # delta = torch.clamp(delta, -5, 5)

            self.theta = self.theta * (1 - self.gamma) + delta * self.a
        return self.theta



class RBM_2Partite(RBM):
    def __init__(self, cfg=None):
        # Don't call super().__init__() to avoid the 4-partite initialization
        # Instead, initialize base attributes directly
        nn.Module.__init__(self)
        self._config = cfg
        self._model_name = "RBM_2Partite"
        self._chain = False
        self.p_zetas_pcd_chains = torch.tensor([])
        self._n_batches = 1
        # Initialize 2-partite specific parameters
        self.n_nodes = self._config.rbm.latent_nodes_per_p
        self._weight_dict = {}
        self._bias_dict = {}
        if self._config.device == "gpu" and torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self._config.gpu_list[0]}")
        else:
            self.device = torch.device("cpu")

        self.init_2partite_params()


    def init_2partite_params(self):
        device = self.device

        W = torch.randn(self.n_nodes, self.n_nodes, device=device) * 1e-4
        v_bias = torch.zeros(self.n_nodes, device=device)
        h_bias = torch.zeros(self.n_nodes, device=device)

        self._weight_dict.clear()
        self._bias_dict.clear()
        self._weight_dict["vh"] = W
        self._bias_dict["v"] = v_bias
        self._bias_dict["h"] = h_bias

    def initOpt(self):
        """Initialize custom optimizers exactly like parent RBM"""
        self.opt = {"bias": {}, "weight": {}}
        self.opt["weight"]["vh"] = AdamOpt(self._weight_dict["vh"],
                                           self._config.rbm.lr,
                                           self._config.rbm.gamma)
        self.opt["bias"]["v"] = AdamOpt(self._bias_dict["v"],
                                        self._config.rbm.lr,
                                        self._config.rbm.gamma)
        self.opt["bias"]["h"] = AdamOpt(self._bias_dict["h"],
                                        self._config.rbm.lr,
                                        self._config.rbm.gamma)


    def sample_hidden(self, visible):
        """Sample hidden units given visible units"""
        hidden_activations = torch.matmul(visible, self._weight_dict["vh"]) + self._bias_dict["h"]
        hidden_probs = torch.sigmoid(hidden_activations)
        hidden_sample = torch.bernoulli(hidden_probs)
        return hidden_sample, hidden_probs

    def sample_visible(self, hidden):
        """Sample visible units given hidden units"""
        visible_activations = torch.matmul(hidden, self._weight_dict["vh"].T) + self._bias_dict["v"]
        visible_probs = torch.sigmoid(visible_activations)
        visible_sample = torch.bernoulli(visible_probs)
        return visible_sample, visible_probs

    def gibbs_sampling(self, batch_size=64, k_steps=None):
        """Run k steps of Gibbs sampling starting from random state"""
        if k_steps is None:
            k_steps = self._config.rbm.bgs_steps
            
        device = self.device
        
        # Start with random visible state
        visible = torch.rand(batch_size, self.n_nodes, device=device).bernoulli_()
        
        for _ in range(k_steps):
            hidden, _ = self.sample_hidden(visible)
            visible, _ = self.sample_visible(hidden)
            
        return visible.detach()

    def reconstruct(self, visible_data):
        """Reconstruct visible data (visible -> hidden -> visible)"""
        hidden, _ = self.sample_hidden(visible_data)
        reconstructed, _ = self.sample_visible(hidden)
        return reconstructed.detach()

    def gradient_rbm_centered(self, visible_data):
        """
        Compute gradients using the centered approach.
        visible_data: (batch_size, n_nodes)
        """
        batch_size = visible_data.size(0)

        # --- Positive phase ---
        hidden_pos, hidden_pos_probs = self.sample_hidden(visible_data)

        # --- Negative phase (CD-k or PCD-k) ---
        if self._config.rbm.method == "PCD":
            if not self._chain:
                visible_neg = visible_data.clone()
            else:
                idx = torch.randint(self._n_batches, (1,)).item()
                visible_neg = self.p_zetas_pcd_chains[idx, :, :].to(visible_data.device)
        else:
            visible_neg = visible_data.clone()

        for _ in range(self._config.rbm.bgs_steps):
            hidden_neg, _ = self.sample_hidden(visible_neg)
            visible_neg, _ = self.sample_visible(hidden_neg)
        hidden_neg, hidden_neg_probs = self.sample_hidden(visible_neg)

        # --- Update persistent chains ---
        if self._config.rbm.method == "PCD":
            if not self._chain:
                self.p_zetas_pcd_chains = torch.cat(
                    [self.p_zetas_pcd_chains, visible_neg.unsqueeze(0).cpu()],
                    dim=0,
                )
                if self.p_zetas_pcd_chains.shape[0] >= self._n_batches:
                    self._chain = True
            else:
                self.p_zetas_pcd_chains[idx, :, :] = visible_neg.cpu()

        # --- Compute expectations ---
        v_data_mean = visible_data.mean(0)
        h_data_mean = hidden_pos_probs.mean(0)
        v_model_mean = visible_neg.mean(0)
        h_model_mean = hidden_neg_probs.mean(0)

        # --- Centered variables ---
        v_data_centered = visible_data - v_data_mean
        h_data_centered = hidden_pos_probs - h_data_mean
        v_model_centered = visible_neg - v_data_mean
        h_model_centered = hidden_neg_probs - h_data_mean

        # --- Gradients ---
        self.grad = {"bias": {}, "weight": {}}

        # Weight gradient (covariance difference of centered stats)
        self.grad["weight"]["vh"] = (
            torch.matmul(v_data_centered.T, h_data_centered) / batch_size
            - torch.matmul(v_model_centered.T, h_model_centered) / batch_size
        )

        # Bias gradients (with correct centered correction)
        self.grad["bias"]["v"] = (v_data_mean - v_model_mean) - torch.matmul(
            self.grad["weight"]["vh"], h_data_mean
        )
        self.grad["bias"]["h"] = (h_data_mean - h_model_mean) - torch.matmul(
            v_data_mean, self.grad["weight"]["vh"]
        )


    def update_params(self):
        """Update parameters using custom Adam optimizers"""
        self._bias_dict["v"] = self.opt["bias"]["v"].step(
            self.grad["bias"]["v"].detach()
        )
        self._bias_dict["h"] = self.opt["bias"]["h"].step(
            self.grad["bias"]["h"].detach()
        )
        self._weight_dict["vh"] = self.opt["weight"]["vh"].step(
            self.grad["weight"]["vh"].detach()
        )


    def update_params_SGD(self):
            """
            Updates the model's parameters using Stochastic Gradient Descent (SGD).

            This method is generalized to work for any number of partitions by
            iterating directly over the parameter and gradient dictionaries instead
            of using hardcoded loops.
            """
            lr = self._config.rbm.lr
            gamma = self._config.rbm.gamma

            # Update all bias parameters
            for key, bias_param in self._bias_dict.items():
                if key in self.grad["bias"]:
                    # Note: We detach the gradient to prevent this update step
                    # from being tracked in the computation graph.
                    self._bias_dict[key] = bias_param + lr * self.grad["bias"][key].detach()

            # Update all weight parameters
            for key, weight_param in self._weight_dict.items():
                if key in self.grad["weight"]:
                    self._weight_dict[key] = weight_param + lr * self.grad["weight"][key].detach() - gamma * self._weight_dict[key]



    def energy(self, visible, hidden):
        """Compute energy of a visible-hidden configuration"""
        # E = -v^T W h - b_v^T v - b_h^T h
        interaction = torch.sum(visible * torch.matmul(hidden, self.W.T), dim=1)
        visible_bias_term = torch.matmul(visible, self.visible_bias)
        hidden_bias_term = torch.matmul(hidden, self.hidden_bias)
        
        energy = -(interaction + visible_bias_term + hidden_bias_term)
        return energy

    # Override methods that don't apply to 2-partite RBM
    def block_gibbs_sampling(self, batch_size=64):
        """Generate samples using Gibbs sampling"""
        return self.gibbs_sampling(batch_size)

    def conditional_gibbs_sampling(self, initial_partitions, clamped_mask):
        """For 2-partite RBM, this becomes clamped reconstruction"""
        if isinstance(initial_partitions, (list, tuple)):
            visible_data = initial_partitions[0]  # Take first element if it's a tuple
        else:
            visible_data = initial_partitions
            
        # For 2-partite, we can clamp visible and sample hidden, or vice versa
        if clamped_mask[0]:  # Clamp visible
            return self.reconstruct(visible_data)
        else:
            # If not clamping visible, just do regular sampling
            return self.gibbs_sampling(visible_data.size(0))

    def sample_visible_2p(self, batch_size=64):
        """Generate visible samples (for compatibility with main script)"""
        return self.gibbs_sampling(batch_size)

    def type(self):
        """String identifier for current model"""
        return self._model_name