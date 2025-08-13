import torch
from model.rbm.zephyr import ZephyrRBM
from torch.optim import Optimizer

class RBM(ZephyrRBM):
    def __init__(self, cfg=None):
        super(RBM, self).__init__(cfg)
        self._model_name = "RBM"
        self._chain = False
        self.p_zetas_pcd_chains = torch.tensor([])
        self._n_batches = 1
        self.initOpt()

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

    def block_gibbs_sampling(self, p0, p1=None, p2=None, p3=None):
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
        W01_T = W01.T
        W12_T = W12.T
        W13_T = W13.T
        W23_T = W23.T

        b1 = self.bias_dict['1']
        b2 = self.bias_dict['2']
        b3 = self.bias_dict['3']

        # --- 3) Gibbs loop ---
        for _ in range(self._config.rbm.bgs_steps):
            p0 = self._p_state(W01_T,   W12_T, W13_T, p1, p2, p3, b0)
            
            p1 = self._p_state(W01,   W12_T, W13_T, p0, p2, p3, b1)
            
            p2 = self._p_state(W02,   W12,   W23_T, p0, p1, p3, b2)
            
            p3 = self._p_state(W03,   W13,   W23,   p0, p1, p2, b3)

        # --- 4) Detach once at the end ---
        return p0.detach(), p1.detach(), p2.detach(), p3.detach()
    
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

    def block_gibbs_sampling(self, p0, p1=None, p2=None):
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
        W01_T = W01.T
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
            p0 = self._p_state(W01_T,   W12_T, W13_T, p1, p2, p3, b0)
            
        # --- 4) Detach once at the end ---
        return p0.detach(), p1.detach(), p2.detach(), p3.detach()
    
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
