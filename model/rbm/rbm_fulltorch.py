import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
from torch.optim import Adam, SGD
from model.rbm.rbm import RBM

from CaloQuVAE import logging
logger = logging.getLogger(__name__)


# Order of weights & biases used throughout
_WEIGHT_KEYS = ["01", "02", "03", "12", "13", "23"]
_BIAS_KEYS = ["0", "1", "2", "3"]


class ContrastiveDivergenceFn(Function):
    """Custom autograd.Function that computes the RBM statistics (positive and
    negative phase) in forward and returns *manual* gradients in backward.

    Signature of forward/backward arguments (positional):
        w01, w02, w03, w12, w13, w23,
        b0, b1, b2, b3,
        post_zetas,
        m01, m02, m03, m12, m13, m23,
        bgs_steps, latent_nodes_per_p

    Important notes:
    - The *sampling* (Bernoulli draws) is non-differentiable; we do not rely
      on autograd to flow through it. Instead we compute the model/data
      statistics and in backward() we return custom gradients for the weight
      and bias tensors.
    - The backward() here returns gradients in the same order as the weight
      and bias inputs. All other inputs (post_zetas, masks, ints) get None.
    - The gradient returned is chosen so that an optimizer.step() (which
      performs param = param - lr * param.grad) will produce the *same*
      update behavior as your original code, which updated params using
      param += lr * grad_where_grad_is(data_mean - data_gen).
      Therefore backward returns - (data_mean - data_gen) etc.
    """

    @staticmethod
    def forward(ctx,
                w01, w02, w03, w12, w13, w23,
                b0, b1, b2, b3,
                post_zetas,
                m01, m02, m03, m12, m13, m23,
                bgs_steps: int,
                latent_nodes_per_p: int):

        device = post_zetas.device
        n_nodes_p = latent_nodes_per_p

        # Compute positive-phase statistics from the provided post_zetas
        data_mean = post_zetas.mean(dim=0)
        # clamp as in your original code to avoid degenerate values
        torch.clamp_(data_mean, min=1e-4, max=(1. - 1e-4))
        vh_data_mean = (post_zetas.transpose(0, 1) @ post_zetas) / float(post_zetas.size(0))

        # Prepare masked weights and biases for Gibbs sampling
        W01 = w01 * m01
        W02 = w02 * m02
        W03 = w03 * m03
        W12 = w12 * m12
        W13 = w13 * m13
        W23 = w23 * m23

        # Precompute transposes used in conditional sampling
        W01_T = W01.T
        W12_T = W12.T
        W13_T = W13.T
        W23_T = W23.T

        b0_t = b0
        b1_t = b1
        b2_t = b2
        b3_t = b3

        # Unpack initial chains from the provided post_zetas
        p0 = post_zetas[:, 0 * n_nodes_p:1 * n_nodes_p].to(device)
        p1 = post_zetas[:, 1 * n_nodes_p:2 * n_nodes_p].to(device)
        p2 = post_zetas[:, 2 * n_nodes_p:3 * n_nodes_p].to(device)
        p3 = post_zetas[:, 3 * n_nodes_p:4 * n_nodes_p].to(device)

        # Run block Gibbs sampling (in-place reassignment). Sampling is
        # nondifferentiable by design (we treat samples as constants in the
        # negative-phase statistics), so we use bernoulli draws and detach later.
        for _ in range(int(bgs_steps)):
            # sample p1 given p0, p2, p3
            p1_logits = (p0 @ W01) + (p2 @ W12_T) + (p3 @ W13_T) + b1_t
            p1 = torch.bernoulli(torch.sigmoid(p1_logits))

            # sample p2
            p2_logits = (p0 @ W02) + (p1 @ W12) + (p3 @ W23_T) + b2_t
            p2 = torch.bernoulli(torch.sigmoid(p2_logits))

            # sample p3
            p3_logits = (p0 @ W03) + (p1 @ W13) + (p2 @ W23) + b3_t
            p3 = torch.bernoulli(torch.sigmoid(p3_logits))

        # Build generated statistics (we detach samples because we won't
        # compute gradients through the sampling process itself)
        post_zetas_gen = torch.cat([p0, p1, p2, p3], dim=1).detach()
        data_gen = post_zetas_gen.mean(dim=0)
        torch.clamp_(data_gen, min=1e-4, max=(1. - 1e-4))
        vh_gen_mean = (post_zetas_gen.transpose(0, 1) @ post_zetas_gen) / float(post_zetas_gen.size(0))

        # Save statistics and masks needed for backward
        ctx.save_for_backward(data_mean, data_gen, vh_data_mean, vh_gen_mean,
                              m01, m02, m03, m12, m13, m23)
        ctx.n_nodes_p = n_nodes_p

        # The forward returns a scalar tensor so calling .backward() will trigger our custom backward().
        loss = (data_mean - data_gen).pow(2).sum() + (vh_data_mean - vh_gen_mean).pow(2).sum()
        return loss

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output):
        # Unpack
        (data_mean, data_gen, vh_data_mean, vh_gen_mean,
         m01, m02, m03, m12, m13, m23) = ctx.saved_tensors
        n_nodes_p = ctx.n_nodes_p

        # compute bias gradients: original code used (data_mean - data_gen)
        # and later assigned param.grad = -1 * that value before optimizer.step.
        # To keep the same external behaviour we return - (data_mean - data_gen)
        # here so optimizer.step (subtracting lr * param.grad) yields param += lr * grad
        grad_biases = []
        for i in range(4):
            g = (data_mean[n_nodes_p * i:n_nodes_p * (i + 1)] -
                 data_gen[n_nodes_p * i:n_nodes_p * (i + 1)])
            grad_biases.append((-g * grad_output).contiguous())

        # compute weight gradients for each pair in the same order defined in _WEIGHT_KEYS
        grad_weights = []
        # mapping order: 01,02,03,12,13,23 -> loops
        # i,j pairs in the same order as earlier implementation
        pairs = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
        masks = [m01, m02, m03, m12, m13, m23]
        for idx, (i,j) in enumerate(pairs):
            vh_block_data = vh_data_mean[n_nodes_p*i:n_nodes_p*(i+1), n_nodes_p*j:n_nodes_p*(j+1)]
            vh_block_gen = vh_gen_mean[n_nodes_p*i:n_nodes_p*(i+1), n_nodes_p*j:n_nodes_p*(j+1)]
            g = (vh_block_data - vh_block_gen) * masks[idx]
            grad_weights.append((-g * grad_output).contiguous())

        # Return gradients in the exact same order as forward inputs.
        # forward inputs order was: w01,w02,w03,w12,w13,w23, b0,b1,b2,b3, post_zetas,
        # m01,...,m23, bgs_steps, latent
        # We return grads for the first 10 inputs (weights + biases), and None for the rest.
        out = tuple(grad_weights) + tuple(grad_biases)
        # follow with Nones for post_zetas, masks, bgs_steps, latent
        out = out + (None,)  # post_zetas
        out = out + (None, None, None, None, None, None)  # masks
        out = out + (None, None)  # bgs_steps, latent
        return out


class RBMTorchFull(RBM):
    """RBM subclass that exposes weight/bias parameters as torch.Parameters
    and uses ContrastiveDivergenceFn to compute gradients through a custom
    backward() implementation.

    This class *does not* mutate the base class's _weight_dict/_bias_dict so
    other RBM subclasses remain unaffected.
    """

    def __init__(self, cfg=None):
        super(RBMTorchFull, self).__init__(cfg)
        logger.info("RBMTorchFull initialized")

        # Create a stable ordering for parameters so that we can flatten to tuples
        # and reconstruct named access if needed.
        self._weight_keys = _WEIGHT_KEYS
        self._bias_keys = _BIAS_KEYS

        # Build ParameterList objects from the underlying dicts. We clone the
        # existing arrays so we don't mutate base-class buffers unexpectedly.
        self._device = self._weight_dict[self._weight_keys[0]].device
        self._weight_params = torch.nn.ParameterList([
            torch.nn.Parameter(self._weight_dict[k].clone().detach().to(self._device))
            for k in self._weight_keys
        ])

        self._bias_params = torch.nn.ParameterList([
            torch.nn.Parameter(self._bias_dict[k].clone().detach().to(self._device))
            for k in self._bias_keys
        ])

        # Keep the original masks in-place (they are used inside ContrastiveDivergenceFn)
        # Make sure masks live on the same device/dtype as params for correct broadcasting
        self._weight_masks = [self._weight_mask_dict[k].to(self._weight_params[0].device)
                               for k in self._weight_keys]

        # init optimizer over the new ParameterList
        self.initOpt()

    @property
    def weight_dict(self):
        # Return the masked weights constructed from Parameters (this differs
        # from the base class which may return detached tensors). This allows
        # other code that reads weight_dict on RBMtorch to see the current
        # trainable values.
        out = {}
        for i, k in enumerate(self._weight_keys):
            out[k] = (self._weight_params[i] * self._weight_masks[i]).detach().clone()
        return out

    @property
    def bias_dict(self):
        out = {}
        for i, k in enumerate(self._bias_keys):
            out[k] = self._bias_params[i].detach().clone()
        return out

    def initOpt(self):
        params_to_optimize = list(self._weight_params) + list(self._bias_params)
        # self.opt = Adam(params_to_optimize,
        #                 lr=self._config.rbm.lr,
        #                 weight_decay=self._config.rbm.gamma)

        self.opt = SGD(params_to_optimize, lr=self._config.rbm.lr)

    def forward(self, post_samples):
        # post_samples: list or tuple of 4 tensors each (batch_size, latent)
        post_zetas = torch.cat(post_samples, dim=1)

        # Unpack params & masks into the order expected by the Function
        w_args = tuple(self._weight_params)
        b_args = tuple(self._bias_params)
        m_args = tuple(self._weight_masks)

        # Call the custom autograd Function. It returns a scalar loss tensor.
        loss = ContrastiveDivergenceFn.apply(
            *w_args,
            *b_args,
            post_zetas,
            *m_args,
            int(self._config.rbm.bgs_steps),
            int(self._config.rbm.latent_nodes_per_p)
        )
        return loss

    def step_on_batch(self, post_samples):
        loss = self.forward(post_samples)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()
