"""
Transfusion network for Generating Layer Energies, based on CaloDREAM Architecture.
Used and Orchestrated by TransfusionModel
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint


class ARTransformer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.dim_embedding = cfg.model.dim_embedding # dimension of embedding learned by transformer encoder and used as input to MLP
        self.dim_in = cfg.data.z # length of layer energy sequence

        self.encode_t_dim = cfg.model.encode_t_dim # dimension of the temporal encoding
        self.encode_t_scale = cfg.model.encode_t_scale

        self.transformer = nn.Transformer(
            d_model = self.dim_embedding,
            nhead = cfg.model.nhead,
            num_encoder_layers = cfg.model.num_encoder_layers,
            num_decoder_layers = cfg.model.num_decoder_layers,
            dim_feedforward = cfg.model.dim_feedforward,
            dropout = cfg.model.dropout,
            batch_first = True
        )

        self.t_embed = nn.Sequential(
            GaussianFourierProjection(self.encode_t_dim, self.encode_t_scale),
            nn.Linear(self.encode_t_dim, self.dim_embedding)
        )
        self.subnet = self.build_subnet()
        self.incidence_proj = nn.Linear(1, self.dim_embedding) # projection of incidence energy to embedding dimension for transformer encoder input
        if cfg.model.position_embedding == 'lookup':
            self.position_embedding = nn.Embedding(self.dim_in, self.dim_embedding-1) # learnable position embedding for transformer decoder input
        elif cfg.model.position_embedding == 'onehot':
            pass
        else:
            raise ValueError(f"Unsupported position embedding type: {cfg.model.position_embedding}")

    def build_subnet(self):
        """
        Dense MLP for velocity matching
        """
        self.intermediate_dim = self.cfg.model.intermediate_dim
        self.activation = nn.SiLU()
        self.layers_per_block = self.cfg.model.layers_per_block

        cond_dim = self.encode_t_dim + self.dim_embedding
        linear = nn.Linear(1+cond_dim, self.intermediate_dim)
        layers = [linear, self.activation]
        for _ in range(1, self.layers_per_block-1):
            layers.append(nn.Linear(self.intermediate_dim, self.intermediate_dim))
            layers.append(self.activation)
        layers.append(nn.Linear(self.intermediate_dim, 1))

        return nn.Sequential(*layers)

    def compute_embedding(
        self, p:torch.Tensor, dim:int
    ):
        """
        Embeds the target sequence 'p' using either one-hot or learned lookups.
        p must have shape (batch_size, dim_in, 1).       
        """
        if self.cfg.model.position_embedding == 'lookup':
            pos_embed = self.position_embedding(
                torch.arange(p.shape[1], device=p.device).unsqueeze(0).expand(p.shape[0], -1)
            ) # (batch_size, dim_in, dim_embedding-1)
            return torch.cat([pos_embed, p], dim=-1) # (batch_size, dim_in, dim_embedding)
        elif self.cfg.model.position_embedding == 'onehot':
            one_hot = torch.eye(dim, device=p.device, dtype=p.dtype)[
                None, : p.shape[1], :
            ].expand(p.shape[0], -1, -1)
            n_rest = self.dim_embedding - p.shape[-1] - dim
            assert n_rest >= 0, "Embedding dimension must be at least the sum of input dimension and position encoding dimension."
            zeros = torch.zeros((*p.shape[:2], n_rest), device=p.device, dtype=p.dtype)
            return torch.cat([p, one_hot, zeros], dim=-1)

    def sample_dimension(self, c:torch.Tensor):
        """
        Samples a dimension to predict based on the context from the transformer.
        """
        batch_size, dtype, device = c.shape[0], c.dtype, c.device

        net = self.subnet
        x_0 = torch.randn((batch_size, 1), device=device, dtype=dtype)
        
        def net_wrapper(t, x_t):
            t_torch = t * torch.ones_like(x_t[:, [0]], dtype=dtype, device=device)
            t_torch = self.t_embed(t_torch)
            v = net(torch.cat([x_t,t_torch.reshape(batch_size, -1), c.squeeze(1)], dim=-1))
            return v

        # Solve ODE from t=0 to t=1 using fixed-step RK4
        with torch.inference_mode():
            x_t = odeint(
                net_wrapper, 
                x_0, 
                torch.tensor([0.0, 1.0], dtype=dtype, device=device),
                method='rk4',
                options={'step_size': 1/50}
            )
        return x_t[1].unsqueeze(-1) # returns the predicted value at t=1, shape (batch_size, 1, 1)


                
    def forward(
        self, 
        incidence_energy: torch.Tensor, 
        x_t: torch.Tensor = None, 
        t: torch.Tensor = None, 
        x: torch.Tensor = None, 
        rev: bool = False):
        """
        Forward pass of the Transfusion model.

        Args:
            incidence_energy: (batch_size, 1, 1) tensor of incident energies. Always required.
            x_t, t, x: Required only if rev=False (training mode). Ignored if rev=True (inference mode).
            x_t: (batch_size, seq_len, 1) tensor of the current noisy state. 
            t: (batch_size, seq_len, 1) tensor of the continuous time step. 
            x: (batch_size, dim_in, 1) tensor of ground-truth layer-wise energies. 
            rev: Boolean flag dictating the operational mode. 
                 False for parallelized training (requires incident_energy, x_t, t, x).
                 True for autoregressive ODE inference (requires ONLY incident_energy).
        """        
        if not rev:
            xp = nn.functional.pad(x[:, :-1], (0, 0, 1, 0))
            embedding = self.transformer(
                src = self.incidence_proj(incidence_energy),
                tgt = self.compute_embedding(xp, dim=self.dim_in),
                tgt_mask = torch.ones(
                    (xp.shape[1], xp.shape[1]), device=x.device, dtype=torch.bool
                ).triu(diagonal=1)
            )
            t = self.t_embed(t)
            pred = self.subnet(torch.cat([x_t, t, embedding], dim=-1))

        else:
            x = torch.zeros((incidence_energy.shape[0], 1, 1), device=incidence_energy.device, dtype=incidence_energy.dtype)
            for i in range(self.dim_in):
                embedding = self.transformer(
                    src = self.incidence_proj(incidence_energy),
                    tgt = self.compute_embedding(x, dim=self.dim_in),
                    tgt_mask = torch.ones(
                        (x.shape[1], x.shape[1]), device=x.device, dtype=torch.bool
                    ).triu(diagonal=1)
                )
                x_new = self.sample_dimension(embedding[:, -1:, :])
                x = torch.cat([x, x_new], dim=1)

        
            pred = x[:, 1:].squeeze() #slice off 0.0 dummy initial token
        return pred





    


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps.
    Taken from CaloDREAM: https://github.com/luigifvr/calo_dreamer/blob/master/src/Networks/transformer.py#L193
    """

    def __init__(self, embed_dim, scale=30.):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x * self.W * 2 * torch.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
