"""
Separate Autoencoder conditioned on layer energies (u)
Inherits from AutoencoderSeparate
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.autoencoder.ae_separate import AutoEncoderSeparate
from model.encoder.encoderhierarchybase import HierarchicalEncoderLayers
from model.decoder.decoder_layers import DecoderLayers, DecoderLayersNoHits, DecoderLayersGated, DecoderLayersSparsity

from model.gumbel import GumbelMod, GumbelNoNoise
from CaloQuVAE import logging
logger = logging.getLogger(__name__)


class AutoencoderLayers(AutoEncoderSeparate):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.register_buffer('feature_min', torch.zeros(5))
        self.register_buffer('feature_max', torch.ones(5))

    def _create_encoder(self):
        if self._config.model.encoder == "hierarchicalencoderlayers":
            return HierarchicalEncoderLayers(self._config)
        else:
            raise ValueError(f"Unknown encoder type: {self._config.model.encoder}")
    
    def _create_decoder(self):
        logger.debug("::_create_decoder")
        if self._config.model.decoder == "decoderlayers":
            return DecoderLayers(self._config)
        # elif self._config.model.decoder == "decoderlayersupsample":
        #     return UpsamplingDecoderLayers(self._config)
        elif self._config.model.decoder == "decoderlayersnohits":
            return DecoderLayersNoHits(self._config)
        elif self._config.model.decoder == "decoderlayersgated":
            return DecoderLayersGated(self._config)
        elif self._config.model.decoder == "decoderlayerssparsity":
            return DecoderLayersSparsity(self._config)
        else:
            raise ValueError(f"Unknown decoder type: {self._config.model.decoder}")



    def decode(self, post_samples, x, x0, u, act_fct_slope=0.02, beta_hits=3.0):
        """
        Decodes through self.decoder
        Args:
            post_samples: binary samples from the latent space, shape (b, n_hierarchy, n_latent_per_p)
            x: shower voxels, shape (b, n_voxels)
            x0: incident energy, shape (b, 1)
            u: Layer energies, shape (b, n_l)
        """
        output_hits, output_activations = self.decoder(torch.cat(post_samples, dim=1), x0, u)
        if self.training:
            output_activations = self._activation_fct(act_fct_slope, output_activations) * torch.where(x > 0, 1., 0.)
        else:
            output_activations = self._activation_fct(0.0, output_activations) * self._hit_smoothing(output_hits)
        return output_hits, output_activations

    def forward(self, inputs, beta_latent=1.0, beta_hits=1.0, act_fct_slope=0.02):
        """
        Forward pass through the model
        Args:
            inputs: tuple of (x, x0, u)
                x: shower voxels, shape (b, n_voxels)
                x0: incident energy, shape (b, 1)
                u: Layer energies, shape (b, n_l)
            beta_latent: beta for gumbel mod in encoder
            beta_hits: beta for gumbel mod in decoder hits head
        """
        x, x0, u = inputs
        post_logits, post_samples = self.encoder(x, x0, u, beta_smoothing_fct=beta_latent)
        output_hits, output_activations = self.decode(post_samples, x, x0, u, act_fct_slope, beta_hits)
        return post_logits, post_samples, output_hits, output_activations

    def loss(self, x, output_hits, output_activations):
        """
        Computes loss between input and output
        Args:
            x: shower voxels, shape (b, n_voxels)
            x0: incident energy, shape (b, 1)
            u: Layer energies, shape (b, n_l)
            output_hits: predicted hits, shape (b, n_voxels)
            output_activations: predicted activations, shape (b, n_voxels)
        Returns:
            loss_dict: dictionary of loss components
        """
        squared_diff = torch.pow((x - output_activations), 2)
        arg = self._config.model.mse_weight * x
        energy_weighting = torch.exp(torch.clamp(arg, max=20))            
        ae_loss = squared_diff * energy_weighting
        ae_loss = torch.clamp(torch.mean(torch.sum(ae_loss, dim=1), dim=0) * self._config.model.coefficient, max=1e30)
        target_hits = (x > 0).float()
        bce_raw = F.binary_cross_entropy_with_logits(output_hits, target_hits, reduction='none')
        bce_loss = torch.mean(torch.sum(bce_raw, dim=1), dim=0)
        loss_dict = {
            "ae_loss": ae_loss,
            "hit_loss": bce_loss,
        }
        return loss_dict


def deterministic_sigmoid_ste(logits, beta=1.0, training=True):
    """
    Differentiable binary mask without stochastic noise.
    Forward pass: Hard threshold at 0.0 (logit) / 0.5 (prob).
    Backward pass: Gradients flow through beta-scaled Sigmoid.

    Uses the same inverse-temperature convention as GumbelMod:
    higher beta = sharper sigmoid.
    """
    if not training:
        return (logits > 0).float()
    soft_mask = torch.sigmoid(logits * beta)
    hard_mask = (soft_mask > 0.5).float()
    # Straight-Through Estimator: forward uses hard mask, backward uses soft mask gradient
    return hard_mask.detach() - soft_mask.detach() + soft_mask


def _add_feature_loss_breakdown(loss_dict, breakdown):
    """
    Adds per-(feature, layer) percentage contributions to loss_dict as plain floats.
    Keys follow the pattern  fl/<feat_key>/l<idx>  (e.g. fl/Eta_center/l0).
    The denominator is the sum of all unweighted per-layer losses, so the
    percentages reflect natural magnitudes rather than the weighted aggregation.
    """
    all_vals = torch.stack(list(breakdown.values()))  # (n_features, n_layers)
    total = all_vals.sum().item() + 1e-8
    for feat_key, per_layer in breakdown.items():
        for l_idx, val in enumerate(per_layer):
            loss_dict[f"fl/{feat_key}/l{l_idx}"] = val.item() / total * 100.0


class AutoencoderLayersBCE(AutoencoderLayers):
    """
    Two-head AE: hits head trained with focal BCE, activation head and physics loss
    trained on the combined hits * activation-fracs shower.  The hit mask uses the
    Gumbel trick annealed with the same beta as the encoder latent space — smooth
    sigmoid during training, hard Heaviside at inference.  No teacher forcing.
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self.hit_gumbel = GumbelNoNoise()

    def decode(self, post_samples, x, x0, u, act_fct_slope=0.02, beta=3.0):
        """
        Decodes through self.decoder.  The hit mask is produced by GumbelMod so it
        is a smooth sigmoid during training and a hard binary sample at inference —
        no teacher forcing, no separate training/inference branch.

        Returns output_hits (raw logits) and output_shower (Gumbel_mask × fracs).
        """
        self._beta = beta
        output_hits, output_logits = self.decoder(torch.cat(post_samples, dim=1), x0, u)

        B = x.shape[0]
        z = self._config.data.z
        logits_3d = output_logits.view(B, z, -1)

        # Gumbel hit mask: smooth (sigmoid) during training, hard (Heaviside) at inference
        gumbel_mask = self.hit_gumbel(output_hits, beta=beta)   # (B, n_voxels)
        gumbel_mask_3d = gumbel_mask.view(B, z, -1)

        # Per-layer sigmoid normalization gated by the Gumbel hit mask
        sigs_3d = torch.sigmoid(logits_3d) * gumbel_mask_3d
        fracs_3d = sigs_3d / (sigs_3d.sum(dim=-1, keepdim=True) + 1e-8)
        output_shower = fracs_3d.view(B, -1)   # hits × activation fracs

        return output_hits, output_shower

    def forward(self, inputs, beta_latent=1.0, beta_hits=1.0, act_fct_slope=0.02):
        """
        Forward pass through the model.
        beta_latent is shared by the encoder latent space and the hits Gumbel mask.
        beta_hits is kept for API compatibility but unused.
        Args:
            inputs: tuple of (x, x0, u)
                x: shower voxels, shape (b, n_voxels)
                x0: incident energy, shape (b, 1)
                u: Layer energies, shape (b, n_l)
            beta_latent: annealing beta for encoder Gumbel and hits Gumbel mask
        """
        x, x0, u = inputs
        post_logits, post_samples = self.encoder(x, x0, u, beta_smoothing_fct=beta_latent)
        output_hits, output_activations = self.decode(post_samples, x, x0, u, act_fct_slope, beta_latent)
        return post_logits, post_samples, output_hits, output_activations

    def mutual_information_penalty(self, post_logits, eps=1e-8):
        """
        Calculates the Mutual Information penalty for binary latent nodes.
        post_logits: Tensor of shape [batch_size, num_latent_nodes]
        """
        probs = torch.sigmoid(post_logits)
        
        #  Calculate Conditional Entropy H(Z|X)
        # We want to MINIMIZE this (push probs towards 0 or 1 for each sample)
        # Shape: [batch_size, num_latent_nodes] -> reduce to scalar
        h_z_given_x = - (probs * torch.log(probs + eps) + (1 - probs) * torch.log(1 - probs + eps))
        expected_h_z_given_x = h_z_given_x.sum(dim=-1).mean() 
        
        #  Calculate Marginal Entropy H(Z)
        # We want to MAXIMIZE this (push average activation across batch towards 0.5)
        # Shape: [num_latent_nodes]
        p_marginal = probs.mean(dim=0)
        h_z = - (p_marginal * torch.log(p_marginal + eps) + (1 - p_marginal) * torch.log(1 - p_marginal + eps))
        expected_h_z = h_z.sum()
        
        #  Mutual Information Penalty
        # We want to maximize MI, so we minimize negative MI: H(Z|X) - H(Z)
        mi_loss = expected_h_z_given_x - expected_h_z
        
        return mi_loss

    def geom_loss(self, x, output_hits):
        """
        Geometric loss that only applies to the hits head
        MAE loss that punishes deviations in sparsity and shower moments
        """
        target_hits = (x > 0).float()
        model_hits = torch.sigmoid(output_hits)

        feat_gt = self.feature_extractor(target_hits)
        feat_pred = self.feature_extractor(model_hits)

        geom_loss_total = 0.0
        for key in feat_gt.keys():
            val_gt = feat_gt[key].view(x.size(0), -1)
            val_pred = feat_pred[key].view(x.size(0), -1)
            
            scale = getattr(self, f"geom_{key}")
            safe_scale = torch.clamp(scale, min=1e-6)
            
            # Calculate scaled MAE
            loss_components = torch.abs(val_gt - val_pred) / safe_scale
            geom_loss_total += loss_components.mean()    
        return geom_loss_total

    def _get_normalized_layer_weights(self, device: torch.device, attr: str = "layer_weights") -> torch.Tensor:
        """
        Fetches layer weights from the config and normalizes them to sum to 1.
        """
        weights = getattr(self._config.model, attr, None)
        if not weights:
            return torch.ones(self._config.data.z, device=device) / self._config.data.z  # Default to equal weights if not specified

        w = torch.tensor(weights, dtype=torch.float32, device=device)
        return w / torch.sum(w)
    
    def compute_physics_loss(self, x, physics_recon, layer_weights=None):
        """
        layer_weights: Tensor of shape (Layers,) normalized to sum to 1.

        Returns (physics_loss, breakdown) where breakdown is a dict mapping
        feature key -> per-layer mean loss tensor of shape (n_layers,), computed
        *before* layer weighting so the proportions reflect natural magnitudes.
        """
        delta = getattr(self._config.model.loss_coeff, "feature_huber_delta", 1.0)
        alpha = getattr(self._config.model, "asym_alpha", 2.0)
        asym_centre_only = getattr(self._config.model, "asym_centre_only", 0)
        feat_gt = self.feature_extractor(x)
        feat_recon = self.feature_extractor(physics_recon)

        physics_loss = torch.tensor(0.0, device=x.device)
        breakdown = {}  # feat_key -> (n_layers,) unweighted per-layer mean loss

        # Pre-define quantiles for IQR (25th and 75th percentiles)
        q = torch.tensor([0.25, 0.75], dtype=torch.float32, device=x.device)

        for key in feat_gt.keys():
            if "E_" in key:
                continue

            val_gt = feat_gt[key].view(x.size(0), -1)
            val_recon = feat_recon[key].view(x.size(0), -1)

            # Calculate Q1 and Q3 across the batch (dim=0)
            quantiles = torch.quantile(val_gt, q, dim=0)
            iqr = quantiles[1] - quantiles[0]

            # Detach and clamp to prevent zero-division if a feature is completely static in a batch
            layer_spread = torch.clamp(iqr.detach(), min=1e-5)

            # Scale features by the IQR before calculating loss
            val_gt_scaled = val_gt / layer_spread
            val_recon_scaled = val_recon / layer_spread

            # Calculate base error on the scaled (relative) features
            base_loss = F.huber_loss(val_recon_scaled, val_gt_scaled, reduction='none', delta=delta)

            # Apply per-feature scalar weight (e.g. upweight widths/centres)
            feature_weights = getattr(self._config.model, "feature_weights", {})
            feat_scalar = next((w for pat, w in feature_weights.items() if pat in key), 1.0)
            base_loss = base_loss * feat_scalar

            # Apply Asymmetry Penalty
            apply_asym = alpha != 1.0 and (not asym_centre_only or "center" in key)
            if apply_asym:
                underpredict_mask = (torch.abs(val_recon_scaled) < torch.abs(val_gt_scaled)).float()
                weights = 1.0 + (alpha - 1.0) * underpredict_mask
                base_loss = base_loss * weights

            # Aggregate and track weighted per-layer mean for breakdown
            if layer_weights is not None:
                lw = layer_weights.view(1, -1)
                weighted_loss = base_loss * lw  # (batch, n_layers)
                layer_weighted_loss = torch.sum(weighted_loss, dim=1)
                physics_loss += layer_weighted_loss.mean()
                breakdown[key] = weighted_loss.mean(dim=0).detach()
            else:
                physics_loss += base_loss.mean()
                breakdown[key] = base_loss.mean(dim=0).detach()

        return physics_loss, breakdown


    def loss(self, x, output_hits, output_activations, post_logits=None):
        """
        Computes focal BCE for the hits head and CE on the combined hits × activation
        fracs shower (output_activations) for the activation head.  Physics feature
        loss is applied directly to output_activations (already hits × fracs).

        x is expected to be layer fractions (output of engine._reduceBCE).
        """
        num_layers = self._config.data.z
        raw_layer_weights = self._get_normalized_layer_weights(x.device)
        layer_weights = raw_layer_weights.view(1, num_layers, 1)

        target_hits = (x > 0).float()

        # Hits Head Loss (Focal BCE)
        bce_raw = F.binary_cross_entropy_with_logits(output_hits, target_hits, reduction='none')
        alpha = self._config.model.loss_coeff.focal_alpha
        gamma = self._config.model.loss_coeff.focal_gamma
        pt = torch.exp(-bce_raw)
        alpha_t = alpha * target_hits + (1 - alpha) * (1 - target_hits)
        focal_loss = alpha_t * (1 - pt).pow(gamma) * bce_raw
        focal_loss = focal_loss.view(x.size(0), num_layers, -1) * layer_weights
        hit_loss = torch.mean(torch.sum(focal_loss, dim=(1, 2)), dim=0)

        # Activation Head Loss: CE on the combined hits × fracs shower
        p = output_activations.clamp(1e-7, 1 - 1e-7)
        bce_act = -x * torch.log(p) - (1 - x) * torch.log(1 - p)
        bce_act = bce_act.view(x.size(0), num_layers, -1) * layer_weights
        ae_loss = torch.mean(torch.sum(bce_act, dim=(1, 2)), dim=0)

        loss_dict = {
            "ae_loss": ae_loss,
            "hit_loss": hit_loss,
        }

        latent_mi_coeff = getattr(self._config.model.loss_coeff, "latent_mi_loss", 0.0)
        if post_logits is not None and latent_mi_coeff > 0:
            mi_loss = self.mutual_information_penalty(post_logits)
            loss_dict["latent_mi_loss"] = mi_loss

        if hasattr(self._config.model.loss_coeff, "geom_loss") and self._config.model.loss_coeff.geom_loss > 0:
            geom_loss = self.geom_loss(x, output_hits)
            loss_dict["geom_loss"] = geom_loss

        feature_loss_weight = getattr(self._config.model.loss_coeff, "feature_loss", 0.0)
        if feature_loss_weight > 0.0:
            # output_activations is already hits × fracs from decode()
            feature_layer_weights = self._get_normalized_layer_weights(
                x.device, attr="feature_layer_weights"
            )
            feature_loss_val, feat_breakdown = self.compute_physics_loss(x, output_activations, layer_weights=feature_layer_weights)
            loss_dict["feature_loss"] = feature_loss_val
            _add_feature_loss_breakdown(loss_dict, feat_breakdown)

        return loss_dict

class AutoencoderLayersNoHits(AutoencoderLayersBCE):
    """
    Removes separate hits head and teacher-forcing masking.
    Model only predicts activation proportions trained via Binary Cross-Entropy (BCE).
    """
    def decode(self, post_samples, x, x0, u, act_fct_slope=0.02, beta_hits=3.0):
        # Decode to get raw activations
        # We ignore output_hits from the decoder entirely
        _, output_logits = self.decoder(torch.cat(post_samples, dim=1), x0, u)
        
        B = x.shape[0]
        z = self._config.data.z
        
        logits_3d = output_logits.view(B, z, -1)
        
        # Apply normalized sigmoid PER LAYER for inference energy conservation
        sigs_3d = torch.sigmoid(logits_3d)
        fracs_3d = sigs_3d / (torch.sum(sigs_3d, dim=-1, keepdim=True) + 1e-8)
        
        # Flatten back to (Batch, Total_Voxels) to match expected pipeline shapes
        output_activations_frac = fracs_3d.view(B, -1)

        # Note: We return output_logits in the first position. 
        # The base `forward` method will mistakenly label this `output_hits`, which we subsequently catch in our overridden `loss` method.
        return output_logits, output_activations_frac
    

    def loss(self, x, output_logits, output_activations, post_logits=None):
        """
        Computes BCE for the continuous fractions.
        Note: `output_logits` here is passed from the `output_hits` slot in `forward`.
        """
        num_layers = self._config.data.z
        
        raw_layer_weights = self._get_normalized_layer_weights(x.device)
        
        layer_weights_bce = raw_layer_weights.view(1, num_layers, 1)

        # BCE with logits treats each voxel as an independent Bernoulli trial
        bce = F.binary_cross_entropy_with_logits(output_logits, x, reduction='none')
        bce = bce.view(x.size(0), num_layers, -1)
        bce = bce * layer_weights_bce  
        ae_loss = torch.mean(torch.sum(bce, dim=(1, 2)), dim=0)
        
        loss_dict = {
            "ae_loss": ae_loss,
        }
        
        latent_mi_coeff = getattr(self._config.model.loss_coeff, "latent_mi_loss", 0.0)
        if post_logits is not None and latent_mi_coeff > 0:
            mi_loss = self.mutual_information_penalty(post_logits)
            loss_dict["latent_mi_loss"] = mi_loss
        
        feature_loss_weight = getattr(self._config.model.loss_coeff, "feature_loss", 0.0)
        if feature_loss_weight > 0.0:
            physics_recon = torch.sigmoid(output_logits)

            feature_layer_weights = self._get_normalized_layer_weights(
                x.device, attr="feature_layer_weights"
            )
            feature_loss_val, feat_breakdown = self.compute_physics_loss(
                x,
                physics_recon,
                layer_weights=feature_layer_weights
            )
            loss_dict["feature_loss"] = feature_loss_val
            _add_feature_loss_breakdown(loss_dict, feat_breakdown)
        return loss_dict


class AutoencoderLayersSparsity(AutoencoderLayersNoHits):
    """
    No hits head; sparsity is enforced by a per-layer top-k mask whose cutoff
    comes from a small sparsity head on the decoder.

    Training: mask uses the *true* per-layer active count from x (teacher forced).
    Inference: mask uses the predicted sparsity from the sparsity head.

    Activation head is supervised by CE on the post-mask normalized fractions;
    sparsity head is supervised independently by BCE against the true per-layer
    active fraction. Feature loss is computed on the post-mask shower so it sees
    the realistic sparsity pattern at training time.

    Decoder forward returns (hits_placeholder, output_logits, sparsity_logits);
    we stash sparsity_logits on self so the engine's existing 4-tuple unpacking
    and (x, output[2], output[3], post_logits=...) loss signature still apply.
    """
    def __init__(self, cfg):
        super().__init__(cfg)
        self._last_sparsity_logits = None

    def _topk_mask(self, logits_3d, k_per_layer):
        """
        Per-(batch, layer) top-k mask. Forward is hard binary; backward uses a
        sigmoid relaxation so masked-out voxels still receive gradient.

        logits_3d: (B, z, V)
        k_per_layer: (B, z) integer counts in [0, V]
        """
        B, z, V = logits_3d.shape
        sorted_logits, _ = logits_3d.sort(dim=-1, descending=True)
        # k=0 is handled by the zero_layer multiplier below; clamp here just to
        # keep the gather index valid.
        k_safe = k_per_layer.clamp(min=1, max=V).long()
        idx = (k_safe - 1).unsqueeze(-1)
        threshold = sorted_logits.gather(-1, idx).detach()  # sort is non-diff

        hard_mask = (logits_3d >= threshold).float()
        zero_layer = (k_per_layer == 0).unsqueeze(-1).float()
        hard_mask = hard_mask * (1.0 - zero_layer)

        if not getattr(self._config.model, "topk_use_ste", True):
            return hard_mask

        beta = float(getattr(self._config.model, "topk_ste_beta", 1.0))
        soft_mask = torch.sigmoid((logits_3d - threshold) * beta) * (1.0 - zero_layer)
        return hard_mask.detach() - soft_mask.detach() + soft_mask

    def decode(self, post_samples, x, x0, u, act_fct_slope=0.02, beta_hits=3.0):
        _, output_logits, sparsity_logits = self.decoder(torch.cat(post_samples, dim=1), x0, u)
        self._last_sparsity_logits = sparsity_logits

        B = output_logits.shape[0]
        z = self._config.data.z
        V = self._config.data.r * self._config.data.phi

        logits_3d = output_logits.view(B, z, V)

        if self.training:
            # Teacher force the cutoff with the true per-layer active count.
            target_active_3d = (x.view(B, z, V) > 0).float()
            k_per_layer = target_active_3d.sum(dim=-1).round().long()
        else:
            sparsity_frac = torch.sigmoid(sparsity_logits)  # (B, z)
            k_per_layer = (sparsity_frac * V).round().long()
        k_per_layer = k_per_layer.clamp(min=0, max=V)

        mask_3d = self._topk_mask(logits_3d, k_per_layer)

        sigs_3d = torch.sigmoid(logits_3d) * mask_3d
        fracs_3d = sigs_3d / (sigs_3d.sum(dim=-1, keepdim=True) + 1e-8)
        output_activations = fracs_3d.view(B, -1)

        return output_logits, output_activations

    def loss(self, x, output_logits, output_activations, post_logits=None):
        num_layers = self._config.data.z
        B = x.shape[0]
        
        raw_layer_weights = self._get_normalized_layer_weights(x.device)
        layer_weights = raw_layer_weights.view(1, num_layers, 1)

        bce = F.binary_cross_entropy_with_logits(output_logits, x, reduction='none')
        bce = bce.view(B, num_layers, -1)
        bce = bce * layer_weights
        ae_loss = torch.mean(torch.sum(bce, dim=(1, 2)), dim=0)

        # BCE against true per-layer active fraction in [0, 1]
        sparsity_logits = self._last_sparsity_logits
        target_sparsity = (x.view(B, num_layers, -1) > 0).float().mean(dim=-1)
        sparsity_loss = F.binary_cross_entropy_with_logits(
            sparsity_logits, target_sparsity, reduction='mean'
        )

        loss_dict = {
            "ae_loss": ae_loss,
            "sparsity_loss": sparsity_loss,
        }

        latent_mi_coeff = getattr(self._config.model.loss_coeff, "latent_mi_loss", 0.0)
        if post_logits is not None and latent_mi_coeff > 0:
            loss_dict["latent_mi_loss"] = self.mutual_information_penalty(post_logits)

        feature_loss_weight = getattr(self._config.model.loss_coeff, "feature_loss", 0.0)
        if feature_loss_weight > 0.0:
            # output_activations is the hard-masked, energy-conserving tensor from decode().
            # Because ae_loss stabilizes the logit sorting, the hard mask selects the correct voxels here.
            feature_layer_weights = self._get_normalized_layer_weights(
                x.device, attr="feature_layer_weights"
            )
            feature_loss_val, feat_breakdown = self.compute_physics_loss(
                x, output_activations, layer_weights=feature_layer_weights
            )
            loss_dict["feature_loss"] = feature_loss_val
            _add_feature_loss_breakdown(loss_dict, feat_breakdown)

        return loss_dict