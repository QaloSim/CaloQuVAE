"""
Separate Autoencoder conditioned on layer energies (u)
Inherits from AutoencoderSeparate
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.autoencoder.ae_separate import AutoEncoderSeparate
from model.encoder.encoderhierarchybase import HierarchicalEncoderLayers
from model.decoder.decoder_layers import DecoderLayers, DecoderLayersNoHits, DecoderLayersGated

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


class AutoencoderLayersBCE(AutoencoderLayers):
    """
    Two-head AE: hits head trained with focal BCE, activation head trained with
    teacher-forced BCE-with-logits. Physics feature loss uses an STE hit mask
    multiplied by the activation fracs.
    """
    def _hit_smoothing(self, output_hits, beta=1.0):
        return deterministic_sigmoid_ste(output_hits, beta=beta, training=self.training)

    def decode(self, post_samples, x, x0, u, act_fct_slope=0.02, beta_hits=3.0):
        """
        Decodes through self.decoder. Activation fracs are computed via sigmoid normalization per layer, masked by teacher-forced hits during training or
        STE hit predictions during inference.

        Stashes beta_hits for use in loss() feature_loss path.
        """
        self._beta_hits = beta_hits
        output_hits, output_logits = self.decoder(torch.cat(post_samples, dim=1), x0, u)

        B = x.shape[0]
        z = self._config.data.z
        logits_3d = output_logits.view(B, z, -1)

        if self.training:
            # Teacher forcing: mask using true hit positions
            mask_3d = (x > 0).view(B, z, -1).float()
        else:
            # Inference: hard STE mask from hits head predictions
            mask_3d = self._hit_smoothing(output_hits, beta=beta_hits).view(B, z, -1)

        # Sigmoid-normalize per layer, zeroing non-hit positions
        sigs_3d = torch.sigmoid(logits_3d) * mask_3d
        fracs_3d = sigs_3d / (sigs_3d.sum(dim=-1, keepdim=True) + 1e-8)
        output_activations_frac = fracs_3d.view(B, -1)

        return output_hits, output_activations_frac

    def forward(self, inputs, beta_latent=1.0, beta_hits=1.0, act_fct_slope=0.02):
        """
        Forward pass through the model
        Args:
            inputs: tuple of (x, x0, u)
                x: shower voxels, shape (b, n_voxels)
                x0: incident energy, shape (b, 1)
                u: Layer energies, shape (b, n_l)
            beta_latent: beta for gumbel mod in encoder
            beta_hits: beta for STE hit mask (inverse temperature)
        """
        x, x0, u = inputs
        post_logits, post_samples = self.encoder(x, x0, u, beta_smoothing_fct=beta_latent)
        output_hits, output_activations = self.decode(post_samples, x, x0, u, act_fct_slope, beta_hits)
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

    def _get_normalized_layer_weights(self, device: torch.device) -> torch.Tensor:
        """
        Fetches layer weights from the config and normalizes them to sum to 1.
        """
        weights = getattr(self._config.model, "layer_weights", None)
        if not weights:
            return torch.ones(self._config.data.z, device=device) / self._config.data.z  # Default to equal weights if not specified
        
        w = torch.tensor(weights, dtype=torch.float32, device=device)
        return w / torch.sum(w)
    
    def compute_physics_loss(self, x, physics_recon, layer_weights=None, delta=1.0):
        """
        layer_weights: Tensor of shape (Layers,) normalized to sum to 1.
        delta: Huber loss threshold. Errors < delta are squared (MSE), > delta are linear (L1).
        """
        alpha = getattr(self._config.model, "asym_alpha", 2.0)
        asym_centre_only = getattr(self._config.model, "asym_centre_only", 0)
        feat_gt = self.feature_extractor(x)
        feat_recon = self.feature_extractor(physics_recon)

        
        physics_loss = torch.tensor(0.0, device=x.device)
        
        for key in feat_gt.keys():
            if "E_" in key:
                continue
                
            # Shape: (Batch, Layers)
            val_gt = feat_gt[key].view(x.size(0), -1)
            val_recon = feat_recon[key].view(x.size(0), -1)
            
            # Calculate base error in linear space
            # Huber loss protects against high-variance outliers (e.g. low E_inc noise)
            base_loss = F.huber_loss(val_recon, val_gt, reduction='none', delta=delta)
            
            # Apply Asymmetry Penalty in linear space
            apply_asym = alpha != 1.0 and (not asym_centre_only or "center" in key)
            if apply_asym:
                underpredict_mask = (torch.abs(val_recon) < torch.abs(val_gt)).float()
                weights = 1.0 + (alpha - 1.0) * underpredict_mask
                base_loss = base_loss * weights
            
            # Shape: (1, Layers) - Detached so it doesn't affect gradients
            # We scale by the mean absolute value of the layer to normalize gradient magnitudes
            layer_scale = torch.clamp(torch.mean(torch.abs(val_gt), dim=0, keepdim=True).detach(), min=1e-5)
            
            # Normalize the loss per layer
            scaled_loss = base_loss / layer_scale
            
            # 4. Aggregate
            if layer_weights is not None:
                lw = layer_weights.view(1, -1)
                layer_weighted_loss = torch.sum(scaled_loss * lw, dim=1)
                physics_loss += layer_weighted_loss.mean()
            else:
                physics_loss += scaled_loss.mean()
            
        return physics_loss



    def loss(self, x, output_hits, output_activations, post_logits=None):
        """
        Computes focal BCE for the hits head and teacher-forced BCE-with-logits for
        the activation head. Physics feature loss uses an STE hit mask applied to
        output_activations.

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

        # Activation Head Loss (BCE on fracs)
        # decode() already zeroes non-hit fracs via teacher-forcing, so no mask needed here.
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
            # STE mask provides differentiable binary gating: hard forward, soft backward
            beta = getattr(self, '_beta_hits', 1.0)
            ste_mask = self._hit_smoothing(output_hits, beta=beta)
            physics_recon = ste_mask * output_activations
            feature_loss_val = self.compute_physics_loss(x, physics_recon, layer_weights=raw_layer_weights)
            loss_dict["feature_loss"] = feature_loss_val

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
            
            # Pass the 1D raw weights to the physics loss
            feature_loss_val = self.compute_physics_loss(
                x, 
                physics_recon, 
                layer_weights=raw_layer_weights
            )
            loss_dict["feature_loss"] = feature_loss_val
        return loss_dict