import torch
import os
import numpy as np
from torch.utils.data import Dataset

class LatentDataset(Dataset):
    """
    A custom Dataset to load pre-generated latent samples.
    It infers the label file path from the data file path
    by replacing 'train_data' with 'train_labels'.

    Optionally computes per-sample importance weights from a pre-saved features
    file (e.g. eta_layer1 values) using inverse-density power-law scaling on
    E-conditioned z-scores.
    """
    def __init__(self, data_path, transform=None,
                 features_paths=None, weight_alpha=2.0, weight_max=500.0,
                 tail_threshold=1e-4):
        """
        Args:
            data_path (string): Path to the .pt file with data.
            transform (callable, optional): Optional transform applied to each sample.
            features_paths (string or list of strings, optional): Path(s) to .pt file(s)
                with per-sample features used to compute importance weights (e.g. eta_layer1).
                When multiple paths are given, per-feature weights are combined by taking
                the element-wise maximum across features.  If None, all weights are 1.
            weight_alpha (float): Log-linear slope for tail weighting. Weight at survival
                probability p (for p <= tail_threshold) is 1 + alpha * log(tail_threshold / p).
            weight_max (float): Hard cap on raw (pre-normalisation) weights.
            tail_threshold (float): Survival-probability threshold below which a sample is
                considered a tail and receives an importance weight > 1.  Samples whose
                survival probability exceeds this value are assigned weight = 1.
        """
        self.transform = transform
        self.data_path = data_path

        # --- Infer label path from data path ---
        try:
            self.label_path = self.data_path.replace("train_data", "train_labels")
            if self.label_path == self.data_path:
                raise ValueError("Filename convention error: 'train_data' not found in path.")
            if not os.path.exists(self.label_path):
                raise FileNotFoundError(f"Inferred label file not found at: {self.label_path}")
        except Exception as e:
            print(f"Error inferring label path from data path: {self.data_path}")
            print(f"Error details: {e}")
            raise

        # --- Load data and targets ---
        try:
            self.data = torch.load(self.data_path)
            self.targets = torch.load(self.label_path)
        except FileNotFoundError as e:
            print(f"Error loading files.")
            print(f"Data path: {self.data_path}")
            print(f"Inferred Label path: {self.label_path}")
            raise

        print(f"Successfully loaded data from {self.data_path}")
        print(f"Successfully inferred and loaded labels from {self.label_path}")
        print(f"Data shape: {self.data.shape}, Labels shape: {self.targets.shape}")

        # --- Compute importance weights ---
        if features_paths is not None:
            # Normalise to a list
            if isinstance(features_paths, str):
                features_paths = [features_paths]

            per_feature_weights = []
            for fp in features_paths:
                features = torch.load(fp)
                print(f"Loaded features from {fp}, shape: {features.shape}")
                w = self._compute_importance_weights(
                    features, self.targets, weight_alpha, weight_max, tail_threshold
                )
                per_feature_weights.append(w)

            # Combine by element-wise max across features
            self.weights = torch.stack(per_feature_weights, dim=0).max(dim=0).values
            print(
                f"Importance weights (combined max over {len(features_paths)} feature(s)) — "
                f"min: {self.weights.min():.4f}, "
                f"max: {self.weights.max():.4f}, mean: {self.weights.mean():.4f}"
            )
        else:
            self.weights = torch.ones(len(self.data), dtype=torch.float32)

    def _compute_importance_weights(
        self,
        features: torch.Tensor,
        incidence_energies: torch.Tensor,
        alpha: float,
        w_max: float,
        tail_threshold: float = 1e-4,
    ) -> torch.Tensor:
        """
        Compute mean-normalised importance weights via log-linear scaling of
        E-conditioned survival probabilities.

        Steps:
          1. Fit linear mu(E) and sigma^2(E) with polyfit.
          2. Compute absolute z-scores.
          3. Map each z-score to its empirical survival probability P(Z > z).
          4. If P(Z > z) > tail_threshold, weight = 1 (sample is not a tail).
             Otherwise weight = 1 + alpha * log(tail_threshold / P(Z > z)),
             clipped to [1, w_max].
          5. Mean-normalise the resulting weight vector.
        """
        y = features.squeeze().numpy().astype(np.float64)  # (N,)
        E = incidence_energies.flatten().numpy().astype(np.float64)  # (N,)

        # Linear fit of conditional mean and variance
        coeffs_mu = np.polyfit(E, y, 1)
        mu_local = np.polyval(coeffs_mu, E)

        residuals_sq = (y - mu_local) ** 2
        coeffs_var = np.polyfit(E, residuals_sq, 1)
        var_local = np.clip(np.polyval(coeffs_var, E), 1e-6, None)
        sig_local = np.sqrt(var_local)

        z_scores = np.abs(y - mu_local) / sig_local

        # Empirical survival function (CCDF)
        sorted_z = np.sort(z_scores)
        p_greater = 1.0 - np.arange(1, len(sorted_z) + 1) / len(sorted_z)

        lookup_idx = np.searchsorted(sorted_z, z_scores)
        lookup_idx = np.clip(lookup_idx, 0, len(sorted_z) - 1)
        sample_probs = p_greater[lookup_idx]

        # Apply threshold: only upweight genuine tails (survival prob <= tail_threshold).
        # Log-linear form: weight = 1 + alpha * log(tail_threshold / sample_probs).
        # Continuous at the boundary (=1 when sample_probs == tail_threshold) and grows
        # only logarithmically into the deep tail, so extreme outliers don't dominate
        # the way a power-law would.
        is_tail = sample_probs <= tail_threshold
        log_excess = np.log(np.maximum(tail_threshold / np.maximum(sample_probs, 1e-300), 1.0))
        raw_weights = np.where(is_tail, 1.0 + alpha * log_excess, 1.0)
        sample_weights = np.clip(raw_weights, 1.0, w_max)
        normalised = sample_weights / sample_weights.mean()

        n_tail = int(is_tail.sum())
        print(f"  tail_threshold={tail_threshold}: {n_tail}/{len(y)} samples "
              f"({100*n_tail/len(y):.3f}%) identified as tails.")

        return torch.tensor(normalised, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Returns a tuple (data, target, idx, weight).
        """
        sample = self.data[idx]
        target = self.targets[idx]
        weight = self.weights[idx]

        if self.transform:
            sample = self.transform(sample)

        return sample, target, idx, weight
