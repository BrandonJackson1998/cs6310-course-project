"""Baseline PCA via SVD on centered data (numpy only)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PCABaseline:
    """PCA with q principal components; reconstruction X_hat = Z @ V.T + mu."""

    components_: np.ndarray  # shape (q, D)
    mean_: np.ndarray  # shape (D,)
    singular_values_: np.ndarray  # shape (q,)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project centered data onto PCs (scores)."""
        Xc = X - self.mean_
        return Xc @ self.components_.T

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Reconstruct from scores."""
        return Z @ self.components_ + self.mean_

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """PCA reconstruction of rows of X."""
        return self.inverse_transform(self.transform(X))


def fit_pca(X: np.ndarray, q: int) -> PCABaseline:
    """Fit PCA using thin SVD on centered X (N x D)."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    N, D = X.shape
    if q <= 0 or q >= min(N, D):
        raise ValueError("q must be in [1, min(N,D)-1]")
    mu = X.mean(axis=0)
    Xc = X - mu
    # Full SVD: Xc = U S Vt
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    q_eff = min(q, len(S))
    components = Vt[:q_eff]
    singular_values = S[:q_eff]
    return PCABaseline(
        components_=components,
        mean_=mu,
        singular_values_=singular_values,
    )

