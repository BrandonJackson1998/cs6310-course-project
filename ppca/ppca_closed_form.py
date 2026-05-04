"""Probabilistic PCA — maximum likelihood closed form (eigenproblem).

Reference: Tipping & Bishop (1999), JRSS B.
Covariance model C = W @ W.T + sigma^2 I with q latent dimensions.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PPCAClosedForm:
    mean_: np.ndarray  # (D,)
    W_: np.ndarray  # (D, q)
    sigma2_: float
    eigvals_: np.ndarray  # eigenvalues of sample covariance S, largest-first (length D)

    def covariance(self) -> np.ndarray:
        D = self.W_.shape[0]
        return self.W_ @ self.W_.T + self.sigma2_ * np.eye(D)

    def transform_latent_mean(self, X: np.ndarray) -> np.ndarray:
        """Posterior mean E[z | x] under fitted model (same linear map as noisy PPCA)."""
        X = np.asarray(X, dtype=float)
        Xc = X - self.mean_
        M = self.sigma2_ * np.eye(self.W_.shape[1]) + self.W_.T @ self.W_
        qw = Xc @ self.W_
        return np.linalg.solve(M, qw.T).T

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Posterior mean of x given observation (Wiener-style reconstruction)."""
        Z = self.transform_latent_mean(X)
        return Z @ self.W_.T + self.mean_


def fit_ppca_closed_form(X: np.ndarray, q: int) -> PPCAClosedForm:
    """MLE via eigendecomposition of sample covariance."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    N, D = X.shape
    if q <= 0 or q >= D:
        raise ValueError("need 0 < q < D")

    mu = X.mean(axis=0)
    Xc = X - mu
    S = (Xc.T @ Xc) / N

    eigvals, eigvecs = np.linalg.eigh(S)
    # ascending → descending
    idx = np.argsort(eigvals)[::-1]
    lam = eigvals[idx].astype(float)
    U = eigvecs[:, idx]

    sigma2 = float(np.mean(lam[q:]))
    lam_q = lam[:q]
    diag_shift = np.maximum(lam_q - sigma2, 1e-12)
    scales = np.sqrt(diag_shift)
    W = U[:, :q] * scales[np.newaxis, :]

    return PPCAClosedForm(mean_=mu, W_=W, sigma2_=sigma2, eigvals_=lam.copy())

