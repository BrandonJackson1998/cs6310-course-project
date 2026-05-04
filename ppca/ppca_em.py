"""Probabilistic PCA — EM algorithm (numpy only)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PPCAEM:
    mean_: np.ndarray  # (D,)
    W_: np.ndarray  # (D, q)
    sigma2_: float
    n_iter_: int
    converged_: bool

    def transform_latent_mean(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        Xc = X - self.mean_
        M = self.sigma2_ * np.eye(self.W_.shape[1]) + self.W_.T @ self.W_
        qw = Xc @ self.W_
        return np.linalg.solve(M, qw.T).T

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        Z = self.transform_latent_mean(X)
        return Z @ self.W_.T + self.mean_


def fit_ppca_em(
    X: np.ndarray,
    q: int,
    *,
    max_iter: int = 300,
    tol: float = 1e-5,
    random_state: int | None = 42,
) -> PPCAEM:
    """Fit PPCA by EM; Gaussian latent prior z ~ N(0,I), noise eps ~ N(0, sigma^2 I)."""
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    N, D = X.shape
    if q <= 0 or q >= D:
        raise ValueError("need 0 < q < D")

    mu = X.mean(axis=0)
    Xc = X - mu

    # Smart init: rank-q PCA directions scaled by variance spectrum (still numpy-only).
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    q_eff = min(q, len(S))
    lam = (S[:q_eff] ** 2) / N  # eigenvalues along leading PCs (same scaling as covariance spectrum)
    if q_eff < len(S):
        sigma2 = float(np.mean((S[q_eff:] ** 2) / N))
    else:
        sigma2 = float(max(np.mean(lam) * 0.01, 1e-8))
    sigma2 = max(sigma2, 1e-8)
    scales = np.sqrt(np.maximum(lam[:q_eff] - sigma2, 1e-12))
    W = (Vt[:q_eff].T * scales[np.newaxis, :])

    if q_eff < q:
        extra = rng.standard_normal(size=(D, q - q_eff)) * 1e-3
        W = np.concatenate([W, extra], axis=1)

    prev_ll_space = np.linalg.norm(W)
    converged = False

    for it in range(max_iter):
        M = sigma2 * np.eye(W.shape[1]) + W.T @ W
        qw = Xc @ W
        Ez = np.linalg.solve(M, qw.T).T
        Minv = np.linalg.solve(M, np.eye(W.shape[1]))
        ezz = N * sigma2 * Minv + Ez.T @ Ez
        # Regression form W = argmin sum ||x - W z||^2 : W.T = solve(Ezz, Ez.T @ Xc)
        W_new = np.linalg.solve(ezz, Ez.T @ Xc).T

        sigma2_new = (
            np.sum(Xc**2)
            - 2.0 * np.sum(Xc * (Ez @ W_new.T))
            + np.trace(ezz @ W_new.T @ W_new)
        ) / (N * D)
        sigma2_new = float(max(sigma2_new, 1e-12))

        delta_W = np.linalg.norm(W_new - W)
        delta_sigma = abs(sigma2_new - sigma2)
        W = W_new
        sigma2 = sigma2_new

        if delta_W < tol * max(1.0, prev_ll_space) and delta_sigma < tol * max(1.0, sigma2):
            converged = True
            break
        prev_ll_space = np.linalg.norm(W)

    return PPCAEM(
        mean_=mu,
        W_=W,
        sigma2_=sigma2,
        n_iter_=it + 1,
        converged_=converged,
    )

