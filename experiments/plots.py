"""Generate figures for experiment reports (matplotlib Agg backend)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

from ppca.pca import fit_pca
from ppca.ppca_closed_form import fit_ppca_closed_form


def plot_mse_by_dataset(rows: list[dict], out_path: Path) -> None:
    """Grouped bars + per-dataset relative uplift vs PCA (each panel autoscales its own y-axis)."""
    from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec

    labels = []
    for r in rows:
        name = r["dataset"].split("(")[0].strip()
        labels.append(name.replace("784", "").strip())

    x = np.arange(len(rows))
    width = 0.25
    mse_pca = np.array([r["mse_pca"] for r in rows])
    mse_cf = np.array([r["mse_ppca_cf"] for r in rows])
    mse_em = np.array([r["mse_ppca_em"] for r in rows])

    fig = plt.figure(figsize=(11, 7.5))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[2.3, 1], hspace=0.42)
    ax0 = fig.add_subplot(gs[0])

    ax0.bar(x - width, mse_pca, width, label="PCA (SVD)")
    ax0.bar(x, mse_cf, width, label="PPCA closed-form")
    ax0.bar(x + width, mse_em, width, label="PPCA EM")
    ax0.set_xticks(x)
    ax0.set_xticklabels(labels, rotation=15, ha="right")
    ax0.set_ylabel("Reconstruction MSE")
    ax0.set_title("Mean squared reconstruction error by dataset (standardized features)")
    ax0.legend()
    ax0.grid(axis="y", alpha=0.3)

    pct_cf = 100.0 * (mse_cf - mse_pca) / np.maximum(mse_pca, 1e-15)
    pct_em = 100.0 * (mse_em - mse_pca) / np.maximum(mse_pca, 1e-15)

    inner = GridSpecFromSubplotSpec(1, len(rows), subplot_spec=gs[1], wspace=0.45)
    bw = 0.38
    for i in range(len(rows)):
        ax_b = fig.add_subplot(inner[0, i])
        ax_b.axhline(0.0, color="gray", linewidth=0.8)
        ax_b.bar([0, 1], [pct_cf[i], pct_em[i]], width=bw, color=["tab:orange", "tab:green"])
        ax_b.set_xticks([0, 1])
        ax_b.set_xticklabels(["PPCA cf", "EM"], fontsize=8)
        ax_b.set_title(labels[i], fontsize=10)
        ax_b.set_ylabel("Δ vs PCA (%)")
        ymax = max(float(pct_cf[i]), float(pct_em[i]), 0.0)
        ymin = min(float(pct_cf[i]), float(pct_em[i]), 0.0)
        pad = max(abs(ymax), abs(ymin), 1e-18) * 0.18 + 1e-18
        ax_b.set_ylim(ymin - pad, ymax + pad)
        ax_b.tick_params(axis="y", labelsize=8)
        ax_b.grid(axis="y", alpha=0.3)
        ax_b.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)

    fig.suptitle(
        "Bottom row: relative uplift vs PCA with its own y-axis per dataset "
        "(Wine’s large % no longer hides MNIST/Olivetti).",
        fontsize=9,
        y=0.008,
    )
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_mse_vs_q_mnist(X: np.ndarray, q_values: list[int], out_path: Path) -> None:
    """Curve of reconstruction MSE vs latent dimension q (MNIST subset)."""
    N, D = X.shape
    qs = []
    mse_pca_l = []
    mse_cf_l = []
    for q in sorted(set(q_values)):
        q_eff = max(1, min(q, D - 1, N - 1))
        if q_eff in qs:
            continue
        qs.append(q_eff)
        pca = fit_pca(X, q_eff)
        cf = fit_ppca_closed_form(X, q_eff)
        mse_pca_l.append(np.mean((X - pca.reconstruct(X)) ** 2))
        mse_cf_l.append(np.mean((X - cf.reconstruct(X)) ** 2))

    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(8, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax_top.plot(qs, mse_pca_l, "o-", label="PCA")
    ax_top.plot(qs, mse_cf_l, "s-", label="PPCA closed-form")
    ax_top.set_ylabel("Reconstruction MSE")
    ax_top.set_title("MNIST subset: MSE vs q (standardized pixels)")
    ax_top.legend()
    ax_top.grid(alpha=0.3)

    delta = np.array(mse_cf_l) - np.array(mse_pca_l)
    ax_bot.axhline(0.0, color="gray", linewidth=0.8)
    ax_bot.plot(qs, delta, "d-", color="darkred", label="PPCA − PCA")
    ax_bot.set_xlabel("q (latent dimensions)")
    ax_bot.set_ylabel("Δ MSE")
    ax_bot.set_title("Gap between curves (often tiny but non-zero)")
    ax_bot.legend()
    ax_bot.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_olivetti_reconstructions(
    X_std: np.ndarray,
    X_raw: np.ndarray,
    scaler: StandardScaler,
    *,
    q: int,
    indices: list[int],
    out_path: Path,
    img_shape: tuple[int, int] = (64, 64),
) -> None:
    """Original vs PCA vs PPCA plus |PCA−PPCA| so near-identical reconstructions remain visible."""
    N, D = X_std.shape
    q_eff = max(1, min(q, D - 1, N - 1))

    pca = fit_pca(X_std, q_eff)
    ppca = fit_ppca_closed_form(X_std, q_eff)

    n_show = len(indices)
    n_col = 4
    fig, axes = plt.subplots(n_show, n_col, figsize=(11, 2.8 * n_show))
    if n_show == 1:
        axes = np.array([axes])

    titles = ["Original", f"PCA (q={q_eff})", f"PPCA (q={q_eff})", "|PCA − PPCA|"]
    for row, ii in enumerate(indices):
        orig_img = X_raw[ii].reshape(img_shape).clip(0, 1)
        x_row = X_std[ii : ii + 1]
        pca_img = scaler.inverse_transform(pca.reconstruct(x_row))[0].reshape(img_shape).clip(0, 1)
        ppca_img = scaler.inverse_transform(ppca.reconstruct(x_row))[0].reshape(img_shape).clip(0, 1)
        diff = np.abs(pca_img - ppca_img)
        # Scale so microscopic differences are visible (clip by high percentile).
        vmax = max(float(np.percentile(diff, 99.5)), 1e-6)

        imgs = [orig_img, pca_img, ppca_img, diff]
        cmaps = ["gray", "gray", "gray", "inferno"]
        vmins = [0, 0, 0, 0]
        vmaxs = [1, 1, 1, vmax]

        for col in range(n_col):
            ax = axes[row, col]
            ax.imshow(imgs[col], cmap=cmaps[col], vmin=vmins[col], vmax=vmaxs[col])
            ax.set_title(titles[col] if row == 0 else "")
            ax.axis("off")
        rmse_disp = float(np.sqrt(np.mean(diff**2)))
        axes[row, 0].text(
            0.02,
            0.98,
            f"RMSE(|Δ|)={rmse_disp:.4f}",
            transform=axes[row, 0].transAxes,
            va="top",
            ha="left",
            fontsize=8,
            color="white",
            bbox=dict(facecolor="black", alpha=0.45, pad=2),
        )

    fig.suptitle(
        "Olivetti: PCA vs PPCA look similar visually; last column amplifies pixel-wise |difference| "
        f"(inferno scale, vmax≈99.5th pct per row). q={q_eff}.",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_explained_variance_wine(X: np.ndarray, out_path: Path, max_q: int = 12) -> None:
    """Scree-style cumulative variance ratio on standardized Wine (PCA eigenvalues)."""
    N, D = X.shape
    Xc = X - X.mean(axis=0)
    _, S, _ = np.linalg.svd(Xc, full_matrices=False)
    lam = (S**2) / N
    total = lam.sum()
    q_max = min(max_q, len(lam))
    ratios = np.cumsum(lam[:q_max]) / total

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, q_max + 1), ratios, "o-")
    ax.set_xlabel("Number of components")
    ax.set_ylabel("Cumulative variance fraction")
    ax.set_title("Wine (standardized): PCA cumulative variance explained")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
