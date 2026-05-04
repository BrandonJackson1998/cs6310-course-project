"""Experiment harness: PCA vs PPCA closed-form vs PPCA EM."""

from __future__ import annotations

import datetime as dt
import sys
import warnings
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_olivetti_faces
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ppca.pca import fit_pca  # noqa: E402
from ppca.ppca_closed_form import fit_ppca_closed_form  # noqa: E402
from ppca.ppca_em import fit_ppca_em  # noqa: E402

from .analysis import build_analysis_section  # noqa: E402
from .plots import (  # noqa: E402
    plot_explained_variance_wine,
    plot_mse_by_dataset,
    plot_mse_vs_q_mnist,
    plot_olivetti_reconstructions,
)


DATASETS_MARKDOWN = """
## Datasets used

Public benchmarks loaded through **scikit-learn** (plus OpenML for MNIST). Each matrix **X** has **N** rows (samples)
and **D** columns (features). Before PCA/PPCA, every dataset is **standardized column-wise**
(`sklearn.preprocessing.StandardScaler`: zero mean, unit variance per feature). **Reconstruction MSE** is computed
on this **standardized** scale so eigenvalues remain numerically stable.

### Wine (`sklearn.datasets.load_wine`)

- **Description:** Chemical analysis of wines from three cultivars in Italy; classic tabular classification benchmark.
- **Shape:** **N = 178**, **D = 13** (continuous chemistry measurements).
- **Source / docs:** [load_wine](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html)
- **Preparation:** `load_wine()` returns `data` as float64; standardized with `StandardScaler`.

### MNIST 784 (`fetch_openml(name="mnist_784", version=1)`)

- **Description:** Handwritten digits rendered as **28 × 28** grayscale images flattened to **784** pixels.
- **Shape:** Full OpenML dump has **70 000** rows; experiments subsample **8000** rows (fixed RNG seed **0**) for speed.
- **Links:**
  - OpenML dataset page: [MNIST 784 (id 554)](https://www.openml.org/d/554)
  - Loader docs: [fetch_openml](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_openml.html)
- **Preparation:** Pixel intensities divided by **255**, then column-wise standardization.

### Olivetti faces (`sklearn.datasets.fetch_olivetti_faces`)

- **Description:** Grayscale face portraits (**64 × 64**) of **40** subjects (**400** images total).
- **Shape:** **N = 400**, **D = 4096**.
- **Source / docs:** [fetch_olivetti_faces](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_olivetti_faces.html)
  (downloads once into `~/scikit_learn_data` by default).
- **Preparation:** Flattened vectors scaled to **[0, 1]** already in sklearn bundle; experiments additionally apply `StandardScaler`.
  Visualization exports reverse the scaler so pixels remain in **[0, 1]** for plotting.

---
""".strip()


def reconstruction_mse(X: np.ndarray, X_hat: np.ndarray) -> float:
    return float(np.mean((X - X_hat) ** 2))


def effective_q(N: int, D: int, requested: int = 10) -> int:
    return max(1, min(requested, D - 1, N - 1))


def standardize_fit_transform(X: np.ndarray) -> tuple[np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    Xs = scaler.fit_transform(np.asarray(X, dtype=np.float64))
    return Xs, scaler


def load_wine():
    from sklearn.datasets import load_wine

    data = load_wine()
    X = np.asarray(data.data, dtype=np.float64)
    X, _ = standardize_fit_transform(X)
    return X, "Wine (sklearn, standardized)"


def load_mnist_subset(n_max: int = 8000, seed: int = 0):
    from sklearn.datasets import fetch_openml

    X_df, _ = fetch_openml(
        "mnist_784",
        version=1,
        parser="auto",
        return_X_y=True,
        as_frame=False,
    )
    rng = np.random.default_rng(seed)
    X = np.asarray(X_df, dtype=np.float64)
    if len(X) > n_max:
        idx = rng.choice(len(X), size=n_max, replace=False)
        X = X[idx]
    X /= 255.0
    X, _ = standardize_fit_transform(X)
    return X, f"MNIST 784 ({len(X)} samples, standardized pixel-wise)"


def load_olivetti():
    bunch = fetch_olivetti_faces(shuffle=True, random_state=0)
    X = np.asarray(bunch.data, dtype=np.float64)
    X, _ = standardize_fit_transform(X)
    return X, "Olivetti faces (standardized)"


def load_olivetti_visual_bundle():
    """Raw [0,1] pixels plus standardized matrix + scaler for reconstruction plots."""
    bunch = fetch_olivetti_faces(shuffle=True, random_state=0)
    X_raw = np.asarray(bunch.data, dtype=np.float64)
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_raw)
    return {"X_raw": X_raw, "X_std": X_std, "scaler": scaler}


def run_one_dataset(X: np.ndarray, name: str, q_requested: int = 10) -> dict:
    N, D = X.shape
    q_eff = effective_q(N, D, q_requested)
    row = {"dataset": name, "N": N, "D": D, "q": q_eff}

    pca = fit_pca(X, q_eff)
    X_hat_pca = pca.reconstruct(X)
    row["mse_pca"] = reconstruction_mse(X, X_hat_pca)

    ppca_cf = fit_ppca_closed_form(X, q_eff)
    X_hat_cf = ppca_cf.reconstruct(X)
    row["mse_ppca_cf"] = reconstruction_mse(X, X_hat_cf)
    row["sigma2_ppca_cf"] = ppca_cf.sigma2_

    ppca_em = fit_ppca_em(X, q_eff, random_state=0)
    X_hat_em = ppca_em.reconstruct(X)
    row["mse_ppca_em"] = reconstruction_mse(X, X_hat_em)
    row["sigma2_ppca_em"] = ppca_em.sigma2_
    row["ppca_em_iters"] = ppca_em.n_iter_
    row["ppca_em_converged"] = ppca_em.converged_

    row["mse_gap_cf_vs_pca"] = row["mse_ppca_cf"] - row["mse_pca"]
    row["mse_gap_em_vs_cf"] = row["mse_ppca_em"] - row["mse_ppca_cf"]

    return row


def format_row(r: dict) -> str:
    lines = [
        f"### {r['dataset']}",
        "",
        f"- N={r['N']}, D={r['D']}, q={r['q']}",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| PCA reconstruction MSE | {r['mse_pca']:.6g} |",
        f"| PPCA closed-form MSE | {r['mse_ppca_cf']:.6g} |",
        f"| PPCA EM MSE | {r['mse_ppca_em']:.6g} |",
        f"| sigma² (closed-form) | {r['sigma2_ppca_cf']:.6g} |",
        f"| sigma² (EM) | {r['sigma2_ppca_em']:.6g} |",
        f"| EM iterations | {r['ppca_em_iters']} |",
        f"| EM converged flag | {r['ppca_em_converged']} |",
        f"| MSE(PPCA_cf) - MSE(PCA) | {r['mse_gap_cf_vs_pca']:.6g} |",
        f"| MSE(PPCA_em) - MSE(PPCA_cf) | {r['mse_gap_em_vs_cf']:.6g} |",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    warnings.filterwarnings(
        "ignore",
        category=RuntimeWarning,
        message=".*encountered in matmul",
    )
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    loaders = [load_wine, load_mnist_subset, load_olivetti]

    rows: list[dict] = []
    for loader in loaders:
        X, name = loader()
        rows.append(run_one_dataset(X, name, q_requested=10))

    results_dir = ROOT / "results"
    figures_dir = results_dir / "figures"
    results_dir.mkdir(exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    plot_mse_by_dataset(rows, figures_dir / "mse_by_dataset.png")

    X_mnist, _ = load_mnist_subset()
    plot_mse_vs_q_mnist(X_mnist, [2, 5, 10, 15, 20, 30, 40, 50], figures_dir / "mnist_mse_vs_q.png")

    X_wine, _ = load_wine()
    plot_explained_variance_wine(X_wine, figures_dir / "wine_variance_explained.png")

    olb = load_olivetti_visual_bundle()
    N_o, D_o = olb["X_std"].shape
    for q_req in (10, 20, 40):
        q_eff = effective_q(N_o, D_o, q_req)
        plot_olivetti_reconstructions(
            olb["X_std"],
            olb["X_raw"],
            olb["scaler"],
            q=q_req,
            indices=[0, 80, 160],
            out_path=figures_dir / f"olivetti_reconstructions_q{q_eff}.png",
        )

    olv_q_eff = tuple(effective_q(N_o, D_o, qr) for qr in (10, 20, 40))

    lines = [
        "# PPCA experiment log",
        "",
        f"Generated: {stamp}",
        "",
        "Models compared:",
        "",
        "- PCA baseline (truncated SVD on centered data)",
        "- PPCA maximum likelihood closed form (sample covariance eigenvectors)",
        "- PPCA EM (initialized from PCA spectrum)",
        "",
        "---",
        "",
        DATASETS_MARKDOWN,
        "",
        "## Metric tables",
        "",
    ]
    for r in rows:
        lines.append(format_row(r))
        lines.append("---")
        lines.append("")

    lines.extend(
        [
            "## Figures",
            "",
            "Image paths are relative to this markdown file.",
            "",
            "- **Olivetti reconstruction grids:** Three PNGs with **q ∈ {10, 20, 40}** (each filename "
            "`olivetti_reconstructions_q*.png` matches the **effective** latent dimension after caps). Same three "
            "example indices (**0**, **80**, **160**) in each grid; columns = Original | PCA | PPCA | **|PCA − PPCA|**.",
            "",
            f"![Reconstruction MSE by dataset](figures/mse_by_dataset.png)",
            "",
            f"![MNIST MSE vs q](figures/mnist_mse_vs_q.png)",
            "",
            f"![Wine cumulative variance](figures/wine_variance_explained.png)",
            "",
            f"![Olivetti q={olv_q_eff[0]}](figures/olivetti_reconstructions_q{olv_q_eff[0]}.png)",
            "",
            f"![Olivetti q={olv_q_eff[1]}](figures/olivetti_reconstructions_q{olv_q_eff[1]}.png)",
            "",
            f"![Olivetti q={olv_q_eff[2]}](figures/olivetti_reconstructions_q{olv_q_eff[2]}.png)",
            "",
            "---",
            "",
        ]
    )
    lines.append(build_analysis_section(rows))
    lines.append("")

    out_md = results_dir / "experiment_log.md"
    body = "\n".join(lines)
    out_md.write_text(body, encoding="utf-8")

    print(body)
    print(f"\nWrote {out_md}")
    print(f"Figures saved under {figures_dir}")


if __name__ == "__main__":
    main()
