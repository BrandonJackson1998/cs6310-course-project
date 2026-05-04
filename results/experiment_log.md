# PPCA experiment log

Generated: 2026-05-01 21:07:09 UTC

Models compared:

- PCA baseline (truncated SVD on centered data)
- PPCA maximum likelihood closed form (sample covariance eigenvectors)
- PPCA EM (initialized from PCA spectrum)

---

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

## Metric tables

### Wine (sklearn, standardized)

- N=178, D=13, q=10

| Metric | Value |
| --- | --- |
| PCA reconstruction MSE | 0.0383028 |
| PPCA closed-form MSE | 0.0748682 |
| PPCA EM MSE | 0.0748682 |
| sigma² (closed-form) | 0.165979 |
| sigma² (EM) | 0.165979 |
| EM iterations | 1 |
| EM converged flag | True |
| MSE(PPCA_cf) - MSE(PCA) | 0.0365653 |
| MSE(PPCA_em) - MSE(PPCA_cf) | -3.19189e-16 |

---

### MNIST 784 (8000 samples, standardized pixel-wise)

- N=8000, D=784, q=10

| Metric | Value |
| --- | --- |
| PCA reconstruction MSE | 0.609164 |
| PPCA closed-form MSE | 0.609457 |
| PPCA EM MSE | 0.609457 |
| sigma² (closed-form) | 0.617034 |
| sigma² (EM) | 0.617034 |
| EM iterations | 1 |
| EM converged flag | True |
| MSE(PPCA_cf) - MSE(PCA) | 0.000293283 |
| MSE(PPCA_em) - MSE(PPCA_cf) | -1.11022e-16 |

---

### Olivetti faces (standardized)

- N=400, D=4096, q=10

| Metric | Value |
| --- | --- |
| PCA reconstruction MSE | 0.339446 |
| PPCA closed-form MSE | 0.339448 |
| PPCA EM MSE | 0.339448 |
| sigma² (closed-form) | 0.340276 |
| sigma² (EM) | 0.340276 |
| EM iterations | 250 |
| EM converged flag | True |
| MSE(PPCA_cf) - MSE(PCA) | 2.16534e-06 |
| MSE(PPCA_em) - MSE(PPCA_cf) | 2.32302e-08 |

---

## Figures

Image paths are relative to this markdown file.

- **Olivetti reconstruction grids:** Three PNGs with **q ∈ {10, 20, 40}** (each filename `olivetti_reconstructions_q*.png` matches the **effective** latent dimension after caps). Same three example indices (**0**, **80**, **160**) in each grid; columns = Original | PCA | PPCA | **|PCA − PPCA|**.

![Reconstruction MSE by dataset](figures/mse_by_dataset.png)

![MNIST MSE vs q](figures/mnist_mse_vs_q.png)

![Wine cumulative variance](figures/wine_variance_explained.png)

![Olivetti q=10](figures/olivetti_reconstructions_q10.png)

![Olivetti q=20](figures/olivetti_reconstructions_q20.png)

![Olivetti q=40](figures/olivetti_reconstructions_q40.png)

---

## Brief analysis

### PPCA closed-form vs EM

Under the stationary Gaussian latent model, EM should reproduce the closed-form maximum likelihood estimate up to numerical tolerance (rotation of **W** is not identifiable). 

- Across runs here, the largest |MSE(PPCA EM) - MSE(PPCA closed-form)| was **2.32e-08**, so EM matches the spectral estimator closely.

### PCA vs PPCA reconstruction MSE

**PCA** minimizes orthogonal projection error onto a rank-**q** subspace for centered data.
**PPCA** reconstruction plotted here uses the **posterior mean** **E[z|x]** mapped back through **W** (same formula used for denoising under the PPCA likelihood). That estimator need not coincide with the PCA projector, so reconstruction MSE can be **lower for PCA even when both span nearly the same principal subspace**.

- Largest PCA vs PPCA(closed-form) MSE gaps by dataset:
  - **Wine**: Δ ≈ **0.03657** (PPCA higher).
  - **MNIST 784**: Δ ≈ **0.0002933** (PPCA higher).
  - **Olivetti faces**: Δ ≈ **2.165e-06** (PPCA higher).

### When the method appears to work well

- **High-dimensional faces / digits** (Olivetti, MNIST): PPCA closed-form and PCA reconstructions track closely under standardized pixels, consistent with both methods aligning with the dominant covariance eigenvectors.
- **Estimated isotropic noise σ²**: interpretable as marginal variance orthogonal to the fitted rank-**q** factor part when the PPCA assumptions are plausible.

### When results should be interpreted cautiously

- **Small N relative to D** (Wine has **N=178**, **D=13** — modest sample size): covariance estimates are noisy; PPCA’s Gaussian noise assumption may not match chemistry features.
- **Heavy tails / outliers**: Gaussian PPCA can be pulled off-axis; robust preprocessing matters.
- **Choice of q**: fixed **q=10** for main tables is arbitrary; see MNIST **q** sweep figure.

