# cs6310-course-project

Course implementation of **Probabilistic Principal Component Analysis** from:

Michael E. Tipping and Christopher M. Bishop, *Probabilistic Principal Component Analysis*,
**Journal of the Royal Statistical Society, Series B**, 61(3), 611–622 (1999).

---

## What this repo contains

| Piece | Location |
| --- | --- |
| PCA baseline (truncated SVD) | `ppca/pca.py` |
| PPCA closed-form MLE | `ppca/ppca_closed_form.py` |
| PPCA EM | `ppca/ppca_em.py` |
| Experiment script | `experiments/run_all.py` |
| Figures + tables + narrative analysis | Generated under `results/` (see below) |

**Constraint:** core decomposition code uses **NumPy only**. `scikit-learn` is used for **datasets** and **`StandardScaler`** — not for `sklearn.decomposition.PCA` / `PPCA`.

---

## Exact reproduction steps

Use **Python 3.11** (see `.python-version`). From the repository root:

### Option A — `venv` + pip

```bash
cd cs6310-course-project
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
python3 -m experiments.run_all
```

### Option B — Conda / Mamba

```bash
cd cs6310-course-project
conda env create -f environment.yml
conda activate cs6310-ppca
python -m experiments.run_all    # or: python3 -m experiments.run_all
```

### Outputs produced by `experiments.run_all`

| Output | Purpose |
| --- | --- |
| `results/experiment_log.md` | Tables, embedded figure links, dataset descriptions, brief analysis |
| `results/figures/*.png` | Plots (MSE bars, MNIST vs **q**, Wine scree, Olivetti reconstructions) |

**First run notes**

- MNIST is fetched from **OpenML** (`mnist_784`, version `1`) — requires network once.
- Olivetti faces may download into `~/scikit_learn_data` once.

---

## Datasets (summary)

Full prose + preparation lives in **`results/experiment_log.md`** after you run experiments (also summarized here):

| Dataset | N × D | Access |
| --- | --- | --- |
| Wine chemistry | 178 × 13 | [`load_wine`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_wine.html) |
| MNIST 784 | 8000 × 784 (subset) | [`fetch_openml`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_openml.html), dataset [`554`](https://www.openml.org/d/554) |
| Olivetti faces | 400 × 4096 | [`fetch_olivetti_faces`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_olivetti_faces.html) |

All inputs are **column-standardized** before PCA/PPCA; reconstruction **MSE** is reported on that standardized scale.

---

## Project layout

```
cs6310-course-project/
  ppca/                  # Algorithms
  experiments/
    run_all.py           # Entry point
    plots.py             # Matplotlib helpers
    analysis.py          # Auto-written discussion text
  results/
    experiment_log.md    # Generated
    figures/             # Generated PNGs
  requirements.txt
  environment.yml
  README.md
```

---

## Presentation / slides

**Marp deck:** [`slides/ppca-presentation.md`](slides/ppca-presentation.md) — open in VS Code with the [Marp for VS Code](https://marketplace.visualstudio.com/items?itemName=marp-team.marp-vscode) extension (or Marp CLI) to preview / export PDF or PowerPoint. Figure paths assume you keep `results/figures/` populated (`python3 -m experiments.run_all`).
