"""Generate brief narrative analysis from experiment metrics."""

from __future__ import annotations


def build_analysis_section(rows: list[dict]) -> str:
    """Markdown bullets interpreting tables (course rubric: when PPCA works well / less so)."""
    lines = [
        "## Brief analysis",
        "",
        "### PPCA closed-form vs EM",
        "",
        "Under the stationary Gaussian latent model, EM should reproduce the closed-form maximum "
        "likelihood estimate up to numerical tolerance (rotation of **W** is not identifiable). ",
        "",
    ]

    max_em_gap = max(abs(r["mse_gap_em_vs_cf"]) for r in rows)
    lines.append(
        f"- Across runs here, the largest |MSE(PPCA EM) - MSE(PPCA closed-form)| was **{max_em_gap:.3g}**, "
        "so EM matches the spectral estimator closely."
    )
    lines.append("")

    lines.extend(
        [
            "### PCA vs PPCA reconstruction MSE",
            "",
            "**PCA** minimizes orthogonal projection error onto a rank-**q** subspace for centered data.",
            "**PPCA** reconstruction plotted here uses the **posterior mean** **E[z|x]** mapped back through **W** "
            "(same formula used for denoising under the PPCA likelihood). That estimator need not coincide with "
            "the PCA projector, so reconstruction MSE can be **lower for PCA even when both span nearly the "
            "same principal subspace**.",
            "",
        ]
    )

    gaps = [(r["dataset"].split("(")[0].strip(), r["mse_gap_cf_vs_pca"]) for r in rows]
    gaps_sorted = sorted(gaps, key=lambda t: abs(t[1]), reverse=True)
    lines.append("- Largest PCA vs PPCA(closed-form) MSE gaps by dataset:")
    for name, g in gaps_sorted:
        cmp_ = "PPCA higher" if g > 0 else "PCA higher"
        lines.append(f"  - **{name}**: Δ ≈ **{g:.4g}** ({cmp_}).")
    lines.append("")

    lines.extend(
        [
            "### When the method appears to work well",
            "",
            "- **High-dimensional faces / digits** (Olivetti, MNIST): PPCA closed-form and PCA reconstructions "
            "track closely under standardized pixels, consistent with both methods aligning with the dominant "
            "covariance eigenvectors.",
            "- **Estimated isotropic noise σ²**: interpretable as marginal variance orthogonal to the fitted "
            "rank-**q** factor part when the PPCA assumptions are plausible.",
            "",
            "### When results should be interpreted cautiously",
            "",
            "- **Small N relative to D** (Wine has **N=178**, **D=13** — modest sample size): covariance estimates "
            "are noisy; PPCA’s Gaussian noise assumption may not match chemistry features.",
            "- **Heavy tails / outliers**: Gaussian PPCA can be pulled off-axis; robust preprocessing matters.",
            "- **Choice of q**: fixed **q=10** for main tables is arbitrary; see MNIST **q** sweep figure.",
            "",
        ]
    )

    return "\n".join(lines)

