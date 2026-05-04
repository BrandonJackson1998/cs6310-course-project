"""PPCA course implementation (numpy only — no sklearn PPCA)."""

from .pca import PCABaseline, fit_pca
from .ppca_closed_form import PPCAClosedForm, fit_ppca_closed_form
from .ppca_em import PPCAEM, fit_ppca_em

__all__ = [
    "PCABaseline",
    "PPCAClosedForm",
    "PPCAEM",
    "fit_pca",
    "fit_ppca_closed_form",
    "fit_ppca_em",
]
