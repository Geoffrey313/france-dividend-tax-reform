"""Small shared helpers."""
from __future__ import annotations

import hashlib

import numpy as np

from src.common.config import STUDY, Study


def digest(values) -> str:
    """Stable SHA-256 over a numeric array, rounded to suppress float noise."""
    arr = np.round(np.asarray(values, dtype=float).ravel(), 6)
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def winsorize(values, study: Study = STUDY) -> np.ndarray:
    """Clip a continuous outcome to the study winsorization thresholds."""
    x = np.asarray(values, dtype=float)
    lo, hi = np.nanquantile(x, [study.winsor_lower, study.winsor_upper])
    return np.clip(x, lo, hi)
