"""Portable paths resolved from the repository root."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = {"en": ROOT / "manuscript" / "en" / "ssrn" / "figures",
           "fr": ROOT / "manuscript" / "fr" / "ssrn" / "figures"}


def figures_dir(lang: str) -> Path:
    d = FIGURES[lang]
    d.mkdir(parents=True, exist_ok=True)
    return d
