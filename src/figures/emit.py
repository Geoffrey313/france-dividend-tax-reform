"""Write the manuscript figure CSVs from the regenerated results, and digest them.

The figures are pgfplots reading these CSVs, so regenerating the CSV regenerates the figure. The
CSVs live beside the .tex in the French manuscript; the English manuscript references no figures.
"""
from __future__ import annotations

import pandas as pd

from src.common import paths
from src.common.util import digest

FIGDIR = paths.ROOT / "manuscript" / "fr" / "ssrn" / "figures"


def _write(df: pd.DataFrame, name: str, index: bool = False) -> str:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    path = FIGDIR / name
    df.to_csv(path, index=index)
    return f"{name:26s} {digest(df.select_dtypes('number').to_numpy())}"


def event_study_csv(es: pd.DataFrame, name: str) -> str:
    """year, coef, std_error (reference year already omitted by the estimator)."""
    return _write(es.reset_index().rename(columns={"index": "year"}), name)


def retention_path_csv(path_df: pd.DataFrame, name: str = "capital_retenu.csv") -> str:
    """year, retention, equity: the level event study over the long window."""
    return _write(path_df.reset_index().rename(columns={"index": "year"}), name)


def cliquet_csv(shock2013: dict, shock2018: dict, distribution: dict,
                name: str = "cliquet.csv") -> str:
    """marge, choc2013, choc2018 on distribution, retention, and own funds (bar-chart input)."""
    frame = pd.DataFrame({
        "marge": ["distribution", "retention", "fondspropres"],
        "choc2013": [distribution["2013"], shock2013["retention"]["coef"],
                     shock2013["fondspropres"]["coef"]],
        "choc2018": [distribution["2018"], shock2018["retention"]["coef"],
                     shock2018["fondspropres"]["coef"]],
    })
    return _write(frame, name)


def dividend_index_csv(wide: pd.DataFrame, name: str = "dividendes_menages.csv") -> str:
    """year, then one column per country: the household-dividend index, 2017 = 100."""
    return _write(wide.reset_index(), name)
