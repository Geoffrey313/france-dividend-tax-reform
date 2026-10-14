"""Readers for the delivered transformed data (the Orbis-extract swap point).

On WRDS these read the firm-year account panel pulled from Orbis (industrial global financials);
here they read the parquet written by the simulation, so the pipeline is identical either way.
The acquisition code that produces the real extract lives outside the repository.
"""
from __future__ import annotations

import pandas as pd

from src.common import paths


def read_panel(path=None) -> pd.DataFrame:
    """Firm-year account panel: firm, country, year, capital, equity, assets, revenue, age."""
    return pd.read_parquet(path or paths.DATA / "firm_panel.parquet")


def read_framing(path=None) -> pd.DataFrame:
    """Aggregate public framing series (Eurostat), long format."""
    return pd.read_csv(path or paths.DATA / "framing_fr.csv")
