"""Write the manuscript figure CSVs from the regenerated results, and digest them.

The figures are pgfplots reading these CSVs, so regenerating the CSV regenerates the figure. The
CSVs live beside the generated .tex fragments in both language trees.
"""
from __future__ import annotations

import pandas as pd

from src.common import paths
from src.common.util import digest

# Both manuscripts read the same figure CSVs, so each one is written to both language trees from the
# single regenerated table; neither copy is hand-maintained.
_FIGDIRS = [paths.ROOT / "manuscript" / lang / "ssrn" / "figures" for lang in ("fr", "en")]
FIGDIR = _FIGDIRS[0]


def _write(df: pd.DataFrame, name: str, index: bool = False) -> str:
    for d in _FIGDIRS:
        d.mkdir(parents=True, exist_ok=True)
        df.to_csv(d / name, index=index)
    return f"{name:26s} {digest(df.select_dtypes('number').to_numpy())}"


def decoupling_csv(df: pd.DataFrame, name: str = "decouplage_fr.csv") -> str:
    """year, France lines (dividendes, profit, pib) and comparable means (_comp), indexed to 2017."""
    return _write(df.reset_index().rename(columns={"index": "year", "year": "year"}), name)


def event_study_csv(es: pd.DataFrame, name: str) -> str:
    """year, coef, std_error (reference year already omitted by the estimator)."""
    return _write(es.reset_index().rename(columns={"index": "year"}), name)


def harmonized_accounting_event_csv(events: pd.DataFrame) -> list[str]:
    """Write private and listed event-study panels from regenerated result rows."""
    outputs = []
    group_columns = ["shock", "outcome"]
    if "population" in events:
        group_columns = ["population", *group_columns]
    for keys, frame in events.groupby(group_columns, sort=False):
        if len(group_columns) == 3:
            population, shock, outcome = keys
            prefix = "" if population == "private_large_vl" else "public_"
        else:
            shock, outcome = keys
            prefix = ""
        name = f"{prefix}firm_accounting_{shock}_{outcome}.csv"
        rendered = frame[["year", "coef", "std_error", "ci95"]].sort_values("year").copy()
        rendered[["coef", "std_error", "ci95"]] *= 100.0
        outputs.append(_write(
            rendered, name,
        ))
    return outputs


def harmonized_main_event_csv(events: pd.DataFrame) -> list[str]:
    """Write the two headline 2018 event studies from the common firm specification."""
    outputs = []
    for outcome, name in (
        ("dlog_capital", "entree_capital.csv"),
        ("fin_debt_ratio", "levier_event.csv"),
    ):
        frame = events[
            events["shock"].eq("2018_full") & events["outcome"].eq(outcome)
        ]
        if frame.empty:
            raise ValueError(f"missing manuscript event study for {outcome}")
        outputs.append(_write(
            frame[["year", "coef", "std_error"]].sort_values("year"), name,
        ))
    return outputs


def retention_path_csv(path_df: pd.DataFrame, name: str = "capital_retenu.csv") -> str:
    """year, retention, equity: the level event study over the long window."""
    return _write(path_df.reset_index().rename(columns={"index": "year"}), name)


def cliquet_csv(shock2013: dict, shock2018: dict, distribution: dict,
                name: str = "cliquet.csv") -> str:
    """marge, choc2013, choc2018 on distribution, retention, and own funds (bar-chart input)."""
    frame = pd.DataFrame({
        "marge": ["distribution", "retention", "fondspropres"],
        "choc2013": [abs(distribution["2013"]), abs(shock2013["retention"]["coef"]),
                     abs(shock2013["fondspropres"]["coef"])],
        "choc2018": [abs(distribution["2018"]), abs(shock2018["retention"]["coef"]),
                     abs(shock2018["fondspropres"]["coef"])],
    })
    return _write(frame, name)


def dividend_index_csv(wide: pd.DataFrame, name: str = "dividendes_menages.csv") -> str:
    """year, then one column per country: the household-dividend index, 2017 = 100."""
    return _write(wide.reset_index(), name)
