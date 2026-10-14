"""Manuscript results regenerated from the committed firm panels and public series.

Every displayed number in manuscript/fr/ssrn traces to one function here, so reproduce.py can
rebuild the figure CSVs and the results JSON and check_manuscript.py can gate on them.

The displayed specification is the size-only baseline: the difference-in-differences regressor and
the size control, firm and year fixed effects, standard errors clustered by country, with the
synthetic-control placebos carrying inference. That is ``did.fit(..., controls=[schema.SIZE])``.
The referee control set stays available through the same override for robustness, but the tables
and figures in the paper display the size-only baseline.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common import paths, schema as S
from src.common.config import STUDY, Study, ace_active
from src.engine import panel as panel_engine
from src.engine import did, synth

BASELINE = [S.SIZE]  # the displayed controls: size only

# Outcomes carried in the appendix descriptive table, in display order.
_DESCRIPTIVE = (S.DLOG_CAPITAL, S.CAP_INCREASE, S.EQUITY_RATIO, S.LEVERAGE, S.DLOG_DEBT,
                S.DLOG_FIXED, S.LOG_LABOR_PROD, S.RESERVES_RATIO, S.LOG_ASSETS)


def fit_outcome(built: pd.DataFrame, outcome: str, study: Study = STUDY,
                controls=BASELINE) -> tuple[did.DidFit, synth.SynthFit]:
    """Size-only difference-in-differences and the country-aggregate synthetic control."""
    d = did.fit(built, outcome, study, controls=controls)
    wide = panel_engine.aggregate_by_country_year(built, outcome)
    return d, synth.fit(wide, study)


def _cell(built: pd.DataFrame, outcome: str, study: Study) -> dict:
    d, s = fit_outcome(built, outcome, study)
    return {"coef": d.coef, "t": d.z, "n": d.n_obs,
            "synth": s.post_effect, "placebo_p": s.placebo_rank_p}


def firm_tables(built: pd.DataFrame, study: Study = STUDY) -> dict:
    """Table 1 (capital margins) and Table 2 (substitution and real margins), size-only."""
    table1 = {o: _cell(built, o, study) for o in (S.DLOG_CAPITAL, S.CAP_INCREASE, S.EQUITY_RATIO)}
    table2 = {o: _cell(built, o, study) for o in (S.LEVERAGE, S.DLOG_DEBT, S.DLOG_FIXED,
                                                  S.LOG_LABOR_PROD)}
    return {"table1": table1, "table2": table2}


def descriptives(built: pd.DataFrame) -> list[dict]:
    """Appendix descriptive statistics: N, mean, sd, and the quartiles per outcome."""
    rows = []
    for v in _DESCRIPTIVE:
        s = built[v].dropna()
        rows.append({"var": v, "n": int(s.size), "mean": float(s.mean()), "sd": float(s.std()),
                     "p25": float(s.quantile(0.25)), "p50": float(s.quantile(0.50)),
                     "p75": float(s.quantile(0.75))})
    return rows


def event_study(built: pd.DataFrame, outcome: str, study: Study = STUDY) -> pd.DataFrame:
    """Size-only treated-by-year coefficients, the input to an event-study figure."""
    return did.fit(built, outcome, study, controls=BASELINE).event_study


# --- The ratchet: the 2013 hike and the 2018 cut in one frame (Table 3, cliquet figure) ----------

RATCHET_FIRST_YEAR = 2010  # the extended panel reaches back to 2010 for the 2013-shock pre-period
# Aggregate distribution semi-elasticities (SQ1), smoothed base, |value|, from the Eurostat D42
# broad panel. The firm-level ratchet below is regenerated live from the panel; these two aggregate
# values are carried as documented constants from the SQ1 estimation (session log 2026-10-13, Outline
# spec 345e3e4b) whose standalone script was not recovered, pending a committed re-wiring of that
# aggregate pipeline. They are gated against the manuscript like every other displayed number.
SQ1_DISTRIBUTION = {"2013": 0.3997359598056483, "2018": 0.1780259354754353}
SHOCK_2013 = Study(first_year=2010, reference_year=2012, reform_year=2013, last_year=2016)
SHOCK_2018 = Study(first_year=2015, reference_year=2017, reform_year=2018, last_year=2022)
RETENTION_LONG = Study(first_year=2010, reference_year=2012, reform_year=2013, last_year=2022)


def _build_shock(long_panel: pd.DataFrame, shock: Study) -> pd.DataFrame:
    """Build on the full 2010-2022 window, then restrict to the shock window.

    The balance filter and the winsorization must see the whole panel: building on the short shock
    window alone selects a different, more-complete sample and halves the coefficient. The estimator
    then uses only the shock window.
    """
    full = shock.model_copy(update={"first_year": RATCHET_FIRST_YEAR, "last_year": STUDY.last_year})
    built = panel_engine.build(long_panel, full)
    return built[built[S.YEAR].between(shock.first_year, shock.last_year)].copy()


def ratchet_shock(long_panel: pd.DataFrame, shock: Study) -> dict:
    """Retention and own-funds difference-in-differences around one tax shock, size-only."""
    built = _build_shock(long_panel, shock)
    return {"retention": _cell(built, S.RESERVES_RATIO, shock),
            "fondspropres": _cell(built, S.EQUITY_RATIO, shock)}


def retention_level_path(long_panel: pd.DataFrame, study: Study = RETENTION_LONG) -> pd.DataFrame:
    """Retention and own-funds level event study over 2010 to 2022, reference 2012."""
    built = panel_engine.build(long_panel, study)
    ret = event_study(built, S.RESERVES_RATIO, study)["coef"].rename("retention")
    eq = event_study(built, S.EQUITY_RATIO, study)["coef"].rename("equity")
    return pd.concat([ret, eq], axis=1)


def robustness_2013(long_panel: pd.DataFrame) -> dict:
    """Section 6 robustness on the 2013 retention coefficient: an equity-allowance indicator, an
    added unemployment (cycle) control, and a subset restricted to the no-allowance controls. Reuses
    one full-window 2013-shock build.
    """
    built = _build_shock(long_panel, SHOCK_2013)
    built = built.assign(**{"ace": [int(ace_active(c, y))
                                    for c, y in zip(built[S.COUNTRY], built[S.YEAR])]})
    unemp = pd.read_csv(paths.DATA / "unemployment_2010_2022.csv")[[S.COUNTRY, S.YEAR, "unemp"]]
    built = built.merge(unemp, on=[S.COUNTRY, S.YEAR], how="left")
    base = did.fit(built, S.RESERVES_RATIO, SHOCK_2013, controls=BASELINE).coef
    with_ace = did.fit(built, S.RESERVES_RATIO, SHOCK_2013, controls=[S.SIZE, "ace"]).coef
    with_cycle = did.fit(built, S.RESERVES_RATIO, SHOCK_2013, controls=[S.SIZE, "ace", "unemp"]).coef
    no_subsidy = built[built[S.COUNTRY].isin(("FR", "DE", "NL"))]
    subset = did.fit(no_subsidy, S.RESERVES_RATIO, SHOCK_2013, controls=BASELINE).coef
    return {"baseline": base, "with_ace": with_ace, "with_cycle": with_cycle,
            "no_subsidy_controls": subset}


# --- Shareholder heterogeneity (Section 6 robustness): the owner-manager channel -----------------

def _did_cell(frame: pd.DataFrame, outcome: str, study: Study) -> dict:
    """Size-only difference-in-differences coefficient, t ratio, and sample size on a subsample.

    No synthetic control here: a country-aggregate placebo on an ownership subsample would mix a
    changing sample composition into the donor pool, so only the regression cell is carried.
    """
    d = did.fit(frame, outcome, study, controls=BASELINE)
    return {"coef": d.coef, "t": d.z, "n": d.n_obs, "n_firms": d.n_firms}


def heterogeneity_shareholder(real_panel: pd.DataFrame, flags: pd.DataFrame,
                              study: Study = STUDY) -> dict:
    """H1 re-estimated on the subset of firms with an individual reference shareholder above 10 pct.

    The flat tax reaches the firm through the resident owner-manager who steers its distribution, so
    the capital-entry null is most at risk of masking an effect among firms that have such a
    shareholder. We merge the firm-level ownership flag onto the built panel and re-estimate the two
    capital margins on the exposed firms, size-only; the non-exposed group and the full matched
    sample are carried for comparison. The flag is an Orbis ownership snapshot, so coverage is partial
    and uneven across countries, and inference still rests on six country clusters.
    """
    built = panel_engine.build(real_panel, study)
    flag = flags[[S.FIRM, "indiv_ref"]].drop_duplicates(S.FIRM)
    m = built.merge(flag, on=S.FIRM, how="left")
    groups = {"exposed": m[m["indiv_ref"] == True], "non_exposed": m[m["indiv_ref"] == False],
              "matched": m[m["indiv_ref"].notna()]}
    out = {"coverage": float(m["indiv_ref"].notna().mean()),
           "exposed_share": float(m.loc[m["indiv_ref"].notna(), "indiv_ref"].mean())}
    for g, frame in groups.items():
        out[g] = {o: _did_cell(frame.copy(), o, study)
                  for o in (S.DLOG_CAPITAL, S.CAP_INCREASE)}
    return out


# --- Covid tail and the cut at 2019 (Section 6) --------------------------------------------------

def cut_at_2019(real_panel: pd.DataFrame) -> dict:
    """H1 capital null re-estimated on the window ending in 2019, before the pandemic."""
    study = Study(last_year=2019)
    built = panel_engine.build(real_panel, study)
    d = did.fit(built, S.DLOG_CAPITAL, study, controls=BASELINE)
    return {"coef": d.coef, "event_study": d.event_study}


# --- Public series: household distribution and the 2021 decoupling (Class A) ----------------------

def dividend_index_wide(div: pd.DataFrame) -> pd.DataFrame:
    """Household-dividend index, 2017 = 100, one column per country (the figure input)."""
    return (div.pivot(index=S.YEAR, columns=S.COUNTRY, values="index_2017")
            .sort_index())


def dividend_jump(div: pd.DataFrame, study: Study = STUDY) -> dict:
    """The reform-year jump in French household dividends, levels and percent."""
    fr = div[div[S.COUNTRY] == "FR"].set_index(S.YEAR)["dividends_received"]
    ref, post = int(fr[study.reference_year]), int(fr[study.reform_year])
    return {"ref_year": study.reference_year, "reform_year": study.reform_year,
            "ref_value": ref, "reform_value": post, "pct": post / ref - 1.0}


def _index_2017(series: pd.Series) -> pd.Series:
    return 100.0 * series / series.loc[2017]


def decoupling_2021(div: pd.DataFrame, surplus: pd.DataFrame, gdp: pd.DataFrame,
                    study: Study = STUDY, year: int = 2021) -> dict:
    """France against the comparables in 2021 on dividends, corporate surplus, and GDP volume,
    all indexed to 2017, to read whether the distribution decoupled from earnings and activity.
    """
    def fr_and_comp(frame: pd.DataFrame, value: str) -> tuple[float, float]:
        wide = frame.pivot(index=S.YEAR, columns=S.COUNTRY, values=value)
        idx = wide.apply(_index_2017)
        comp = [c for c in study.controls if c in idx.columns]
        return float(idx.loc[year, "FR"]), float(idx.loc[year, comp].mean())

    out = {}
    for name, frame, value in (("dividends", div, "dividends_received"),
                               ("surplus", surplus, "value"), ("gdp", gdp, "value")):
        fr, comp = fr_and_comp(frame, value)
        out[name] = {"fr": fr, "comp": comp, "gap": fr - comp}
    return out
