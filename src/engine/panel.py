"""Turn a firm-year account panel into the estimation panel: outcomes, treatment, balance filter."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common import schema as S
from src.common.config import STUDY, Study
from src.common.util import winsorize

# Continuous outcomes are winsorized against Orbis outliers; the indicator is left as is.
# The last two are only present with the enriched extract, so winsorization skips absent columns.
_WINSORIZED = (S.DLOG_CAPITAL, S.EQUITY_RATIO, S.LEVERAGE, S.DLOG_DEBT,
               S.FIN_DEBT_RATIO, S.DLOG_FIXED, S.LOG_LABOR_PROD, S.RESERVES_RATIO,
               S.RESERVES_NET_RATIO, S.CASH_RATIO,
               S.PTBI, S.PTBI_VOL, S.LEV_CTRL)


def _dlog_within(panel: pd.DataFrame, level: pd.Series) -> pd.Series:
    """Year-on-year change of log(1 + level) within a firm, ordered by year."""
    tmp = panel[[S.FIRM, S.YEAR]].assign(v=np.log1p(level.clip(lower=0).to_numpy()))
    return tmp.sort_values([S.FIRM, S.YEAR]).groupby(S.FIRM)["v"].diff()


def _lag_within(panel: pd.DataFrame, level: pd.Series) -> pd.Series:
    """One-year lag of a level within a firm, ordered by year."""
    tmp = panel[[S.FIRM, S.YEAR]].assign(v=level.to_numpy())
    return tmp.sort_values([S.FIRM, S.YEAR]).groupby(S.FIRM)["v"].shift(1)


def _ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    """Safe ratio, missing where the denominator is not strictly positive."""
    out = num / den.where(den > 0)
    return out


def build(panel: pd.DataFrame, study: Study = STUDY) -> pd.DataFrame:
    """Add treatment and the equity, debt, and real outcome columns, then keep balanced firms.

    Input: one row per firm-year with the account lines. Output: the same rows for firms seen at
    least the balance minimum, with the difference-in-differences regressor and all outcomes.
    """
    df = panel[panel[S.YEAR].between(study.first_year, study.last_year)].copy()
    df = df[df[S.COUNTRY].isin(study.countries)]

    df[S.TREATED] = (df[S.COUNTRY] == study.treated).astype(int)
    df[S.POST] = (df[S.YEAR] >= study.reform_year).astype(int)
    df[S.TREAT_POST] = df[S.TREATED] * df[S.POST]
    df[S.LOG_ASSETS] = np.log1p(df[S.ASSETS].clip(lower=0))

    df[S.DLOG_CAPITAL] = _dlog_within(df, df[S.CAPITAL])
    df[S.CAP_INCREASE] = (df[S.DLOG_CAPITAL] > 0).astype(float)
    df.loc[df[S.DLOG_CAPITAL].isna(), S.CAP_INCREASE] = np.nan

    df[S.EQUITY_RATIO] = _ratio(df[S.EQUITY], df[S.ASSETS])
    df[S.RESERVES] = df[S.EQUITY] - df[S.CAPITAL]
    df[S.RESERVES_RATIO] = _ratio(df[S.RESERVES], df[S.ASSETS])

    # Source-of-funds columns from the enriched extract; absent in the synthetic panel, so guarded.
    if S.NET_INCOME in df.columns:
        df[S.RESERVES_NET] = df[S.RESERVES] - df[S.NET_INCOME].fillna(0)
        df[S.RESERVES_NET_RATIO] = _ratio(df[S.RESERVES_NET], df[S.ASSETS])
    if S.CASH in df.columns:
        df[S.CASH_RATIO] = _ratio(df[S.CASH], df[S.ASSETS])
    df[S.LEVERAGE] = _ratio(df[S.NONCUR_LIAB].fillna(0) + df[S.CUR_LIAB].fillna(0), df[S.ASSETS])
    df[S.FIN_DEBT_RATIO] = _ratio(df[S.LT_DEBT].fillna(0) + df[S.LOANS].fillna(0), df[S.ASSETS])
    df[S.DLOG_DEBT] = _dlog_within(df, df[S.LT_DEBT].fillna(0) + df[S.LOANS].fillna(0))
    df[S.DLOG_FIXED] = _dlog_within(df, df[S.FIXED_ASSETS])
    df[S.LOG_LABOR_PROD] = np.log(_ratio(df[S.REVENUE].where(df[S.REVENUE] > 0),
                                         df[S.EMPLOYEES]))

    # Referee control set (schema.CONTROLS). Ratio controls scale by lagged total assets; log firm
    # age is nonlinear in the year, so the firm and year effects do not absorb it. PTBI and its
    # volatility need the pretax-income line from the enriched extract, so they are guarded like the
    # source-of-funds columns and are absent in the synthetic panel only until it is enriched too.
    ta_lag = _lag_within(df, df[S.ASSETS])
    df[S.LEV_CTRL] = _ratio(df[S.LT_DEBT], ta_lag)
    df[S.LOG_AGE] = np.log1p(df[S.AGE].clip(lower=0))
    if S.PRETAX_INCOME in df.columns:
        df[S.PTBI] = _ratio(df[S.PRETAX_INCOME], ta_lag)
        df[S.PTBI_VOL] = (df.sort_values([S.FIRM, S.YEAR]).groupby(S.FIRM)[S.PTBI]
                          .rolling(study.ptbi_vol_window, min_periods=study.ptbi_vol_min_years)
                          .std().reset_index(level=0, drop=True))

    for col in _WINSORIZED:
        if col not in df.columns:
            continue
        present = df[col].notna()
        df.loc[present, col] = winsorize(df.loc[present, col], study)

    seen = df.groupby(S.FIRM)[S.YEAR].transform("nunique")
    return df[seen >= study.balanced_min_years].reset_index(drop=True)


def aggregate_by_country_year(panel: pd.DataFrame, outcome: str) -> pd.DataFrame:
    """Country-by-year mean of an outcome, the input to the synthetic control."""
    return (panel.groupby([S.COUNTRY, S.YEAR])[outcome].mean()
            .unstack(S.COUNTRY).sort_index())
