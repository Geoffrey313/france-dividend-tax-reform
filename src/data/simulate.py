"""Synthetic firm-year panel with a known injected treatment effect, for end-to-end validation."""
from __future__ import annotations

import numpy as np
import pandas as pd

from src.common import schema as S
from src.common.config import SIM, STUDY, Simulation, Study


def simulate(sim: Simulation = SIM, study: Study = STUDY) -> pd.DataFrame:
    """Balanced firm panel over the study window; the treated country gets a post-reform lift.

    Each firm has a persistent capital level, a common yearly drift, and idiosyncratic shocks.
    For the treated country from the reform year on, a fixed increment is added to the yearly
    change in log capital; the difference-in-differences coefficient on that change recovers it.
    """
    rng = np.random.default_rng(sim.seed)
    years = np.array(study.years)
    n_years = len(years)

    frames = []
    for country in study.countries:
        n = sim.firms_per_country
        level = rng.normal(sim.firm_capital_log_mean, sim.firm_capital_log_sd, n)
        incr = rng.normal(sim.yearly_growth, sim.idiosyncratic_sd, (n, n_years))
        incr[:, 0] = 0.0
        if country == study.treated:
            incr[:, years >= study.reform_year] += sim.true_post_effect

        logk = level[:, None] + np.cumsum(incr, axis=1)
        capital = np.expm1(logk)
        # Size and the other account lines follow their own firm paths, not the treated capital
        # jump, so controls stay valid and the non-equity margins carry no spurious effect.
        asset_incr = rng.normal(sim.yearly_growth, sim.idiosyncratic_sd, (n, n_years))
        asset_incr[:, 0] = 0.0
        asset_logk = level[:, None] + np.log(4.0) + np.cumsum(asset_incr, axis=1)
        assets = np.expm1(asset_logk)
        shape = (n, n_years)
        equity = capital * (1.0 + rng.uniform(0.0, 0.5, shape))
        revenue = assets * rng.uniform(0.3, 1.5, shape)
        noncur = assets * rng.uniform(0.1, 0.4, shape)
        current = assets * rng.uniform(0.1, 0.4, shape)
        lt_debt = noncur * rng.uniform(0.3, 0.9, shape)
        loans = current * rng.uniform(0.1, 0.5, shape)
        fixed = assets * rng.uniform(0.2, 0.6, shape)
        employees = np.ceil(assets / 1e5 * rng.uniform(0.5, 1.5, shape))
        # Pretax result follows its own firm path, not the treated capital jump, so the PTBI control
        # and its volatility stay valid; a return-on-assets draw that can be negative.
        pretax = assets * rng.normal(0.05, 0.08, shape)
        inc_year = rng.integers(study.first_year - 25, study.first_year, n)

        firm_ids = [f"{country}{i:06d}" for i in range(n)]
        frame = pd.DataFrame({
            S.FIRM: np.repeat(firm_ids, n_years),
            S.COUNTRY: country,
            S.YEAR: np.tile(years, n),
            S.CAPITAL: capital.ravel(), S.EQUITY: equity.ravel(), S.ASSETS: assets.ravel(),
            S.REVENUE: revenue.ravel(), S.LT_DEBT: lt_debt.ravel(), S.LOANS: loans.ravel(),
            S.NONCUR_LIAB: noncur.ravel(), S.CUR_LIAB: current.ravel(),
            S.FIXED_ASSETS: fixed.ravel(), S.EMPLOYEES: employees.ravel(),
            S.PRETAX_INCOME: pretax.ravel(),
            S.AGE: (np.tile(years, n) - np.repeat(inc_year, n_years)).astype(float),
        })
        frames.append(frame)

    return pd.concat(frames, ignore_index=True)
