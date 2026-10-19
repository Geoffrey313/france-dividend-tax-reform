"""Country-year difference-in-differences for the distributional extension.

This module is deliberately separate from the firm-level engine.  The unit fixed effect is a
country rather than a firm, and inference with six countries is descriptive; placebo permutation
and synthetic control, implemented by the analysis layer, carry the design-based diagnostics.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.common import schema as S
from src.common.config import Study
from src.engine.did import _demean, _ols_cluster


@dataclass(frozen=True)
class CountryDidFit:
    coef: float
    std_error: float
    n_obs: int
    n_countries: int
    event_study: pd.DataFrame

    @property
    def z(self) -> float:
        return self.coef / self.std_error if self.std_error else np.nan


def _sample(frame: pd.DataFrame, outcome: str, study: Study,
            controls: Sequence[str]) -> pd.DataFrame:
    units = (study.treated,) + study.controls
    keep = [S.COUNTRY, S.YEAR, outcome, *controls]
    out = frame.loc[
        frame[S.COUNTRY].isin(units) & frame[S.YEAR].between(study.first_year, study.last_year),
        keep,
    ].dropna(subset=[outcome, *controls]).copy()
    if out.duplicated([S.COUNTRY, S.YEAR]).any():
        raise ValueError("country-year outcomes must be unique before estimation")
    return out


def fit(frame: pd.DataFrame, outcome: str, study: Study, *, treated: str | None = None,
        controls: Sequence[str] = ()) -> CountryDidFit:
    """Estimate country and year fixed-effects DiD and its event-study path.

    ``treated`` is overridable for in-space placebo permutation.  All countries in ``study`` stay
    in the estimation sample, so every placebo uses the same outcome matrix and time window.
    """
    treated = treated or study.treated
    base = _sample(frame, outcome, study, controls)
    if treated not in set(base[S.COUNTRY]):
        raise ValueError(f"treated country {treated!r} is absent from the estimation sample")

    base[S.TREAT_POST] = (
        (base[S.COUNTRY] == treated) & (base[S.YEAR] >= study.reform_year)
    ).astype(int)
    regressors = [S.TREAT_POST, *controls]
    resid = _demean(base, [outcome, *regressors], [S.COUNTRY, S.YEAR])
    beta, se = _ols_cluster(
        resid[:, 1:], resid[:, 0], base[S.COUNTRY].to_numpy()
    )

    # With partially observed robustness controls, a placebo country may lack a country-year.
    # Do not create an all-zero event dummy for such a year, which would make X'X singular.
    treated_years = set(base.loc[base[S.COUNTRY].eq(treated), S.YEAR])
    present = sorted(y for y in treated_years if y != study.reference_year)
    event = base.copy()
    names: list[str] = []
    for year in present:
        name = f"treat_{year}"
        event[name] = (
            (event[S.COUNTRY] == treated) & (event[S.YEAR] == year)
        ).astype(int)
        names.append(name)
    eresid = _demean(event, [outcome, *names, *controls], [S.COUNTRY, S.YEAR])
    ebeta, ese = _ols_cluster(
        eresid[:, 1:], eresid[:, 0], event[S.COUNTRY].to_numpy()
    )
    event_study = pd.DataFrame(
        [{"year": year, "coef": ebeta[i], "std_error": ese[i]}
         for i, year in enumerate(present)]
    ).set_index("year")

    return CountryDidFit(
        coef=float(beta[0]),
        std_error=float(se[0]),
        n_obs=len(base),
        n_countries=base[S.COUNTRY].nunique(),
        event_study=event_study,
    )


def placebo_permutation(frame: pd.DataFrame, outcome: str, study: Study,
                        controls: Sequence[str] = ()) -> dict:
    """Treat every country in turn and rank France's absolute DiD coefficient among them."""
    countries = (study.treated,) + study.controls
    coefficients = {
        country: fit(frame, outcome, study, treated=country, controls=controls).coef
        for country in countries
    }
    treated_abs = abs(coefficients[study.treated])
    rank = sum(abs(value) >= treated_abs for value in coefficients.values())
    return {
        "coefficients": coefficients,
        "placebo_p": rank / len(countries),
        "n_countries": len(countries),
    }
