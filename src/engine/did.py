"""Two-way fixed-effects difference-in-differences with cluster-robust inference and event study."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.common import schema as S
from src.common.config import STUDY, Study


def _demean(frame: pd.DataFrame, cols: list[str], groups: list[str],
            tol: float = 1e-8, max_iter: int = 1000) -> np.ndarray:
    """Absorb the fixed effects by alternating within-group demeaning (vectorized over groups)."""
    x = frame[cols].to_numpy(dtype=float)
    codes, counts = [], []
    for g in groups:
        c = pd.factorize(frame[g], sort=False)[0]
        codes.append(c)
        counts.append(np.bincount(c).astype(float))
    for _ in range(max_iter):
        shift = 0.0
        for c, cnt in zip(codes, counts):
            for j in range(x.shape[1]):
                means = np.bincount(c, weights=x[:, j], minlength=cnt.size) / cnt
                adj = means[c]
                x[:, j] -= adj
                shift = max(shift, np.abs(adj).max())
        if shift < tol:
            break
    return x


def _ols_cluster(x: np.ndarray, y: np.ndarray, cluster: np.ndarray):
    """Normal-equation OLS with a cluster-robust sandwich covariance."""
    xtx_inv = np.linalg.inv(x.T @ x)
    beta = xtx_inv @ (x.T @ y)
    resid = y - x @ beta
    meat = np.zeros((x.shape[1], x.shape[1]))
    for c in np.unique(cluster):
        s = x[cluster == c].T @ resid[cluster == c]
        meat += np.outer(s, s)
    cov = xtx_inv @ meat @ xtx_inv
    # Round-off can leave a variance at a tiny negative value in exact-fit synthetic tests.
    return beta, np.sqrt(np.clip(np.diag(cov), 0.0, None))


@dataclass(frozen=True)
class DidFit:
    coef: float
    std_error: float
    n_obs: int
    n_firms: int
    event_study: pd.DataFrame

    @property
    def z(self) -> float:
        return self.coef / self.std_error if self.std_error else np.nan


def _fit_core(frame: pd.DataFrame, outcome: str, regressors: list[str], cluster_col: str):
    """Absorb firm and year effects, run OLS on the residualized design, return coef and SE."""
    resid = _demean(frame, [outcome] + regressors, [S.FIRM, S.YEAR])
    return _ols_cluster(resid[:, 1:], resid[:, 0], frame[cluster_col].to_numpy())


def fit(panel: pd.DataFrame, outcome: str, study: Study = STUDY,
        cluster_col: str = S.COUNTRY, controls: Sequence[str] | None = None) -> DidFit:
    """Estimate the single difference-in-differences coefficient and the event-study path.

    Firm and year fixed effects are absorbed. The point estimate is the coefficient on the
    treated-country-after-reform interaction; the event study replaces it with treated-by-year
    interactions relative to the omitted reference year. Standard errors are clustered; with one
    treated country the cluster count is small, so the synthetic-control placebos carry inference.

    By default the controls are the referee set for the outcome (schema.controls_for). Passing
    ``controls`` overrides that set, so the displayed baseline (size only) is one call with
    ``controls=[schema.SIZE]`` and no code duplication.
    """
    # Control vector for this outcome, limited to the columns the panel actually carries; the
    # enriched extract adds the pretax-income controls, and the balance filter plus the one-year lag
    # inside the ratio controls trim the earliest event-time years.
    wanted = list(controls) if controls is not None else list(S.controls_for(outcome))
    controls = [c for c in wanted if c in panel.columns]
    base = panel.dropna(subset=[outcome, *controls]).copy()

    beta, se = _fit_core(base, outcome, [S.TREAT_POST] + controls, cluster_col)

    present = sorted(y for y in base[S.YEAR].unique() if y != study.reference_year)
    event = base.copy()
    names = []
    for y in present:
        name = f"treat_{y}"
        event[name] = event[S.TREATED] * (event[S.YEAR] == y).astype(int)
        names.append(name)
    eb, ese = _fit_core(event, outcome, names + controls, cluster_col)
    es = pd.DataFrame([{"year": y, "coef": eb[i], "std_error": ese[i]}
                       for i, y in enumerate(present)]).set_index("year")

    return DidFit(coef=float(beta[0]), std_error=float(se[0]),
                  n_obs=len(base), n_firms=base[S.FIRM].nunique(), event_study=es)
