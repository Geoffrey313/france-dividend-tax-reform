"""Synthetic control at the country-aggregate level, with placebo permutation for inference."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from src.common.config import STUDY, Study


def _weights(donor_pre: np.ndarray, target_pre: np.ndarray) -> np.ndarray:
    """Non-negative weights summing to one that best match the target over the pre-period."""
    n = donor_pre.shape[1]
    w0 = np.full(n, 1.0 / n)
    loss = lambda w: float(np.sum((donor_pre @ w - target_pre) ** 2))
    res = minimize(loss, w0, method="SLSQP",
                   bounds=[(0.0, 1.0)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}])
    return res.x


def _gap_for_unit(wide: pd.DataFrame, unit: str, donors: list[str],
                  pre: np.ndarray, post: np.ndarray) -> pd.Series:
    """Fit weights on the pre-period donors and return the unit-minus-synthetic gap each year."""
    d = wide[donors].to_numpy()
    t = wide[unit].to_numpy()
    w = _weights(d[pre], t[pre])
    return pd.Series(t - d @ w, index=wide.index)


@dataclass(frozen=True)
class SynthFit:
    gap: pd.Series                 # treated country minus its synthetic twin, by year
    placebo_gaps: pd.DataFrame     # same gap computed for each control unit in turn
    post_effect: float             # mean treated gap over the post-reform years
    placebo_rank_p: float          # permutation p-value from the post-period gap ratio


def fit(wide: pd.DataFrame, study: Study = STUDY) -> SynthFit:
    """Build the treated gap and the control placebos, and rank the treated effect among them."""
    years = wide.index.to_numpy()
    pre = years <= study.reference_year
    post = years >= study.reform_year
    units = [study.treated] + list(study.controls)

    gaps = {u: _gap_for_unit(wide, u, [c for c in units if c != u], pre, post)
            for u in units}
    placebo = pd.DataFrame({u: gaps[u] for u in study.controls})
    treated_gap = gaps[study.treated]

    post_effect = float(treated_gap[post].mean())
    treated_stat = abs(post_effect)
    placebo_stat = placebo.loc[post].mean().abs()
    rank = 1 + int((placebo_stat >= treated_stat).sum())
    p = rank / (len(study.controls) + 1)

    return SynthFit(gap=treated_gap, placebo_gaps=placebo,
                    post_effect=post_effect, placebo_rank_p=p)
