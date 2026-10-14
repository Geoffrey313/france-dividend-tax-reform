"""Estimation driver: difference-in-differences, event study, and synthetic control per outcome."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from src.common import schema as S
from src.common.config import STUDY, Study
from src.engine import panel as panel_engine
from src.engine import did, synth


@dataclass(frozen=True)
class Estimation:
    panel: pd.DataFrame
    did: dict[str, did.DidFit]       # one difference-in-differences fit per outcome
    synthetic: dict[str, synth.SynthFit]  # one country-aggregate synthetic control per outcome


def run(accounts: pd.DataFrame, outcomes: tuple[str, ...] = S.OUTCOMES,
        study: Study = STUDY) -> Estimation:
    """Build the estimation panel, then fit every outcome with the DiD and the synthetic control."""
    built = panel_engine.build(accounts, study)
    fits, synths = {}, {}
    for y in outcomes:
        fits[y] = did.fit(built, y, study)
        wide = panel_engine.aggregate_by_country_year(built, y)
        synths[y] = synth.fit(wide, study)
    return Estimation(panel=built, did=fits, synthetic=synths)
