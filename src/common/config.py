"""Single source of study parameters and synthetic-panel settings (no magic numbers elsewhere)."""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class Study(Frozen):
    """Pre-registered design choices for the cross-country firm panel (see protocol/)."""
    treated: str = "FR"                                   # France, treated from the reform year
    controls: tuple[str, ...] = ("DE", "IT", "ES", "BE", "NL")
    first_year: int = 2013
    reference_year: int = 2017        # omitted event-time level, just before the reform
    reform_year: int = 2018           # single flat tax starts in January
    last_year: int = 2022
    balanced_min_years: int = 6       # a firm enters the estimation panel if seen this many years
    ownership_threshold: float = 0.10  # reference-shareholder cut for the ownership-snapshot framing
    winsor_lower: float = 0.01        # continuous-outcome winsorization, declared with its threshold
    winsor_upper: float = 0.99
    ptbi_vol_window: int = 5          # years in the rolling pretax-income volatility control
    ptbi_vol_min_years: int = 3       # minimum non-missing years required to form that volatility

    @property
    def countries(self) -> tuple[str, ...]:
        return (self.treated,) + self.controls

    @property
    def years(self) -> tuple[int, ...]:
        return tuple(range(self.first_year, self.last_year + 1))

    @property
    def event_years(self) -> tuple[int, ...]:
        """Years whose event-time interaction is estimated (reference year omitted)."""
        return tuple(y for y in self.years if y != self.reference_year)


class Simulation(Frozen):
    """Synthetic firm panel for the end-to-end validation: a known treatment effect to recover."""
    seed: int = 20181
    firms_per_country: int = 4_000
    firm_capital_log_mean: float = 11.0   # lognormal location of paid-in capital across firms
    firm_capital_log_sd: float = 1.3
    yearly_growth: float = 0.03           # common year drift in log capital
    idiosyncratic_sd: float = 0.25        # within-firm year-on-year noise in log capital
    base_increase_rate: float = 0.30      # untreated probability a firm raises capital in a year
    true_post_effect: float = 0.08        # injected treated-country post-reform lift (log capital)


STUDY = Study()
SIM = Simulation()

# Statutory equity-allowance (ACE-type) schemes among the control countries, by law. Used as a
# fixed indicator in the Section 6 robustness that nets out a direct subsidy to own funds; it is
# set on the statute, not estimated. Each entry is the country and the inclusive year ranges the
# scheme was in force. France, Germany, and the Netherlands run no such scheme.
ACE_SCHEMES: dict[str, tuple[tuple[int, int], ...]] = {
    "BE": ((2006, 9999),),               # déduction des intérêts notionnels
    "IT": ((2011, 2018), (2020, 9999)),  # aiuto alla crescita economica
    "ES": ((2015, 9999),),               # reserva de capitalización
}


def ace_active(country: str, year: int) -> bool:
    """Whether an equity-allowance scheme was in force in a country and year, on the statute."""
    return any(lo <= year <= hi for lo, hi in ACE_SCHEMES.get(country, ()))
