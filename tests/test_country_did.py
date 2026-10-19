from __future__ import annotations

import unittest

import pandas as pd

from src.common.config import Study
from src.engine import country_did


class CountryDidTest(unittest.TestCase):
    def test_recovers_injected_country_post_effect(self) -> None:
        study = Study(first_year=2010, reference_year=2012, reform_year=2013, last_year=2016)
        country_effect = {country: i / 100 for i, country in enumerate(study.countries)}
        rows = []
        for country in study.countries:
            for year in study.years:
                treated_post = country == "FR" and year >= study.reform_year
                rows.append({
                    "country": country,
                    "year": year,
                    "outcome": country_effect[country] + (year - 2010) / 200
                    + (0.08 if treated_post else 0.0),
                })
        fit = country_did.fit(pd.DataFrame(rows), "outcome", study)
        self.assertAlmostEqual(fit.coef, 0.08, places=10)
        for year in (2010, 2011):
            self.assertAlmostEqual(fit.event_study.loc[year, "coef"], 0.0, places=10)
        for year in (2013, 2014, 2015, 2016):
            self.assertAlmostEqual(fit.event_study.loc[year, "coef"], 0.08, places=10)

    def test_placebo_rank_uses_all_six_countries(self) -> None:
        study = Study(first_year=2010, reference_year=2012, reform_year=2013, last_year=2016)
        rows = [
            {
                "country": country,
                "year": year,
                "outcome": 0.10 * (country == "FR" and year >= 2013),
            }
            for country in study.countries
            for year in study.years
        ]
        result = country_did.placebo_permutation(pd.DataFrame(rows), "outcome", study)
        self.assertEqual(result["n_countries"], 6)
        self.assertAlmostEqual(result["placebo_p"], 1 / 6)


if __name__ == "__main__":
    unittest.main()
