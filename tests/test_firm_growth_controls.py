from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.analysis import firm_growth_controls as controls
from src.common import schema as S


class FirmGrowthControlsTest(unittest.TestCase):
    def _panel(self) -> pd.DataFrame:
        study = controls.SHOCKS["2013"]
        rows = []
        for country_index, country in enumerate(study.countries):
            for firm in range(8):
                previous_growth = 0.0
                for year in study.years:
                    treatment = 0.04 if country == "FR" and year >= study.reform_year else 0.0
                    growth = 0.01 * country_index + 0.2 * previous_growth + treatment
                    rows.append({
                        S.FIRM: f"{country}_{firm}", S.COUNTRY: country, S.YEAR: year,
                        "sector_stable_2013": str(10 + firm % 2),
                        "tangible_growth_real": growth,
                        "lag_tangible_growth_real": previous_growth if year > study.first_year else np.nan,
                        "lag_book_leverage": 0.4 + 0.01 * firm,
                        "lag_log_assets_real": 8.0 + 0.02 * firm,
                        "lag_log_firm_age": np.log1p(10 + year - study.first_year),
                    })
                    previous_growth = growth
        return pd.DataFrame(rows)

    def test_a_b_c_share_one_complete_case_sample(self) -> None:
        result = controls._estimate(
            self._panel(), "tangible_growth_real", controls.SHOCKS["2013"]
        )
        self.assertEqual(result["status"], "estimated")
        self.assertEqual(
            set(result["specifications"]),
            {
                "A_fe_common_sample", "B_controls_no_ptbi", "C_dynamic_no_ptbi",
                "B_controls_no_ptbi_no_age", "C_dynamic_no_ptbi_no_age",
                "B_controls_no_ptbi_no_size", "C_dynamic_no_ptbi_no_size",
            },
        )
        self.assertGreater(result["n_obs"], 0)
        self.assertGreater(result["specifications"]["C_dynamic_no_ptbi"]["coef"], 0)

    def test_control_gate_fails_closed(self) -> None:
        panel = self._panel()
        panel.loc[panel[S.COUNTRY].eq("NL"), "lag_book_leverage"] = np.nan
        result = controls._estimate(
            panel, "tangible_growth_real", controls.SHOCKS["2013"]
        )
        self.assertEqual(result["status"], "not_estimated_low_coverage")

    def test_dynamic_gate_keeps_static_controlled_model(self) -> None:
        panel = self._panel()
        panel.loc[panel[S.COUNTRY].eq("NL"), "lag_tangible_growth_real"] = np.nan
        result = controls._estimate(
            panel, "tangible_growth_real", controls.SHOCKS["2013"]
        )
        self.assertEqual(result["status"], "estimated_static_only_dynamic_low_coverage")
        self.assertIn("B_controls_no_ptbi", result["specifications"])
        self.assertNotIn("C_dynamic_no_ptbi", result["specifications"])

    def test_estimator_accepts_an_explicit_control_set(self) -> None:
        panel = self._panel().rename(columns={"lag_log_firm_age": "lag_log_reporting_age"})
        selected = ("lag_book_leverage", "lag_log_assets_real", "lag_log_reporting_age")
        result = controls._estimate(
            panel, "tangible_growth_real", controls.SHOCKS["2013"], selected
        )
        self.assertEqual(
            result["specifications"]["B_controls_no_ptbi"]["controls"], list(selected)
        )

    def test_harmonized_spec_absorbs_industry_by_year(self) -> None:
        result = controls._estimate(
            self._panel(), "tangible_growth_real", controls.SHOCKS["2013"],
            sector_column="sector_stable_2013",
        )
        self.assertEqual(result["status"], "estimated")
        self.assertIn("C_controls_no_ptbi", result["specifications"])
        self.assertIn("D_dynamic_no_ptbi", result["specifications"])
        self.assertEqual(
            result["specifications"]["C_controls_no_ptbi"]["fixed_effects"],
            ["firm", "industry_by_year"],
        )
        self.assertEqual(
            result["specifications"]["D_dynamic_no_ptbi"]["controls"][-1],
            "lag_tangible_growth_real",
        )
        pretrend = result["specifications"]["C_controls_no_ptbi"]["pretrend_joint_test"]
        self.assertEqual(pretrend["status"], "estimated_diagnostic_six_country_clusters")
        self.assertEqual(pretrend["years"], [2009, 2010, 2011])
        self.assertGreaterEqual(pretrend["conventional_p"], 0.0)
        self.assertLessEqual(pretrend["conventional_p"], 1.0)


if __name__ == "__main__":
    unittest.main()
