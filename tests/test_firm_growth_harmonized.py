from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.analysis import firm_growth_harmonized as analysis
from src.analysis import firm_growth_square_robustness as country_robustness
from src.analysis.firm_growth_controls import SHOCKS
from src.figures import firm_reporting


class FirmGrowthHarmonizedTest(unittest.TestCase):
    @staticmethod
    def _reporting_fixture() -> dict:
        populations: dict = {}
        for population in {panel[0] for panel in firm_reporting.PANELS}:
            outcomes = {outcome for panel in firm_reporting.PANELS if panel[0] == population
                        for outcome in panel[1]}
            populations[population] = {}
            for shock, years in (("2013", tuple(range(2008, 2017))),
                                 ("2018_precovid", tuple(range(2014, 2020)))):
                populations[population][shock] = {
                    outcome: {"n_obs": 100, "n_firms": 20,
                              "specifications": {firm_reporting.SPECIFICATION: {
                        "coef": 0.01,
                        "std_error": 0.005,
                        "placebo_p": 0.5,
                        "control_estimates": {
                            "lag_book_leverage": {"coef": 0.01, "std_error": 0.005},
                            "lag_log_assets_real": {"coef": 0.02, "std_error": 0.01},
                            "lag_log_firm_age": {"coef": -0.01, "std_error": 0.005},
                        },
                        "pretrend_joint_test": {"statistic": 1.0, "conventional_p": 0.5},
                        "event_study": [
                            {"year": year, "coef": 0.01, "std_error": 0.02}
                            for year in years
                        ],
                    }}}
                    for outcome in outcomes
                }
        contrasts = {
            population: {
                outcome: {"contrast": 0.02, "t_stat": 2.0, "two_sided_placebo_p": 0.5}
                for panel_population, panel_outcomes, *_ in firm_reporting.PANELS
                if panel_population == population
                for outcome in panel_outcomes
            }
            for population in populations
        }
        return {
            "populations": populations,
            "private_cross_shock_contrasts": {"2018_precovid": contrasts["private_large_vl"]},
            "public_cross_shock_contrasts": {"2018_precovid": contrasts["public_listed"]},
        }

    def test_event_study_table_covers_pre_and_post_periods(self) -> None:
        table = firm_reporting._event_study_table(self._reporting_fixture(), "en", "2013", 2013)
        self.assertIn("around the 2013 Reform", table)
        self.assertNotIn("around the 2018 Reform", table)
        self.assertIn("Dependent variables:", table)
        self.assertIn("Year $(t-3)$", table)
        self.assertIn("Year $(t-4)$", table)
        self.assertIn("Year $(t-5)$", table)
        self.assertNotIn("Year $(t-1)$", table)
        self.assertIn("Year $(t)$", table)
        self.assertIn("Year $(t+1)$", table)
        self.assertIn("Year $(t+3)$", table)
        self.assertIn("Joint pre-trend Wald", table)
        self.assertIn("Firm-year observations", table)
        self.assertNotIn("Distinct firms", table)
        self.assertIn("\\ContinuedFloat", table)
        self.assertIn("\\captionsetup[subfloat]{position=top}", table)
        self.assertEqual(table.count("\\setcounter{table}{\\firmeventtablecounter}"), 1)
        self.assertIn("$p$-values below 0.05 are in bold", table)

    def test_results_table_reports_controls_and_one_observation_row(self) -> None:
        table = firm_reporting._results_table(self._reporting_fixture(), "en", "2013", 2013)
        self.assertIn("Private Firms Accounting Results", table)
        self.assertIn("Listed Firms Accounting Results", table)
        self.assertIn("table:firm-square-results-2013-private", table)
        self.assertIn("table:firm-square-results-2013-listed", table)
        self.assertEqual(table.count("Panel A :"), 2)
        self.assertEqual(table.count("Panel B :"), 2)
        self.assertEqual(table.count("Panel C :"), 2)
        self.assertIn(r"\SetRow{font=\scriptsize\itshape} Two-sided placebo $p$", table)
        self.assertIn("Leverage $(t-1)$", table)
        self.assertNotIn("Control variables", table)
        self.assertNotIn("Controls $(t-1)$", table)
        self.assertIn("Observations", table)
        self.assertNotIn("2013 / 2018", table)
        self.assertNotIn("2013 minus 2018 gap", table)
        self.assertNotIn("Distinct firms", table)

    def test_event_figure_draws_reform_and_zero_lines(self) -> None:
        output = self._reporting_fixture()
        figure = firm_reporting._event_figure(
            output, "private_large_vl", ("dlog_capital",), 0, "en"
        )
        self.assertIn("extra x ticks={2012.5,2017.5}", figure)
        self.assertIn("extra y ticks={0}", figure)

    def test_equal_country_weights_give_identical_country_totals(self) -> None:
        countries = pd.Series(["FR", "FR", "FR", "DE", "IT", "IT"])
        weights = country_robustness._equal_country_weights(countries)
        totals = pd.Series(weights).groupby(countries).sum()
        self.assertTrue(np.allclose(totals, totals.iloc[0]))
        self.assertAlmostEqual(float(weights.mean()), 1.0)

    def test_coverage_gives_each_country_year_equal_weight(self) -> None:
        rows = []
        for index in range(10):
            rows.append({
                "firm_id": f"a{index}", "country": "FR", "year": 2010,
                "outcome": np.nan,
            })
        rows.append({"firm_id": "b", "country": "FR", "year": 2011, "outcome": 1.0})
        result = analysis._outcome_coverage(
            pd.DataFrame(rows), "outcome", SHOCKS["2013"]
        )
        self.assertAlmostEqual(result["by_country"]["FR"], 0.5)

    def test_2013_window_exposes_four_pretrend_years(self) -> None:
        study = SHOCKS["2013"]
        self.assertEqual(study.first_year, 2008)
        self.assertEqual(study.reference_year, 2012)
        self.assertEqual(
            list(range(study.first_year, study.reference_year)),
            [2008, 2009, 2010, 2011],
        )

    def test_family_test_uses_only_declared_hypothesis_outcomes(self) -> None:
        specification = "C_controls_no_ptbi"
        results = {
            "included": {
                "status": "estimated",
                "specifications": {specification: {
                    "placebo_coefficients": {
                        "FR": 2.0, "DE": 1.0, "IT": 0.0,
                        "ES": -1.0, "BE": -2.0, "NL": 0.5,
                    },
                }},
            },
            "excluded": {
                "status": "estimated",
                "specifications": {specification: {
                    "placebo_coefficients": {
                        "FR": 100.0, "DE": 0.0, "IT": 0.0,
                        "ES": 0.0, "BE": 0.0, "NL": 0.0,
                    },
                }},
            },
        }
        family = analysis._family_max_t(results, specification, ("included",))
        self.assertEqual(family["outcomes"], ["included"])
        self.assertEqual(family["placebo_p"], 2 / 6)

    def test_cross_shock_contrast_uses_same_country_placebos(self) -> None:
        specification = "C_controls_no_ptbi"
        def result(france: float, germany: float, standard_error: float) -> dict:
            coefficients = {
                "FR": france, "DE": germany, "IT": 0.0,
                "ES": 0.0, "BE": 0.0, "NL": 0.0,
            }
            return {"status": "estimated", "specifications": {
                specification: {
                    "placebo_coefficients": coefficients,
                    "std_error": standard_error,
                },
            }}
        results = {
            "2013": {"capital": result(0.03, 0.01, 0.02)},
            "2018_precovid": {"capital": result(-0.01, 0.00, 0.01)},
        }
        contrast = analysis._cross_shock_contrasts(results, "2018_precovid")["capital"]
        self.assertAlmostEqual(contrast["contrast"], 0.04)
        self.assertEqual(contrast["one_sided_placebo_p"], 1 / 6)
        self.assertAlmostEqual(contrast["std_error"], np.sqrt(0.02**2 + 0.01**2))
        self.assertAlmostEqual(
            contrast["t_stat"], 0.04 / np.sqrt(0.02**2 + 0.01**2)
        )

        reversed_results = {
            "2013": {
                "z_outcome": result(0.03, 0.01, 0.02),
                "a_outcome": result(0.02, 0.00, 0.01),
            },
            "2018_precovid": {
                "a_outcome": result(-0.01, 0.00, 0.01),
                "z_outcome": result(-0.02, 0.00, 0.01),
            },
        }
        ordered = analysis._cross_shock_contrasts(reversed_results, "2018_precovid")
        self.assertEqual(list(ordered), ["a_outcome", "z_outcome"])

    def test_manuscript_events_restore_omitted_reference_year(self) -> None:
        def outcome(years: list[int]) -> dict:
            return {"status": "estimated_static_only_by_design", "specifications": {"C_controls_no_ptbi": {"event_study": [
                {"year": year, "coef": 0.01, "std_error": 0.02} for year in years
            ]}}}

        results = {"private_large_vl": {
            "2013": {
                name: outcome([2011, 2013]) for name in analysis.MANUSCRIPT_EVENT_OUTCOMES
                if name in analysis.square.outcomes("private_large_vl")
            },
            "2018_precovid": {
                name: outcome([2016, 2018]) for name in analysis.MANUSCRIPT_EVENT_OUTCOMES
                if name in analysis.square.outcomes("private_large_vl")
            },
        }}
        events = analysis._manuscript_event_rows(results)
        references = events[events["year"].isin([2012, 2017])]
        expected = len(set(analysis.MANUSCRIPT_EVENT_OUTCOMES).intersection(
            analysis.square.outcomes("private_large_vl")
        ))
        self.assertEqual(len(references), 2 * expected)
        self.assertTrue(references["coef"].eq(0.0).all())
        self.assertTrue(references["ci95"].eq(0.0).all())

    def test_descriptive_rows_use_nonmissing_observations(self) -> None:
        variables = analysis.DESCRIPTIVE_VARIABLES["private_large_vl"]
        frame = pd.DataFrame({name: [1.0, 2.0, np.nan] for name in variables})
        output = analysis._descriptive_rows(
            frame, "private_large_vl", "2018_precovid", "capital", variables,
        )
        row = output.set_index("variable").loc["dlog_capital"]
        self.assertEqual(row["n"], 2)
        self.assertAlmostEqual(row["mean"], 1.5)
        self.assertAlmostEqual(row["median"], 1.5)
        self.assertEqual(row["shock"], "2018_precovid")
        self.assertEqual(row["family"], "capital")


if __name__ == "__main__":
    unittest.main()
