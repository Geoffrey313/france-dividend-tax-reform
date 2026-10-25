from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from src.analysis import firm_square_samples as square
from src.analysis.firm_growth_controls import SHOCKS


class FirmSquareSamplesTest(unittest.TestCase):
    def _frame(self) -> pd.DataFrame:
        rows = []
        for firm, country in (("a", "FR"), ("b", "DE")):
            for year in (2011, 2013, 2014):
                rows.append({
                    "firm_id": firm, "country": country, "year": year,
                    "y1": float(year), "y2": float(year + 1),
                    "lag_book_leverage": 0.2,
                    "lag_log_assets_real": 4.0,
                    "lag_log_firm_age": 2.0,
                    "sector_stable_2013": "20",
                })
        return pd.DataFrame(rows)

    def test_square_sample_has_identical_complete_rows_and_pre_post(self) -> None:
        sample, manifest = square.build_square_sample(
            self._frame(), population="private_large_vl", shock="2013",
            study=SHOCKS["2013"], family="test", dependent_variables=("y1", "y2"),
        )
        self.assertEqual(len(sample), 6)
        self.assertEqual(manifest["final_firms"], 2)
        self.assertFalse(sample[["y1", "y2", *square.CONTROLS]].isna().any().any())

    def test_controls_are_not_filled(self) -> None:
        frame = self._frame()
        frame.loc[0, "lag_log_firm_age"] = np.nan
        sample, manifest = square.build_square_sample(
            frame, population="private_large_vl", shock="2013",
            study=SHOCKS["2013"], family="test", dependent_variables=("y1", "y2"),
        )
        # Dropping the only pre-reform row for firm a also removes that firm under the mandatory
        # pre/post rule; no value is imputed to keep it.
        self.assertEqual(len(sample), 3)
        self.assertEqual(set(sample.firm_id), {"b"})
        self.assertEqual(manifest["missing_policy"], "listwise_no_fillna")

    def test_firm_without_post_observation_is_removed(self) -> None:
        frame = self._frame()
        frame = frame[~((frame.firm_id == "b") & (frame.year >= 2013))]
        sample, _ = square.build_square_sample(
            frame, population="private_large_vl", shock="2013",
            study=SHOCKS["2013"], family="test", dependent_variables=("y1", "y2"),
        )
        self.assertEqual(set(sample.firm_id), {"a"})


if __name__ == "__main__":
    unittest.main()
