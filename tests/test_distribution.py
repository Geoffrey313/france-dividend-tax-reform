from __future__ import annotations

import unittest

from src.analysis import distribution


class DistributionDataTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.panel = distribution.load_panel()

    def test_panel_is_balanced_and_complete(self) -> None:
        self.assertEqual(len(self.panel), 6 * 13)
        self.assertFalse(self.panel[list(distribution.OUTCOMES)].isna().any().any())
        self.assertGreater(
            self.panel[list(distribution.ASSET_CONTROL_COLUMNS)].notna().mean().min(), 0.95
        )

    def test_quality_is_outcome_specific(self) -> None:
        france_2018 = self.panel[
            self.panel["country"].eq("FR") & self.panel["year"].eq(2018)
        ].iloc[0]
        self.assertEqual(france_2018["quality_net_wealth"], 0)
        self.assertGreaterEqual(france_2018["quality_pretax_income"], 3)
        self.assertGreaterEqual(france_2018["quality_posttax_income"], 3)

    def test_outcome_hierarchy_does_not_promote_imputed_wealth(self) -> None:
        self.assertTrue(all("income" in outcome for outcome in distribution.SIGNAL_OUTCOMES))
        self.assertTrue(all("net_wealth" in outcome for outcome in distribution.CONTEXT_OUTCOMES))

    def test_required_pool_sensitivities_are_declared(self) -> None:
        self.assertNotIn("IT", distribution.POOL_CONTROLS["drop_italy"])
        self.assertNotIn("BE", distribution.POOL_CONTROLS["drop_belgium"])
        self.assertNotIn("IT", distribution.POOL_CONTROLS["clean_controls"])
        self.assertNotIn("BE", distribution.POOL_CONTROLS["clean_controls"])


if __name__ == "__main__":
    unittest.main()
