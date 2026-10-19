"""Regenerate the manuscript's figures and numbers from committed code, then gate on consistency.

Three stages:
  self-test  a synthetic panel with a known injected effect, recovered end to end (seconds);
  class A    the public series (Eurostat), always runs: the household-dividend figure, the reform
             jump, and the 2021 decoupling;
  class B    the firm panels (Orbis, licensed, under data/raw): the capital and leverage tables,
             the event-study figures, the cash and financial-debt tail, the ratchet, and the
             descriptive table. Runs only when data/raw/firm_panel_real.parquet is present,
             otherwise it is skipped and class A still completes.

Class B ends by re-reading the manuscript and asserting every regenerated headline number matches
what the paper displays, to its displayed precision; a divergence exits non-zero. Each regenerated
CSV is printed with a SHA-256 digest.
"""
from __future__ import annotations

import json
import warnings

import pandas as pd

from src.common import paths
from src.common.config import SIM, STUDY
from src.common import schema as S
from src.data import simulate
from src.analysis import distribution, estimate, results
from src.figures import emit

warnings.filterwarnings("ignore")  # the ratio outcomes divide by zero on empty balance lines
RAW = paths.DATA / "raw"


def self_test() -> None:
    """Recover a known injected capital effect on a synthetic panel, with null other margins."""
    est = estimate.run(simulate.simulate())
    cap = est.did[S.DLOG_CAPITAL]
    pre = cap.event_study.loc[cap.event_study.index < STUDY.reform_year]
    assert (pre["coef"].abs() < (3.0 * pre["std_error"]).clip(lower=0.01)).all(), pre
    assert abs(cap.coef - SIM.true_post_effect) < max(3.0 * cap.std_error, 0.01), cap.coef
    for margin in (S.LEVERAGE, S.DLOG_DEBT, S.DLOG_FIXED):
        f = est.did[margin]
        assert abs(f.coef) < max(3.0 * f.std_error, 0.01), (margin, f.coef)
    print(f"self-test     capital recovered {cap.coef:+.4f} (injected {SIM.true_post_effect:+.4f}); "
          "other margins null; PASS")


def class_a(out: dict) -> None:
    """Public Eurostat/WID series, including the distributional extension."""
    div = pd.read_csv(paths.DATA / "dividends_cross_country.csv")
    surplus = pd.read_csv(paths.DATA / "corp_surplus_2010_2022.csv")
    gdp = pd.read_csv(paths.DATA / "gdp_volume_2010_2022.csv")

    print("class A       public series")
    print("  " + emit.dividend_index_csv(results.dividend_index_wide(div)))
    out["dividend_jump"] = results.dividend_jump(div)
    out["decoupling_2021"] = results.decoupling_2021(div, surplus, gdp)
    print(f"  dividend jump {out['dividend_jump']['pct']:+.1%} "
          f"({out['dividend_jump']['ref_value']} to {out['dividend_jump']['reform_value']})")
    d = out["decoupling_2021"]
    print(f"  2021 index FR/comp  dividends {d['dividends']['fr']:.1f}/{d['dividends']['comp']:.1f} "
          f"surplus {d['surplus']['fr']:.1f}/{d['surplus']['comp']:.1f} "
          f"gdp {d['gdp']['fr']:.1f}/{d['gdp']['comp']:.1f}")

    out["distributional_extension"] = distribution.run()
    print("  distributional extension  WID 2013/2018 estimates and diagnostics written")


def class_b(out: dict) -> bool:
    """Firm panels (Orbis): tables, event-study figures, ratchet, descriptives. Returns ran?"""
    main = RAW / "firm_panel_real.parquet"
    if not main.exists():
        print("class B       SKIPPED: Orbis panel absent (data/raw/firm_panel_real.parquet)")
        return False

    real = pd.read_parquet(main)
    import src.engine.panel as panel_engine
    built = panel_engine.build(real, STUDY)
    out["main_rows"], out["main_firms"] = int(len(built)), int(built[S.FIRM].nunique())
    out.update(results.firm_tables(built, STUDY))
    out["descriptives"] = results.descriptives(built)

    print(f"class B       firm panel  rows {out['main_rows']}  firms {out['main_firms']}")
    figures = {"entree_capital.csv": S.DLOG_CAPITAL, "levier_event.csv": S.LEVERAGE,
               "cash_event.csv": S.CASH_RATIO, "fin_debt_event.csv": S.FIN_DEBT_RATIO}
    for name, outcome in figures.items():
        print("  " + emit.event_study_csv(results.event_study(built, outcome, STUDY), name))

    out["cut_at_2019"] = {"coef": results.cut_at_2019(real)["coef"]}

    # Shareholder heterogeneity (Section 6 robustness): the owner-manager channel. Needs the Orbis
    # ownership snapshot, part of the same licensed extract; skipped with class B if it is absent.
    flags_path = RAW / "ownership_flags.parquet"
    if flags_path.exists():
        out["phase4"] = results.heterogeneity_shareholder(real, pd.read_parquet(flags_path))
        ex = out["phase4"]["exposed"][S.DLOG_CAPITAL]
        print(f"  shareholder heterogeneity  exposed dlog_capital {ex['coef']:+.4f} "
              f"(t {ex['t']:+.1f}, coverage {out['phase4']['coverage']:.2f})")
    else:
        print("  shareholder heterogeneity  SKIPPED: ownership_flags.parquet absent")

    long = pd.read_parquet(RAW / "firm_panel_2010.parquet")
    # The ratchet (Table 3, cliquet and capital_retenu figures). Build on the full 2010-2022 window
    # so the balance filter and winsorization see the whole panel, then estimate on each shock window
    # (results._build_shock); this reproduces the displayed coefficients exactly. The aggregate SQ1
    # distribution row is carried as a documented constant (results.SQ1_DISTRIBUTION).
    out["table3"] = {"2013": results.ratchet_shock(long, results.SHOCK_2013),
                     "2018": results.ratchet_shock(long, results.SHOCK_2018)}
    out["robustness_2013"] = results.robustness_2013(long)
    print("  " + emit.cliquet_csv(out["table3"]["2013"], out["table3"]["2018"],
                                  results.SQ1_DISTRIBUTION))
    print("  " + emit.retention_path_csv(results.retention_level_path(long)))
    return True


def main() -> None:
    paths.RESULTS.mkdir(parents=True, exist_ok=True)
    self_test()

    out: dict = {}
    class_a(out)
    ran_b = class_b(out)

    results_path = paths.ROOT / "manuscript" / "fr" / "ssrn" / "paper_results.json"
    if ran_b:
        results_path.write_text(json.dumps(out, indent=2))
        print(f"wrote         {results_path.relative_to(paths.ROOT)}")
        from src.analysis import check_manuscript
        check_manuscript.check(out)
        print("REPRODUCTION PASSED")
    else:
        print("REPRODUCTION PASSED (class A only; class B needs the licensed Orbis panel)")


if __name__ == "__main__":
    main()
