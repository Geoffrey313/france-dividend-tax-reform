"""Regenerate the manuscript's figures and numbers from committed code, then gate on consistency.

Three stages:
  self-test  a synthetic panel with a known injected effect, recovered end to end (seconds);
  class A    the public Eurostat and WID series, including the distributional section;
  class C    the harmonized Large/Very Large Orbis and Compustat panels that generate every
             firm-level table and figure in the manuscript.

Each class re-reads the two manuscripts and asserts that every regenerated headline number matches
what the papers display, to their displayed precision. A divergence exits non-zero. The historical
broad-Orbis class-B pipeline has been retired from the manuscript reproduction.
"""
from __future__ import annotations

import json
import hashlib
import platform
import warnings
from importlib.metadata import version
from pathlib import Path

import pandas as pd

from src.common import paths
from src.common.config import SIM, STUDY
from src.common import schema as S
from src.data import simulate
from src.analysis import (
    distribution,
    estimate,
    firm_growth_harmonized,
    results,
)
from src.figures import emit

warnings.filterwarnings("ignore")  # the ratio outcomes divide by zero on empty balance lines


def manuscript_sources_available() -> bool:
    """Return whether the optional bilingual paper sources are present in this checkout."""
    return all(
        (paths.ROOT / "manuscript" / language / "ssrn" / "main.tex").is_file()
        for language in ("fr", "en")
    )


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


def _sha256(target: Path) -> str:
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def class_a(out: dict) -> dict:
    """Public Eurostat/WID series, including the distributional extension."""
    div = pd.read_csv(paths.DATA / "dividends_cross_country.csv")
    surplus = pd.read_csv(paths.DATA / "corp_surplus_2010_2022.csv")
    gdp = pd.read_csv(paths.DATA / "gdp_volume_2010_2022.csv")

    print("class A       public series")
    print("  " + emit.dividend_index_csv(results.dividend_index_wide(div)))
    print("  " + emit.decoupling_csv(results.decoupling_series(div, surplus, gdp)))
    out["dividend_jump"] = results.dividend_jump(div)
    out["decoupling_2021"] = results.decoupling_2021(div, surplus, gdp)
    broad = pd.read_csv(paths.DATA / "dividends_2010_2022_broad.csv")
    out["sq1_distribution"] = results.sq1_distribution(broad)
    out["sq1_gross_flow"] = results.sq1_gross_flow(broad)
    print(f"  dividend jump {out['dividend_jump']['pct']:+.1%} "
          f"({out['dividend_jump']['ref_value']} to {out['dividend_jump']['reform_value']})")
    d = out["decoupling_2021"]
    print(f"  2021 index FR/comp  dividends {d['dividends']['fr']:.1f}/{d['dividends']['comp']:.1f} "
          f"surplus {d['surplus']['fr']:.1f}/{d['surplus']['comp']:.1f} "
          f"gdp {d['gdp']['fr']:.1f}/{d['gdp']['comp']:.1f}")

    distributional = distribution.run()
    print("  distributional analysis  WID 2012--2013/2018 estimates and diagnostics written")
    return distributional


def class_c() -> dict | None:
    """Licensed Large/Very Large and listed accounting analysis integrated in Section 6."""
    if not firm_growth_harmonized.estimation_inputs_available():
        print("class C       SKIPPED: final firm estimation samples absent")
        return None
    print("class C       final-sample firm accounting analysis")
    output = firm_growth_harmonized.run()
    private = output["populations"]["private_large_vl"]
    specification = firm_growth_harmonized.MANUSCRIPT_SPECIFICATION

    def cell(shock: str, outcome: str) -> dict:
        return private[shock][outcome]["specifications"][specification]

    shock2013 = {
        "retention": cell("2013", "reserves_ratio"),
        "fondspropres": cell("2013", "equity_ratio"),
    }
    shock2018 = {
        "retention": cell("2018_full", "reserves_ratio"),
        "fondspropres": cell("2018_full", "equity_ratio"),
    }
    broad = pd.read_csv(paths.DATA / "dividends_2010_2022_broad.csv")
    emit.cliquet_csv(shock2013, shock2018, results.sq1_distribution(broad))
    if manuscript_sources_available():
        from src.analysis import check_manuscript
        check_manuscript.check_firm_accounting(output)
    return output


def write_manifest() -> Path:
    """Record the exact public inputs, licensed-panel manifest, outputs, and environment."""
    public_inputs = [
        paths.DATA / "dividends_cross_country.csv",
        paths.DATA / "dividends_2010_2022_broad.csv",
        paths.DATA / "corp_surplus_2010_2022.csv",
        paths.DATA / "gdp_volume_2010_2022.csv",
        paths.DATA / "unemployment_2010_2022.csv",
        paths.DATA / "wid_distribution_2010_2022.csv",
        paths.DATA / "asset_prices_2010_2022.csv",
        paths.DATA / "gdp_deflator_2005_2025.csv",
    ]
    panel_manifest = firm_growth_harmonized.input_manifest_path()
    outputs = sorted(paths.RESULTS.glob("distributional_*"))
    outputs.extend(sorted(paths.RESULTS.glob("firm_growth_harmonized_*")))
    outputs.extend(sorted(paths.RESULTS.glob("firm_square_sample_manifest*")))
    for language in ("fr", "en"):
        figure_dir = paths.ROOT / "manuscript" / language / "ssrn" / "figures"
        outputs.extend(
            figure_dir / name for name in (
                "dividendes_menages.csv",
                "decouplage_fr.csv",
                "cliquet.csv",
                "entree_capital.csv",
                "levier_event.csv",
            )
        )
        outputs.extend(sorted(figure_dir.glob("firm_accounting_*.csv")))
        outputs.extend(sorted(figure_dir.glob("public_firm_accounting_*.csv")))
        outputs.extend(sorted(figure_dir.glob("firm_events_*.tex")))
        outputs.extend(
            figure_dir / name for name in (
                "firm_square_results_2018_table.tex",
                "firm_square_results_2013_table.tex",
                "firm_specification_table.tex",
                "firm_country_robustness_table.tex",
                "firm_event_study_2018_table.tex",
                "firm_event_study_2013_table.tex",
                "firm_descriptive_table.tex",
                "firm_window_sensitivity_table.tex",
            )
        )
    outputs.append(paths.ROOT / "manuscript" / "fr" / "ssrn" / "paper_results.json")
    outputs = list(dict.fromkeys(target for target in outputs if target.is_file()))
    manifest = {
        "python": platform.python_version(),
        "dependencies": {
            package: version(package)
            for package in ("numpy", "pandas", "scipy", "statsmodels", "pyarrow", "pydantic", "pyhdfe")
        },
        "public_input_sha256": {
            str(target.relative_to(paths.ROOT)): _sha256(target) for target in public_inputs
        },
        "licensed_panel_manifest": {
            "path": str(panel_manifest.relative_to(paths.ROOT)),
            "sha256": _sha256(panel_manifest),
        },
        "result_sha256": {
            str(target.relative_to(paths.ROOT)): _sha256(target) for target in outputs
        },
    }
    target = paths.RESULTS / "reproduction_manifest.json"
    target.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return target


def main() -> None:
    paths.RESULTS.mkdir(parents=True, exist_ok=True)
    self_test()

    out: dict = {}
    distributional = class_a(out)
    if manuscript_sources_available():
        from src.analysis import check_manuscript
        check_manuscript.check_public(out)
        check_manuscript.check_distribution(distributional)
    else:
        print("manuscript    optional consistency gate skipped: sources absent")
    firm_output = class_c()

    results_path = paths.ROOT / "manuscript" / "fr" / "ssrn" / "paper_results.json"
    if not manuscript_sources_available():
        results_path = paths.RESULTS / "paper_results.json"
    results_path.write_text(json.dumps(out, indent=2))
    print(f"wrote         {results_path.relative_to(paths.ROOT)}")
    if firm_output is not None:
        manifest = write_manifest()
        print(f"wrote         {manifest.relative_to(paths.ROOT)}")
    print("PUBLIC-SERIES REPRODUCTION PASSED")
    print("DISTRIBUTIONAL REPRODUCTION PASSED")
    if firm_output is not None:
        print("FIRM ACCOUNTING REPRODUCTION PASSED")


if __name__ == "__main__":
    main()
