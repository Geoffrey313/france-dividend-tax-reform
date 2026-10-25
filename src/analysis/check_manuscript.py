"""Gate every regenerated public, distributional, and firm-accounting headline number.

Each target is a regenerated value (from reproduce.py's results dict) with a format spec. The spec is
rendered into the exact token each paper prints, in French (decimal comma ``{,}``) and in English
(decimal point), and both tokens are asserted present in the relevant section of their language. If
the code output drifts, the token changes and is not found; if one language is edited away from the
code, or the French is changed without propagating to the English, the missing token is reported. A
single missing token exits non-zero.

The public gate covers dividends and the cross-package semi-elasticities. The WID gate covers every
coefficient and diagnostic discussed in the distributional subsection. The licensed-panel gate covers
the common firm-level specification and the descriptive statistics reported in the appendices.
"""
from __future__ import annotations

import json
import sys
from decimal import Decimal, ROUND_HALF_UP
from math import copysign

import pandas as pd

from src.common import paths


def _sections(lang: str) -> dict:
    sec = paths.ROOT / "manuscript" / lang.lower() / "ssrn" / "sections"
    figures = sec.parent / "figures"
    generated = "\n".join(
        target.read_text() for target in sorted(figures.glob("firm_*.tex"))
    )
    return {"DON": (sec / "03_donnees.tex").read_text(),
            "RES": (sec / "05_resultats.tex").read_text(),
            "ACC": (sec / "06_entreprises.tex").read_text() + generated,
            "ROB": (sec / "06_robustesse.tex").read_text() + generated,
            "ANN": (sec / "08_annexe.tex").read_text() + generated}


# The two languages and the decimal separator each prints.
_LANGS = {"FR": {"sep": "{,}"}, "EN": {"sep": "."}}


def _token(value: float, dec: int, kind: str, sep: str) -> str:
    """Render one regenerated value into the token a manuscript prints.

    kind: 'dec' signed decimal, 't' signed decimal in parentheses, 'pct' unsigned decimal,
    'int' thousands grouped with a thin space.
    """
    if kind == "int":
        return f"{int(round(value)):,}".replace(",", "\\,")
    value = _display_round(value, dec)
    if kind == "num":
        return f"{value:,.{dec}f}".replace(",", "\\,").replace(".", sep)
    signed = kind in ("dec", "t")
    body = (f"{value:+.{dec}f}" if signed else f"{value:.{dec}f}").replace(".", sep)
    return f"({body})" if kind == "t" else body


def _display_round(value: float, dec: int) -> float:
    """Decimal half-up rounding, insulated from binary representations of exact half steps."""
    stabilized = value + copysign(10 ** (-(dec + 9)), value)
    quantum = Decimal(1).scaleb(-dec)
    return float(Decimal(str(stabilized)).quantize(quantum, rounding=ROUND_HALF_UP))


# The legacy broad-Orbis gate is preserved in Git history; the live gates below cover
# public series, WID distributional results, and harmonized firm accounting.

def check_public(out: dict) -> None:
    """Gate the public-series headline numbers independently of licensed firm panels."""
    jump = out["dividend_jump"]
    sq1 = out["sq1_distribution"]
    sq1g = out["sq1_gross_flow"]
    targets = [
        ("dividend jump ref value", jump["ref_value"], 0, "int", "RES"),
        ("dividend jump reform value", jump["reform_value"], 0, "int", "RES"),
        ("dividend jump pct", jump["pct"], 3, "pct", "RES"),
        ("SQ1 distribution 2013", -sq1["2013"], 2, "dec", "RES"),
        ("SQ1 distribution 2018", sq1["2018"], 2, "dec", "RES"),
        ("SQ1 gross-flow 2013", -sq1g["2013"], 2, "dec", "RES"),
        ("SQ1 gross-flow 2018", sq1g["2018"], 2, "dec", "RES"),
    ]
    _check_targets(targets, "PUBLIC-SERIES")


def check_distribution(out: dict) -> None:
    """Gate every WID number discussed in the distributional manuscript section."""
    mirror = out["mirror"]
    shocks = out["shocks"]
    outcomes = (
        "pretax_income_top_1", "pretax_income_top_10",
        "posttax_income_top_1", "posttax_income_top_10",
        "pretax_income_bottom_50", "posttax_income_bottom_50",
    )
    targets: list[tuple] = []
    for outcome in outcomes:
        targets.extend([
            (f"WID {outcome} 2012--2013", mirror[outcome]["coef_2013_pp"],
             3, "dec", "RES"),
            (f"WID {outcome} 2018", mirror[outcome]["coef_2018_pp"],
             3, "dec", "RES"),
            (f"WID {outcome} 2018 asset controls",
             shocks["2018_precovid"][outcome]["asset_price_controls"]["coef_pp"],
             3, "dec", "RES"),
        ])

    top1_13 = _display_round(mirror["posttax_income_top_1"]["coef_2013_pp"], 3)
    top1_18 = _display_round(mirror["posttax_income_top_1"]["coef_2018_pp"], 3)
    top10_13 = _display_round(mirror["posttax_income_top_10"]["coef_2013_pp"], 3)
    top10_18 = _display_round(mirror["posttax_income_top_10"]["coef_2018_pp"], 3)
    bottom_13 = _display_round(mirror["posttax_income_bottom_50"]["coef_2013_pp"], 3)
    bottom_18 = _display_round(mirror["posttax_income_bottom_50"]["coef_2018_pp"], 3)
    asset_top1 = _display_round(
        shocks["2018_precovid"]["posttax_income_top_1"]["asset_price_controls"]["coef_pp"],
        3,
    )
    targets.extend([
        ("WID top-1 cross-package gap", top1_18 - top1_13, 3, "pct", "RES"),
        ("WID top-10/bottom-50 gap 2012--2013",
         abs(top10_13) + abs(bottom_13), 3, "pct", "RES"),
        ("WID top-10/bottom-50 gap 2018",
         abs(top10_18) + abs(bottom_18), 3, "pct", "RES"),
        ("WID top-1 asset-control attenuation", top1_18 - asset_top1,
         3, "pct", "RES"),
        ("WID top-1 asset-control attenuation pct",
         100.0 * (top1_18 - asset_top1) / top1_18, 2, "pct", "RES"),
        ("WID top-1 alternative-pool maximum",
         max(cell["coef_pp"] for cell in
             out["pool_robustness"]["2018_precovid"]["posttax_income_top_1"].values()),
         3, "dec", "RES"),
        ("WID top-10 cycle-and-assets",
         shocks["2018_precovid"]["posttax_income_top_10"]["cycle_asset_controls"]["coef_pp"],
         3, "dec", "RES"),
    ])

    event = pd.read_csv(paths.RESULTS / "distributional_mirror_2013.csv").set_index("year")
    for year in (2010, 2012, 2013):
        targets.append((
            f"WID posttax top-1 event {year}", event.loc[year, "coef_pp"],
            3, "dec", "RES",
        ))
    _check_targets(targets, "DISTRIBUTIONAL")


def _check_targets(targets: list[tuple], name: str) -> None:
    texts = {lang: _sections(lang) for lang in _LANGS}
    missing = []
    for label, value, dec, kind, section in targets:
        line = f"  check {label:42s}"
        for lang, cfg in _LANGS.items():
            token = _token(value, dec, kind, cfg["sep"])
            ok = token in texts[lang][section]
            line += f"  {lang} {'ok ' if ok else 'MISS'} {token}"
            if not ok:
                missing.append(f"{label} [{lang}]: regenerated token {token} not in {section}")
        print(line)
    if missing:
        print(f"\n{name} CONSISTENCY CHECK FAILED: {len(missing)} mismatch(es):")
        for item in missing:
            print(f"  {item}")
        sys.exit(1)
    print(f"{name.lower()} consistency check PASSED: {len(targets)} numbers match both manuscripts")


def check_firm_accounting(out: dict) -> None:
    """Gate the manuscript-integrated Large/Very Large accounting robustness."""
    texts = {lang: _sections(lang) for lang in _LANGS}
    private = out["populations"]["private_large_vl"]
    public = out["populations"]["public_listed"]
    specification = out.get("manuscript_specification", "C_controls_no_ptbi")
    panel_manifest = json.loads(
        (paths.RESULTS / "harmonized_firm_panel_manifest.json").read_text()
    )
    private_panel = panel_manifest["panels"]["private_large_vl"]
    public_panel = panel_manifest["panels"]["public_listed"]
    future_fill = 100 * panel_manifest["industry_fixed_effects"][
        "future_code_backfill_share"
    ]["private_large_vl"]["2013"]

    def cell(population: dict, shock: str, outcome: str) -> dict:
        return population[shock][outcome]["specifications"][specification]

    # PFU-US-11 renders every firm estimate in generated bilingual fragments. Gate the rendered
    # precision used by those fragments rather than legacy hand-written table precision.
    panels = (
        ("private_large_vl", ("dlog_capital", "cap_increase", "equity_growth_real", "equity_ratio",
                              "asset_growth_real", "tangible_growth_real", "intangible_growth_real",
                              "reserves_ratio", "cash_at", "fin_debt_ratio", "dlog_debt")),
        ("public_listed", ("net_ext_equity_at", "equity_growth_real", "equity_ratio",
                           "asset_growth_real", "tangible_growth_real", "intangible_growth_real",
                           "capex_at", "sales_growth_real", "operating_roa", "retained_earnings_at",
                           "cash_at", "fin_debt_ratio", "dlog_debt")),
    )
    targets = [
        ("accounting private firms", private_panel["firms"], 0, "int", "DON"),
        ("accounting private rows", private_panel["rows"], 0, "int", "DON"),
        ("accounting listed firms", public_panel["firms"], 0, "int", "DON"),
        ("accounting listed rows", public_panel["rows"], 0, "int", "DON"),
        ("accounting sector future fill", future_fill, 1, "pct", "DON"),
    ]
    contrast_keys = {
        "private_large_vl": "private_cross_shock_contrasts",
        "public_listed": "public_cross_shock_contrasts",
    }
    for population, outcomes in panels:
        for outcome in outcomes:
            for shock in ("2013", "2018_precovid"):
                result = cell(out["populations"][population], shock, outcome)
                targets.extend([
                    (f"{population} {outcome} {shock} coefficient", 100 * result["coef"], 3, "dec", "ACC"),
                    (f"{population} {outcome} {shock} statistic", result["coef"] / result["std_error"], 2, "t", "ACC"),
                    (f"{population} {outcome} {shock} rank", result["placebo_p"], 3, "pct", "ACC"),
                    (f"{population} {outcome} {shock} observations",
                     out["populations"][population][shock][outcome]["n_obs"], 0, "int", "ACC"),
                ])
                for control, estimate in result.get("control_estimates", {}).items():
                    targets.extend([
                        (f"{population} {outcome} {shock} {control} coefficient",
                         100 * estimate["coef"], 3, "dec", "ACC"),
                        (f"{population} {outcome} {shock} {control} statistic",
                         estimate["coef"] / estimate["std_error"], 2, "t", "ACC"),
                    ])
            # Cross-shock contrasts are no longer printed mechanically for every outcome:
            # 2018 is the main table and 2013 is an appendix table. Gate only the two
            # contrasts interpreted in the text.
            if population == "private_large_vl" and outcome in ("dlog_capital", "reserves_ratio"):
                contrast = out[contrast_keys[population]]["2018_precovid"][outcome]
                targets.extend([
                    (f"{population} {outcome} contrast", 100 * contrast["contrast"], 3, "pct", "ACC"),
                    (f"{population} {outcome} contrast rank",
                     contrast["two_sided_placebo_p"], 3, "pct", "ACC"),
                ])
            full = cell(out["populations"][population], "2018_full", outcome)
            targets.extend([
                (f"{population} {outcome} full-window coefficient", 100 * full["coef"], 3, "dec", "ROB"),
                (f"{population} {outcome} full-window rank", full["placebo_p"], 3, "pct", "ROB"),
            ])
    descriptive = pd.read_csv(paths.RESULTS / "firm_growth_harmonized_descriptive.csv")
    for row in descriptive.itertuples(index=False):
        prefix = f"descriptive {row.population} {row.variable}"
        targets.append((f"{prefix} n", row.n, 0, "int", "ANN"))
        for statistic in ("mean", "sd", "p25", "median", "p75"):
            targets.append((
                f"{prefix} {statistic}", getattr(row, statistic), 3, "num", "ANN",
            ))
    _check_targets(targets, "ACCOUNTING")
    return

    contrasts = out["private_cross_shock_contrasts"]["2018_precovid"]
    contrasts_full = out["private_cross_shock_contrasts"]["2018_full"]
    public_contrasts = out["public_cross_shock_contrasts"]["2018_precovid"]
    targets = [
        ("accounting private firms", 236133, 0, "int", "DON"),
        ("accounting private rows", 1878451, 0, "int", "DON"),
        ("accounting listed firms", 2706, 0, "int", "DON"),
        ("accounting listed rows", 24285, 0, "int", "DON"),
        ("accounting sector future fill", 78.9, 1, "pct", "DON"),
    ]
    for outcome in ("dlog_capital", "cap_increase", "equity_ratio"):
        result = cell(private, "2018_full", outcome)
        targets.extend([
            (f"main H1 {outcome} coef", result["coef"], 4, "dec", "RES"),
            (f"main H1 {outcome} t", result["coef"] / result["std_error"], 2, "t", "RES"),
            (f"main H1 {outcome} p", result["placebo_p"], 3, "pct", "RES"),
            (f"main H1 {outcome} n", private["2018_full"][outcome]["n_obs"], 0, "int", "RES"),
        ])
    for outcome in (
        "fin_debt_ratio", "asset_growth_real", "tangible_growth_real",
        "intangible_growth_real",
    ):
        result = cell(private, "2018_full", outcome)
        targets.extend([
            (f"main H2 {outcome} coef", result["coef"], 4, "dec", "RES"),
            (f"main H2 {outcome} t", result["coef"] / result["std_error"], 1, "t", "RES"),
            (f"main H2 {outcome} p", result["placebo_p"], 3, "pct", "RES"),
            (f"main H2 {outcome} n", private["2018_full"][outcome]["n_obs"], 0, "int", "RES"),
        ])
    for outcome in ("equity_growth_real", "equity_ratio"):
        result = cell(public, "2018_full", outcome)
        targets.extend([
            (f"main listed H1 {outcome} coef", result["coef"], 4, "dec", "RES"),
            (f"main listed H1 {outcome} t", result["coef"] / result["std_error"],
             2, "t", "RES"),
            (f"main listed H1 {outcome} p", result["placebo_p"], 3, "pct", "RES"),
            (f"main listed H1 {outcome} n", public["2018_full"][outcome]["n_obs"],
             0, "int", "RES"),
        ])
    for outcome in (
        "fin_debt_ratio", "asset_growth_real", "tangible_growth_real",
        "intangible_growth_real", "capex_at",
    ):
        result = cell(public, "2018_full", outcome)
        targets.extend([
            (f"main listed H2 {outcome} coef", result["coef"], 4, "dec", "RES"),
            (f"main listed H2 {outcome} t", result["coef"] / result["std_error"],
             1, "t", "RES"),
            (f"main listed H2 {outcome} p", result["placebo_p"], 3, "pct", "RES"),
            (f"main listed H2 {outcome} n", public["2018_full"][outcome]["n_obs"],
             0, "int", "RES"),
        ])
    for outcome in ("reserves_ratio", "equity_ratio"):
        for shock in ("2013", "2018_full"):
            result = cell(private, shock, outcome)
            targets.extend([
                (f"main H3 {outcome} {shock} coef", result["coef"], 4, "dec", "RES"),
                (f"main H3 {outcome} {shock} t", result["coef"] / result["std_error"], 1, "t", "RES"),
                (f"main H3 {outcome} {shock} p", result["placebo_p"], 3, "pct", "RES"),
                (f"main H3 {outcome} {shock} n", private[shock][outcome]["n_obs"], 0, "int", "RES"),
            ])
        targets.extend([
            (f"main H3 {outcome} contrast", contrasts_full[outcome]["contrast"], 4, "dec", "RES"),
            (f"main H3 {outcome} contrast p",
             contrasts_full[outcome]["two_sided_placebo_p"], 3, "pct", "RES"),
        ])
    for outcome in ("retained_earnings_at", "equity_ratio"):
        for shock in ("2013", "2018_full"):
            result = cell(public, shock, outcome)
            targets.extend([
                (f"main listed H3 {outcome} {shock} coef", result["coef"],
                 4, "dec", "RES"),
                (f"main listed H3 {outcome} {shock} t",
                 result["coef"] / result["std_error"], 1, "t", "RES"),
                (f"main listed H3 {outcome} {shock} p", result["placebo_p"],
                 3, "pct", "RES"),
                (f"main listed H3 {outcome} {shock} n", public[shock][outcome]["n_obs"],
                 0, "int", "RES"),
            ])
    for spec in (
        "A_firm_year_fe_common_sample", "B_firm_industry_year_fe_common_sample",
        specification,
    ):
        targets.append((
            f"robustness dlog capital {spec}",
            private["2018_precovid"]["dlog_capital"]["specifications"][spec]["coef"],
            4, "dec", "ROB",
        ))
        targets.append((
            f"robustness retention {spec}",
            private["2018_precovid"]["reserves_ratio"]["specifications"][spec]["coef"],
            4, "dec", "ROB",
        ))
    for outcome in (
        "dlog_capital", "cap_increase", "tangible_growth_real", "asset_growth_real",
        "reserves_ratio",
    ):
        result = cell(private, "2018_precovid", outcome)
        targets.extend([
            (f"robustness precovid {outcome}", result["coef"], 4, "dec", "ROB"),
            (f"robustness precovid {outcome} p", result["placebo_p"], 3, "pct", "ROB"),
        ])
    for outcome in ("equity_growth_real", "equity_ratio"):
        result = cell(public, "2018_full", outcome)
        targets.extend([
            (f"robustness listed {outcome}", abs(100 * result["coef"]), 2, "pct", "ROB"),
            (f"robustness listed {outcome} p", result["placebo_p"], 3, "pct", "ROB"),
        ])
    for outcome in ("capex_at", "sales_growth_real", "operating_roa"):
        result = cell(public, "2018_full", outcome)
        targets.extend([
            (f"robustness listed {outcome}", result["coef"], 4, "dec", "ROB"),
            (f"robustness listed {outcome} p", result["placebo_p"], 3, "pct", "ROB"),
        ])
    for outcome in (
        "dlog_capital", "cap_increase", "equity_growth_real", "equity_ratio",
        "reserves_ratio", "tangible_growth_real",
    ):
        targets.extend([
            (f"accounting {outcome} 2013", 100 * cell(private, "2013", outcome)["coef"],
             2, "dec", "ACC"),
            (f"accounting {outcome} 2018", 100 * cell(private, "2018_precovid", outcome)["coef"],
             2, "dec", "ACC"),
            (f"accounting {outcome} contrast", 100 * contrasts[outcome]["contrast"],
             2, "dec", "ACC"),
            (f"accounting {outcome} p 2013", cell(private, "2013", outcome)["placebo_p"],
             3, "pct", "ACC"),
            (f"accounting {outcome} p 2018",
             cell(private, "2018_precovid", outcome)["placebo_p"], 3, "pct", "ACC"),
            (f"accounting {outcome} t 2013",
             cell(private, "2013", outcome)["coef"] /
             cell(private, "2013", outcome)["std_error"], 2, "t", "ACC"),
            (f"accounting {outcome} t 2018",
             cell(private, "2018_precovid", outcome)["coef"] /
             cell(private, "2018_precovid", outcome)["std_error"], 2, "t", "ACC"),
            (f"accounting {outcome} n 2013", private["2013"][outcome]["n_obs"],
             0, "int", "ACC"),
            (f"accounting {outcome} n 2018",
             private["2018_precovid"][outcome]["n_obs"], 0, "int", "ACC"),
            (f"accounting {outcome} p contrast",
             contrasts[outcome]["two_sided_placebo_p"], 3, "pct", "ACC"),
        ])
    for outcome in (
        "tangible_growth_real", "equity_growth_real", "equity_ratio",
        "capex_at", "retained_earnings_at",
    ):
        early = cell(public, "2013", outcome)
        later = cell(public, "2018_precovid", outcome)
        contrast_standard_error = (
            early["std_error"] ** 2 + later["std_error"] ** 2
        ) ** 0.5
        contrast_t = (
            public_contrasts[outcome]["contrast"] / contrast_standard_error
            if contrast_standard_error > 0 else float("nan")
        )
        targets.extend([
            (f"listed {outcome} 2013", 100 * cell(public, "2013", outcome)["coef"],
             2, "dec", "ACC"),
            (f"listed {outcome} 2018", 100 * cell(public, "2018_precovid", outcome)["coef"],
             2, "dec", "ACC"),
            (f"listed {outcome} p 2013", cell(public, "2013", outcome)["placebo_p"],
             3, "pct", "ACC"),
            (f"listed {outcome} p 2018",
             cell(public, "2018_precovid", outcome)["placebo_p"], 3, "pct", "ACC"),
            (f"listed {outcome} t 2013",
             cell(public, "2013", outcome)["coef"] /
             cell(public, "2013", outcome)["std_error"], 2, "t", "ACC"),
            (f"listed {outcome} t 2018",
             cell(public, "2018_precovid", outcome)["coef"] /
             cell(public, "2018_precovid", outcome)["std_error"], 2, "t", "ACC"),
            (f"listed {outcome} n 2013", public["2013"][outcome]["n_obs"],
             0, "int", "ACC"),
            (f"listed {outcome} n 2018",
             public["2018_precovid"][outcome]["n_obs"], 0, "int", "ACC"),
            (f"listed {outcome} contrast",
             100 * public_contrasts[outcome]["contrast"], 2, "dec", "ACC"),
            (f"listed {outcome} contrast t", contrast_t,
             2, "t", "ACC"),
            (f"listed {outcome} contrast p",
             public_contrasts[outcome]["two_sided_placebo_p"], 3, "pct", "ACC"),
        ])

    descriptive = pd.read_csv(paths.RESULTS / "firm_growth_harmonized_descriptive.csv")
    for row in descriptive.itertuples(index=False):
        if row.variable in {"capex_at_reported", "capex_ppent", "capex_ppent_reported"}:
            continue
        prefix = f"descriptive {row.population} {row.variable}"
        targets.append((f"{prefix} n", row.n, 0, "int", "ANN"))
        for statistic in ("mean", "sd", "p25", "median", "p75"):
            targets.append((
                f"{prefix} {statistic}", getattr(row, statistic), 3, "num", "ANN",
            ))

    missing = []
    for label, value, dec, kind, section in targets:
        line = f"  check {label:42s}"
        for lang, cfg in _LANGS.items():
            token = _token(value, dec, kind, cfg["sep"])
            ok = token in texts[lang][section]
            line += f"  {lang} {'ok ' if ok else 'MISS'} {token}"
            if not ok:
                missing.append(f"{label} [{lang}]: regenerated token {token} not in {section}")
        print(line)
    if missing:
        print(f"\nACCOUNTING CONSISTENCY CHECK FAILED: {len(missing)} mismatch(es):")
        for item in missing:
            print(f"  {item}")
        sys.exit(1)
    print(f"accounting consistency check PASSED: {len(targets)} numbers match both manuscripts")
