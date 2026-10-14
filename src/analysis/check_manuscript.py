"""Gate: every regenerated headline number must appear in the manuscript at its displayed precision.

Each target formats a regenerated value (from reproduce.py's results dict) into the exact French
token the paper prints, then asserts that token is present in the relevant .tex. If the code output
drifts, the token changes and is not found; if the paper is edited away from the code, likewise. A
single missing token exits non-zero. The SQ1 distribution semi-elasticities are retained verbatim
from committed artifacts (their derivation script was lost), so they are reported, not gated.
"""
from __future__ import annotations

import sys

from src.common import paths

SEC = paths.ROOT / "manuscript" / "fr" / "ssrn" / "sections"
RESULTATS = (SEC / "05_resultats.tex").read_text()
ROBUSTESSE = (SEC / "06_robustesse.tex").read_text()


def fr(x: float, dec: int, signed: bool = True) -> str:
    """French decimal token as the math mode prints it, e.g. -0.0034 -> '-0{,}0034'."""
    return (f"{x:+.{dec}f}" if signed else f"{x:.{dec}f}").replace(".", "{,}")


def fr_int(n: int) -> str:
    """French thousands grouping as the paper prints it, e.g. 16177643 -> '16\\,177\\,643'."""
    s = f"{int(round(n)):,}".replace(",", "\\,")
    return s


def check(out: dict) -> None:
    cap = out["table1"]["dlog_capital"]
    inc = out["table1"]["cap_increase"]
    eq = out["table1"]["equity_ratio"]
    lev = out["table2"]["leverage"]
    deb = out["table2"]["dlog_debt"]
    inv = out["table2"]["dlog_fixed"]
    prod = out["table2"]["log_labor_prod"]
    t3 = out["table3"]            # Table 3, regenerated from the full-window ratchet build
    rob = out["robustness_2013"]  # Section 6 robustness on the 2013 retention coefficient
    jump = out["dividend_jump"]
    dec = out["decoupling_2021"]

    # (label, token, tex). token is derived from regenerated values, so a match ties code to paper.
    T: list[tuple[str, str, str]] = [
        ("H1 capital coef", fr(cap["coef"], 4), RESULTATS),
        ("H1 capital t", f"({fr(cap['t'], 1)})", RESULTATS),
        ("H1 capital synth", fr(cap["synth"], 4), RESULTATS),
        ("H1 cap_increase coef", fr(inc["coef"], 4), RESULTATS),
        ("H1 cap_increase t", f"({fr(inc['t'], 1)})", RESULTATS),
        ("H1 equity coef", fr(eq["coef"], 4), RESULTATS),
        ("H1 equity synth", fr(eq["synth"], 4), RESULTATS),
        ("H1 observations", fr_int(out["main_rows"]), RESULTATS),
        ("H2 leverage coef", fr(lev["coef"], 4), RESULTATS),
        ("H2 leverage synth", fr(lev["synth"], 4), RESULTATS),
        ("H2 dlog_debt coef", fr(deb["coef"], 4), RESULTATS),
        ("H2 dlog_debt synth", fr(deb["synth"], 4), RESULTATS),
        ("H2 investment coef", fr(inv["coef"], 4), RESULTATS),
        ("H2 investment t", f"({fr(inv['t'], 1)})", RESULTATS),
        ("H2 productivity coef", fr(prod["coef"], 4), RESULTATS),
        ("H3 retention 2013", fr(t3["2013"]["retention"]["coef"], 4), RESULTATS),
        ("H3 retention 2018", fr(t3["2018"]["retention"]["coef"], 4), RESULTATS),
        ("H3 fonds propres 2013", fr(t3["2013"]["fondspropres"]["coef"], 4), RESULTATS),
        ("H3 fonds propres 2018", fr(t3["2018"]["fondspropres"]["coef"], 4), RESULTATS),
        ("H3 retention 2013 with ACE", fr(rob["with_ace"], 4), ROBUSTESSE),
        ("H3 retention 2013 with cycle", fr(rob["with_cycle"], 4), ROBUSTESSE),
        ("H3 retention 2013 no-subsidy", fr(rob["no_subsidy_controls"], 4), ROBUSTESSE),
        ("dividend jump ref value", fr_int(jump["ref_value"]), RESULTATS),
        ("dividend jump reform value", fr_int(jump["reform_value"]), RESULTATS),
        ("dividend jump pct", fr(jump["pct"], 3, signed=False), RESULTATS),
        ("cut at 2019 coef", fr(out["cut_at_2019"]["coef"], 4), ROBUSTESSE),
        ("decoupling dividends FR", fr(dec["dividends"]["fr"], 1, signed=False), ROBUSTESSE),
        ("decoupling dividends comp", fr(dec["dividends"]["comp"], 1, signed=False), ROBUSTESSE),
        ("decoupling dividends gap", fr(dec["dividends"]["gap"], 1, signed=False), ROBUSTESSE),
        ("decoupling surplus FR", fr(dec["surplus"]["fr"], 1, signed=False), ROBUSTESSE),
        ("decoupling surplus comp", fr(dec["surplus"]["comp"], 1, signed=False), ROBUSTESSE),
        ("decoupling gdp FR", fr(dec["gdp"]["fr"], 1, signed=False), ROBUSTESSE),
        ("decoupling gdp comp", fr(dec["gdp"]["comp"], 1, signed=False), ROBUSTESSE),
    ]

    missing = [(label, token) for label, token, text in T if token not in text]
    for label, token, text in T:
        flag = "ok " if token in text else "MISS"
        print(f"  check {flag} {label:34s} {token}")

    print("  note: the SQ1 aggregate distribution semi-elasticities (the cliquet distribution row) "
          "are carried as a documented constant, reported not gated (aggregate script not recovered).")
    if missing:
        print(f"\nCONSISTENCY CHECK FAILED: {len(missing)} manuscript number(s) do not match the "
              "regenerated results:")
        for label, token in missing:
            print(f"  {label}: regenerated token {token} not found in the manuscript")
        sys.exit(1)
    print(f"consistency check PASSED: {len(T)} headline numbers match the manuscript")
