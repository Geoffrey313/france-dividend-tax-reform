"""Two-shock empirical analysis of inequality around the 2013 and 2018 tax packages.

The estimand is deliberately reduced-form: the differential path of France around each package,
not a decomposition of the PFU and the ISF-to-IFI reform.  No harmonized annual capital-income or
financial-wealth distribution is available for all six countries.  Pre- and post-tax income
shares are the closest detectable signal, while net wealth is context only; neither measures
dividend income itself.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.common import paths, schema as S
from src.common.config import Study
from src.engine import country_did, synth

SHOCK_2013 = Study(first_year=2010, reference_year=2012, reform_year=2013, last_year=2014)
SHOCK_2013_LONG = Study(first_year=2010, reference_year=2012, reform_year=2013, last_year=2016)
SHOCK_2018_PRECOVID = Study(
    first_year=2014, reference_year=2017, reform_year=2018, last_year=2019
)
SHOCK_2018_FULL = Study(first_year=2014, reference_year=2017, reform_year=2018, last_year=2022)

SHOCKS = {
    "2013_observed": SHOCK_2013,
    "2013_long": SHOCK_2013_LONG,
    "2018_precovid": SHOCK_2018_PRECOVID,
    "2018_full": SHOCK_2018_FULL,
}
CONCEPTS = ("net_wealth", "pretax_income", "posttax_income")
GROUPS = ("top_1", "top_10", "bottom_50")
OUTCOMES = tuple(f"{concept}_{group}" for concept in CONCEPTS for group in GROUPS)
SIGNAL_OUTCOMES = (
    "pretax_income_top_1", "pretax_income_top_10",
    "posttax_income_top_1", "posttax_income_top_10",
)
DIFFUSION_OUTCOMES = ("pretax_income_bottom_50", "posttax_income_bottom_50")
CONTEXT_OUTCOMES = ("net_wealth_top_1", "net_wealth_top_10", "net_wealth_bottom_50")
CONTROL_COLUMNS = ("log_gdp", "unemp")
ASSET_CONTROL_COLUMNS = ("house_price_return", "equity_return")
POOL_CONTROLS = {
    "full": ("DE", "IT", "ES", "BE", "NL"),
    "drop_italy": ("DE", "ES", "BE", "NL"),
    "drop_belgium": ("DE", "IT", "ES", "NL"),
    "clean_controls": ("DE", "ES", "NL"),
    "large_economies": ("DE", "IT", "ES"),
    "northwestern_neighbors": ("DE", "BE", "NL"),
}
QUALITY_THRESHOLD = {
    "net_wealth": 1,
    "pretax_income": 3,
    "posttax_income": 3,
}


def _concept(outcome: str) -> str:
    for concept in CONCEPTS:
        if outcome.startswith(f"{concept}_"):
            return concept
    raise ValueError(f"unknown distributional outcome: {outcome}")


def load_panel(path: Path | None = None) -> pd.DataFrame:
    """Load WID shares and merge the pre-specified cycle controls."""
    path = path or paths.DATA / "wid_distribution_2010_2022.csv"
    raw = pd.read_csv(path)
    required = {S.COUNTRY, S.YEAR, "concept", "group", "share", "data_quality"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"WID panel is missing columns: {sorted(missing)}")

    raw["outcome"] = raw["concept"] + "_" + raw["group"]
    wide = raw.pivot(index=[S.COUNTRY, S.YEAR], columns="outcome", values="share").reset_index()
    # Data quality belongs to the WID concept, not to the country-year as a whole.  Keeping one
    # column per concept prevents high-quality income observations from masking an imputed wealth
    # series (which is precisely the case for France after 2014).
    quality = raw.pivot_table(
        index=[S.COUNTRY, S.YEAR], columns="concept", values="data_quality", aggfunc="first"
    ).rename(columns=lambda name: f"quality_{name}").reset_index()
    wide = wide.merge(quality, on=[S.COUNTRY, S.YEAR], how="left")

    gdp = pd.read_csv(paths.DATA / "gdp_volume_2010_2022.csv")[[S.COUNTRY, S.YEAR, "value"]]
    gdp = gdp.rename(columns={"value": "gdp_volume"})
    gdp["log_gdp"] = np.log(gdp["gdp_volume"])
    unemp = pd.read_csv(paths.DATA / "unemployment_2010_2022.csv")[
        [S.COUNTRY, S.YEAR, "unemp"]
    ]
    panel = wide.merge(gdp, on=[S.COUNTRY, S.YEAR], how="left").merge(
        unemp, on=[S.COUNTRY, S.YEAR], how="left"
    )
    asset = pd.read_csv(paths.DATA / "asset_prices_2010_2022.csv")
    asset = asset[
        [S.COUNTRY, S.YEAR, "hpi_2015", "house_price_growth_pct", "equity_return_pct"]
    ].copy()
    asset["house_price_return"] = asset["house_price_growth_pct"] / 100.0
    asset["equity_return"] = asset["equity_return_pct"] / 100.0
    panel = panel.merge(asset, on=[S.COUNTRY, S.YEAR], how="left")

    if panel.duplicated([S.COUNTRY, S.YEAR]).any():
        raise ValueError("distributional panel has duplicate country-years")
    if panel[list(OUTCOMES)].isna().any().any():
        missing_rows = panel.loc[panel[list(OUTCOMES)].isna().any(axis=1), [S.COUNTRY, S.YEAR]]
        raise ValueError(f"distributional panel is incomplete: {missing_rows.to_dict('records')}")
    if panel[list(ASSET_CONTROL_COLUMNS)].notna().mean().min() < 0.95:
        raise ValueError("distributional panel has insufficient asset-price control coverage")
    return panel.sort_values([S.COUNTRY, S.YEAR]).reset_index(drop=True)


def _spec(panel: pd.DataFrame, outcome: str, study: Study,
          controls: tuple[str, ...] = ()) -> tuple[dict, pd.DataFrame]:
    fit = country_did.fit(panel, outcome, study, controls=controls)
    permutation = country_did.placebo_permutation(panel, outcome, study, controls=controls)

    subset = panel[
        panel[S.COUNTRY].isin((study.treated,) + study.controls)
        & panel[S.YEAR].between(study.first_year, study.last_year)
    ]
    wide = subset.pivot(index=S.YEAR, columns=S.COUNTRY, values=outcome).sort_index()
    synthetic = synth.fit(wide, study)
    pre = fit.event_study.loc[fit.event_study.index < study.reform_year, "coef"]

    concept = _concept(outcome)
    quality_column = f"quality_{concept}"
    threshold = QUALITY_THRESHOLD[concept]
    quality_sample = subset[[S.COUNTRY, S.YEAR, quality_column]].drop_duplicates()
    treated_quality = quality_sample[quality_sample[S.COUNTRY].eq(study.treated)]
    treated_pre = treated_quality[treated_quality[S.YEAR] < study.reform_year][quality_column]
    treated_post = treated_quality[treated_quality[S.YEAR] >= study.reform_year][quality_column]
    observed = quality_sample[quality_column].ge(threshold)

    result = {
        "coef": fit.coef,
        "coef_pp": 100.0 * fit.coef,
        "std_error": fit.std_error,
        "z": fit.z,
        "n_obs": fit.n_obs,
        "n_countries": fit.n_countries,
        "placebo_p": permutation["placebo_p"],
        "placebo_coefficients": permutation["coefficients"],
        "synthetic_effect": synthetic.post_effect,
        "synthetic_effect_pp": 100.0 * synthetic.post_effect,
        "synthetic_placebo_p": synthetic.placebo_rank_p,
        "pretrend_max_abs_pp": 100.0 * float(pre.abs().max()) if len(pre) else None,
        "wid_quality": {
            "observed_threshold": threshold,
            "treated_pre_min": int(treated_pre.min()),
            "treated_post_min": int(treated_post.min()),
            "treated_post_observed_share": float(treated_post.ge(threshold).mean()),
            "treated_post_all_observed": bool(treated_post.ge(threshold).all()),
            "sample_observed_share": float(observed.mean()),
        },
    }
    event = fit.event_study.reset_index().assign(
        outcome=outcome,
        shock=str(study.reform_year),
        first_year=study.first_year,
        last_year=study.last_year,
        controls="+".join(controls) if controls else "none",
    )
    event["coef_pp"] = 100.0 * event["coef"]
    event["std_error_pp"] = 100.0 * event["std_error"]
    return result, event


def estimate(panel: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Estimate baseline and cycle-controlled specifications for all outcomes and windows."""
    results: dict = {"metadata": {
        "estimand": "reduced-form country-year differential; tax packages are not decomposed",
        "countries": list((SHOCK_2013.treated,) + SHOCK_2013.controls),
        "outcome_hierarchy": {
            "detectable_income_signal": list(SIGNAL_OUTCOMES),
            "no_diffusion": list(DIFFUSION_OUTCOMES),
            "net_wealth_context_only": list(CONTEXT_OUTCOMES),
        },
        "reported_income_signals": list(SIGNAL_OUTCOMES),
        "source": "WID.world country bulk files",
        "quality_rule": (
            "WID grades >=1 for distributed wealth and >=3 for distributed income are treated "
            "as observed-data estimates; grade 0 wealth is income-inequality-based imputation"
        ),
        "identification_limit": (
            "The design estimates the reduced-form incidence of each tax package. It cannot "
            "separate PFU from ISF-to-IFI or identify a dividend-specific causal channel."
        ),
        "measurement_limit": (
            "No harmonized annual dividend-income or distributed financial-wealth share is "
            "available for all six countries in these WID files. Pretax national income includes "
            "allocated retained earnings and is therefore only a triangulation outcome."
        ),
        "interpretation_status": (
            "descriptive directional signal; not a conventionally significant causal estimate"
        ),
        "capitalization_nuance": (
            "WID pretax income allocates retained earnings to owners. Mechanically, greater 2013 "
            "retention would tend to raise rather than lower top shares; the observed decline is "
            "therefore not explained by that mechanical allocation alone."
        ),
    }, "shocks": {}}
    events: list[pd.DataFrame] = []
    for shock_name, study in SHOCKS.items():
        results["shocks"][shock_name] = {}
        for outcome in OUTCOMES:
            baseline, event = _spec(panel, outcome, study)
            controlled, controlled_event = _spec(panel, outcome, study, CONTROL_COLUMNS)
            asset_controlled, asset_event = _spec(panel, outcome, study, ASSET_CONTROL_COLUMNS)
            full_controlled, full_event = _spec(
                panel, outcome, study, CONTROL_COLUMNS + ASSET_CONTROL_COLUMNS
            )
            results["shocks"][shock_name][outcome] = {
                "baseline": baseline,
                "cycle_controls": controlled,
                "asset_price_controls": asset_controlled,
                "cycle_asset_controls": full_controlled,
            }
            events.extend([event.assign(window=shock_name),
                           controlled_event.assign(window=shock_name),
                           asset_event.assign(window=shock_name),
                           full_event.assign(window=shock_name)])

    mirror: dict = {}
    for outcome in OUTCOMES:
        b13 = results["shocks"]["2013_long"][outcome]["baseline"]["coef"]
        b18 = results["shocks"]["2018_precovid"][outcome]["baseline"]["coef"]
        expected = -1 if outcome.endswith("bottom_50") else 1
        mirror[outcome] = {
            "coef_2013_pp": 100.0 * b13,
            "coef_2018_pp": 100.0 * b18,
            "opposite_signs": bool(b13 * b18 < 0),
            "expected_direction": bool(np.sign(b13) == -expected and np.sign(b18) == expected),
        }
    results["mirror"] = mirror

    # Pool sensitivity is deliberately restricted to the income signal and lower-half outcomes.
    # Net wealth is context-only and is imputed for France in the 2018 window.
    results["pool_robustness"] = {}
    for shock_name in ("2013_long", "2018_precovid"):
        base_study = SHOCKS[shock_name]
        results["pool_robustness"][shock_name] = {}
        for outcome in SIGNAL_OUTCOMES + DIFFUSION_OUTCOMES:
            results["pool_robustness"][shock_name][outcome] = {}
            for pool_name, controls in POOL_CONTROLS.items():
                pool_study = base_study.model_copy(update={"controls": controls})
                pool_result, _ = _spec(panel, outcome, pool_study)
                results["pool_robustness"][shock_name][outcome][pool_name] = pool_result
    return results, pd.concat(events, ignore_index=True)


def estimates_table(results: dict) -> pd.DataFrame:
    """Flatten the machine-readable results into a manuscript-ready audit table."""
    rows: list[dict] = []
    for shock, outcomes in results["shocks"].items():
        for outcome, specifications in outcomes.items():
            for specification, estimate in specifications.items():
                quality = estimate["wid_quality"]
                rows.append({
                    "shock": shock,
                    "outcome": outcome,
                    "specification": specification,
                    "coef_pp": estimate["coef_pp"],
                    "std_error_pp": 100.0 * estimate["std_error"],
                    "placebo_p": estimate["placebo_p"],
                    "synthetic_effect_pp": estimate["synthetic_effect_pp"],
                    "synthetic_placebo_p": estimate["synthetic_placebo_p"],
                    "pretrend_max_abs_pp": estimate["pretrend_max_abs_pp"],
                    "n_obs": estimate["n_obs"],
                    "wid_quality_threshold": quality["observed_threshold"],
                    "france_post_quality_min": quality["treated_post_min"],
                    "france_post_all_observed": quality["treated_post_all_observed"],
                })
    return pd.DataFrame(rows)


def pool_robustness_table(results: dict) -> pd.DataFrame:
    """Flatten leave-one-out and alternative comparison-pool estimates."""
    rows: list[dict] = []
    for shock, outcomes in results["pool_robustness"].items():
        for outcome, pools in outcomes.items():
            full_sign = np.sign(pools["full"]["coef"])
            for pool, estimate in pools.items():
                rows.append({
                    "shock": shock,
                    "outcome": outcome,
                    "pool": pool,
                    "controls": "+".join(POOL_CONTROLS[pool]),
                    "coef_pp": estimate["coef_pp"],
                    "std_error_pp": 100.0 * estimate["std_error"],
                    "same_sign_as_full": bool(np.sign(estimate["coef"]) == full_sign),
                    "placebo_p": estimate["placebo_p"],
                    "synthetic_effect_pp": estimate["synthetic_effect_pp"],
                    "synthetic_placebo_p": estimate["synthetic_placebo_p"],
                    "pretrend_max_abs_pp": estimate["pretrend_max_abs_pp"],
                    "n_countries": estimate["n_countries"],
                })
    return pd.DataFrame(rows)


def _write_figure_series(events: pd.DataFrame, output_dir: Path) -> None:
    """Write the two compact event paths consumed by the manuscript's descriptive figure."""
    definitions = {
        "2013": ("2013_long", 2012),
        "2018": ("2018_precovid", 2017),
    }
    for label, (window, reference_year) in definitions.items():
        series = events[
            events["outcome"].eq("pretax_income_top_1")
            & events["controls"].eq("none")
            & events["window"].eq(window)
        ][["year", "coef_pp", "std_error_pp"]].copy()
        reference = pd.DataFrame({
            "year": [reference_year], "coef_pp": [0.0], "std_error_pp": [0.0]
        })
        series = pd.concat([series, reference], ignore_index=True).sort_values("year")
        series.to_csv(output_dir / f"distributional_mirror_{label}.csv", index=False)


def run(output_dir: Path | None = None) -> dict:
    """Run the extension and write machine-readable results without touching the manuscript."""
    output_dir = output_dir or paths.RESULTS
    output_dir.mkdir(parents=True, exist_ok=True)
    panel = load_panel()
    results, events = estimate(panel)
    (output_dir / "distributional_results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    estimates_table(results).to_csv(output_dir / "distributional_estimates.csv", index=False)
    pool_robustness_table(results).to_csv(
        output_dir / "distributional_pool_robustness.csv", index=False
    )
    events.to_csv(output_dir / "distributional_event_studies.csv", index=False)
    _write_figure_series(events, output_dir)
    return results
