"""Complete Orbis Large/Very Large versus Compustat analysis.

The released replication path starts directly from the final common samples and their checksummed
manifest. Larger harmonized panels remain an optional local-development input. CAPEX is never
estimated for private firms and remains a listed-firm outcome with reported-only sensitivities.
"""
from __future__ import annotations

import json
import math
import gc
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import firm_growth_controls as engine
from src.analysis import firm_square_samples as square
from src.common import paths, schema as S
from src.figures import emit


COMMON_PRIMARY = ("tangible_growth_real", "asset_growth_real")
COMMON_SECONDARY = (
    "intangible_growth_real", "cash_at", "fin_debt_ratio", "dlog_debt",
    "equity_growth_real", "equity_ratio",
)
COMMON_OUTCOMES = (*COMMON_PRIMARY, *COMMON_SECONDARY)
PRIVATE_SPECIFIC = ("dlog_capital", "cap_increase", "reserves_ratio")
PUBLIC_SPECIFIC = (
    "capex_at", "sales_growth_real", "operating_roa", "retained_earnings_at",
    "net_ext_equity_at",
)
CONTROLS = square.CONTROLS
HARMONIZED_PRIVATE = paths.DATA / "derived" / "harmonized_private_large_vl.parquet"
HARMONIZED_PUBLIC = paths.DATA / "derived" / "harmonized_public_listed.parquet"
RELEASE_MANIFEST = square.OUTPUT_DIR / "manifest.json"
COVERAGE_THRESHOLD = 0.50
# One static specification supplies every firm-level number eligible for the manuscript.  The
# alternative columns remain diagnostics in the technical output and must not be mixed into the
# published tables.
MANUSCRIPT_SPECIFICATION = "C_controls_no_ptbi"
SHARD_SCHEMA_VERSION = 2
SHARD_DIR = paths.RESULTS / "firm_square_shards"
CONTROL_CONFLICTS = {
    "leverage": {"lag_book_leverage"},
    "fin_debt_ratio": {"lag_book_leverage"},
    "dlog_debt": {"lag_book_leverage"},
    "equity_ratio": {"lag_book_leverage"},
}
MANUSCRIPT_EVENT_OUTCOMES = (
    "dlog_capital", "net_ext_equity_at", "equity_growth_real",
    "asset_growth_real", "tangible_growth_real", "intangible_growth_real",
    "reserves_ratio", "retained_earnings_at", "fin_debt_ratio", "capex_at",
    "sales_growth_real", "operating_roa",
)
DESCRIPTIVE_VARIABLES = {
    "private_large_vl": (
        "dlog_capital", "cap_increase", "equity_growth_real", "equity_ratio",
        "reserves_ratio", "asset_growth_real", "tangible_growth_real",
        "intangible_growth_real", "cash_at", "fin_debt_ratio", "dlog_debt",
        *CONTROLS,
    ),
    "public_listed": (
        "net_ext_equity_at", "equity_growth_real", "equity_ratio", "retained_earnings_at",
        "asset_growth_real", "tangible_growth_real", "intangible_growth_real",
        "cash_at", "fin_debt_ratio", "dlog_debt", "capex_at",
        "sales_growth_real", "operating_roa", *CONTROLS,
    ),
}


def _sha256(target: Path) -> str:
    digest = hashlib.sha256()
    with target.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _release_manifest() -> dict:
    payload = json.loads(RELEASE_MANIFEST.read_text())
    if payload.get("schema_version") != 1:
        raise ValueError(f"unsupported final-sample manifest: {RELEASE_MANIFEST}")
    return payload


def _release_index() -> dict[tuple[str, str, str], dict]:
    return {
        (row["population"], row["shock"], row["family"]): row
        for row in _release_manifest()["samples"]
    }


def _released_inputs_complete() -> bool:
    if not RELEASE_MANIFEST.exists():
        return False
    try:
        rows = _release_manifest()["samples"]
    except (KeyError, ValueError, json.JSONDecodeError):
        return False
    return bool(rows) and all((paths.ROOT / row["path"]).is_file() for row in rows)


def estimation_inputs_available() -> bool:
    """Return whether local development panels or released final samples are present."""
    return (
        HARMONIZED_PRIVATE.is_file() and HARMONIZED_PUBLIC.is_file()
    ) or _released_inputs_complete()


def input_manifest_path() -> Path:
    """Return the manifest that fingerprints the inputs used by the replication run."""
    local = paths.RESULTS / "harmonized_firm_panel_manifest.json"
    if HARMONIZED_PRIVATE.is_file() and HARMONIZED_PUBLIC.is_file() and local.is_file():
        return local
    return RELEASE_MANIFEST


def _released_sample(
    index: dict[tuple[str, str, str], dict],
    population: str,
    shock: str,
    family: str,
) -> tuple[pd.DataFrame, dict]:
    manifest = index[(population, shock, family)]
    target = paths.ROOT / manifest["path"]
    actual = _sha256(target)
    if actual != manifest["sha256"]:
        raise AssertionError(f"final-sample checksum mismatch: {target}")
    return pd.read_parquet(target), manifest


def _controls_for_outcome(outcome: str) -> tuple[str, ...]:
    """Remove controls mechanically equivalent to a debt/equity outcome."""
    excluded = CONTROL_CONFLICTS.get(outcome, set())
    return tuple(control for control in CONTROLS if control not in excluded)


def _assert_manuscript_specification(results: dict) -> None:
    """Fail if an eligible estimate departs from the common firm-level specification."""
    for population, population_results in results.items():
        for shock, shock_results in population_results.items():
            for outcome, result in shock_results.items():
                if not result["status"].startswith("estimated"):
                    continue
                specification = result["specifications"].get(MANUSCRIPT_SPECIFICATION)
                if specification is None:
                    raise AssertionError(
                        f"missing {MANUSCRIPT_SPECIFICATION}: {population}/{shock}/{outcome}"
                    )
                if specification["fixed_effects"] != ["firm", "industry_by_year"]:
                    raise AssertionError(
                        f"wrong fixed effects: {population}/{shock}/{outcome}"
                    )
                expected = list(_controls_for_outcome(outcome))
                if specification["controls"] != expected:
                    raise AssertionError(
                        f"wrong controls: {population}/{shock}/{outcome}: "
                        f"{specification['controls']} != {expected}"
                    )


def _outcome_role(population: str, outcome: str) -> str:
    """Map each accounting outcome to the hypothesis it can inform."""
    if outcome in {"dlog_capital", "cap_increase", "net_ext_equity_at", "stock_issuance_at",
                   "repurchases_at", "net_equity_issuance_at",
                   "equity_growth_real", "equity_ratio"}:
        return "H1_financing"
    if outcome in {"reserves_ratio", "cash_dividends_at", "retained_earnings_at"}:
        return "H3_retention_asymmetry"
    if outcome in COMMON_PRIMARY or outcome in {
        "intangible_growth_real", "cash_at", "fin_debt_ratio", "dlog_debt",
        "capex_at", "capex_at_reported", "capex_ppent", "capex_ppent_reported",
        "sales_growth_real", "operating_roa",
    }:
        return "H2_real_activity"
    return f"descriptive_{population}"


def _normal_p(coef: float, standard_error: float) -> float:
    if not np.isfinite(standard_error) or standard_error <= 0:
        return float("nan")
    return math.erfc(abs(coef / standard_error) / math.sqrt(2.0))


def _descriptive_rows(
    frame: pd.DataFrame,
    population: str,
    shock: str,
    family: str,
    variables: tuple[str, ...],
) -> pd.DataFrame:
    """Summarize one final common sample used by the manuscript regressions."""
    rows = []
    for variable in variables:
        values = pd.to_numeric(frame[variable], errors="coerce").dropna()
        quantiles = values.quantile([0.25, 0.50, 0.75])
        rows.append({
            "population": population, "shock": shock, "family": family,
            "variable": variable,
            "n": int(values.size),
            "mean": float(values.mean()),
            "sd": float(values.std()),
            "p25": float(quantiles.loc[0.25]),
            "median": float(quantiles.loc[0.50]),
            "p75": float(quantiles.loc[0.75]),
        })
    return pd.DataFrame(rows)


def _outcome_coverage(frame: pd.DataFrame, outcome: str, study) -> dict:
    sample = frame[
        frame[S.COUNTRY].isin(study.countries)
        & frame[S.YEAR].between(study.first_year, study.last_year)
    ]
    # Match the predeclared coverage gate: compute each country-year reporting rate first, then
    # average annual rates within country.  This prevents years with more surviving firms from
    # receiving a larger weight solely because the panel is unbalanced.
    cells = sample.groupby([S.COUNTRY, S.YEAR], observed=True)[outcome].apply(
        lambda x: float(x.notna().mean())
    )
    values = cells.groupby(level=S.COUNTRY).mean()
    by_country = values.reindex(study.countries).fillna(0).to_dict()
    minimum = min(by_country.values())
    return {
        "threshold": COVERAGE_THRESHOLD, "minimum": minimum,
        "by_country": by_country, "pass": minimum >= COVERAGE_THRESHOLD,
    }


def _prepare_public(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for outcome in ("capex_at_reported", "capex_ppent_reported"):
        out[outcome] = engine._clip_country_year(out, outcome)
        out[f"lag_{outcome}"] = engine._adjacent_lag(out, outcome)
    return out


def _hypothesis_outcomes(population: str) -> dict[str, tuple[str, ...]]:
    common_h1 = ("equity_growth_real", "equity_ratio")
    common_h2 = (
        "tangible_growth_real", "asset_growth_real", "intangible_growth_real",
        "cash_at", "fin_debt_ratio", "dlog_debt",
    )
    if population == "private_large_vl":
        return {
            "H1_financing": (*common_h1, "dlog_capital", "cap_increase"),
            "H2_real_activity": common_h2,
            "H3_retention_asymmetry": ("reserves_ratio",),
        }
    return {
        "H1_financing": (
            "net_ext_equity_at", *common_h1,
        ),
        "H2_real_activity": (*common_h2, "capex_at", "capex_at_reported",
                             "capex_ppent", "capex_ppent_reported",
                             "sales_growth_real", "operating_roa"),
        "H3_retention_asymmetry": ("cash_dividends_at", "retained_earnings_at"),
    }


def _family_max_t(
    results: dict, specification: str, outcomes: tuple[str, ...],
) -> dict:
    eligible = [
        outcome for outcome in outcomes
        if outcome in results
        if results[outcome]["status"].startswith("estimated")
        and specification in results[outcome]["specifications"]
    ]
    standardized = {country: [] for country in engine.SHOCKS["2013"].countries}
    for outcome in eligible:
        coefficients = results[outcome]["specifications"][specification][
            "placebo_coefficients"
        ]
        scale = float(np.std(list(coefficients.values()), ddof=1))
        if not np.isfinite(scale) or scale <= 0:
            continue
        for country, value in coefficients.items():
            standardized[country].append(abs(value) / scale)
    maxima = {country: max(values) for country, values in standardized.items() if values}
    if "FR" not in maxima:
        return {"status": "not_estimated", "outcomes": eligible}
    treated = maxima["FR"]
    return {
        "status": "estimated", "outcomes": eligible, "statistic": treated,
        "placebo_p": sum(value >= treated for value in maxima.values()) / len(maxima),
        "placebo_maxima": maxima,
    }


def _cross_shock_contrasts(
    results: dict, later_shock: str, specification: str = MANUSCRIPT_SPECIFICATION,
    frame: pd.DataFrame | None = None,
) -> dict:
    """Compare the 2013 increase with the later 2018 decrease in the same outcome."""
    output = {}
    for outcome in sorted(set(results["2013"]).intersection(results[later_shock])):
        early = results["2013"][outcome]
        later = results[later_shock][outcome]
        if not (early["status"].startswith("estimated")
                and later["status"].startswith("estimated")):
            continue
        if specification not in early["specifications"] or specification not in later["specifications"]:
            continue
        early_coefficients = early["specifications"][specification]["placebo_coefficients"]
        later_coefficients = later["specifications"][specification]["placebo_coefficients"]
        contrasts = {
            country: early_coefficients[country] - later_coefficients[country]
            for country in early_coefficients
        }
        treated = contrasts["FR"]
        output[outcome] = {
            "definition": f"2013 coefficient minus {later_shock} coefficient",
            "contrast": treated,
            "one_sided_placebo_p": (
                sum(value >= treated for value in contrasts.values()) / len(contrasts)
            ),
            "two_sided_placebo_p": (
                sum(abs(value) >= abs(treated) for value in contrasts.values())
                / len(contrasts)
            ),
            "placebo_contrasts": contrasts,
        }
        if (
            frame is None
            and "std_error" in early["specifications"][specification]
            and "std_error" in later["specifications"][specification]
        ):
            early_se = float(early["specifications"][specification]["std_error"])
            later_se = float(later["specifications"][specification]["std_error"])
            standard_error = math.sqrt(early_se * early_se + later_se * later_se)
            output[outcome].update({
                "std_error": standard_error,
                "t_stat": treated / standard_error if standard_error > 0 else float("nan"),
                "inference": (
                    "descriptive independence approximation; placebo contrast is primary"
                ),
            })
        if frame is not None:
            early_inference = _coefficient_cluster_influence(
                frame, outcome, "2013",
            )
            later_inference = _coefficient_cluster_influence(
                frame, outcome, later_shock,
            )
            if not np.isclose(
                early_inference["coef"],
                early["specifications"][specification]["coef"],
                rtol=1e-8, atol=1e-10,
            ) or not np.isclose(
                later_inference["coef"],
                later["specifications"][specification]["coef"],
                rtol=1e-8, atol=1e-10,
            ):
                raise AssertionError(f"joint contrast does not reproduce {outcome}")
            countries = sorted(
                set(early_inference["influence"]) | set(later_inference["influence"])
            )
            difference_influence = {
                country: early_inference["influence"].get(country, 0.0)
                - later_inference["influence"].get(country, 0.0)
                for country in countries
            }
            variance = sum(value * value for value in difference_influence.values())
            standard_error = math.sqrt(max(variance, 0.0))
            output[outcome].update({
                "std_error": standard_error,
                "t_stat": treated / standard_error if standard_error > 0 else float("nan"),
                "cluster_influence_difference": difference_influence,
                "inference": (
                    "joint country-cluster sandwich covariance across the overlapping "
                    "2013 and 2018 estimation windows"
                ),
            })
    return output


def _coefficient_cluster_influence(
    frame: pd.DataFrame, outcome: str, shock: str,
) -> dict:
    """Return the manuscript coefficient and its country-cluster influence terms.

    Reconstructing the exact common sample and fixed-effect residualization used by
    ``engine._estimate`` lets the two overlapping-window coefficients share a covariance term.
    The contrast variance is the sum of squared differences in their country influence terms.
    """
    study = engine.SHOCKS[shock]
    controls = _controls_for_outcome(outcome)
    sector_column = "sector_stable_2013" if shock == "2013" else "sector_stable_2018"
    coverage = engine._coverage(frame, outcome, study, controls, sector_column)
    if not coverage["controls_pass"]:
        raise ValueError(f"controls fail coverage for {shock}/{outcome}")
    lag_outcome = f"lag_{outcome}"
    dynamic_available = coverage["dynamic_pass"]
    needed = [S.FIRM, S.COUNTRY, S.YEAR, outcome, *controls, sector_column]
    if dynamic_available:
        needed.append(lag_outcome)
    sample = frame[
        frame[S.COUNTRY].isin(study.countries)
        & frame[S.YEAR].between(study.first_year, study.last_year)
    ][needed].dropna().copy()
    sample["sector_year"] = (
        sample[sector_column].astype(str) + "|" + sample[S.YEAR].astype(str)
    )
    for column in controls:
        sample[column] = engine._clip_country_year(sample, column)
    treatment = "post_FR"
    sample[treatment] = (
        sample[S.COUNTRY].eq(study.treated)
        & sample[S.YEAR].ge(study.reform_year)
    ).astype(float)
    columns = [outcome, treatment, *controls]
    residual = engine._residualize(sample, columns, sector_year=True)
    y = residual[:, 0]
    x = residual[:, 1:]
    xtx_inverse = np.linalg.pinv(x.T @ x)
    beta = xtx_inverse @ (x.T @ y)
    regression_residual = y - x @ beta
    cluster = sample[S.COUNTRY].to_numpy()
    influence = {}
    for country in np.unique(cluster):
        score = x[cluster == country].T @ regression_residual[cluster == country]
        influence[str(country)] = float((xtx_inverse @ score)[0])
    return {"coef": float(beta[0]), "influence": influence, "n_obs": int(len(sample))}


def _contrast_rows(
    private_contrasts: dict, public_contrasts: dict,
) -> pd.DataFrame:
    rows = []
    for population, population_contrasts in (
        ("private_large_vl", private_contrasts),
        ("public_listed", public_contrasts),
    ):
        for later, outcomes in population_contrasts.items():
            for outcome, values in outcomes.items():
                rows.append({
                    "population": population,
                    "later_shock": later, "outcome": outcome,
                    "role": _outcome_role(population, outcome),
                    "contrast_2013_minus_2018": values["contrast"],
                    "std_error": values.get("std_error", np.nan),
                    "t_stat": values.get("t_stat", np.nan),
                    "one_sided_placebo_p": values["one_sided_placebo_p"],
                    "two_sided_placebo_p": values["two_sided_placebo_p"],
                })
    return pd.DataFrame(rows)


def refresh_public_cross_shock_contrasts() -> dict:
    """Refresh listed-firm contrasts without re-estimating the million-row private panel."""
    target = paths.RESULTS / "firm_growth_harmonized_results.json"
    output = json.loads(target.read_text())
    public = _prepare_public(pd.read_parquet(panels.PUBLIC_TARGET))
    public_results = output["populations"]["public_listed"]
    public_contrasts = {
        later: _cross_shock_contrasts(public_results, later)
        for later in ("2018_precovid", "2018_full")
    }
    output["public_cross_shock_contrasts"] = public_contrasts
    private_contrasts = output["private_cross_shock_contrasts"]
    _contrast_rows(private_contrasts, public_contrasts).to_csv(
        paths.RESULTS / "firm_growth_harmonized_cross_shock.csv", index=False,
        float_format="%.6f",
    )
    target.write_text(json.dumps(output, indent=2))
    return public_contrasts


def _manuscript_event_rows(population_results: dict) -> pd.DataFrame:
    """Collect private and listed event studies and restore omitted reference years."""
    rows = []
    for population, results in population_results.items():
        available = set(square.outcomes(population))
        for shock, reference_year in (
            ("2013", 2012), ("2018_precovid", 2017), ("2018_full", 2017),
        ):
            if shock not in results:
                continue
            for outcome in MANUSCRIPT_EVENT_OUTCOMES:
                if outcome not in available or outcome not in results[shock]:
                    continue
                result = results[shock][outcome]
                if not result["status"].startswith("estimated"):
                    continue
                specification = result["specifications"][MANUSCRIPT_SPECIFICATION]
                rows.append({
                    "population": population, "shock": shock, "outcome": outcome,
                    "year": reference_year, "coef": 0.0, "std_error": 0.0, "ci95": 0.0,
                })
                for value in specification["event_study"]:
                    rows.append({
                        "population": population, "shock": shock, "outcome": outcome,
                        "year": int(value["year"]), "coef": float(value["coef"]),
                        "std_error": float(value["std_error"]),
                        "ci95": 1.96 * float(value["std_error"]),
                    })
    return pd.DataFrame(rows).sort_values(["population", "shock", "outcome", "year"])


def _pretrend_rows(population_results: dict) -> pd.DataFrame:
    """Flatten the preferred-specification joint pretrend diagnostics."""
    rows = []
    for population, shocks in population_results.items():
        for shock, outcomes in shocks.items():
            for outcome, result in outcomes.items():
                if not result["status"].startswith("estimated"):
                    continue
                diagnostic = result["specifications"][MANUSCRIPT_SPECIFICATION][
                    "pretrend_joint_test"
                ]
                rows.append({
                    "population": population, "shock": shock, "outcome": outcome,
                    "status": diagnostic["status"],
                    "years": "|".join(str(year) for year in diagnostic.get("years", [])),
                    "statistic": diagnostic.get("statistic", np.nan),
                    "degrees_of_freedom": diagnostic.get("degrees_of_freedom", np.nan),
                    "conventional_p": diagnostic.get("conventional_p", np.nan),
                })
    return pd.DataFrame(rows)


def _estimate_square_family(
    sample: pd.DataFrame,
    family_outcomes: tuple[str, ...],
    study,
    sector_column: str,
    *,
    include_year_baseline: bool = True,
) -> dict[str, dict]:
    """Estimate a whole square family after absorbing its fixed effects only once."""
    work = sample.copy(deep=False)
    work["sector_year"] = (
        work[sector_column].astype(str) + "|" + work[S.YEAR].astype(str)
    )
    for control in CONTROLS:
        work[control] = engine._clip_country_year(work, control)
    placebo_names = []
    for country in study.countries:
        name = f"post_{country}"
        work[name] = (
            work[S.COUNTRY].eq(country) & work[S.YEAR].ge(study.reform_year)
        ).astype(float)
        placebo_names.append(name)
    event_years = [
        year for year in sorted(work[S.YEAR].unique()) if year != study.reference_year
    ]
    event_names = []
    for year in event_years:
        name = f"event_{year}"
        work[name] = (
            work[S.COUNTRY].eq(study.treated) & work[S.YEAR].eq(year)
        ).astype(float)
        event_names.append(name)
    columns = [*family_outcomes, *placebo_names, *event_names, *CONTROLS]
    positions = {name: index for index, name in enumerate(columns)}
    cluster = work[S.COUNTRY].to_numpy()
    output_by_outcome: dict[str, dict] = {outcome: {} for outcome in family_outcomes}

    def estimate_specification(
        residual: np.ndarray, outcome: str, label: str, selected_controls: list[str],
    ) -> dict:
        y = residual[:, positions[outcome]]
        control_idx = [positions[name] for name in selected_controls]
        treated_name = f"post_{study.treated}"
        x = residual[:, [positions[treated_name], *control_idx]]
        beta, se = engine._ols_cluster_pinv(x, y, cluster)
        placebo_coefficients = {}
        for country, placebo_name in zip(study.countries, placebo_names):
            xp = residual[:, [positions[placebo_name], *control_idx]]
            placebo_beta, _ = engine._ols_cluster_pinv(xp, y, cluster)
            placebo_coefficients[country] = float(placebo_beta[0])
        treated_abs = abs(placebo_coefficients[study.treated])
        placebo_p = sum(
            abs(value) >= treated_abs for value in placebo_coefficients.values()
        ) / len(placebo_coefficients)
        event_x = residual[:, [*(positions[name] for name in event_names), *control_idx]]
        event_beta, event_covariance = engine._ols_cluster_cov(event_x, y, cluster)
        event_se = np.sqrt(np.clip(np.diag(event_covariance), 0.0, None))
        return {
            "coef": float(beta[0]), "std_error": float(se[0]),
            "placebo_p": float(placebo_p),
            "placebo_coefficients": placebo_coefficients,
            "controls": selected_controls,
            "control_estimates": {
                name: {"coef": float(beta[index + 1]), "std_error": float(se[index + 1])}
                for index, name in enumerate(selected_controls)
            },
            "fixed_effects": (
                ["firm", "year"] if label.startswith("A_")
                else ["firm", "industry_by_year"]
            ),
            "event_study": [
                {"year": int(year), "coef": float(event_beta[index]),
                 "std_error": float(event_se[index])}
                for index, year in enumerate(event_years)
            ],
            "pretrend_joint_test": engine._pretrend_test(
                event_beta, event_covariance, event_years, study.reform_year,
            ),
        }

    # The two absorptions are deliberately sequential.  Keeping both million-row matrices alive
    # doubles peak memory without adding information.
    if include_year_baseline:
        residual = engine._residualize(work, columns)
        for outcome in family_outcomes:
            label = "A_firm_year_fe_common_sample"
            output_by_outcome[outcome][label] = estimate_specification(
                residual, outcome, label, [],
            )
        del residual
        gc.collect()

    residual = engine._residualize(work, columns, sector_year=True)
    for outcome in family_outcomes:
        controls = list(_controls_for_outcome(outcome))
        without_age = [value for value in controls if value != "lag_log_firm_age"]
        without_size = [value for value in controls if value != "lag_log_assets_real"]
        without_leverage = [value for value in controls if value != "lag_book_leverage"]
        specifications = {
            "B_firm_industry_year_fe_common_sample": [],
            MANUSCRIPT_SPECIFICATION: controls,
            "C_controls_no_ptbi_no_age": without_age,
            "C_controls_no_ptbi_no_size": without_size,
            "C_controls_no_ptbi_no_leverage": without_leverage,
        }
        for label, selected_controls in specifications.items():
            output_by_outcome[outcome][label] = estimate_specification(
                residual, outcome, label, selected_controls,
            )
    del residual
    gc.collect()

    results = {}
    for outcome in family_outcomes:
        results[outcome] = {
            "status": "estimated_static_only_by_design",
            "n_obs": int(len(work)), "n_firms": int(work[S.FIRM].nunique()),
            "specifications": output_by_outcome[outcome], "sector_column": sector_column,
            "dynamic_bias_correction": "not_applicable_static_model_selected_by_design",
        }
    return results


def _shard_path(population: str, shock: str, family: str) -> Path:
    return SHARD_DIR / f"{population}_{shock}_{family}.json"


def _load_shard(
    population: str, shock: str, family: str, panel_sha256: str,
) -> dict | None:
    target = _shard_path(population, shock, family)
    if not target.exists():
        return None
    payload = json.loads(target.read_text())
    # Control coefficients are published only for the 2013 and pre-Covid 2018
    # specifications.  The pandemic-inclusive window remains a sensitivity table
    # without control rows, so its verified v1 shards do not need re-estimation.
    controls_not_published = shock == "2018_full" or family.startswith("complete")
    accepted_versions = {SHARD_SCHEMA_VERSION, 1} if controls_not_published else {SHARD_SCHEMA_VERSION}
    if payload.get("schema_version") not in accepted_versions:
        return None
    if payload.get("panel_sha256") != panel_sha256:
        return None
    return payload


def _write_shard(
    population: str, shock: str, family: str, panel_sha256: str,
    results: dict, manifest: dict,
) -> None:
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    target = _shard_path(population, shock, family)
    temporary = target.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({
        "schema_version": SHARD_SCHEMA_VERSION,
        "panel_sha256": panel_sha256,
        "population": population,
        "shock": shock,
        "family": family,
        "results": results,
        "manifest": manifest,
    }, indent=2))
    temporary.replace(target)


def _estimate_population(
    frame: pd.DataFrame | None, population: str, outcomes: tuple[str, ...],
    summary_rows: list[dict], coverage_rows: list[dict], square_manifests: list[dict],
    descriptive_rows: list[pd.DataFrame],
    *, complete_population: bool, panel_sha256: str,
    release_index: dict[tuple[str, str, str], dict] | None = None,
) -> tuple[dict, dict, dict]:
    population_results: dict = {}
    complete_results: dict = {}
    family_results: dict = {}

    for shock, study in engine.SHOCKS.items():
        population_results[shock] = {}
        complete_results[shock] = {}
        sector_column = "sector_stable_2013" if shock == "2013" else "sector_stable_2018"
        coverage_by_outcome = {}
        for outcome in outcomes:
            if frame is not None:
                coverage = _outcome_coverage(frame, outcome, study)
            else:
                if release_index is None:
                    raise AssertionError("released sample index is required")
                family = "complete" if complete_population else next(
                    name for name, values in square.FAMILIES[population].items()
                    if outcome in values
                )
                manifest = release_index[(population, shock, family)]
                by_country = {
                    country: float(manifest["rows_by_country"].get(country, 0) > 0)
                    for country in study.countries
                }
                coverage = {
                    "threshold": COVERAGE_THRESHOLD,
                    "minimum": min(by_country.values()),
                    "by_country": by_country,
                    "pass": min(by_country.values()) >= COVERAGE_THRESHOLD,
                }
            coverage_by_outcome[outcome] = coverage
            for country, value in coverage["by_country"].items():
                coverage_rows.append({
                    "population": population, "shock": shock, "outcome": outcome,
                    "country": country, "coverage": value, "minimum": coverage["minimum"],
                    "pass": coverage["pass"],
                })

        declared = (
            {"complete": square.outcomes(population)}
            if complete_population else square.FAMILIES[population]
        )
        for family, family_outcomes in declared.items():
            cached = _load_shard(population, shock, family, panel_sha256)
            if cached is None:
                if frame is None:
                    if release_index is None:
                        raise AssertionError("released sample index is required")
                    sample, manifest = _released_sample(
                        release_index, population, shock, family,
                    )
                else:
                    sample, manifest = square.materialize_one(
                        frame, population=population, shock=shock, study=study,
                        family=family, dependent_variables=family_outcomes,
                    )
                estimated = {}
                for outcome in family_outcomes:
                    coverage = coverage_by_outcome[outcome]
                    outcome_family = f"{family}__{outcome}"
                    outcome_cache = _load_shard(
                        population, shock, outcome_family, panel_sha256,
                    )
                    if outcome_cache is None:
                        print(f"estimating {population} {shock} {outcome}", flush=True)
                        result = _estimate_square_family(
                            sample, (outcome,), study, sector_column,
                            include_year_baseline=shock != "2018_full",
                        )[outcome]
                        result["outcome_coverage"] = coverage
                        result["square_sample"] = {
                            "family": family, "n_obs": int(len(sample)),
                            "n_firms": int(sample[S.FIRM].nunique()),
                        }
                        _write_shard(
                            population, shock, outcome_family, panel_sha256,
                            {outcome: result}, manifest,
                        )
                    else:
                        result = outcome_cache["results"][outcome]
                        print(
                            f"reusing verified outcome shard {population} {shock} {outcome}",
                            flush=True,
                        )
                    estimated[outcome] = result
                _write_shard(
                    population, shock, family, panel_sha256, estimated, manifest,
                )
                cached = {"results": estimated, "manifest": manifest}
                del sample
                gc.collect()
            else:
                print(f"reusing verified shard {population} {shock} {family}", flush=True)
            if shock in {"2013", "2018_precovid"}:
                sample_path = paths.ROOT / cached["manifest"]["path"]
                descriptive_variables = tuple(dict.fromkeys([*family_outcomes, *CONTROLS]))
                descriptive_sample = pd.read_parquet(
                    sample_path, columns=list(descriptive_variables),
                )
                expected_rows = int(cached["manifest"]["final_rows"])
                if len(descriptive_sample) != expected_rows:
                    raise AssertionError(
                        f"descriptive sample mismatch: {population}/{shock}/{family}: "
                        f"{len(descriptive_sample)} != {expected_rows}"
                    )
                descriptive_rows.append(_descriptive_rows(
                    descriptive_sample, population, shock, family,
                    descriptive_variables,
                ))
                del descriptive_sample
            square_manifests.append(cached["manifest"])
            population_results[shock].update(cached["results"])

        for outcome in outcomes:
            result = population_results[shock][outcome]
            if result["status"].startswith("estimated"):
                for specification, values in result["specifications"].items():
                    summary_rows.append({
                        "population": population, "shock": shock, "outcome": outcome,
                        "role": _outcome_role(population, outcome),
                        "specification": specification, "coef": values["coef"],
                        "std_error": values["std_error"],
                        "conventional_p": _normal_p(values["coef"], values["std_error"]),
                        "placebo_p": values["placebo_p"], "n_obs": result["n_obs"],
                        "n_firms": result["n_firms"],
                        "outcome_coverage_min": coverage["minimum"],
                        "square_family": result["square_sample"]["family"],
                    })
        if not complete_population and shock != "2018_full":
            cached = _load_shard(population, shock, "complete", panel_sha256)
            if cached is None:
                if frame is None:
                    if release_index is None:
                        raise AssertionError("released sample index is required")
                    complete, manifest = _released_sample(
                        release_index, population, shock, "complete",
                    )
                else:
                    complete, manifest = square.materialize_one(
                        frame, population=population, shock=shock, study=study,
                        family="complete", dependent_variables=square.outcomes(population),
                    )
                estimated = {}
                for outcome in outcomes:
                    outcome_family = f"complete__{outcome}"
                    outcome_cache = _load_shard(
                        population, shock, outcome_family, panel_sha256,
                    )
                    if outcome_cache is None:
                        print(
                            f"estimating complete sensitivity {population} {shock} {outcome}",
                            flush=True,
                        )
                        result = _estimate_square_family(
                            complete, (outcome,), study, sector_column,
                        )[outcome]
                        result["outcome_coverage"] = coverage_by_outcome[outcome]
                        result["square_sample"] = {
                            "family": "complete", "n_obs": int(len(complete)),
                            "n_firms": int(complete[S.FIRM].nunique()),
                        }
                        _write_shard(
                            population, shock, outcome_family, panel_sha256,
                            {outcome: result}, manifest,
                        )
                    else:
                        result = outcome_cache["results"][outcome]
                    estimated[outcome] = result
                _write_shard(
                    population, shock, "complete", panel_sha256, estimated, manifest,
                )
                cached = {"results": estimated, "manifest": manifest}
                del complete
                gc.collect()
            else:
                print(f"reusing verified shard {population} {shock} complete", flush=True)
            square_manifests.append(cached["manifest"])
            complete_results[shock] = cached["results"]
        preferred = MANUSCRIPT_SPECIFICATION
        family_results[shock] = {
            hypothesis: _family_max_t(
                population_results[shock], preferred, hypothesis_outcomes,
            )
            for hypothesis, hypothesis_outcomes in _hypothesis_outcomes(population).items()
        }
    return population_results, family_results, complete_results


def run() -> dict:
    identifiers = [S.FIRM, S.COUNTRY, S.YEAR, "sector_stable_2013", "sector_stable_2018"]
    private_outcomes = (*COMMON_OUTCOMES, *PRIVATE_SPECIFIC)
    public_outcomes = (*COMMON_OUTCOMES, *PUBLIC_SPECIFIC)
    release_index = None
    if HARMONIZED_PRIVATE.is_file() and HARMONIZED_PUBLIC.is_file():
        private_sha256 = _sha256(HARMONIZED_PRIVATE)
        public_sha256 = _sha256(HARMONIZED_PUBLIC)
        private = pd.read_parquet(
            HARMONIZED_PRIVATE,
            columns=list(dict.fromkeys([*identifiers, *CONTROLS, *private_outcomes])),
        )
        public = pd.read_parquet(
            HARMONIZED_PUBLIC,
            columns=list(dict.fromkeys([*identifiers, *CONTROLS, *public_outcomes])),
        )
    else:
        if not _released_inputs_complete():
            raise FileNotFoundError("final estimation samples are incomplete")
        release_index = _release_index()
        release_sha256 = _sha256(RELEASE_MANIFEST)
        private_sha256 = release_sha256
        public_sha256 = release_sha256
        private = None
        public = None
    descriptive_rows: list[pd.DataFrame] = []
    summary_rows: list[dict] = []
    coverage_rows: list[dict] = []
    square_manifests: list[dict] = []
    private_results, private_family, private_complete = _estimate_population(
        private, "private_large_vl", private_outcomes,
        summary_rows, coverage_rows, square_manifests, descriptive_rows,
        complete_population=False,
        panel_sha256=private_sha256,
        release_index=release_index,
    )
    del private
    public_results, public_family, _ = _estimate_population(
        public, "public_listed", public_outcomes,
        summary_rows, coverage_rows, square_manifests, descriptive_rows,
        complete_population=True,
        panel_sha256=public_sha256,
        release_index=release_index,
    )
    square.write_manifest(square_manifests)
    public_cross_shock = {
        later: _cross_shock_contrasts(public_results, later)
        for later in ("2018_precovid", "2018_full")
    }
    del public
    _assert_manuscript_specification({
        "private_large_vl": private_results,
        "public_listed": public_results,
    })

    summary = pd.DataFrame(summary_rows)
    comparison_rows = []
    for shock in engine.SHOCKS:
        preferred = MANUSCRIPT_SPECIFICATION
        for outcome in COMMON_OUTCOMES:
            row = {"shock": shock, "outcome": outcome, "specification": preferred}
            for population in ("private_large_vl", "public_listed"):
                cell = summary[
                    summary["population"].eq(population)
                    & summary["shock"].eq(shock)
                    & summary["outcome"].eq(outcome)
                    & summary["specification"].eq(preferred)
                ]
                prefix = "private" if population == "private_large_vl" else "public"
                if cell.empty:
                    for name in ("coef", "std_error", "conventional_p", "placebo_p",
                                 "n_obs", "n_firms"):
                        row[f"{prefix}_{name}"] = np.nan
                else:
                    values = cell.iloc[0]
                    for name in ("coef", "std_error", "conventional_p", "placebo_p",
                                 "n_obs", "n_firms"):
                        row[f"{prefix}_{name}"] = values[name]
            comparison_rows.append(row)

    paths.RESULTS.mkdir(parents=True, exist_ok=True)
    descriptive = pd.concat(descriptive_rows, ignore_index=True)
    descriptive.to_csv(
        paths.RESULTS / "firm_growth_harmonized_descriptive.csv", index=False,
        float_format="%.6f",
    )
    summary.to_csv(
        paths.RESULTS / "firm_growth_harmonized_summary.csv", index=False,
        float_format="%.6f",
    )
    pd.DataFrame(comparison_rows).to_csv(
        paths.RESULTS / "firm_growth_harmonized_comparison.csv", index=False,
        float_format="%.6f",
    )
    pd.DataFrame(coverage_rows).to_csv(
        paths.RESULTS / "firm_growth_harmonized_coverage.csv", index=False,
        float_format="%.6f",
    )
    hypothesis_family = {
        "private_large_vl": private_family, "public_listed": public_family,
    }
    cross_shock = {
        later: _cross_shock_contrasts(private_results, later)
        for later in ("2018_precovid", "2018_full")
    }
    _contrast_rows(cross_shock, public_cross_shock).to_csv(
        paths.RESULTS / "firm_growth_harmonized_cross_shock.csv", index=False,
        float_format="%.6f",
    )
    family_rows = []
    for population, shocks in hypothesis_family.items():
        for shock, hypotheses in shocks.items():
            for hypothesis, values in hypotheses.items():
                family_rows.append({
                    "population": population, "shock": shock,
                    "hypothesis": hypothesis, "status": values["status"],
                    "outcomes": "|".join(values.get("outcomes", [])),
                    "statistic": values.get("statistic", np.nan),
                    "placebo_p": values.get("placebo_p", np.nan),
                })
    pd.DataFrame(family_rows).to_csv(
        paths.RESULTS / "firm_growth_harmonized_family_tests.csv", index=False,
        float_format="%.6f",
    )
    manuscript_events = _manuscript_event_rows({
        "private_large_vl": private_results,
        "public_listed": public_results,
    })
    manuscript_events.to_csv(
        paths.RESULTS / "firm_growth_harmonized_event_studies.csv", index=False,
        float_format="%.6f",
    )
    _pretrend_rows({
        "private_large_vl": private_results,
        "public_listed": public_results,
    }).to_csv(
        paths.RESULTS / "firm_growth_harmonized_pretrends.csv", index=False,
        float_format="%.6f",
    )
    emit.harmonized_accounting_event_csv(manuscript_events)
    emit.harmonized_main_event_csv(
        manuscript_events[manuscript_events["population"].eq("private_large_vl")]
    )
    panel_manifest_path = paths.RESULTS / "harmonized_firm_panel_manifest.json"
    listed_source_path = paths.DATA / "raw" / "listed_growth_panel_2005.parquet"
    flow_counts = None
    listed_source_rows = None
    if panel_manifest_path.is_file() and listed_source_path.is_file():
        panel_manifest = json.loads(panel_manifest_path.read_text())
        flow_counts = panel_manifest["public_source_flow_counts"]
        listed_source_rows = int(pd.read_parquet(
            listed_source_path, columns=["gvkey"]
        ).shape[0])
    output = {
        "status": "exploratory_manuscript_robustness",
        "design": "canonical common-sample private-public firm outcomes",
        "sample_rule": {
            "missing": "listwise outcomes plus lagged controls; no fillna",
            "timing": "at least one firm observation before and after each reform",
            "private": "common sample by family; complete intersection in sensitivity",
            "public": "one complete common sample across all outcome families",
        },
        "controls": list(CONTROLS), "coverage_threshold": COVERAGE_THRESHOLD,
        "manuscript_specification": MANUSCRIPT_SPECIFICATION,
        "fixed_effects": {
            "primary": ["firm", "industry_by_year"],
            "baseline": ["firm", "year"],
            "industry_columns": {
                "2013": "sector_stable_2013", "2018": "sector_stable_2018",
            },
        },
        "control_conflicts": {
            outcome: sorted(controls) for outcome, controls in CONTROL_CONFLICTS.items()
        },
        "common_primary": list(COMMON_PRIMARY),
        "common_secondary": list(COMMON_SECONDARY),
        "private_specific": list(PRIVATE_SPECIFIC),
        "public_specific": list(PUBLIC_SPECIFIC),
        "hypothesis_mapping": {
            "H1_financing": [
                "dlog_capital", "cap_increase", "net_ext_equity_at",
                "equity_growth_real", "equity_ratio",
            ],
            "H2_real_activity": [
                "tangible_growth_real", "asset_growth_real", "intangible_growth_real",
                "cash_at", "fin_debt_ratio", "dlog_debt", "capex_at",
                "sales_growth_real", "operating_roa",
            ],
            "H3_retention_asymmetry": [
                "reserves_ratio", "retained_earnings_at",
            ],
        },
        "private_capex": "not_estimated_by_author_decision",
        "preferred_specification": {
            "2013": MANUSCRIPT_SPECIFICATION,
            "2018_precovid": MANUSCRIPT_SPECIFICATION,
            "2018_full": MANUSCRIPT_SPECIFICATION,
        },
        "populations": {
            "private_large_vl": private_results, "public_listed": public_results,
        },
        "private_complete_sample_sensitivity": private_complete,
        "square_sample_manifest": square_manifests,
        "listed_flow_variables_excluded": {
            "stock_issuance_at": "coverage_below_50_percent_in_each_country",
            "repurchases_at": (
                (
                    f"PRSTKC has {flow_counts['prstkc']['nonmissing_source_rows']} nonmissing "
                    f"values in {listed_source_rows} source rows"
                )
                if flow_counts is not None else
                "excluded before construction of the released final samples"
            ),
            "net_equity_issuance_at": "PRSTKC prevents construction",
            "cash_dividends_at": "coverage_below_50_percent_in_each_country",
            "DVP_nonmissing_source_rows": (
                flow_counts["dvp"]["nonmissing_source_rows"]
                if flow_counts is not None else None
            ),
        },
        "hypothesis_family_max_t": hypothesis_family,
        "private_cross_shock_contrasts": cross_shock,
        "public_cross_shock_contrasts": public_cross_shock,
    }
    (paths.RESULTS / "firm_growth_harmonized_results.json").write_text(
        json.dumps(output, indent=2)
    )
    # The two country-composition checks consume the exact canonical samples just materialized
    # above and append their audited summary to the main result object.
    from src.analysis import firm_growth_square_robustness
    robustness = firm_growth_square_robustness.run(output)
    from src.figures import firm_reporting
    firm_reporting.write_firm_reporting(output, robustness)
    return output


if __name__ == "__main__":
    run()
