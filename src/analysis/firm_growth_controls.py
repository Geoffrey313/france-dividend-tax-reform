"""PFU-US-04 controlled dynamic DiD without PTBI.

The author-approved control set is lagged book leverage, lagged real size, lagged log age and,
for the dynamic column, the lagged dependent variable.  Columns A, B and C are always estimated
on the complete-case sample required by column C.  The module is exploratory and writes only to
``results/``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

from src.common import paths, schema as S
from src.common.config import Study
from src.engine import did


SHOCKS = {
    "2013": Study(first_year=2008, reference_year=2012, reform_year=2013, last_year=2016),
    "2018_precovid": Study(first_year=2013, reference_year=2017, reform_year=2018, last_year=2019),
    "2018_full": Study(first_year=2013, reference_year=2017, reform_year=2018, last_year=2025),
}
TIERS = {"private_large_vl": "", "private_medium": "_medium"}
OUTCOMES = ("tangible_growth_real",)
CONTROLS = ("lag_book_leverage", "lag_log_assets_real", "lag_log_firm_age")
COVERAGE_THRESHOLD = 0.50


def _adjacent_lag(frame: pd.DataFrame, column: str) -> pd.Series:
    grouped = frame.groupby(S.FIRM, sort=False)
    lag = grouped[column].shift(1)
    return lag.where(grouped[S.YEAR].diff().eq(1))


def _clip_country_year(frame: pd.DataFrame, column: str) -> pd.Series:
    """One-percent winsorisation with bounds fixed by country-year."""
    grouped = frame.groupby([S.COUNTRY, S.YEAR], observed=True)[column]
    lo = grouped.transform(lambda x: x.quantile(0.01))
    hi = grouped.transform(lambda x: x.quantile(0.99))
    return frame[column].clip(lower=lo, upper=hi)


def _read_population(raw: Path, suffix: str, deflator: pd.DataFrame) -> pd.DataFrame:
    parts = []
    columns = [S.FIRM, S.YEAR, S.COUNTRY, "total_assets", "tangible_fixed",
               "fiscal_months", "orbis_tier", "category_of_company"]
    control_columns = [S.FIRM, S.YEAR, "shareholders_funds", "incorporation_year",
                       "orbis_tier"]
    for country in SHOCKS["2013"].countries:
        growth_path = raw / f"private_growth_{country}{suffix}.parquet"
        control_path = raw / f"private_controls_{country}{suffix}.parquet"
        if not growth_path.exists() or not control_path.exists():
            raise FileNotFoundError(f"missing PFU-US-04 inputs for {country}{suffix}")
        growth = pd.read_parquet(growth_path, columns=columns)
        controls = pd.read_parquet(control_path, columns=control_columns)
        controls = controls.rename(columns={"orbis_tier": "control_orbis_tier"})
        joined = growth.merge(controls, on=[S.FIRM, S.YEAR], how="left", validate="one_to_one")
        if not joined["control_orbis_tier"].eq(joined["orbis_tier"]).all():
            raise ValueError(f"tier mismatch in PFU-US-04 controls for {country}{suffix}")
        parts.append(joined.drop(columns=["control_orbis_tier"]))

    frame = pd.concat(parts, ignore_index=True)
    tier = frame["orbis_tier"].astype(str).str.lower()
    category = frame["category_of_company"].astype(str).str.lower()
    if not tier.isin({"large", "medium"}).all() or not category.isin({"v", "l", "m"}).all():
        raise ValueError("forbidden or unknown Orbis tier in PFU-US-04 input")

    frame[S.YEAR] = pd.to_numeric(frame[S.YEAR], errors="coerce").astype("Int64")
    frame = frame.dropna(subset=[S.FIRM, S.COUNTRY, S.YEAR]).copy()
    frame[S.YEAR] = frame[S.YEAR].astype(int)
    frame = frame.sort_values([S.FIRM, S.YEAR]).drop_duplicates([S.FIRM, S.YEAR], keep="last")
    frame = frame.merge(deflator[[S.COUNTRY, S.YEAR, "gdp_deflator_2015"]],
                        on=[S.COUNTRY, S.YEAR], how="left", validate="many_to_one")

    assets = pd.to_numeric(frame["total_assets"], errors="coerce")
    tangible = pd.to_numeric(frame["tangible_fixed"], errors="coerce")
    equity = pd.to_numeric(frame["shareholders_funds"], errors="coerce")
    incorporation = pd.to_numeric(frame["incorporation_year"], errors="coerce")
    deflator_value = pd.to_numeric(frame["gdp_deflator_2015"], errors="coerce")
    frame["assets_real"] = assets * 100.0 / deflator_value
    frame["tangible_real"] = tangible * 100.0 / deflator_value
    frame["book_leverage"] = (assets - equity) / assets.where(assets.gt(0))
    frame["firm_age"] = frame[S.YEAR] - incorporation

    lag_tangible = _adjacent_lag(frame, "tangible_real")
    valid = frame["tangible_real"].ge(0) & lag_tangible.ge(0)
    frame["tangible_growth_real"] = np.nan
    frame.loc[valid, "tangible_growth_real"] = (
        np.log1p(frame.loc[valid, "tangible_real"])
        - np.log1p(lag_tangible.loc[valid])
    )
    months = pd.to_numeric(frame["fiscal_months"], errors="coerce")
    lag_months = _adjacent_lag(frame.assign(_months=months), "_months")
    frame["tangible_growth_real"] = frame["tangible_growth_real"].where(
        months.eq(12) & lag_months.eq(12)
    )
    frame["tangible_growth_real"] = _clip_country_year(frame, "tangible_growth_real")

    lag_assets = _adjacent_lag(frame, "assets_real")
    lag_age = _adjacent_lag(frame, "firm_age")
    frame["lag_book_leverage"] = _adjacent_lag(frame, "book_leverage")
    frame["lag_log_assets_real"] = np.log(lag_assets.where(lag_assets.gt(0)))
    frame["lag_log_firm_age"] = np.log1p(lag_age.where(lag_age.ge(0)))
    frame["lag_tangible_growth_real"] = _adjacent_lag(frame, "tangible_growth_real")
    return frame[[S.FIRM, S.COUNTRY, S.YEAR, "tangible_growth_real",
                  "lag_tangible_growth_real", *CONTROLS]]


def _residualize(
    sample: pd.DataFrame, columns: list[str], *, sector_year: bool = False,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Absorb firm plus either year or stable-industry-by-year fixed effects."""
    groups = [S.FIRM, "sector_year"] if sector_year else [S.FIRM, S.YEAR]
    if len(sample) >= 100_000 or weights is not None:
        import pyhdfe

        ids = np.column_stack([
            pd.factorize(sample[group], sort=False)[0] for group in groups
        ])
        absorber = pyhdfe.create(ids, drop_singletons=False, compute_degrees=False,
                                 residualize_method="map")
        hdfe_weights = None
        if weights is not None:
            hdfe_weights = np.asarray(weights, dtype=float).reshape(-1, 1)
        return absorber.residualize(
            sample[columns].to_numpy(dtype=float), weights=hdfe_weights,
        )
    return did._demean(sample, columns, groups)


def _ols_cluster_cov(
    x: np.ndarray, y: np.ndarray, cluster: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    xtx_inv = np.linalg.pinv(x.T @ x)
    beta = xtx_inv @ (x.T @ y)
    residual = y - x @ beta
    meat = np.zeros((x.shape[1], x.shape[1]))
    for value in np.unique(cluster):
        score = x[cluster == value].T @ residual[cluster == value]
        meat += np.outer(score, score)
    return beta, xtx_inv @ meat @ xtx_inv


def _ols_cluster_pinv(
    x: np.ndarray, y: np.ndarray, cluster: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    beta, covariance = _ols_cluster_cov(x, y, cluster)
    return beta, np.sqrt(np.clip(np.diag(covariance), 0.0, None))


def _pretrend_test(
    event_beta: np.ndarray, event_covariance: np.ndarray,
    event_years: list[int], reform_year: int,
) -> dict:
    """Joint Wald diagnostic for all available treated-country pre-reform leads."""
    positions = [index for index, year in enumerate(event_years) if year < reform_year]
    if not positions:
        return {"status": "not_estimable_no_pre_reform_leads"}
    coefficients = event_beta[positions]
    covariance = event_covariance[np.ix_(positions, positions)]
    degrees = int(np.linalg.matrix_rank(covariance))
    if degrees == 0:
        return {"status": "not_estimable_singular_covariance"}
    statistic = float(coefficients @ np.linalg.pinv(covariance) @ coefficients)
    return {
        "status": "estimated_diagnostic_six_country_clusters",
        "years": [int(event_years[index]) for index in positions],
        "statistic": statistic,
        "degrees_of_freedom": degrees,
        "conventional_p": float(chi2.sf(statistic, degrees)),
    }


def _coverage(
    frame: pd.DataFrame, outcome: str, study: Study,
    controls: tuple[str, ...] = CONTROLS,
    sector_column: str | None = None,
    require_dynamic: bool = True,
) -> dict:
    sample = frame[
        frame[S.COUNTRY].isin(study.countries)
        & frame[S.YEAR].between(study.first_year, study.last_year)
        & frame[outcome].notna()
    ]
    required = [*controls]
    if require_dynamic:
        required.append(f"lag_{outcome}")
    if sector_column is not None:
        required.append(sector_column)
    by_variable = {}
    for variable in required:
        values = sample.groupby(S.COUNTRY)[variable].apply(lambda x: float(x.notna().mean()))
        by_variable[variable] = values.reindex(study.countries).fillna(0).to_dict()
    static_required = [*controls, *([sector_column] if sector_column is not None else [])]
    controls_minimum = min(
        min(by_variable[variable].values()) for variable in static_required
    )
    dynamic_minimum = (
        min(by_variable[f"lag_{outcome}"].values()) if require_dynamic else 0.0
    )
    return {
        "threshold": COVERAGE_THRESHOLD,
        "controls_minimum": controls_minimum,
        "dynamic_minimum": dynamic_minimum,
        "by_variable_country": by_variable,
        "controls_pass": controls_minimum >= COVERAGE_THRESHOLD,
        "dynamic_pass": require_dynamic and dynamic_minimum >= COVERAGE_THRESHOLD,
    }


def _estimate(
    frame: pd.DataFrame, outcome: str, study: Study,
    controls: tuple[str, ...] = CONTROLS,
    *, allow_dynamic: bool = True, sector_column: str | None = None,
) -> dict:
    gate = _coverage(
        frame, outcome, study, controls, sector_column,
        require_dynamic=allow_dynamic,
    )
    if not gate["controls_pass"]:
        return {"status": "not_estimated_low_coverage", "coverage": gate}

    lag_outcome = f"lag_{outcome}"
    dynamic_available = gate["dynamic_pass"] and allow_dynamic
    needed = [S.FIRM, S.COUNTRY, S.YEAR, outcome, *controls]
    if sector_column is not None:
        needed.append(sector_column)
    if dynamic_available:
        needed.append(lag_outcome)
    sample = frame[
        frame[S.COUNTRY].isin(study.countries)
        & frame[S.YEAR].between(study.first_year, study.last_year)
    ][needed].dropna().copy()

    if sector_column is not None:
        sample["sector_year"] = (
            sample[sector_column].astype(str) + "|" + sample[S.YEAR].astype(str)
        )

    for column in controls:
        sample[column] = _clip_country_year(sample, column)

    placebo_names = []
    for country in study.countries:
        name = f"post_{country}"
        sample[name] = (
            sample[S.COUNTRY].eq(country) & sample[S.YEAR].ge(study.reform_year)
        ).astype(float)
        placebo_names.append(name)
    event_years = [year for year in sorted(sample[S.YEAR].unique())
                   if year != study.reference_year]
    event_names = []
    for year in event_years:
        name = f"event_{year}"
        sample[name] = (
            sample[S.COUNTRY].eq(study.treated) & sample[S.YEAR].eq(year)
        ).astype(float)
        event_names.append(name)

    all_columns = [outcome, *placebo_names, *event_names, *controls]
    if dynamic_available:
        all_columns.append(lag_outcome)
    residual_year = _residualize(sample, all_columns)
    residual_sector = (
        _residualize(sample, all_columns, sector_year=True)
        if sector_column is not None else residual_year
    )
    positions = {name: index for index, name in enumerate(all_columns)}
    cluster = sample[S.COUNTRY].to_numpy()
    if sector_column is None:
        specifications = {
            "A_fe_common_sample": ("year", []),
            "B_controls_no_ptbi": ("year", list(controls)),
            "B_controls_no_ptbi_no_age": (
                "year", [control for control in controls if control != "lag_log_firm_age"],
            ),
            "B_controls_no_ptbi_no_size": (
                "year", [control for control in controls if control != "lag_log_assets_real"],
            ),
        }
        if dynamic_available:
            specifications["C_dynamic_no_ptbi"] = ("year", [*controls, lag_outcome])
            specifications["C_dynamic_no_ptbi_no_age"] = (
                "year",
                [*(control for control in controls if control != "lag_log_firm_age"), lag_outcome],
            )
            specifications["C_dynamic_no_ptbi_no_size"] = (
                "year",
                [*(control for control in controls if control != "lag_log_assets_real"), lag_outcome],
            )
    else:
        without_age = [control for control in controls if control != "lag_log_firm_age"]
        without_size = [control for control in controls if control != "lag_log_assets_real"]
        without_leverage = [control for control in controls if control != "lag_book_leverage"]
        specifications = {
            "A_firm_year_fe_common_sample": ("year", []),
            "B_firm_industry_year_fe_common_sample": ("sector_year", []),
            "C_controls_no_ptbi": ("sector_year", list(controls)),
            "C_controls_no_ptbi_no_age": ("sector_year", without_age),
            "C_controls_no_ptbi_no_size": ("sector_year", without_size),
            "C_controls_no_ptbi_no_leverage": ("sector_year", without_leverage),
        }
        if dynamic_available:
            specifications["D_dynamic_no_ptbi"] = (
                "sector_year", [*controls, lag_outcome],
            )
            specifications["D_dynamic_no_ptbi_no_age"] = (
                "sector_year", [*without_age, lag_outcome],
            )
            specifications["D_dynamic_no_ptbi_no_size"] = (
                "sector_year", [*without_size, lag_outcome],
            )
            specifications["D_dynamic_no_ptbi_no_leverage"] = (
                "sector_year", [*without_leverage, lag_outcome],
            )
    outputs = {}
    for label, (fixed_effects, specification_controls) in specifications.items():
        residual = residual_sector if fixed_effects == "sector_year" else residual_year
        y = residual[:, positions[outcome]]
        control_idx = [positions[name] for name in specification_controls]
        treated_name = f"post_{study.treated}"
        x = residual[:, [positions[treated_name], *control_idx]]
        beta, se = _ols_cluster_pinv(x, y, cluster)
        placebo_coefficients = {}
        for country, placebo_name in zip(study.countries, placebo_names):
            xp = residual[:, [positions[placebo_name], *control_idx]]
            placebo_beta, _ = _ols_cluster_pinv(xp, y, cluster)
            placebo_coefficients[country] = float(placebo_beta[0])
        treated_abs = abs(placebo_coefficients[study.treated])
        placebo_p = sum(abs(value) >= treated_abs for value in placebo_coefficients.values()) / len(
            placebo_coefficients
        )
        event_x = residual[:, [*(positions[name] for name in event_names), *control_idx]]
        event_beta, event_covariance = _ols_cluster_cov(event_x, y, cluster)
        event_se = np.sqrt(np.clip(np.diag(event_covariance), 0.0, None))
        outputs[label] = {
            "coef": float(beta[0]), "std_error": float(se[0]),
            "placebo_p": float(placebo_p),
            "placebo_coefficients": placebo_coefficients,
            "controls": specification_controls,
            "fixed_effects": (
                ["firm", "industry_by_year"]
                if fixed_effects == "sector_year" else ["firm", "year"]
            ),
            "event_study": [
                {"year": int(year), "coef": float(event_beta[index]),
                 "std_error": float(event_se[index])}
                for index, year in enumerate(event_years)
            ],
            "pretrend_joint_test": _pretrend_test(
                event_beta, event_covariance, event_years, study.reform_year,
            ),
        }
    if dynamic_available:
        status = "estimated"
    elif not allow_dynamic:
        status = "estimated_static_only_by_design"
    else:
        status = "estimated_static_only_dynamic_low_coverage"
    return {
        "status": status, "coverage": gate, "n_obs": int(len(sample)),
        "n_firms": int(sample[S.FIRM].nunique()), "specifications": outputs,
        "sector_column": sector_column,
        "dynamic_bias_correction": (
            "not_estimated_temporal_split_does_not_preserve_pre_and_post"
            if dynamic_available else (
                "not_applicable_static_model_selected_by_design" if not allow_dynamic
                else "not_applicable_dynamic_coverage_below_threshold"
            )
        ),
    }


def run() -> dict:
    paths.RESULTS.mkdir(parents=True, exist_ok=True)
    deflator = pd.read_csv(paths.DATA / "gdp_deflator_2005_2025.csv")
    output = {
        "status": "exploratory_not_in_manuscript",
        "design": "PFU-US-04 amended by author: PTBI omitted",
        "book_leverage": "(total assets - shareholders funds) / total assets, lagged one year",
        "controls": list(CONTROLS),
        "outcomes": list(OUTCOMES),
        "populations": {},
    }
    summary_rows = []
    coverage_rows = []
    for population, suffix in TIERS.items():
        print(f"loading {population}", flush=True)
        frame = _read_population(paths.DATA / "raw", suffix, deflator)
        population_results = {}
        for shock, study in SHOCKS.items():
            population_results[shock] = {}
            for outcome in OUTCOMES:
                print(f"estimating {population} {shock} {outcome}", flush=True)
                result = _estimate(frame, outcome, study)
                population_results[shock][outcome] = result
                gate = result["coverage"]
                for variable, countries in gate["by_variable_country"].items():
                    for country, coverage in countries.items():
                        coverage_rows.append({
                            "population": population, "shock": shock, "outcome": outcome,
                            "variable": variable, "country": country, "coverage": coverage,
                        })
                if result["status"].startswith("estimated"):
                    for specification, values in result["specifications"].items():
                        summary_rows.append({
                            "population": population, "shock": shock, "outcome": outcome,
                            "specification": specification, "coef": values["coef"],
                            "std_error": values["std_error"], "placebo_p": values["placebo_p"],
                            "n_obs": result["n_obs"], "n_firms": result["n_firms"],
                        })
        output["populations"][population] = population_results
        del frame

    pd.DataFrame(summary_rows).to_csv(
        paths.RESULTS / "firm_growth_controls_no_ptbi_summary.csv", index=False,
        float_format="%.6f",
    )
    pd.DataFrame(coverage_rows).to_csv(
        paths.RESULTS / "firm_growth_controls_no_ptbi_coverage.csv", index=False,
        float_format="%.6f",
    )
    target = paths.RESULTS / "firm_growth_controls_no_ptbi.json"
    target.write_text(json.dumps(output, indent=2))
    return output


if __name__ == "__main__":
    result = run()
    print(json.dumps({"status": result["status"],
                      "populations": list(result["populations"])}, indent=2))
