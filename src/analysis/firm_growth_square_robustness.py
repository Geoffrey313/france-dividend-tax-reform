"""Country-composition robustness for the canonical common samples.

The preferred static specification is re-estimated (i) after dropping each control country in
turn and (ii) with inverse country sample-size weights, normalized so every country receives the
same total weight. These checks never change the common-sample construction and are descriptive:
there are only five or six country clusters.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import firm_growth_controls as engine
from src.analysis import firm_growth_harmonized as analysis
from src.analysis import firm_square_samples as square
from src.common import paths, schema as S


SHOCKS = ("2013", "2018_precovid")
HEADLINE_OUTCOMES = {
    "private_large_vl": (
        "dlog_capital", "cap_increase", "equity_growth_real", "equity_ratio",
        "asset_growth_real", "tangible_growth_real", "intangible_growth_real",
        "reserves_ratio",
    ),
    "public_listed": (
        "net_ext_equity_at", "equity_growth_real", "equity_ratio",
        "asset_growth_real", "tangible_growth_real", "intangible_growth_real",
        "capex_at", "sales_growth_real", "operating_roa", "retained_earnings_at",
    ),
}


def _equal_country_weights(country: pd.Series) -> np.ndarray:
    """Give every country equal total weight and normalize the observation mean to one."""
    counts = country.value_counts()
    raw = country.map((1.0 / counts).to_dict()).to_numpy(dtype=float)
    weights = raw / raw.mean()
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise AssertionError("invalid equal-country weights")
    return weights


def _estimate_preferred(
    sample: pd.DataFrame,
    outcomes: tuple[str, ...],
    study,
    sector_column: str,
    countries: tuple[str, ...],
    *,
    equal_country_weight: bool,
) -> dict[str, dict]:
    """Estimate the preferred specification once for a declared robustness sample."""
    work = sample[sample[S.COUNTRY].isin(countries)].copy()
    if work.empty or study.treated not in set(work[S.COUNTRY]):
        raise ValueError("robustness sample must contain the treated country")
    work["sector_year"] = (
        work[sector_column].astype(str) + "|" + work[S.YEAR].astype(str)
    )
    for control in analysis.CONTROLS:
        work[control] = engine._clip_country_year(work, control)
    treated_name = f"post_{study.treated}"
    work[treated_name] = (
        work[S.COUNTRY].eq(study.treated) & work[S.YEAR].ge(study.reform_year)
    ).astype(float)
    columns = [*outcomes, treated_name, *analysis.CONTROLS]
    positions = {name: index for index, name in enumerate(columns)}
    weights = (
        _equal_country_weights(work[S.COUNTRY]) if equal_country_weight else None
    )
    residual = engine._residualize(
        work, columns, sector_year=True, weights=weights,
    )
    if weights is not None:
        regression = residual * np.sqrt(weights)[:, None]
    else:
        regression = residual
    cluster = work[S.COUNTRY].to_numpy()
    country_rows = work[S.COUNTRY].value_counts().sort_index().to_dict()
    country_weight_totals = None
    if weights is not None:
        country_weight_totals = (
            pd.Series(weights, index=work.index)
            .groupby(work[S.COUNTRY], observed=True).sum().sort_index().to_dict()
        )
    output = {}
    for outcome in outcomes:
        controls = list(analysis._controls_for_outcome(outcome))
        x = regression[:, [positions[treated_name], *(positions[c] for c in controls)]]
        y = regression[:, positions[outcome]]
        beta, se = engine._ols_cluster_pinv(x, y, cluster)
        output[outcome] = {
            "coef": float(beta[0]),
            "std_error": float(se[0]),
            "conventional_p": analysis._normal_p(float(beta[0]), float(se[0])),
            "n_obs": int(len(work)),
            "n_firms": int(work[S.FIRM].nunique()),
            "countries": list(countries),
            "country_rows": {str(key): int(value) for key, value in country_rows.items()},
            "country_weight_totals": country_weight_totals,
            "controls": controls,
            "fixed_effects": ["firm", "industry_by_year"],
        }
    return output


def _sample_path(population: str, shock: str, family: str) -> Path:
    return square.OUTPUT_DIR / f"{population}_{shock}_{family}.parquet"


def _family_for(output: dict, population: str, shock: str, outcome: str) -> str:
    return output["populations"][population][shock][outcome]["square_sample"]["family"]


def run(output: dict | None = None) -> dict:
    """Run both outstanding country robustness checks and write JSON/CSV outputs."""
    result_path = paths.RESULTS / "firm_growth_harmonized_results.json"
    if output is None:
        output = json.loads(result_path.read_text())
    rows: list[dict] = []
    detailed: dict[str, dict] = {}
    for population, outcomes in HEADLINE_OUTCOMES.items():
        detailed[population] = {}
        for shock in SHOCKS:
            study = engine.SHOCKS[shock]
            sector = "sector_stable_2013" if shock == "2013" else "sector_stable_2018"
            detailed[population][shock] = {}
            by_family: dict[str, list[str]] = {}
            for outcome in outcomes:
                family = _family_for(output, population, shock, outcome)
                by_family.setdefault(family, []).append(outcome)
            for family, family_outcomes_list in by_family.items():
                family_outcomes = tuple(family_outcomes_list)
                target = _sample_path(population, shock, family)
                if not target.exists():
                    raise FileNotFoundError(f"missing canonical common sample: {target}")
                print(
                    f"country robustness {population} {shock} {family}: loading common sample",
                    flush=True,
                )
                sample = pd.read_parquet(target)
                print("  reproduce unweighted reference", flush=True)
                reproduced = _estimate_preferred(
                    sample, family_outcomes, study, sector, study.countries,
                    equal_country_weight=False,
                )
                for outcome in family_outcomes:
                    expected = output["populations"][population][shock][outcome][
                        "specifications"
                    ][analysis.MANUSCRIPT_SPECIFICATION]["coef"]
                    if not np.isclose(
                        reproduced[outcome]["coef"], expected, rtol=1e-8, atol=1e-10,
                    ):
                        raise AssertionError(
                            f"robustness estimator does not reproduce reference: "
                            f"{population}/{shock}/{outcome}: "
                            f"{reproduced[outcome]['coef']} != {expected}"
                        )
                print("  equal total country weight", flush=True)
                variants = {
                    "equal_country_weight": _estimate_preferred(
                        sample, family_outcomes, study, sector, study.countries,
                        equal_country_weight=True,
                    )
                }
                for omitted in study.controls:
                    print(f"  drop control {omitted}", flush=True)
                    countries = tuple(country for country in study.countries if country != omitted)
                    variants[f"drop_{omitted}"] = _estimate_preferred(
                        sample, family_outcomes, study, sector, countries,
                        equal_country_weight=False,
                    )
                for outcome in family_outcomes:
                    baseline = output["populations"][population][shock][outcome][
                        "specifications"
                    ][analysis.MANUSCRIPT_SPECIFICATION]
                    detailed[population][shock][outcome] = {
                        "family": family,
                        "baseline_coef": baseline["coef"],
                        "variants": {name: values[outcome] for name, values in variants.items()},
                    }
                    for name, values in variants.items():
                        value = values[outcome]
                        omitted = name.removeprefix("drop_") if name.startswith("drop_") else ""
                        rows.append({
                            "population": population,
                            "shock": shock,
                            "family": family,
                            "outcome": outcome,
                            "robustness": name,
                            "omitted_control_country": omitted,
                            "weighting": (
                                "inverse_country_observations_equal_total_country_weight"
                                if name == "equal_country_weight" else "observation_weighted"
                            ),
                            "baseline_coef": baseline["coef"],
                            "coef": value["coef"],
                            "delta_from_baseline": value["coef"] - baseline["coef"],
                            "std_error": value["std_error"],
                            "conventional_p": value["conventional_p"],
                            "n_obs": value["n_obs"],
                            "n_firms": value["n_firms"],
                            "n_country_clusters": len(value["countries"]),
                            "same_sign_as_baseline": (
                                math.copysign(1.0, value["coef"])
                                == math.copysign(1.0, baseline["coef"])
                            ),
                        })
                del sample
    table = pd.DataFrame(rows).sort_values(
        ["population", "shock", "outcome", "robustness"]
    )
    summary_rows = []
    for keys, group in table.groupby(["population", "shock", "outcome"], sort=True):
        population, shock, outcome = keys
        leave_one_out = group[group["robustness"].str.startswith("drop_")]
        equal = group[group["robustness"].eq("equal_country_weight")].iloc[0]
        summary_rows.append({
            "population": population,
            "shock": shock,
            "outcome": outcome,
            "baseline_coef": equal["baseline_coef"],
            "equal_country_coef": equal["coef"],
            "equal_country_delta": equal["delta_from_baseline"],
            "leave_one_out_min": leave_one_out["coef"].min(),
            "leave_one_out_max": leave_one_out["coef"].max(),
            "all_variants_same_sign": bool(group["same_sign_as_baseline"].all()),
            "largest_abs_delta": group["delta_from_baseline"].abs().max(),
        })
    summary = pd.DataFrame(summary_rows)
    payload = {
        "status": "estimated_descriptive_five_or_six_country_clusters",
        "design": {
            "sample": "same canonical common sample as the preferred estimate",
            "specification": analysis.MANUSCRIPT_SPECIFICATION,
            "reference_check": "unweighted estimator reproduces every preferred coefficient",
            "leave_one_out": "France plus four controls; each control omitted once",
            "equal_country_weight": (
                "inverse country observation count, normalized to mean one; "
                "each country has identical total estimation weight"
            ),
            "inference": "country-clustered conventional standard errors are descriptive",
        },
        "results": detailed,
        "summary": summary.to_dict(orient="records"),
    }
    paths.RESULTS.mkdir(parents=True, exist_ok=True)
    table.to_csv(
        paths.RESULTS / "firm_growth_harmonized_country_robustness.csv",
        index=False, float_format="%.6f",
    )
    summary.to_csv(
        paths.RESULTS / "firm_growth_harmonized_country_robustness_summary.csv",
        index=False, float_format="%.6f",
    )
    (paths.RESULTS / "firm_growth_harmonized_country_robustness.json").write_text(
        json.dumps(payload, indent=2) + "\n"
    )
    output["country_robustness"] = {
        "status": payload["status"],
        "design": payload["design"],
        "summary": payload["summary"],
        "files": [
            "results/firm_growth_harmonized_country_robustness.csv",
            "results/firm_growth_harmonized_country_robustness_summary.csv",
            "results/firm_growth_harmonized_country_robustness.json",
        ],
    }
    result_path.write_text(json.dumps(output, indent=2))
    return payload


if __name__ == "__main__":
    run()
