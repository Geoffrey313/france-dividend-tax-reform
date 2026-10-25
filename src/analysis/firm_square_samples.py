"""Canonical common samples for the harmonized firm analysis.

A common sample contains exactly the same firm-year rows for every outcome in a declared
family.  Controls are never imputed.  Firms must be observed at least once before and once after
the reform inside the bounded estimation window.  The module writes auditable Parquets and a
manifest, but estimation consumes the returned frames directly so no stale artifact can enter a
result.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from src.common import paths, schema as S
from src.common.config import Study


CONTROLS = ("lag_book_leverage", "lag_log_assets_real", "lag_log_firm_age")
FAMILIES: dict[str, dict[str, tuple[str, ...]]] = {
    "private_large_vl": {
        "capital": ("dlog_capital", "cap_increase", "equity_growth_real", "equity_ratio"),
        "investment": (
            "asset_growth_real", "tangible_growth_real", "intangible_growth_real",
        ),
        "resources": ("reserves_ratio", "cash_at", "fin_debt_ratio", "dlog_debt"),
    },
    "public_listed": {
        "capital": ("net_ext_equity_at", "equity_growth_real", "equity_ratio"),
        "investment": (
            "asset_growth_real", "tangible_growth_real", "intangible_growth_real",
            "capex_at", "sales_growth_real", "operating_roa",
        ),
        "resources": ("retained_earnings_at", "cash_at", "fin_debt_ratio", "dlog_debt"),
    },
}
OUTPUT_DIR = paths.DATA / "derived" / "square_samples_canonical"


def outcomes(population: str) -> tuple[str, ...]:
    """Return the deduplicated outcome union in declaration order."""
    return tuple(dict.fromkeys(
        outcome
        for family in FAMILIES[population].values()
        for outcome in family
    ))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_square_sample(
    frame: pd.DataFrame,
    *,
    population: str,
    shock: str,
    study: Study,
    family: str,
    dependent_variables: tuple[str, ...],
    controls: tuple[str, ...] = CONTROLS,
) -> tuple[pd.DataFrame, dict]:
    """Apply the author-approved listwise and pre/post rules once.

    No value is filled.  The returned frame keeps every original column because downstream
    estimation and figure code use the harmonized panel as their single source of truth.
    """
    sector = "sector_stable_2013" if shock == "2013" else "sector_stable_2018"
    required = [*dependent_variables, *controls, sector]
    missing = [column for column in required if column not in frame]
    if missing:
        raise KeyError(f"missing common-sample columns: {missing}")

    window = frame[
        frame[S.COUNTRY].isin(study.countries)
        & frame[S.YEAR].between(study.first_year, study.last_year)
    ].copy()
    initial_rows = len(window)
    initial_firms = int(window[S.FIRM].nunique())
    complete = window.dropna(subset=required).copy()
    after_listwise_rows = len(complete)
    after_listwise_firms = int(complete[S.FIRM].nunique())

    timing = complete.groupby(S.FIRM, observed=True)[S.YEAR].agg(
        has_pre=lambda values: bool((values < study.reform_year).any()),
        has_post=lambda values: bool((values >= study.reform_year).any()),
    )
    eligible = timing.index[timing["has_pre"] & timing["has_post"]]
    square = complete[complete[S.FIRM].isin(eligible)].copy()
    square = square.sort_values([S.FIRM, S.YEAR]).reset_index(drop=True)
    if square.empty:
        raise ValueError(f"empty common sample: {population}/{shock}/{family}")
    if square.duplicated([S.FIRM, S.YEAR]).any():
        raise AssertionError("common sample is not unique by firm-year")
    if square[required].isna().any().any():
        raise AssertionError("common sample contains a missing required value")
    check = square.groupby(S.FIRM, observed=True)[S.YEAR].agg(
        pre=lambda values: bool((values < study.reform_year).any()),
        post=lambda values: bool((values >= study.reform_year).any()),
    )
    if not (check["pre"] & check["post"]).all():
        raise AssertionError("common sample contains a firm without pre/post observations")

    # Do not duplicate the full licensed accounting panel in every family artifact.  These are
    # analytical samples, so retain only the exact columns consumed by the estimator.  This keeps
    # the three private families plus the complete sensitivity tractable on million-row windows.
    analytical_columns = list(dict.fromkeys([S.FIRM, S.COUNTRY, S.YEAR, *required]))
    square = square[analytical_columns].copy()

    country = square.groupby(S.COUNTRY, observed=True).agg(
        rows=(S.FIRM, "size"), firms=(S.FIRM, "nunique"),
    )
    country = country.reindex(study.countries, fill_value=0)
    manifest = {
        "population": population,
        "shock": shock,
        "family": family,
        "window": [study.first_year, study.last_year],
        "reference_year": study.reference_year,
        "reform_year": study.reform_year,
        "dependent_variables": list(dependent_variables),
        "controls": list(controls),
        "sector": sector,
        "missing_policy": "listwise_no_fillna",
        "firm_rule": "at_least_one_pre_and_one_post_observation",
        "initial_rows": initial_rows,
        "initial_firms": initial_firms,
        "after_listwise_rows": after_listwise_rows,
        "after_listwise_firms": after_listwise_firms,
        "final_rows": int(len(square)),
        "final_firms": int(square[S.FIRM].nunique()),
        "rows_by_country": {str(k): int(v) for k, v in country["rows"].items()},
        "firms_by_country": {str(k): int(v) for k, v in country["firms"].items()},
    }
    return square, manifest


def materialize(
    frame: pd.DataFrame,
    *,
    population: str,
    shock: str,
    study: Study,
    complete_population: bool = False,
) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    """Build and write all canonical samples for one population and shock."""
    declared = FAMILIES[population]
    specifications: dict[str, tuple[str, ...]] = dict(declared)
    if complete_population:
        specifications = {"complete": outcomes(population)}
    elif population == "private_large_vl":
        specifications["complete"] = outcomes(population)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    samples: dict[str, pd.DataFrame] = {}
    manifests: list[dict] = []
    for family, variables in specifications.items():
        sample, manifest = build_square_sample(
            frame,
            population=population,
            shock=shock,
            study=study,
            family=family,
            dependent_variables=variables,
        )
        target = OUTPUT_DIR / f"{population}_{shock}_{family}.parquet"
        sample.to_parquet(target, index=False)
        manifest["path"] = str(target.relative_to(paths.ROOT))
        manifest["sha256"] = _sha256(target)
        samples[family] = sample
        manifests.append(manifest)
    return samples, manifests


def materialize_one(
    frame: pd.DataFrame,
    *,
    population: str,
    shock: str,
    study: Study,
    family: str,
    dependent_variables: tuple[str, ...],
) -> tuple[pd.DataFrame, dict]:
    """Build and write one sample so callers can release it before constructing the next."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sample, manifest = build_square_sample(
        frame,
        population=population,
        shock=shock,
        study=study,
        family=family,
        dependent_variables=dependent_variables,
    )
    target = OUTPUT_DIR / f"{population}_{shock}_{family}.parquet"
    sample.to_parquet(target, index=False)
    manifest["path"] = str(target.relative_to(paths.ROOT))
    manifest["sha256"] = _sha256(target)
    return sample, manifest


def write_manifest(rows: list[dict]) -> Path:
    """Write deterministic JSON and flat CSV audit manifests."""
    paths.RESULTS.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ordered = sorted(rows, key=lambda row: (row["population"], row["shock"], row["family"]))
    payload = json.dumps({"schema_version": 1, "samples": ordered}, indent=2) + "\n"
    json_target = paths.RESULTS / "firm_square_sample_manifest.json"
    json_target.write_text(payload)
    # This copy travels with the released estimation samples; unlike results/, it is an input
    # audit record for a replication package built without the licensed source extracts.
    (OUTPUT_DIR / "manifest.json").write_text(payload)
    flat = []
    for row in ordered:
        item = {key: value for key, value in row.items()
                if key not in {"rows_by_country", "firms_by_country"}}
        item["dependent_variables"] = "|".join(row["dependent_variables"])
        item["controls"] = "|".join(row["controls"])
        for country, value in row["rows_by_country"].items():
            item[f"rows_{country}"] = value
        for country, value in row["firms_by_country"].items():
            item[f"firms_{country}"] = value
        flat.append(item)
    pd.DataFrame(flat).to_csv(
        paths.RESULTS / "firm_square_sample_manifest.csv", index=False,
    )
    return json_target
