"""Acquire and normalize the public WID distributional series used by the extension."""
from __future__ import annotations

import csv
import tempfile
import urllib.request
from pathlib import Path

import pandas as pd

COUNTRIES = ("FR", "DE", "IT", "ES", "BE", "NL")
VARIABLES = {
    "shwealj992": "net_wealth",
    "sptincj992": "pretax_income",
    "sdiincj992": "posttax_income",
}
PERCENTILES = {
    "p99p100": "top_1",
    "p90p100": "top_10",
    "p0p50": "bottom_50",
}
FIRST_YEAR = 2010
LAST_YEAR = 2022
URL = "https://wid.world/bulk_download/WID_data_{country}.csv"


def extract_country(path: Path) -> pd.DataFrame:
    """Read one WID country bulk file and retain only the pre-specified public series."""
    rows: list[dict] = []
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        for row in reader:
            variable = row["variable"]
            percentile = row["percentile"]
            year = int(row["year"])
            if (variable not in VARIABLES or percentile not in PERCENTILES
                    or not FIRST_YEAR <= year <= LAST_YEAR):
                continue
            rows.append({
                "country": row["country"],
                "year": year,
                "concept": VARIABLES[variable],
                "group": PERCENTILES[percentile],
                "share": float(row["value"]),
                "wid_variable": variable,
                "percentile": percentile,
                "data_quality": int(row["data_quality"]) if row["data_quality"] else None,
            })
    return pd.DataFrame(rows)


def build_from_directory(source_dir: Path) -> pd.DataFrame:
    """Build the six-country transformed panel from already downloaded country files."""
    frames = [extract_country(source_dir / f"WID_data_{country}.csv")
              for country in COUNTRIES]
    out = pd.concat(frames, ignore_index=True)
    expected = len(COUNTRIES) * (LAST_YEAR - FIRST_YEAR + 1) * len(VARIABLES) * len(PERCENTILES)
    if len(out) != expected:
        counts = out.groupby(["country", "concept", "group"]).size().to_dict()
        raise ValueError(f"incomplete WID extraction: got {len(out)} of {expected} rows; {counts}")
    if out.duplicated(["country", "year", "concept", "group"]).any():
        raise ValueError("duplicate WID country-year-concept-group rows")
    return out.sort_values(["concept", "group", "country", "year"]).reset_index(drop=True)


def download(output: Path) -> pd.DataFrame:
    """Download official WID country files, retain the transformed panel, and write it to CSV."""
    with tempfile.TemporaryDirectory(prefix="wid-distribution-") as tmp:
        source_dir = Path(tmp)
        for country in COUNTRIES:
            urllib.request.urlretrieve(URL.format(country=country),
                                       source_dir / f"WID_data_{country}.csv")
        frame = build_from_directory(source_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    return frame
