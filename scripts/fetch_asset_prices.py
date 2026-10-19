#!/usr/bin/env python3
"""Fetch the public asset-price controls used by the distributional robustness checks."""
from __future__ import annotations

import json
import urllib.parse
import urllib.request

import pandas as pd

from src.common import paths
from src.data.wid import COUNTRIES, FIRST_YEAR, LAST_YEAR

EUROSTAT = "https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/prc_hpi_a"
WORLD_BANK = "https://api.worldbank.org/v2/country/{country}/indicator/CM.MKT.INDX.ZG"
ISO3 = {"FR": "FRA", "DE": "DEU", "IT": "ITA", "ES": "ESP", "BE": "BEL", "NL": "NLD"}


def _json(url: str) -> object:
    request = urllib.request.Request(url, headers={"User-Agent": "tax-and-capital-replication/1.0"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return json.load(response)


def _house_prices(country: str, unit: str = "I15_A_AVG") -> dict[int, float | None]:
    query: list[tuple[str, str]] = [
        ("format", "JSON"), ("lang", "EN"), ("freq", "A"), ("purchase", "TOTAL"),
        ("unit", unit), ("geo", country),
    ]
    query.extend(("time", str(year)) for year in range(FIRST_YEAR, LAST_YEAR + 1))
    data = _json(EUROSTAT + "?" + urllib.parse.urlencode(query))
    assert isinstance(data, dict)
    time_index = data["dimension"]["time"]["category"]["index"]
    values = data.get("value", {})
    return {int(year): values.get(str(position)) for year, position in time_index.items()}


def _equity_returns(country: str) -> dict[int, float | None]:
    query = urllib.parse.urlencode({
        "format": "json", "date": f"{FIRST_YEAR}:{LAST_YEAR}", "per_page": 1000,
    })
    data = _json(WORLD_BANK.format(country=ISO3[country]) + "?" + query)
    if not isinstance(data, list) or len(data) < 2:
        raise ValueError(f"unexpected World Bank response for {country}")
    return {int(row["date"]): row["value"] for row in data[1]}


def build() -> pd.DataFrame:
    rows: list[dict] = []
    for country in COUNTRIES:
        hpi = _house_prices(country)
        house_growth = _house_prices(country, unit="RCH_A_AVG")
        equity = _equity_returns(country)
        for year in range(FIRST_YEAR, LAST_YEAR + 1):
            rows.append({
                "country": country,
                "year": year,
                "hpi_2015": hpi.get(year),
                "house_price_growth_pct": house_growth.get(year),
                "equity_return_pct": equity.get(year),
            })
    frame = pd.DataFrame(rows).sort_values(["country", "year"]).reset_index(drop=True)
    # Eurostat does not publish Italy's annual-change observation for 2010.  Preserve that source
    # missing value rather than imputing it; the controlled regression drops this one country-year.
    required = ["hpi_2015", "equity_return_pct"]
    if frame[required].isna().any().any():
        missing = frame[frame[required].isna().any(axis=1)]
        raise ValueError(f"incomplete asset-price panel: {missing.to_dict('records')}")
    return frame


def main() -> None:
    output = paths.DATA / "asset_prices_2010_2022.csv"
    frame = build()
    frame.to_csv(output, index=False)
    missing_growth = int(frame["house_price_growth_pct"].isna().sum())
    print(f"wrote {output} ({len(frame)} rows; {missing_growth} missing house-price change)")


if __name__ == "__main__":
    main()
