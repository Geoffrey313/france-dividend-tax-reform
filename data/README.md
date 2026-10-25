# Data

This directory holds the inputs of the cross-country firm-panel study of the 2018 French single
flat tax. Two classes of input sit here, and the reproduction is organised around the split.

## Class A: public, redistributable final inputs

Open Eurostat and related public series, read by the class-A stage of `reproduce.py` without any
licence. The repository contains the frozen inputs used by the paper; acquisition code is outside
the replication package.

| file | content |
|------|---------|
| `dividends_cross_country.csv` | household dividends received by country and year, with the 2017=100 index (Eurostat sector accounts, D42, S14) |
| `dividends_2010_2022_broad.csv` | household dividends received (Eurostat D42, S14), broad panel of 27 countries 2010 to 2022, source of the distribution semi-elasticities of the ratchet |
| `corp_surplus_2010_2022.csv` | gross operating surplus of non-financial corporations, by country (Eurostat, B2G, S11) |
| `gdp_volume_2010_2022.csv` | GDP in chain-linked volume, by country (Eurostat, B1GQ) |
| `unemployment_2010_2022.csv` | unemployment rate by country and year (Eurostat) |
| `wid_distribution_2010_2022.csv` | top-1%, top-10%, and bottom-50% shares of net wealth, pretax income, and posttax income for France and the five comparison countries (WID.world country bulk files) |
| `asset_prices_2010_2022.csv` | annual house-price index (Eurostat `prc_hpi_a`) and equity-index return (World Bank `CM.MKT.INDX.ZG`) used only as asset-price robustness controls |
| `gdp_deflator_2005_2025.csv` | national GDP deflator normalized to 2015=100 (Eurostat `nama_10_gdp`), used to express firm sales and balance-sheet levels in real terms |

The WID file retains the source variable code, percentile code, and data-quality grade on every
row. Definitions are in the
[WID codes dictionary](https://wid.world/codes-dictionary/) and grades in the
[WID data-quality documentation](https://wid.world/data-quality/). Under WID's quality convention, distributed-wealth
grades of at least 1 and distributed-income grades of at least 3 are treated as observed-data
estimates. France's net-wealth series falls to grade 0 from 2015 onward; those observations are
income-inequality-based imputations and are never presented as observed wealth measurements.
The asset-price controls are reported as sensitivity specifications because asset prices may
themselves react to policy.
Eurostat does not report Italy's annual house-price change for 2010; that single country-year is
left missing and dropped only from specifications using asset-price controls.

## Class C: final firm estimation samples

The repository starts at the final common estimation samples under
`derived/square_samples_canonical/`. Raw Orbis and WRDS extracts, acquisition programs, harmonized
source panels, and alternative intermediate samples are deliberately excluded. The directory
contains only the family-specific samples consumed by the paper, their SHA-256 manifest, and the
aggregate recent-coverage table used in the appendix. Parquets are stored with Git LFS.

Private firms use one final common sample for each outcome family and reform window, plus the
complete-intersection sensitivities actually reported by the paper. Listed firms use one complete
common sample per reform window. Every retained firm has at least one observation before and after
the reform; required outcomes, the stable industry code, and lagged controls are complete.

Run `python run_firm_growth_harmonized.py` to regenerate the empirical results, figures, tables,
placebo ranks, event studies, pre-trend tests, and country-composition robustness checks directly
from these released samples. `reproduce.py` uses the same path and verifies the sample checksums
before estimation. The synthetic panel remains confined to the initial estimator self-test.

## Synthetic panel

`src/data/simulate.py` writes a synthetic firm panel with a known injected treatment effect, used
by the self-test in `reproduce.py` so the pipeline can be validated without any licensed data.
