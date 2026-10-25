# Capital Leaves Faster Than It Returns: The Asymmetry of France's 2018 Flat Tax

The 2018 French single flat tax, firm capital, and household distribution, in a cross-country
difference-in-differences design.

**Authors.** Geoffrey Ducournau and Jalal Qanas. October 2026 manuscript.

## Abstract

Does a dividend-tax cut provide firms with new financing and change their investment? The paper studies
France's 2018 `prélèvement forfaitaire unique` (PFU) by linking French firms' accounts to household
dividend receipts and the income distribution. Household dividend receipts rise by 32% in the reform
year. This payout is not matched by observable new financing: the estimated decline in private firms'
subscribed capital is about 26% of its average annual growth, and the decline in listed firms' external
financing is about 83% of its mean. Private firms' tangible assets rise in the short run, but rejected
pre-trends prevent attributing that increase to the PFU. The relative increase in the post-tax income
share of the top 1% is about 8% of its 2017 French share. Placebo ranks and pre-trends require a
descriptive reading. Comparison with the 2013 tax increase reveals asymmetric responses in payouts and
retained resources without establishing causality.

## Research questions

Main: does a cut in dividend taxation bring new capital into firms, or does it merely push
dividends out to households? Secondary: is the response of distribution and capital allocation to
taxation symmetric, so that a cut undoes a comparable hike, or is there a ratchet by which capital
leaves more readily than it returns?

## Hypotheses

- **H1.** A cut in dividend taxation brings no new capital into firm equity (expected sign: zero).
- **H2.** It triggers neither debt substitution nor lower investment (expected signs: zero).
- **H3.** The response to taxation is asymmetric: the 2018 cut undoes only a fraction of the
  reallocation caused by the 2013 hike, so capital leaves more readily than it returns.

## Empirical design

France is compared with Germany, Italy, Spain, Belgium, and the Netherlands. Firm-level regressions
absorb firm and stable two-digit-industry-by-year fixed effects and include lagged size, firm age, and
leverage whenever leverage is not mechanically redundant with the dependent variable. Required
outcomes and controls are jointly complete within each final common sample; no missing value is filled.

The 2013 comparison uses 2008--2016 with 2012 as the reference year. For the 2018 reform, the main
window is 2013--2019 with 2017 as the reference year, so the principal estimates precede the pandemic.
The 2013--2025 window reports horizons `t+2` through `t+7` as a pandemic-inclusive sensitivity. A common
Covid indicator is absorbed by industry-by-year effects, while France-by-Covid interactions overlap
the post-treatment coefficients. The extended window therefore cannot distinguish a delayed PFU
response from France-specific pandemic shocks. Results for 2025 are provisional because recent Orbis
coverage remains incomplete.

## Main results

- **H1, directional support.** Private subscribed-capital growth falls by 0.926 percentage point,
  about 26% of its estimation-sample mean. Listed external financing falls by 5.503 points, about
  83% of its mean. Rejected pre-trends prevent a clean causal interpretation.
- **H2, window-dependent investment response.** The 2018--2019 window yields a +2.852-point
  tangible-asset response and a +1.232-point total-asset response among private firms. These movements
  are not reproduced uniformly by listed-firm measures or by the pandemic-inclusive window.
- **Destination.** Household dividends received rise 32% in the reform year. The evidence points to
  higher payouts without a uniform increase in firm financing or investment.
- **H3, differentiated evidence.** The aggregate payout response after 2018 is about 45% of the
  opposite-signed response after 2013. This is economically meaningful descriptive asymmetry, not a
  direct equality test. Firm reserves and equity do not jointly confirm the predicted ratchet.
- **Distribution.** The post-2018 increase in the top 1%'s post-tax income share is 0.554 percentage
  point, about 8.3% of its 2017 French share. This extension remains descriptive.

## Repository layout

```
src/common/    study and simulation parameters; column schema; portable paths
src/data/      synthetic validation used by the estimator self-test
src/engine/    firm- and country-level two-way fixed-effects DiD, synthetic control
src/analysis/  estimation drivers (DiD + synthetic-control and permutation placebos)
src/figures/   figure and table emitters
reproduce.py   deterministic end-to-end entry point
data/          final public inputs and final firm estimation samples; see data/README.md
```

## Reproduction

```
python -m venv .venv-repro
. .venv-repro/bin/activate
pip install -r requirements.txt
python reproduce.py
```

The distributional extension can also be run on its own:

```
python run_distribution.py
```

It compares France with Germany, Italy, Spain, Belgium, and the Netherlands around the 2012--2013 and
2018 tax packages. The first package starts in 2012 and uses 2011 as its reference because the 2012
exceptional wealth levy already enters posttax income. The 2018 package uses 2017 as its reference
and 2018--2019 as its pre-Covid post-period. It writes the full
estimates, in-space placebo ranks, synthetic-control placebos, event paths, comparison-pool
sensitivities, and WID data-quality diagnostics under `results/`. The estimand is the
reduced-form incidence of each package: it does not separately identify the PFU and the
ISF-to-IFI reform, and the available harmonized WID outcomes do not directly measure dividend
income. Pretax and posttax income shares are consequently the closest detectable signal, not a
direct capital-income measure. Net wealth is context only: France's WID series is grade 0 after
2014, so its 2018 estimate is an imputation-based diagnostic rather than an observed-data finding.
Robustness checks drop Italy and Belgium successively, change the comparison pool, and add public
house-price and equity-return controls. These estimates remain descriptive.

The integrated firm-accounting analysis can be rebuilt separately from the released final
estimation samples with `python run_firm_growth_harmonized.py`. It
estimates Orbis Large/Very Large firms and listed Compustat Global firms separately
around 2013 and 2018; Small is excluded and no asset-size filter recreates an Orbis library. The
controlled specification uses firm and stable two-digit-industry-by-year fixed effects, lagged
size, lagged firm age, lagged leverage when it is not mechanically redundant with the dependent
variable. The manuscript specification is static. Outcomes are estimated on canonical common
samples: listwise deletion is applied jointly to each declared family and its lagged controls,
controls are never filled, and every retained firm has at least one observation before and after
the reform. Private firms use one common sample per outcome family plus a complete-sample
sensitivity; listed firms use one common sample. Listed external equity financing is
measured by `net_ext_equity_at = (delta CEQ - delta RE) / lagged assets`, with the accounting-residual
caveat retained in the metadata. A country-specific 50% coverage gate is applied before every
outcome. Estimates, event paths, joint pretrend diagnostics, spatial placebo ranks, family tests,
coverage diagnostics, common-sample manifests, and the source-separated comparison are written
under `results/`. Country-composition robustness re-estimates the headline outcomes after dropping
each control country in turn and with inverse country sample-size weights so every country receives
the same total weight; run it with `python run_firm_growth_square_robustness.py`.

The run is deterministic and has three live stages: a synthetic recovery self-test; public Eurostat
and WID series; and the final Orbis Large/Very Large plus Compustat estimation samples. The historical
broad-Orbis pipeline is retired. When the optional local manuscript sources are present, public-series,
distributional, and firm-accounting headline numbers are checked against both manuscript languages.
The WID analysis remains explicitly descriptive;
the firm hypotheses use the common controlled specification with firm and industry-by-year fixed
effects. The exact input and output hashes and dependency versions are written to
`results/reproduction_manifest.json`. The replication package starts from the checksummed final
firm estimation samples stored through Git LFS; raw licensed extracts and acquisition code are not
part of the repository.

## Data policy

The repository ships the public final inputs consumed by the reproduction as CSV under
`data/`: household distributed income (the six-country index and a broad 27-country panel),
corporate gross operating surplus, GDP volume, unemployment (Eurostat, open, CC BY 4.0), and
income and net-wealth shares (WID.world).
For the firm analysis, only the final common estimation samples and their checksum manifest are
included. Raw Orbis and Compustat extracts, acquisition programs, harmonized source panels, and
unused intermediate variants remain outside the repository. Access and redistribution of the final
samples must follow the underlying data licences.

## Contact

Corresponding author contact available on request.
