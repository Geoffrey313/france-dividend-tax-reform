# Capital Leaves Faster Than It Returns: The Asymmetry of France's 2018 Flat Tax

The 2018 French single flat tax, firm capital, and household distribution, in a cross-country
difference-in-differences design.

**Authors.** Withheld for double-blind review.

## Abstract

The study compares France with Germany, Italy, Spain, Belgium, and the Netherlands around the 2012--2013
tax package and the 2018 single flat tax. It combines 2,636,675 Orbis observations for 272,906 Large and
Very Large private firms, 38,937 Compustat Global observations for 3,115 listed firms, Eurostat national
accounts, and WID.world distributional series. After 2018, all three private-firm capital-entry margins
decline relatively, with clustered Student statistics between -1.95 and -3.50. The full 2018--2025
window shows no fall in tangible investment; the 2018--2019 pre-Covid window instead shows a 2.85-point
increase. Household dividends rise by 32% in 2018. Comparing the two tax packages gives economically
marked descriptive evidence of payout asymmetry, but the firm-accounting margins do not jointly confirm
the predicted ratchet. The distributional extension is descriptive rather than causal.

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

## Main results

- **H1, directional support.** Private-firm subscribed-capital growth, capital increases, and the
  equity ratio all decline relatively after 2018. Clustered Student statistics are the primary
  inference; the six-country permutation ranks are complementary diagnostics.
- **H2, window-dependent investment response.** The full 2018--2025 window shows no tangible-asset
  decline. The 2018--2019 window instead yields a +2.85-point tangible-asset response (t = 13.06),
  while total assets increase by 1.23 points.
- **Destination.** Household dividends received rise 32% in the reform year. The evidence points to
  higher payouts without a uniform increase in firm financing or investment.
- **H3, differentiated evidence.** The aggregate payout response after 2018 is about 45% of the
  opposite-signed response after 2013. This is economically meaningful descriptive asymmetry, not a
  direct equality test. Firm reserves and equity do not jointly confirm the predicted ratchet.

## Repository layout

```
src/common/    study and simulation parameters; column schema; portable paths
src/data/      synthetic validation used by the estimator self-test
src/engine/    firm- and country-level two-way fixed-effects DiD, synthetic control
src/analysis/  estimation drivers (DiD + synthetic-control and permutation placebos)
src/figures/   figure and table emitters
reproduce.py   deterministic end-to-end entry point
data/          final public inputs and final firm estimation samples; see data/README.md
protocol/      frozen analysis plan
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
broad-Orbis pipeline is retired. Public-series, distributional, and firm-accounting headline numbers
are all checked against both manuscript languages. The WID analysis remains explicitly descriptive;
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
