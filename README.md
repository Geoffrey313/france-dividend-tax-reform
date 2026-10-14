# After the Flat Tax: Capital Leaves Faster Than It Returns

The 2018 French single flat tax, firm capital, and household distribution, in a cross-country
difference-in-differences design.

**Authors.** Geoffrey Ducournau.

## Abstract

Capital-income taxation is meant to steer saving toward productive financing, yet where the
money freed by a tax cut actually goes is poorly observed. The single flat tax (prélèvement
forfaitaire unique) introduced in France in 2018, a proportional thirty-percent levy on
dividends and capital gains, offers a clean case. We compare French firms to those of five
comparable neighbouring economies, before and after the reform, on a panel of more than sixteen
million firm-years drawn from Orbis, complemented by listed firms from Compustat Global and by
Eurostat national accounts. The reform brought no new capital into firm equity, neither private
nor listed, and triggered neither debt substitution nor higher investment. The money left to
households as dividends, up thirty-two percent in the reform year, and settled in liquid savings
and portfolio equity, not in firm capital or housing. Placing the 2013 tax hike and the 2018 cut
in a single frame uncovers an asymmetry: the hike had reallocated capital toward retained
earnings, and the symmetric cut undoes only a fraction of it, less than half once anticipation is
netted out. Capital leaves more readily than it returns. This fiscal ratchet, coherent across
distribution and capital allocation, shows that a tax cut does not undo the distortions of a past
hike, a result that matters for any assessment of capital taxation.

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

- **H1 supported.** The difference-in-differences coefficient on subscribed-capital growth is
  −0.0034, a precisely estimated null; the placebo permutation ranks France at the smallest gap of
  the six countries.
- **H2 not supported in the substitution direction.** Leverage (−0.0472) and its mirror, the
  equity ratio (+0.0474), are both null; no debt replaces the equity that is not distributed, and
  investment does not rise.
- **Destination.** Household dividends rise 32% in 2018 and settle in deposits and listed-share
  portfolios, not in unlisted firm capital or housing.
- **H3 supported.** The cut undoes about 45% of the distribution effect of the hike, and about a
  third of the capital reallocation (retained earnings and equity return in a 1-to-2.8 and
  1-to-2.7 ratio). The ratchet survives controls for ACE-type equity subsidies and for the cycle,
  and holds when the window stops in 2019, before the Covid state-guaranteed-loan tail.

## Repository layout

```
src/common/    study and simulation parameters; column schema; portable paths
src/data/      readers for the delivered transformed data
src/engine/    panel construction, two-way fixed-effects DiD, synthetic control
src/analysis/  estimation driver (per-outcome DiD + synthetic-control placebos)
src/figures/   figure and table emitters
reproduce.py   deterministic end-to-end entry point
data/          transformed open-data inputs (Eurostat / OECD); see data/README.md
protocol/      frozen analysis plan
```

## Reproduction

```
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python reproduce.py
```

The run is deterministic. It chains the shipped transformed data through the engine, the
analysis, and the figure emitters. The open-data figures (household dividends, the
distribution-vs-earnings decoupling, the distribution leg of the asymmetry) regenerate from the
committed Eurostat/OECD series without network access. The firm-level figures and tables (capital
entry, leverage, cash and financial debt, retained capital, the descriptive statistics) require
the licensed Orbis panel, which is not redistributable; when it is present the run recomputes them
and checks their digest, otherwise it skips them and says so.

## Data policy

The repository ships every non-licensed transformed input as CSV under `data/`: household
distributed income, corporate gross operating surplus, GDP volume, unemployment, house prices,
household financial flows, and the dividend-tax-rate panel (Eurostat and OECD, open, CC BY 4.0).
The firm-level panels (Orbis, Bureau van Dijk; Compustat Global, S&P Global Market Intelligence)
are proprietary and not redistributable; they are **available on request**, and `data/README.md`
documents their schema so a third party can reconstruct the input.

## Contact

G.ducournau.voisin@gmail.com
