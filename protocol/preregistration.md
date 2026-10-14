# Pre-registration: Did the Flat Tax Bring Capital into Firms?

**Authors.** Geoffrey Ducournau.
**Date.** 2026-10-12, control set amended 2026-10-13. **Status.** Frozen before estimation on the firm panel.

This plan replaces the earlier household design. Matched individual tax data (POTE, panel
ISF-IFI) sit inside the CASD enclave and are not reachable without a habilitated French
carrier laboratory, so the household exposure design is set aside. This version observes the
firm, not the household, and draws identification from a cross-country comparison around the
2018 French single flat tax. It is observational, so the default reading is associative; a
causal reading stays conditional on assumptions the data cannot establish alone.

## Question
Did the 2018 single flat tax coincide with more capital entering the equity of French
companies, relative to comparable European economies without an equivalent reform?

## Data
Firm-year accounts from Orbis (Bureau van Dijk), industrial global-format financials, for
France and five controls: Germany, Italy, Spain, Belgium, the Netherlands. Window 2013 to
2022, reference 2017, treatment from 2018. Per firm-year the panel carries paid-in share
capital, total shareholders' funds, total assets, operating revenue, and the closing-date
year; company tables carry the country, the legal form, and the incorporation year. Financial
sector and holding entities are excluded. The household aggregate corroboration uses Eurostat
sector financial accounts (net acquisition of equity and investment-fund shares by households,
transaction F.51), annual, same countries.

## Outcome
Primary: a firm-year increase in paid-in share capital, the subscribed-equity margin closest
to capital entering a firm. It is read both as an indicator of a positive year-on-year change
and as the log change in paid-in capital on the sample of firms that raised it. Secondary: the
year-on-year change in total shareholders' funds net of the retained result, and the yearly
count of newly incorporated firms per country from the incorporation year. The subscription
versus transfer distinction is not observable, so no claim about new productive financing is
made; the outcome is capital recorded as entering firm equity, not its ultimate source.

## Specification
Two-way fixed-effects panel on firm-year observations, firm and year effects, treatment equal
to one for French firms from 2018:

Outcome(i,t) = year effect + firm effect + beta x (France_i x Post_t) + controls + error.

An event-study variant replaces the single Post term with year-by-year interactions relative
to 2017 (omitted), to read the pre-reform path and the dynamics after 2018. The coefficient
beta is the average post-reform differential; the event-study coefficients trace its timing.

Controls (amendment 2026-10-13), the firm-level set a referee expects. Size is the log of one
plus total assets, entered contemporaneously. Pretax profitability (PTBI) is pretax income over
lagged total assets. Its five-year rolling standard deviation (PTBI volatility) measures the
risk of that profitability. The leverage control is long-term debt over lagged total assets.
Log firm age is the log of one plus the closing year minus the incorporation year. The three
ratio controls scale by lagged total assets. Linear firm age is a combination of the firm and
year effects and stays absorbed, so it is not entered; its logarithm is nonlinear in the year,
is not spanned by those effects, and is entered. The leverage control overlaps mechanically with
the debt-margin outcomes, so it is dropped from the leverage and debt-growth regressions, and any
control that coincides with the outcome is dropped from that outcome's regression. Continuous
controls are winsorized at the first and ninety-ninth percentiles with the outcomes.

The one-year lag inside the ratio controls and the five-year volatility window trim the earliest
event-time years. With the full control set the capital event study retains a single pre-reform
coefficient (2016). The parallel-trends diagnostic is therefore read on the specification that
keeps the pre-reform years, size and log firm age only, and the full referee control set is
reported as a robustness specification. PTBI and its volatility require a pretax-income line not
present in the first extract; they wait on the same re-extraction as the cash position.

## Inference and the single treated country
Treatment is assigned at the country level and only France is treated, so clustered standard
errors on six countries overstate certainty. Inference is therefore design-based: a synthetic
control at the country-aggregate level (France against a weighted combination of the controls,
fit on the pre-2017 path) with placebo permutation across the control countries, together with
the firm-level fixed-effects estimates whose uncertainty is read through a wild cluster
bootstrap at the country level and through within-country sector and size heterogeneity. The
firm-level and the synthetic-control readings are reported side by side; agreement between them
is the evidence, a single clustered p-value is not.

## Pre-trend and the 2018 reform bundle
The event-study coefficients for 2013 to 2016 test the common pre-trend; a visible pre-trend
voids the causal reading and the result is reported as associative. France combined several
measures in 2018: the flat tax, the wealth-tax narrowing to real estate, the higher general
social contribution, and the lower corporate tax. The estimate captures the bundle, not the
flat tax alone, and this is stated wherever the number appears. No control built from
post-2018 profits, rates, or payouts is introduced as if predetermined.

## Data rules fixed ex ante
Coverage differs by Orbis vintage, so estimation uses a balanced sub-panel of firms present
across the window, and raw country aggregates are never used as the outcome. A missing account
line is distinguished from a zero. Mergers, restructurings, and identifier changes are handled
before estimation. Winsorization of the continuous outcome is at the first and ninety-ninth
percentiles, declared with the result.

## Decision rule
If the balanced panel or the pre-trend test fails, the firm design stops and the project falls
back to a descriptive account of concentrated individual ownership from the ownership snapshot.
If entry of capital is measurable but the pre-trend or the synthetic-control placebos are
unconvincing, the result is an associative account of capital flows around the reform, without
causal vocabulary. A claim about productive financing is retained only if the subscription
channel is established, which these data do not permit. An extension past 2022 is a separate
project.
