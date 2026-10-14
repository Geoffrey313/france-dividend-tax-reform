"""Canonical column names shared by data, engine, and analysis layers."""

# Firm-year panel (long): one row per firm and closing-date year.
FIRM = "firm_id"
COUNTRY = "country"
YEAR = "year"

# Account lines carried from Orbis (industrial global financials), in euros unless noted.
CAPITAL = "paid_in_capital"     # capi: subscribed and paid-in share capital
EQUITY = "shareholders_funds"   # shfd: total shareholders' funds
ASSETS = "total_assets"         # toas
REVENUE = "operating_revenue"   # opre
LT_DEBT = "long_term_debt"      # ltdb: long-term financial debt
LOANS = "loans"                 # loan: current loans
NONCUR_LIAB = "noncurrent_liab" # ncli: non-current liabilities
CUR_LIAB = "current_liab"       # culi: current liabilities
FIXED_ASSETS = "tangible_fixed" # tfas: tangible fixed assets
EMPLOYEES = "employees"         # empl: number of employees (count, not euros)
AGE = "firm_age"                # closing year minus incorporation year

# Treatment and derived outcomes.
TREATED = "treated"             # firm in the treated country
POST = "post"                   # year at or after the reform year
TREAT_POST = "treat_post"       # interaction, the difference-in-differences regressor
LOG_ASSETS = "log_assets"       # size control

# Equity margin (did capital enter firms' own funds).
CAP_INCREASE = "cap_increase"   # indicator: paid-in capital rose from the previous year
DLOG_CAPITAL = "dlog_capital"   # within-firm change in log paid-in capital
EQUITY_RATIO = "equity_ratio"   # shareholders' funds over total assets

# Debt margin (did firms substitute borrowing for equity: the Jalal thesis).
LEVERAGE = "leverage"           # total liabilities over total assets
DLOG_DEBT = "dlog_debt"         # within-firm change in log financial debt (long-term plus loans)
FIN_DEBT_RATIO = "fin_debt_ratio"  # financial debt (long-term plus loans) over total assets

# Real margins (did lower equity financing show up in investment and productivity).
DLOG_FIXED = "dlog_fixed"       # within-firm change in log tangible fixed assets
LOG_LABOR_PROD = "log_labor_prod"  # log operating revenue per employee

# Source of the distributed funds (Phase 6a): were accumulated savings drawn down rather than new
# capital raised. The reserves residual is shareholders' funds minus paid-in capital; it is a
# residual, not pure retained earnings, since it also holds the share premium, revaluation
# reserves, and the current-year result. Netting the current-year result needs a net-income line
# not yet in the panel, so it waits on the cash re-extraction (Phase 6a, trésorerie). The residual
# can be negative when accumulated losses erode own funds below paid-in capital, so it is carried
# as a signed ratio to assets rather than a log change.
RESERVES = "reserves_resid"        # shareholders' funds minus paid-in capital
RESERVES_RATIO = "reserves_ratio"  # reserves residual over total assets

# Cash and the current-year result complete the source test (Phase 6a, trésorerie). The distributed
# funds can leave as cash without touching booked reserves, so the cash-to-assets position is the
# decisive source measure. The current-year result nets the reserves residual to approximate
# accumulated savings alone (residual minus the year's result).
CASH = "cash"                          # cash and cash equivalent
NET_INCOME = "net_income"              # pl: profit or loss for the period
PRETAX_INCOME = "pretax_income"        # plbt: profit or loss before tax (Worldscope ITEM1401 equiv)
CASH_RATIO = "cash_ratio"              # cash over total assets
RESERVES_NET = "reserves_net"          # reserves residual minus the current-year result
RESERVES_NET_RATIO = "reserves_net_ratio"  # netted reserves over total assets

OUTCOMES = (DLOG_CAPITAL, CAP_INCREASE, EQUITY_RATIO, LEVERAGE, DLOG_DEBT,
            DLOG_FIXED, LOG_LABOR_PROD)

# Phase 6a source-of-funds outcomes, estimated on the real panel alongside the capital margin.
SOURCE_OUTCOMES = (RESERVES_RATIO, RESERVES_NET_RATIO, CASH_RATIO)

# Firm-level control set the referee expects, built on lagged total assets where a ratio is scaled.
# Size enters contemporaneously. Linear firm age is collinear with the firm and year effects (year
# minus incorporation year) and stays absorbed, but log(1 + age) is nonlinear in the year and is not
# spanned by the two-way effects, so it is identified and entered.
SIZE = LOG_ASSETS                  # log(1 + total assets); the size control
PTBI = "ptbi"                      # pretax income over lagged total assets
PTBI_VOL = "ptbi_vol"              # rolling standard deviation of PTBI over the volatility window
LEV_CTRL = "leverage_ctrl"         # long-term debt over lagged total assets (control, not the LEVERAGE outcome)
LOG_AGE = "log_firm_age"           # log(1 + firm age)
CONTROLS = (SIZE, PTBI, PTBI_VOL, LEV_CTRL, LOG_AGE)

# The leverage control overlaps mechanically with the debt-margin outcomes, so it is dropped when
# the outcome is one of them; a control equal to the outcome is always dropped.
_CONTROL_CONFLICTS = {LEVERAGE: (LEV_CTRL,), DLOG_DEBT: (LEV_CTRL,)}


def controls_for(outcome: str) -> tuple[str, ...]:
    """Control vector for one outcome, with controls that overlap that outcome removed."""
    drop = set(_CONTROL_CONFLICTS.get(outcome, ())) | {outcome}
    return tuple(c for c in CONTROLS if c not in drop)
