"""Run the 2013/2018 distributional extension without modifying either manuscript."""
from __future__ import annotations

from src.analysis import distribution


def main() -> None:
    results = distribution.run()
    print("Distributional extension: detectable income signal (percentage points)")
    for outcome in results["metadata"]["reported_income_signals"]:
        b13 = results["shocks"]["2013_long"][outcome]["baseline"]
        b18 = results["shocks"]["2018_precovid"][outcome]["baseline"]
        mirror = results["mirror"][outcome]
        print(
            f"  {outcome:26s} 2013 {b13['coef_pp']:+.3f} "
            f"(placebo {b13['placebo_p']:.3f}) | "
            f"2018 {b18['coef_pp']:+.3f} (placebo {b18['placebo_p']:.3f}) | "
            f"expected mirror {mirror['expected_direction']}"
        )
    pool_results = results["pool_robustness"]
    stable = 0
    total = 0
    for outcomes in pool_results.values():
        for pools in outcomes.values():
            full_sign = pools["full"]["coef"] >= 0
            total += 1
            stable += all((estimate["coef"] >= 0) == full_sign for estimate in pools.values())
    changed_with_assets = []
    for outcome in distribution.SIGNAL_OUTCOMES:
        estimates = results["shocks"]["2018_precovid"][outcome]
        if estimates["baseline"]["coef"] * estimates["cycle_asset_controls"]["coef"] < 0:
            changed_with_assets.append(outcome)
    print("Net wealth is context only: its French post-2014 observations are WID grade 0.")
    print(f"Pool robustness: sign retained in {stable}/{total} shock-outcome comparisons.")
    if changed_with_assets:
        print("Sign changes with cycle+asset controls: " + ", ".join(changed_with_assets))
    else:
        print("No signal-outcome sign change with cycle+asset controls.")
    print("Interpretation: descriptive, not a dividend-specific or conventionally significant effect.")
    print("wrote results/distributional_results.json")
    print("wrote results/distributional_estimates.csv")
    print("wrote results/distributional_event_studies.csv")
    print("wrote results/distributional_pool_robustness.csv")


if __name__ == "__main__":
    main()
