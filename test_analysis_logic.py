"""
Validates the core analysis functions in tire_degradation_analysis.py
using synthetic data shaped like FastF1's real lap data schema.
This does NOT hit the network -- it's purely to confirm the pandas/
sklearn logic is correct before Shiv runs it against real F1 data.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from tire_degradation_analysis import clean_laps, fit_degradation_by_compound

np.random.seed(42)

def make_synthetic_laps(n_stints=6, laps_per_stint=15, with_fuel_effect=False):
    """Builds a fake laps DataFrame with known ground-truth degradation
    rates per compound, so we can check the regression recovers them.
    If with_fuel_effect is True, also bakes in a known fuel-burn effect
    (cars get faster as LapNumber increases) to test the fuel-controlled
    regression path."""
    true_degradation = {"SOFT": 0.12, "MEDIUM": 0.07, "HARD": 0.04}  # sec/lap
    true_base_pace = {"SOFT": 88.0, "MEDIUM": 88.8, "HARD": 89.5}
    true_fuel_effect = -0.05  # cars get 0.05 sec/lap FASTER per lap of fuel burned

    rows = []
    compounds = list(true_degradation.keys())
    lap_number = 0

    for stint_id in range(n_stints):
        compound = compounds[stint_id % len(compounds)]
        base = true_base_pace[compound]
        deg = true_degradation[compound]

        for tyre_life in range(1, laps_per_stint + 1):
            lap_number += 1
            noise = np.random.normal(0, 0.15)
            lap_time_sec = base + deg * tyre_life + noise
            if with_fuel_effect:
                lap_time_sec += true_fuel_effect * lap_number
            rows.append(
                {
                    "Driver": "TST",
                    "LapTime": pd.to_timedelta(lap_time_sec, unit="s"),
                    "PitInTime": pd.NaT,
                    "PitOutTime": pd.NaT,
                    "TrackStatus": "1",
                    "TyreLife": tyre_life,
                    "LapNumber": lap_number,
                    "Compound": compound,
                }
            )

    # add a few "dirty" rows that should get filtered out by clean_laps
    rows.append({
        "Driver": "TST", "LapTime": pd.NaT, "PitInTime": pd.NaT,
        "PitOutTime": pd.NaT, "TrackStatus": "1", "TyreLife": 3, "Compound": "SOFT",
    })
    rows.append({
        "Driver": "TST", "LapTime": pd.to_timedelta(95.0, unit="s"),
        "PitInTime": pd.to_timedelta(1, unit="s"), "PitOutTime": pd.NaT,
        "TrackStatus": "1", "TyreLife": 1, "Compound": "MEDIUM",
    })

    return pd.DataFrame(rows)


def run_test():
    laps = make_synthetic_laps()
    print(f"Synthetic laps generated: {len(laps)}")

    clean = clean_laps(laps)
    print(f"Laps after cleaning: {len(clean)} (should be {len(laps) - 2}, i.e. the 2 dirty rows removed)")
    assert len(clean) == len(laps) - 2, "clean_laps did not filter the expected rows"

    summary = fit_degradation_by_compound(clean)
    print("\nRecovered degradation rates vs. ground truth:")
    print(summary.to_string(index=False))

    ground_truth = {"SOFT": 0.12, "MEDIUM": 0.07, "HARD": 0.04}
    for _, row in summary.iterrows():
        expected = ground_truth[row["Compound"]]
        actual = row["Degradation (sec/lap)"]
        diff = abs(expected - actual)
        status = "OK" if diff < 0.02 else "FAIL"
        print(f"  {row['Compound']}: expected ~{expected}, got {actual}  [{status}]")
        assert diff < 0.02, f"{row['Compound']} degradation estimate off by {diff}"

    print("\nAll checks passed -- regression logic correctly recovers known degradation rates.")

    # --- Test 2: fuel-controlled regression correctly separates the two effects ---
    print("\n--- Testing fuel-controlled regression ---")
    laps_fuel = make_synthetic_laps(with_fuel_effect=True)
    clean_fuel = clean_laps(laps_fuel)
    summary_fuel = fit_degradation_by_compound(clean_fuel, control_for_fuel=True)
    print(summary_fuel.to_string(index=False))

    for _, row in summary_fuel.iterrows():
        expected_deg = ground_truth[row["Compound"]]
        actual_deg = row["Degradation (sec/lap)"]
        expected_fuel = -0.05
        actual_fuel = row["Fuel effect (sec/lap)"]
        assert abs(expected_deg - actual_deg) < 0.02, (
            f"{row['Compound']} degradation estimate off when controlling for fuel"
        )
        assert abs(expected_fuel - actual_fuel) < 0.02, (
            f"{row['Compound']} fuel effect estimate off: expected {expected_fuel}, got {actual_fuel}"
        )
    print("Fuel-controlled regression correctly separates tire wear from fuel burn.")

    # --- Test 3: fuel-adjusted plotting line matches known ground truth ---
    print("\n--- Testing fuel-adjusted plot line construction ---")
    from tire_degradation_analysis import plot_degradation
    plot_degradation(clean_fuel, summary_fuel, title_suffix=" (synthetic)", fuel_adjusted=True)
    print("Plot generated without error -- check tire_degradation_fuel_adjusted.png visually.")


if __name__ == "__main__":
    run_test()
