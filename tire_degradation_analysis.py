"""
Tire Degradation & Lap-Time Regression
----------------------------------------
Uses FastF1 (https://github.com/theOehrly/Fast-F1) to pull official F1 lap
timing data for a chosen race, then fits a simple linear regression of
LapTime vs. TyreLife (laps on that set of tires) for each tire compound.

The regression slope tells you, in seconds per lap, how much a compound
degrades as the stint goes on -- exactly the kind of "regression" /
"time-series" analysis referenced in motorsports data/strategy roles.

SETUP (run once, on your own machine):
    pip install fastf1 pandas numpy matplotlib scikit-learn

USAGE:
    python tire_degradation_analysis.py

Edit YEAR, GRAND_PRIX, and SESSION below to analyze a different race.
FastF1 caches downloaded data locally in ./f1_cache so repeat runs are fast.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# ---------------------------------------------------------------------
# CONFIG -- change these to analyze a different race
# ---------------------------------------------------------------------
YEAR = 2026
GRAND_PRIX = "Hungary"   # e.g. "Monaco", "Silverstone", "Monza"
SESSION = "R"            # R = Race, Q = Qualifying, FP1/FP2/FP3 = Practice
DRIVER_FILTER = None     # e.g. "VER" to isolate one driver, or None for all


def load_session_laps(year, grand_prix, session_code):
    """Pull lap data for a session using FastF1. Requires internet access
    to F1's timing API -- run this part on your own machine, not in a
    network-restricted environment."""
    import fastf1

    os.makedirs("f1_cache", exist_ok=True)
    fastf1.Cache.enable_cache("f1_cache")

    session = fastf1.get_session(year, grand_prix, session_code)
    session.load()

    if not hasattr(session, "_laps") or session.laps is None or len(session.laps) == 0:
        raise RuntimeError(
            f"No lap data came back for {year} {grand_prix} {session_code}. "
            "This usually means FastF1's data source doesn't have this session "
            "archived (common for very recent races) or is temporarily down. "
            "Try a different, older race (e.g. a full season back) to confirm "
            "your setup works, then retry this one later."
        )

    return session.laps


def clean_laps(laps: pd.DataFrame) -> pd.DataFrame:
    """Remove laps that would distort degradation analysis: in/out laps
    (pit stops), laps with no recorded time, and laps under safety car /
    VSC conditions where pace isn't representative of true tire wear."""
    df = laps.copy()

    # Drop laps with no lap time recorded (e.g. red flag, DNF laps)
    df = df[df["LapTime"].notna()]

    # Drop in-laps and out-laps (pit stop laps distort pace)
    df = df[df["PitInTime"].isna() & df["PitOutTime"].isna()]

    # Drop laps flagged as not "personal best pace" track status issues,
    # if TrackStatus is available and not green-flag racing (status '1')
    if "TrackStatus" in df.columns:
        df = df[df["TrackStatus"].astype(str).str.fullmatch(r"1+") | df["TrackStatus"].isna()]

    # Convert LapTime (a pandas Timedelta) into plain seconds for regression
    df = df.dropna(subset=["LapTime"]).copy()
    df["LapTimeSeconds"] = df["LapTime"].dt.total_seconds()

    # Drop rows missing the two fields the regression needs
    df = df.dropna(subset=["LapTimeSeconds", "TyreLife", "Compound"])

    return df


def fit_degradation_by_compound(df: pd.DataFrame, control_for_fuel: bool = False) -> pd.DataFrame:
    """Fit LapTimeSeconds ~ TyreLife separately for each tire compound.

    If control_for_fuel is True, also includes LapNumber as a second
    regressor (a simple proxy for fuel load, since cars burn fuel and get
    lighter/faster as the race goes on). Without this, tire degradation
    estimates can come out negative or unreliable, because the fuel-burn
    effect (cars get faster over the race) fights against the tire-wear
    effect (cars get slower on a given set of tires) in the same lap-by-lap
    data. Controlling for LapNumber separates the two effects.

    Returns a summary table with the degradation rate (seconds/lap) and
    R^2 for each compound."""
    results = []

    for compound, group in df.groupby("Compound"):
        min_rows = 5 if not control_for_fuel else 8
        if len(group) < min_rows:
            # Not enough data points for a meaningful fit
            continue

        if control_for_fuel and "LapNumber" in group.columns:
            X = group[["TyreLife", "LapNumber"]].values
        else:
            X = group[["TyreLife"]].values
        y = group["LapTimeSeconds"].values

        model = LinearRegression()
        model.fit(X, y)

        r_squared = model.score(X, y)

        results.append(
            {
                "Compound": compound,
                "Laps analyzed": len(group),
                "Degradation (sec/lap)": round(model.coef_[0], 4),
                "Fuel effect (sec/lap)": round(model.coef_[1], 4) if control_for_fuel and X.shape[1] > 1 else None,
                "Intercept (sec)": round(model.intercept_, 3),
                "Mean LapNumber": round(group["LapNumber"].mean(), 1) if control_for_fuel and "LapNumber" in group.columns else None,
                "R^2": round(r_squared, 3),
            }
        )

    return pd.DataFrame(results).sort_values("Degradation (sec/lap)", ascending=False)


def plot_degradation(df: pd.DataFrame, summary: pd.DataFrame, title_suffix="", fuel_adjusted=False):
    """Scatter plot of lap time vs. tire age, with a fitted regression
    line per compound overlaid.

    If fuel_adjusted is True, summary must come from
    fit_degradation_by_compound(..., control_for_fuel=True). The fitted
    line then shows the *partial* effect of tire age alone -- holding
    fuel burn (LapNumber) fixed at that compound's average race position
    -- so the line reflects the same fuel-controlled degradation numbers
    reported in the summary table, rather than the raw, confounded trend.
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    compound_colors = {
        "SOFT": "red",
        "MEDIUM": "gold",
        "HARD": "silver",
        "INTERMEDIATE": "green",
        "WET": "blue",
    }

    for compound, group in df.groupby("Compound"):
        color = compound_colors.get(str(compound).upper(), "gray")
        ax.scatter(group["TyreLife"], group["LapTimeSeconds"], s=14, alpha=0.5,
                   color=color, label=f"{compound} (laps)")

        row = summary[summary["Compound"] == compound]
        if not row.empty:
            slope = row["Degradation (sec/lap)"].values[0]
            intercept = row["Intercept (sec)"].values[0]
            x_line = np.linspace(group["TyreLife"].min(), group["TyreLife"].max(), 50)

            if fuel_adjusted and row["Fuel effect (sec/lap)"].values[0] is not None:
                fuel_coef = row["Fuel effect (sec/lap)"].values[0]
                mean_lap = row["Mean LapNumber"].values[0]
                # Hold LapNumber fixed at this compound's average race
                # position, so the line isolates the tire-age effect only.
                y_line = slope * x_line + fuel_coef * mean_lap + intercept
            else:
                y_line = slope * x_line + intercept

            ax.plot(x_line, y_line, color=color, linewidth=2)

    ax.set_xlabel("Tire Age (laps on this set)")
    ax.set_ylabel("Lap Time (seconds)")
    mode_label = " (Fuel-Adjusted)" if fuel_adjusted else " (Raw, Unadjusted)"
    ax.set_title(f"Tire Degradation by Compound{mode_label}{title_suffix}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    filename = "tire_degradation_fuel_adjusted.png" if fuel_adjusted else "tire_degradation_raw.png"
    fig.savefig(filename, dpi=150)
    print(f"\nSaved plot to {filename}")


def main():
    print(f"Loading {YEAR} {GRAND_PRIX} {SESSION} session data...")
    laps = load_session_laps(YEAR, GRAND_PRIX, SESSION)

    if DRIVER_FILTER:
        laps = laps[laps["Driver"] == DRIVER_FILTER]
        print(f"Filtered to driver: {DRIVER_FILTER}")

    print(f"Raw laps loaded: {len(laps)}")

    clean = clean_laps(laps)
    print(f"Laps remaining after cleaning (removed pit/in/out/no-time laps): {len(clean)}")

    summary = fit_degradation_by_compound(clean)
    print("\n=== Tire Degradation Summary (tire age only) ===")
    print(summary.to_string(index=False))

    summary_fuel = fit_degradation_by_compound(clean, control_for_fuel=True)
    if not summary_fuel.empty:
        print("\n=== Tire Degradation Summary (controlling for fuel burn via LapNumber) ===")
        print(summary_fuel.to_string(index=False))
        print(
            "\nNote: if the tire-age-only degradation numbers above look negative or "
            "inconsistent with expectations, compare them to this fuel-controlled version. "
            "A large difference between the two means fuel burn (cars getting lighter and "
            "faster over the race) was confounding the tire-only estimate."
        )

    plot_degradation(clean, summary, title_suffix=f" -- {YEAR} {GRAND_PRIX}", fuel_adjusted=False)

    if not summary_fuel.empty:
        plot_degradation(clean, summary_fuel, title_suffix=f" -- {YEAR} {GRAND_PRIX}", fuel_adjusted=True)


if __name__ == "__main__":
    main()
