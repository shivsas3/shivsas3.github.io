"""
Quick check: does FastF1 have lap data available for a given year/race?
Doesn't run any analysis -- just tries to load the session and reports
whether real lap data came back.

USAGE:
    python check_data_available.py
"""
import os
import fastf1

# Edit these to check a different year/race
YEAR = 2026
GRAND_PRIX = "Hungary"
SESSION = "R"

os.makedirs("f1_cache", exist_ok=True)
fastf1.Cache.enable_cache("f1_cache")

print(f"Checking {YEAR} {GRAND_PRIX} {SESSION}...")

try:
    session = fastf1.get_session(YEAR, GRAND_PRIX, SESSION)
    session.load(telemetry=False, weather=False, messages=False)  # laps only, faster check

    if hasattr(session, "_laps") and session.laps is not None and len(session.laps) > 0:
        print(f"\nDATA AVAILABLE: {len(session.laps)} laps loaded for {YEAR} {GRAND_PRIX} {SESSION}.")
        print("You're good to run the full tire_degradation_analysis.py with this config.")
    else:
        print(f"\nNO DATA: the session loaded but returned zero laps.")
        print("This race/session may not be archived yet, or the event name/round may be wrong.")

except Exception as e:
    print(f"\nFAILED: {type(e).__name__}: {e}")
    print("This usually means the session doesn't exist yet (race hasn't happened), the")
    print("event name doesn't match FastF1's naming, or the data source is temporarily down.")
