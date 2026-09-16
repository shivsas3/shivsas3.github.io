# Shiv Sastry — Engineering Portfolio

Live site: [shivsas3.github.io](https://shivsas3.github.io)

A running collection of independent mechanical engineering and data analysis projects — reverse engineering, CAD, dimensioning, and applied statistics work done outside of coursework.

## About

I'm a sophomore at Lehigh University pursuing a dual degree in Mechanical Engineering and Finance through the Integrated Business and Engineering Honors Program. I also contribute to Lehigh's Formula SAE aerodynamics subsystem and write [The Parc Fermé Report](https://theparcfermereport.substack.com/), an independent publication analyzing Formula 1 technical regulations and strategy.

## Projects

- **Self-Locking Tape Measure — Reverse Engineering & Part Definition** — Full product teardown and dimensioning of an 8-component mechanism using a dial caliper, with tolerances derived at each mating interface and a complete 2D/3D part definition built in Onshape (10-sheet drawing package).

- **Tire Degradation & Lap-Time Regression — F1 Race Data Analysis** — Built a linear regression pipeline on official Formula 1 timing data (via FastF1) to quantify tire degradation by compound. Diagnosed a fuel-burn confound in the initial model, corrected it by adding lap number as a second regressor, and compared results across two seasons to distinguish a genuine multicollinearity limitation from a fixed property of any one compound.

More projects will be added here over time.

## Repo contents

| File | Purpose |
|---|---|
| `index.html` | The portfolio site itself (GitHub Pages serves this as the homepage) |
| `Shiv_Sastry_Tape_Measure_Drawing_Package.pdf` | Full 10-sheet drawing package for the tape measure project |
| `tire_degradation_analysis.py` | Main script: pulls F1 data, cleans it, fits both the raw and fuel-controlled regressions, and saves both charts |
| `test_analysis_logic.py` | Validates the regression and cleaning logic against synthetic data with known ground-truth degradation rates |
| `check_data_available.py` | Quick utility to check whether FastF1 has data for a given year/race before running the full analysis |
| `tire_degradation_raw.png` | Chart of the original, fuel-confounded regression (shows why the naive model was misleading) |
| `tire_degradation_fuel_adjusted.png` | Chart of the corrected, fuel-adjusted regression (matches the numbers discussed in the write-up) |

## Contact

- Email: shiv.sas3@gmail.com
- LinkedIn: [linkedin.com/in/shiv-sastry](https://linkedin.com/in/shiv-sastry)
