# Learning Log

MMA3001 individual project: predicting EV energy consumption from EPA test data.

This log records what I did in each session, what I learned, where I got stuck, and how I used AI. The factual record of each session was drafted with Claude from our working sessions and checked by me. The reflections under "What I learned" and "Where I got stuck" are my own.

---

## Week 7 
### Scoping the project

**What I did**
- Read the project brief and rubric in detail. 
- Started from an idea to predict EV driving range from the EPA "Data on Cars Used for Testing Fuel Economy" spreadsheets for model years 2022 to 2026.
- Found that the dataset has no range column and no battery capacity, only measured energy efficiency.
- Looked at alternative datasets: a Kaggle EV specs dataset (478 cars, has range and battery size but no mass, and its range values come from a specs website rather than testing), the fueleconomy.gov vehicle dataset (official range but almost no design inputs), and a Mendeley dataset (only 53 cars).
- Chose to keep the EPA test data and change the target to energy consumption (kWh/100 mi), because it is measured test data, includes mass and road-load coefficients, and supports a physics-based comparison using numerical methods from the unit. Notified the coordinator of the change of target; it was approved.

**Why it mattered**
- Range is roughly battery energy divided by consumption, so consumption is the part of range that vehicle design controls.

**AI use**
- Claude read the brief with me, inspected the five EPA files, searched for alternative datasets and compared them against the rubric. I made the final choice of dataset and target.

### Setting up the project

**What I did**
- double checked that I had already installed Python, VS Code (with the Python extension) and GitHub Desktop.
- using my existing MMA3001 Project repository (that we made in week 1) and renamed it to `ev-energy-prediction`, and edited the README
- Set up a virtual environment (`.venv`) and installed the project packages from `requirements.txt`.
- Learned the workflow: edit, run `pytest`, commit with a clear message, push.
- Created a `physics-model` branch for Stage 2.

**AI use**
- Claude explained each concept and gave project setup instructions. I carried out every step myself and checked each result (for example `35 passed` from `pytest`, and the commits appearing on GitHub).

### Stage 1, data cleaning

**What I did**
- Built a cleaning pipeline (`src/ev_energy/data_cleaning.py`) that reduces 21,479 raw rows to one row per tested EV with both a city (UDDS) and highway (HWFET) result.
- Issues found and handled in the raw data:
  - The efficiency column mixes units. Most manufacturers report unadjusted MPGe (about 70 to 200), but Hyundai, Kia, Genesis, Rivian, Fisker, Fiat and some others report kWh/100 mi (about 15 to 40). The unit is inferred from magnitude with a threshold of 40, and MPGe is converted with kWh/100 mi = 100 × 33.705 / MPGe.
  - Placeholder values: efficiency values of 999 (Lucid) and horsepower values such as 1 hp (Fisker).
  - Inconsistent brand names ("Audi" and "AUDI"), and Lucid rows with a year in the make field.
  - Repeated tests of the same vehicle, which are averaged. The median spread between repeat tests is 0.16 kWh/100 mi, which sets a measurement noise floor.
  - Carryover models retested with identical specifications in later years. Each model gets a group ID so a train/test split can keep all its copies on one side, preventing data leakage.
- Result: 740 vehicles (275 unique models). Wrote 35 tests, including checks with synthetic data where the correct answer is known.

**AI use**
- Claude analysed the raw data and generated the cleaning code and tests. I checked it by running the tests, running the pipeline on the real data, and reading each function's docstring against the decisions above.

---

## Week 8: 
### Stage 2, physics-based energy model

**What I did**
- Downloaded the EPA UDDS and HWFET speed traces and checked them against EPA's published values: 1,369 s and 7.45 mi for the UDDS, 765 s and 10.26 mi for the HWFET.
- Derived the energy model: wheel power P_w = (m a + f0 + f1 v + f2 v²) v, split into driving energy E+ and braking energy E−, with wall energy E_b = E+/η_d − η_r E− + P_aux T. The model is linear in 1/η_d, η_r and P_aux, so these are fitted by bounded linear least squares.
- Used numerical methods from the unit: central finite differences for acceleration (Week 7.1), and the composite trapezoidal and Simpson's rules for the integrals (Week 6), with a 3/8-rule correction because both cycles have an odd number of intervals.
- Verified the code with analytical solutions, the EPA benchmarks, a manufactured solution (fake data from known parameters, which the fit recovers exactly) and an energy-conservation check. 66 tests passing at this point.
- Convergence study: at EPA's 1 s time step, consumption is about 1.0 to 1.4% low (city) and 0.3% low (highway) compared with refining to 1/32 s. Simpson's rule did not beat the trapezoidal rule, because max(P_w, 0) puts kinks in the integrand.
- Exploratory fit on all vehicles: η_d = 0.785, η_r = 0.669, R² of 0.62 (city) and 0.72 (highway). P_aux hit its lower bound of zero; unbounded it wanted about −555 W, which suggests the constant-efficiency assumption is a limitation.

**AI use**
- Claude derived the model with me and generated the code, tests and figures. I checked the derivation step by step before agreeing to the form, then ran the tests and checked the outputs against the derivation and the EPA benchmarks.

### Independent data check on fueleconomy.gov

**What I did**
- The physics model showed a group of vehicles sitting about 1.3 times above prediction, nearly all reported as whole number MPGe. The theory was that these are window sticker values (about 0.7 times the raw test result) rather than dynamometer results.
- Checked label values myself on fueleconomy.gov:
  - 2022 Chevrolet Bolt EV: 120 combined, 131 city, 109 highway. Our data: 129 and 132 city, 109 highway.
  - 2024 GMC Hummer EV SUV: 59/48, 56/46 and 51/43 city/highway across versions, matching our data exactly. The MT Tires version showed 55/45.
- Result: the theory was confirmed. Added a rule to exclude vehicles reported as whole-number MPGe on both cycles. The rule uses how values were reported, not the model's residuals, to avoid circular reasoning.
- Effect: 712 vehicles remain. Highway RMSE fell from 3.07 to 2.41 kWh/100 mi and city RMSE from 3.73 to 3.43.
- Trade-off: 10 of the 28 flagged vehicles (for example the Tesla Model 3 Long Range RWD and BMW i4) look like genuine results that happen to be whole numbers. I accepted losing them to keep a rule that does not depend on the model.

**AI use**
- Claude proposed the window-sticker theory and wrote the exclusion rule and its tests. I did the fueleconomy.gov lookups myself and kept screenshots as evidence.



## Midsem Break:
### Fixing a file mix-up and merging Stage 2

**What I did**
- Copied the window-sticker fix into my repo, but `pytest` still showed 66 tests instead of the expected 78.
- Used `grep -c "is_whole_number_mpge"` to check whether the new code was actually in my files. Both files returned 0, which proved they were still the old versions.
- Found the cause: my repo is called `MMA3001_PROJECT_HARSHILDOBARIYA-EV_ENERGY_PREDICTION`, while the instructions and the download both used the name `ev-energy-prediction`. With several similarly named folders in Downloads, the files were not copied into the right place.
- Fixed it by repackaging the download under a unique name (`STAGE2_FIX_FILES`) and copying the files with `cp` in the terminal instead of dragging in Finder.
- Confirmed the fix: `grep` returned 6 and 2, and `pytest` showed 78 passed.
- Reran the full pipeline. My results matched Claude's to the last decimal place (712 vehicles, η_d = 0.796, highway RMSE 2.41 kWh/100 mi), which shows the workflow is reproducible on a different computer.
- Committed the fix in three parts, pushed, and merged Stage 2 into `main` with my first pull request.

**AI use**
- Claude diagnosed the problem and gave the `grep` and `cp` commands. I ran every command myself and checked each output against what was expected.

---


## Resources used

- US EPA, Data on Cars Used for Testing Fuel Economy: https://www.epa.gov/compliance-and-fuel-economy-data/data-cars-used-testing-fuel-economy
- US EPA, Dynamometer Drive Schedules: https://epa.gov/vehicle-and-fuel-emissions-testing/dynamometer-drive-schedules
- US DOE/EPA, fueleconomy.gov: https://www.fueleconomy.gov
- MMA3001 unit notes: Week 1 (version control, documentation, testing), Week 5.5 and 5.6 (evaluation and data splitting), Week 6 (numerical integration), Week 7.1 (finite differences)
- [Add anything else you read or watched yourself.]