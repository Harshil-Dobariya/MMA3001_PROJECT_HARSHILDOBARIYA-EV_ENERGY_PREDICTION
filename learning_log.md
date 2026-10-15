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

## Week 9
### Stage 3, ML and hybrid models

**What I did**
- Created a new branch, `ml-models`, for this stage.
- Split the data by vehicle model so no model appears in both sets: 537 training vehicles (210 models) and 175 test vehicles (53 models). The two sets have similar averages and spreads (city mean 25.5 vs 25.1 kWh/100 mi). The test set was saved but not used for anything in this stage.
- Trained the four model families from Week 5 (linear regression, decision tree, RBF SVR and an MLP neural network), each in a pipeline with `StandardScaler`, plus a mean baseline.
- Built hybrid models: the physics model predicts first, and an ML model learns the residual (what physics gets wrong).
- Tuned hyperparameters with grouped 5-fold cross-validation on the training set only. Inside every fold the physics model was refitted on that fold's training rows, so the hybrids never saw validation targets.
- Wrote a leakage test: it adds 100 kWh/100 mi to every target in one fold and checks the predictions for that fold do not change. It passed.
- My first run gave different numbers from Claude's (mean baseline 5.301 vs 5.338). Since the mean baseline has no randomness, the folds must have been assigned differently. scikit-learn's `GroupKFold` assigned vehicle models to folds differently in different library versions (I have Python 3.14.8 and scikit-learn 1.9.1). Fixed by assigning folds explicitly with a seeded shuffle.
- After the fix my results matched exactly, except the decision trees, which differed by up to 0.10 kWh/100 mi because of tie-breaking changes between scikit-learn 1.8 and 1.9. The selected model was unaffected. Locked my exact package versions in `requirements-lock.txt`.
- Comparing three different fold assignments showed the gap between the top models (about 0.05) was smaller than the variation from fold assignment alone (about 0.1). So I repeated cross-validation with five fold assignments and compared mean RMSE.
- Results: city, Physics + SVR was clearly best (2.88 ± 0.10 vs 3.39 for physics alone, a 15% improvement). Highway, the top four models were within their run-to-run spread (2.36 to 2.43), so no model meaningfully beat physics, which was also by far the most stable (±0.004).
- Fixed the model-selection rule before opening the test set: one model for both cycles, lowest average of city and highway mean RMSE. Selected: Physics + SVR (2.62 kWh/100 mi).
- 110 tests passing.

**AI use**
- Claude designed the split, models and cross-validation, and wrote the code and tests. I ran everything on my own machine, and the mismatch between my results and Claude's is what exposed the fold-assignment problem. Claude diagnosed it and wrote the fix; I confirmed it by rerunning and comparing the outputs line by line. The model-selection rule was set before the test set was opened.

---

 ## Week 11: Stage 4, final validation on the test set

**What I did**
- Created a `validation` branch and opened the held-out test set (175 vehicles, 53 models) for the first and only time. The model choice and hyperparameters from Stage 3 were not changed.
- Evaluated every model on the test set. The selected model, Physics + SVR, did not beat physics alone: test RMSE 3.82 vs 3.62 kWh/100 mi (city) and 2.80 vs 2.39 (highway).
- Did not switch to the model that scored best on the test set (Physics + MLP, 3.06 city), because choosing a model using the test set would make its test score optimistic in the same way as a CV score.
- Added 95% confidence intervals by resampling whole vehicle models (bootstrap). The intervals are wide (physics city RMSE 3.62, interval 2.75 to 4.50), so with 53 test models no model was clearly better than physics.
- Failure analysis: the worst error was the 2025 Maserati Grecale Folgore. Its road-load inputs are at the edge of the training data (f0 = 285 N). Physics predicted 31.7 kWh/100 mi, the hybrid SVR 51.6, measured 26.9.
- Added a domain check that flags test vehicles with any input outside the training 1st to 99th percentile range: 4 models were flagged. On them, the Physics + tree city error rose from 3.29 to 6.89.
- The hybrid SVR had a lower typical error than physics (city MAE 2.20 vs 2.79) but much larger rare misses, which RMSE penalises.
- Temporal check (train on 2022 to 2024, test on 41 new 2025 to 2026 models): physics was best on highway (RMSE 1.18, R² 0.84). Physics + SVR scored R² −0.58, worse than predicting the average.
- Sensitivity analysis (±10% mass and drag): physics responded in the physically correct direction for every vehicle. Mass mattered more in the city (elasticity 0.43) than on the highway (0.12), and drag the reverse (0.19 vs 0.46). Every flexible model gave wrong-direction responses for some vehicles; Physics + SVR predicted lower highway consumption for a heavier car in 33% of test vehicles.
- Compared four well-known EVs with the official EPA figures I looked up on fueleconomy.gov. Physics + SVR was close for mainstream cars (Model 3: 129.9 vs 132 MPGe). Physics underestimated their efficiency by 8 to 13% because it assumes one average drivetrain efficiency.
- Range estimates using an assumed charging efficiency of 85 to 95%: Ioniq 6 322 to 360 miles (EPA 361), Kia EV6 220 to 245 miles (EPA 232).
- My results matched Claude's except the decision-tree models, the same scikit-learn version difference found in Stage 3. My results are the official ones. 130 tests passing.

**AI use**
- Claude designed and wrote the validation code, tests and figures. I ran everything on my own machine and compared the output line by line with Claude's. I looked up the official EPA figures for the four EVs myself. The decision not to change the selected model after seeing the test results was based on the selection rule fixed in Stage 3.

---

## Week 11: Stage 5, optimisation and performance

**What I did**
- Created a `performance` branch.
- Implemented the physics kernel (wheel power integrated over a drive cycle for every vehicle) four ways: pure Python loops (baseline), NumPy with a loop over vehicles, fully vectorised NumPy, and NumPy in chunks of vehicles. All four give the same answers (maximum relative difference 3 × 10⁻¹⁵, round-off level), checked by tests.
- On my Mac (Apple Silicon, 12 cores), vectorised NumPy was 104 times faster than Python loops for 712 vehicles (0.0023 s vs 0.24 s).
- Counted the FLOPs: 10 per vehicle per time sample (v² is shared across vehicles, so it is not counted per vehicle), about 9.75 million for one city cycle.
- Estimated arithmetic intensity: 0.06 FLOP/byte for NumPy as written (every operation writes a temporary array that the next one reads back) and 0.63 FLOP/byte for an ideal fused loop. Both are far below a modern CPU's machine balance, so the kernel is memory-bandwidth bound. It achieved about 4 GFLOP/s, a small fraction of the processor's peak.
- Chunk-size sweep: on Claude's test machine (Intel Xeon, 1 MB L2 cache), chunks of 32 vehicles were fastest, because the full 7.8 MB arrays did not fit in cache. On my Mac, processing all 712 vehicles at once was fastest, because Apple Silicon's much larger cache holds the full arrays. Chose a chunk size of 712: the same speed as full vectorisation on my machine, but memory capped as the dataset grows (23.7 MB instead of 234 MB at 7,120 vehicles). Kept the Xeon results in the repository as a reference.
- Time step: halving the step doubles the run time. At EPA's 1 s step the numerical error is 1.3% (city) and 0.35% (highway), much smaller than the model error of about 12%, so the 1 s step is justified.
- Profiled one full cross-validation run with `cProfile`: neural-network training took about 80% of the time (3.16 of 3.94 s).
- Model costs: physics trains in 0.7 ms and the MLP in 181 ms, about 260 times slower, for similar cross-validated accuracy (2.91 vs 2.72 kWh/100 mi).
- Parallel hyperparameter tuning on 12 cores gave about a 2 times speed-up (1.9 to 2.2 times over two runs), with the same hyperparameters chosen.
- 153 tests passing.

**AI use**
- Claude wrote the four kernel implementations, the timing and memory tools, the FLOP counts and the Stage 5 script. Claude's first FLOP count was 11 per sample; it corrected this to 10 on noticing that v² is computed once per time step, not per vehicle. Claude initially recommended chunks of 32 based on its own machine; my results showed this was slower on my Mac, and the default was changed to 712 based on my measurements. I ran everything on my machine and checked the outputs.

---

## Resources used

### Data and official figures
- US EPA, Data on Cars Used for Testing Fuel Economy (model years 2022 to 2026): https://www.epa.gov/compliance-and-fuel-economy-data/data-cars-used-testing-fuel-economy
- US EPA, Dynamometer Drive Schedules (UDDS and HWFET speed traces): https://epa.gov/vehicle-and-fuel-emissions-testing/dynamometer-drive-schedules
- US DOE/EPA, fueleconomy.gov: used to check window-sticker values (2022 Chevrolet Bolt EV, 2024 GMC Hummer EV SUV) and to look up official label MPGe for four well-known EVs (2024 Tesla Model 3 RWD, Tesla Model Y Long Range AWD, Hyundai Ioniq 6 Long Range RWD, Kia EV6 Standard Range RWD): https://www.fueleconomy.gov
- Kia America, 2024 EV6 pricing release (battery capacity 58.0 kWh and EPA range 232 miles for the EV6 Light RWD): https://kiamedia.com/us/en/models/ev6/2024
- Carscoops, 2024 Hyundai Ioniq 6 range figures (77.4 kWh battery, EPA range 361 miles for the Long Range RWD on 18-inch wheels): https://carscoops.com/?p=2601849

### MMA3001 unit notes
- Week 1: version control, documenting code, responsible use of AI, testing
- Week 2.1: optimisation and profiling (timing, profilers, FLOPs, memory bandwidth, arithmetic intensity, cache hierarchy)
- Week 2.2: numerical errors (round-off when comparing implementations)
- Week 3.4: spline interpolation (PCHIP used to refine the drive cycles without overshoot)
- Weeks 5.1 to 5.4: linear regression, decision trees, support-vector regression, neural networks
- Weeks 5.5 and 5.6: evaluating regression performance, training data evaluation and splitting
- Week 6: numerical integration (trapezoidal and Simpson's rules)
- Week 7.1: finite-difference derivatives

### Software documentation
- scikit-learn documentation (pipelines, GroupKFold, GridSearchCV, SVR, MLPRegressor): https://scikit-learn.org/stable/
- Python documentation, the Python profilers (cProfile): https://docs.python.org/3/library/profile.html
- Python documentation, tracemalloc (memory measurement): https://docs.python.org/3/library/tracemalloc.html
- NumPy documentation, legacy random generation (RandomState, used for reproducible folds): https://numpy.org/doc/stable/reference/random/legacy.html