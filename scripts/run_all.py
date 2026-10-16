"""Reproduce the whole project from the raw data with one command.

Run from the repository root, with the virtual environment active:

    python scripts/run_all.py

Steps, in order:

1. Stage 1: clean the raw EPA data (data/processed/)
2. Stage 2: physics model checks and convergence study (results/stage2/)
3. Stage 3: split, tuning and cross-validation (results/stage3/)
4. Stage 4: held-out test evaluation (results/stage4/)
5. Stage 5: performance analysis (results/stage5/)
6. Tests, with an HTML test report and a coverage report (reports/)
7. HTML documentation of the code (docs/)

Use ``--skip-performance`` to leave out Stage 5, which is the slowest step
and whose timings depend on the computer. Everything else gives the same
numbers on every run, because all random choices use fixed seeds.

The script stops at the first step that fails.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def steps(skip_performance: bool) -> list[tuple[str, list[str]]]:
    """Return (description, command) for every step to run."""
    py = sys.executable
    plan = [
        ("Stage 1: data cleaning", [py, "-m", "ev_energy.data_cleaning"]),
        ("Stage 2: physics model", [py, "scripts/stage2_physics.py"]),
        ("Stage 3: ML and hybrid models", [py, "scripts/stage3_models.py"]),
        ("Stage 4: test-set validation", [py, "scripts/stage4_validation.py"]),
    ]
    if not skip_performance:
        plan.append(("Stage 5: performance", [py, "scripts/stage5_performance.py"]))
    plan += [
        ("Tests, test report and coverage report", [
            py, "-m", "pytest",
            "--html=reports/test_report.html", "--self-contained-html",
            "--cov=ev_energy", "--cov-report=html:reports/coverage",
            "--cov-report=term",
        ]),
        ("HTML documentation", [
            py, "-m", "pdoc", "ev_energy", "-o", "docs", "--docformat", "numpy",
        ]),
    ]
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-performance", action="store_true",
                        help="leave out Stage 5 (slowest, machine-dependent timings)")
    args = parser.parse_args()

    (ROOT / "reports").mkdir(exist_ok=True)
    timings = []
    for description, command in steps(args.skip_performance):
        print(f"\n{'=' * 70}\n{description}\n{'=' * 70}", flush=True)
        start = time.perf_counter()
        result = subprocess.run(command, cwd=ROOT)
        seconds = time.perf_counter() - start
        timings.append((description, seconds, result.returncode))
        if result.returncode != 0:
            print(f"\nFAILED: {description} (exit code {result.returncode}). Stopping.")
            break
        # coverage.py writes a .gitignore containing "*" into its output
        # folder, which would stop Git from committing the coverage report.
        hidden = ROOT / "reports" / "coverage" / ".gitignore"
        if hidden.exists():
            hidden.unlink()

    print(f"\n{'=' * 70}\nSummary\n{'=' * 70}")
    for description, seconds, code in timings:
        status = "ok" if code == 0 else "FAILED"
        print(f"  {status:<7} {seconds:7.1f} s  {description}")
    if all(code == 0 for _, _, code in timings):
        print("\nAll steps completed. See results/, reports/ and docs/.")
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
