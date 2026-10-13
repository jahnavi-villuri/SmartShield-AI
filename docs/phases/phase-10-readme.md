# Phase 10: Complete testing and continuous integration

## What we did
- **Simulated a fresh clone** (project copied without `data/`, `reports/` and every `.joblib`) and ran the tests. Result: 272 passed, 47 skipped, no failures. This is what CI will see.
- **Found a publishing risk:** the error-analysis CSVs in `reports/metrics/` quote real third-party SMS and email text. They are now git-ignored, and a new script, `scripts/check_repo_hygiene.py`, rejects model binaries, datasets, private data, quoted text, secrets and files over 2 MB. It is itself tested (`tests/test_repo_hygiene.py`).
- **Measured coverage** with `pytest-cov`: 94% of the app and serving code. The first run said 58% because offline training scripts were included; those are now excluded in `pyproject.toml` with the reason written next to the setting.
- **Added tests** for the synthetic email loader (it had 0% coverage), the hygiene rules and the rate limiter.
- **Wrote the CI workflow** (`.github/workflows/ci.yml`): lint, format check, hygiene, tests on Ubuntu and Windows with a 90% gate, `pip-audit` (also weekly). `requirements-ci.txt` pins the test tools. Dependabot proposes weekly updates for pip and for the actions.

## Why (concepts)
- **Continuous integration:** every change is checked automatically in a clean machine, so "works on my computer" is not enough.
- **Test pyramid:** many small unit tests (features, splitting, detectors), fewer app tests (Streamlit `AppTest`), a few real-model regression tests.
- **Skip, don't fake:** tests that need real models skip when the models are absent; they never pass on invented data.
- **Coverage is a smoke alarm, not a goal:** 94% means most lines ran, not that they are correct. The gate only stops silent decay.
- **Hygiene as code:** a rule that is a script and a test cannot be forgotten.

## Difficulties
- My first copy command excluded every folder named `data`, including `src/data`, and produced misleading import errors. The simulation must mirror Git: exclude only the top-level `data/`.
- A hygiene test failed because one path matched two rules, so counting messages was wrong; comparing the set of paths fixed it.

## What CI cannot tell us
- Whether the real models still behave: 47 tests skip. Run the full suite locally before a release.
- Python 3.14 availability on the runners: the workflow was written but **has not run yet** (nothing is pushed). The first push may need small fixes.

## Self-check
1. Why do real-model tests skip in CI instead of training a model there?
2. Why are training scripts left out of the coverage figure, and what checks them instead?
3. Which three kinds of file does the hygiene script block, and why?

## Done when
CI file and hygiene script exist and are tested; a fresh-clone simulation passes; coverage gate set below the measured value; docs updated.
