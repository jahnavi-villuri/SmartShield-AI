# Phase 1 README: Environment, Repository and Project Skeleton

## 1. What we did

| Item | What it is |
|---|---|
| Git repository | `git init` on branch `main` (no commits yet, you make the first one) |
| Virtual environment | `.venv/` with all packages installed |
| Folder skeleton | `config/ data/ models/ notebooks/ src/ reports/ tests/ scripts/` |
| `.gitignore` | Keeps datasets, model binaries, `.venv`, secrets and private data out of Git |
| `requirements.in` / `requirements.txt` | What we want / the exact versions installed |
| `requirements-dev.in` / `requirements-dev.txt` | Development tools (pytest, ruff, jupyter, pip-audit) |
| `pyproject.toml` | Settings for ruff (linter/formatter) and pytest |
| `config/settings.py` | Paths, random seed, label convention (1 = bad), toxicity label list |
| `app.py` | Hello-world Streamlit page |
| `tests/test_environment.py` | Smoke tests for imports and settings |
| `README.md` | Project README skeleton |

## 2. Why each piece exists

### Virtual environment
A project-private copy of Python and its packages. Without it, packages from different projects collide and "it works on my machine" bugs appear. **Rule: one venv per project, never commit it.**

### Pinned versions
`requirements.in` lists what we asked for; `requirements.txt` lists the *exact* versions that were installed (via `pip freeze`). This matters for ML more than usual: **a model saved with one scikit-learn version may fail to load, or behave differently, in another.** Pinning means anyone (including future you, and the deployment server) rebuilds the same environment.

### Python version note
Decision D-009 chose Python 3.11/3.12, but this machine has **Python 3.14**. A dry-run showed every package we need installs on 3.14, so we used it (decision D-011). If a later package or the hosting platform doesn't support 3.14, install Python 3.12 and recreate the venv; nothing else changes.

### .gitignore
Written *before* the first commit because Git remembers everything: a dataset or private email committed once stays in history. We ignore data and model binaries (large, licensed or private) but keep folder structure through `.gitkeep` files and keep small `metadata.json` files.

### config/settings.py
One source of truth for paths, the seed (42) and the label convention. If a value lives in one place, it can't be wrong in another. Seeds matter because splits and model training involve randomness; the same seed gives the same results, so experiments can be compared fairly.

### Ruff and pytest settings
Ruff reads your code for mistakes (unused imports, likely bugs) and formats it consistently. pytest finds files named `test_*.py`. We set `pythonpath = ["."]` so tests can `import config`.

### Folder separation
`src/data` (loading/cleaning), `src/features` (turning raw input into model input), `src/training`, `src/evaluation`, `src/detection` (prediction used by the app), and `app.py` (UI only). **Training code never runs when the app starts,** and the UI never contains ML logic. This separation is what makes testing and deployment possible.

## 3. Concepts introduced

| Concept | Plain explanation |
|---|---|
| Virtual environment | Private box of Python packages for one project |
| Dependency pinning | Recording exact versions so the environment is reproducible |
| `.gitignore` | List of files Git must never track |
| Linter / formatter | Tool that checks and tidies code automatically |
| Smoke test | Tiny test proving the basics work |
| Random seed | Fixed starting number that makes "random" steps repeatable |

## 4. How to pick tools for the next project (technology selection)

1. **Does it have to run on the user's/hosting machine?** Prefer common, pip-installable, CPU-friendly libraries.
2. **Is it the community standard for this task?** (pandas, scikit-learn). Standard tools have answers online when you get stuck.
3. **Does it do one thing you need?** Don't add a library "just in case" (each dependency is a security and maintenance cost).
4. **Can you pin and reproduce it?** If it can't be pinned, it can't be trusted in a pipeline.
5. **Is there a simpler tool?** Streamlit over Flask + JavaScript for a solo ML demo.

## 5. Commands you will use constantly

```powershell
.\.venv\Scripts\Activate.ps1          # enter the venv (prompt shows (.venv))
python -m pytest                      # run tests
python -m ruff check .                # lint
python -m ruff format .               # auto-format
streamlit run app.py                  # run the app
git status                            # what changed? (run before every commit)
```

## 6. Your first commit and GitHub (you do this)

```powershell
git status
git add .
git status          # read the list: no data, no .venv, no secrets?
git commit -m "Phase 0-1: docs, environment, project skeleton"
```
Then create an empty repository on GitHub (no README) and:
```powershell
git remote add origin https://github.com/<your-username>/smartshield-ai.git
git push -u origin main
```
The second `git status` is the habit that prevents leaking data.

## 7. Self-check questions

1. Why do we pin exact versions of scikit-learn?
2. Why is `.gitignore` created before the first commit?
3. Why does `config/settings.py` exist?
4. Why is the app not allowed to train models?
5. What is a random seed for?

## 8. Done when

- [ ] `python -m pytest` passes
- [ ] `python -m ruff check .` passes
- [ ] `streamlit run app.py` opens a page in your browser
- [ ] Your first commit contains no data, no `.venv`, no secrets
- [ ] You answered the five questions

## 9. Next: Phase 2

SMS data preparation: download UCI + Mendeley SMS + SmishTank, inspect, clean, label-map, deduplicate (within and across datasets), split, and write the data-quality report. No training yet.
