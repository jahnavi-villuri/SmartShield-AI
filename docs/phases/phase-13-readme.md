# Phase 13: Publishing to GitHub and deploying

Written for: the project owner, who is publishing this project for the first time.

This phase has two parts. **Part A** puts the code on GitHub (nothing here needs the models). **Part B** explains how to run the app online, which needs the model files that Git deliberately does not contain.

Everything below was checked against this machine on 2026-10-13: Git 2.54 is installed, your identity is already set (`jahnavi-villuri`, `villurijahnavi@gmail.com`), the branch is `main`, there are **no commits and no remote yet**, and Git would add **258 files, 3.4 MB**. The GitHub command-line tool (`gh`) is not installed, so these steps use the website plus plain Git.

---

## Part A: commit and push

### Step 1: create an empty repository on GitHub
1. Sign in at github.com as `jahnavi-villuri` and click **New repository** (the green button, or the **+** menu at the top right).
2. **Repository name:** for example `smartshield-ai` (lower case, no spaces).
3. **Description:** "Spam, phishing and harmful-comment detection with traditional machine learning (scikit-learn + Streamlit)".
4. **Visibility:** choose **Private** first. You can make it Public after you have looked at it on GitHub (Settings > General > Danger Zone > Change visibility).
5. **Leave all three boxes unticked:** *Add a README*, *Add .gitignore*, *Choose a license*. The project already has all three, and ticking them creates a conflicting first commit.
6. Click **Create repository**. Copy the address shown, which looks like `https://github.com/jahnavi-villuri/smartshield-ai.git`.

### Step 2: open a terminal in the project folder
In VS Code: **Terminal > New Terminal** (it opens in `C:\Projects\smartshield`). Or in PowerShell:

```powershell
cd C:\Projects\smartshield
.\.venv\Scripts\Activate.ps1      # so "python" means the project's Python
```

### Step 3: run the checks (about 3 minutes)
```powershell
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
python -m scripts.check_repo_hygiene
```
Expected: no lint errors; `389 passed, 3 xfailed, 1 xpassed`; and the last line `0 problem(s)`. **Do not continue if the hygiene check prints FORBIDDEN lines**: it means a model file, dataset, private file or secret is about to be committed.

### Step 4: look at exactly what will be committed
```powershell
git status
git add -n .                       # "-n" = dry run: lists files, adds nothing
```
You should see about 258 files and none of these:
- anything under `data/` except `.gitkeep` files,
- any `.joblib` model file,
- `.venv/`, `__pycache__/`, `.env`,
- `reports/metrics/*false_negatives.csv`, `*false_positives.csv`, `*missed_spam.csv` (they quote real third-party messages).

Prove that the important protections work:
```powershell
git check-ignore -v data/private/my_sms/messages.csv models/sms_spam/model.joblib .env
```
Each path must print a line naming the `.gitignore` rule that excludes it. A path that prints nothing is **not** ignored: stop and fix `.gitignore` before going on.

**Privacy review (done for you):** I searched the files to be published for your private SMS text and account-like numbers and found none, except one document, `docs/phases/phase-03b-readme.md`, which quoted your real SBI alert with the last four digits of your account number. I replaced them with `XXXX`. Before pushing, skim `docs/` and `README.md` once more for anything personal.

### Step 5: make the first commit
```powershell
git add .
git status                          # everything should be green "new file:" lines
git commit -m "Initial commit: SmartShield AI (phases 0-12)"
git log --oneline                   # shows your one commit
```
If Git says "Please tell me who you are", run (your values are already set, so this is normally not needed):
```powershell
git config --global user.name "jahnavi-villuri"
git config --global user.email "villurijahnavi@gmail.com"
```
Tip: GitHub only credits commits to your profile if the commit email matches an email on your GitHub account (or GitHub's private `noreply` address, found under Settings > Emails).

### Step 6: connect to GitHub and push
```powershell
git remote add origin https://github.com/jahnavi-villuri/smartshield-ai.git
git branch -M main
git push -u origin main
```
Use your real repository name. **Signing in:** Git for Windows includes Git Credential Manager. The first push opens a browser window: choose "Sign in with your browser", approve, and it remembers you. GitHub does **not** accept your account password in the terminal. If the browser sign-in does not appear, create a token: GitHub > Settings > Developer settings > Personal access tokens > Fine-grained tokens > Generate; give it access to this one repository with **Contents: Read and write** (and **Workflows: Read and write**, because the repository contains `.github/workflows`). Paste the token when Git asks for a password.

### Step 7: check the result on GitHub
1. Refresh the repository page. You should see `README.md` with the screenshots, `docs/`, `src/`, `tests/`, `LICENSE` and so on, and **no** `data` contents or `.joblib` files.
2. Open the **Actions** tab. The `CI` workflow starts automatically. It has never run before, so expect to read its output once.
   - Jobs: `lint`, `test` (Ubuntu and Windows) and `audit`.
   - **In CI the 47 tests that need the real models skip themselves** (the models are not in Git). That is expected, and the coverage gate (90%) was measured at 94% in a local simulation.
   - If a job fails, click it, open the red step, read the last lines, fix locally, then `git add`, `git commit -m "Fix CI: <what>"`, `git push`. Likely first-run problems: the Python 3.14 version string, a Windows-only path in a test, or a pinned package missing a wheel for the runner.
3. Add topics (Settings gear next to "About"): `machine-learning`, `scikit-learn`, `streamlit`, `spam-detection`, `phishing-detection`, `nlp`.

### Step 8: everyday workflow afterwards
```powershell
git status                 # what changed?
git diff                   # see the changes
git add path/to/file       # stage on purpose (avoid "git add ." until you have checked git status)
git commit -m "Short, present-tense summary of the change"
git push
```
Good habits: one idea per commit; run the tests before pushing; never use `git add -f` (it overrides `.gitignore`); look at `git status` every time; work on a branch (`git switch -c fix-email-threshold`) and merge through a pull request when a change is risky.

### Step 9 (optional): mark the release
```powershell
git tag -a v1.0.0 -m "First complete version"
git push origin v1.0.0
```
On GitHub: Releases > Draft a new release > choose the tag. Part B uses a release to carry the model files.

### If something goes wrong
| Situation | What to do |
|---|---|
| You staged a file you should not commit, **not yet committed** | `git restore --staged path/to/file`, then fix `.gitignore` |
| You committed it, **not yet pushed** | `git reset --soft HEAD~1` (undo the commit, keep the files), unstage the file, commit again |
| You **pushed** a private file or a secret | Treat it as public and compromised. Change any password or token **first**. Then remove it from history (`git filter-repo`, or delete and recreate the repository while it is still empty or private) and ask GitHub Support to purge cached views if it was public. Deleting the file in a new commit is **not** enough: it stays in history |
| `git push` says "rejected ... fetch first" | The remote has a commit you do not have (usually a README created on the website). If the repository is new and nothing there matters: `git push -u origin main --force`. Otherwise `git pull --rebase origin main`, then push |
| "Permission denied" or "Authentication failed" | Wrong account in the browser sign-in, or an expired token. Run `git credential-manager erase` (or remove the GitHub entry in Windows Credential Manager) and push again |
| A file over 100 MB is rejected | None should be: the largest model is 37 MB and models are not committed anyway |

---

## Part B: deploying the app (plan)

**The obstacle:** GitHub now has the code, but `models/**/*.joblib` (about 64 MB in total, 37 MB for the comment model) is git-ignored. A hosting service that builds from GitHub would find no models, and the app would show "model not available" on every tab.

**Options**

| Option | How | Verdict |
|---|---|---|
| **A. GitHub Release asset + download at start (recommended)** | Upload the five `model.joblib` files to a Release; the app downloads them on first start | Keeps Git small. Safe, because every file is verified against the SHA-256 stored in the committed `metadata.json` before it is loaded (the loader already does this) |
| B. Commit the `.joblib` files | Remove the ignore rule | Simple, but each retrain adds about 64 MB to history for ever |
| C. Git LFS | Large-file storage for `*.joblib` | Works, but quotas are small and it complicates cloning |
| D. Hugging Face Hub model repository | Host models there | A good long-term home; extra account and code |

**What I would build for Option A (not done yet):**
- `scripts/fetch_models.py`: download the release assets, **verify each SHA-256 against `metadata.json`**, refuse to continue on a mismatch.
- A startup call in `app.py` that runs it once when a model is missing.
- A test with a fake download server proving that a tampered file is rejected.

**Hosting on Streamlit Community Cloud (outline)**
1. Make sure the repository is on GitHub and the release assets exist.
2. Go to share.streamlit.io, sign in with GitHub, choose **Create app**, pick the repository, branch `main` and main file `app.py`.
3. Open **Advanced settings** and choose the Python version. The project was built and tested with **Python 3.14**; if the platform does not offer it, the pinned packages (`numpy 2.5`, `pandas 3.0`, `scikit-learn 1.9`) may not install and the pins must be re-resolved for the version it does offer. I have not tested this: check the current platform documentation.
4. No secrets are needed.
5. After deploying, test every tab with the example buttons, then open the Dashboard.

**Things to decide or check before making the app public**
- **Memory and start-up time:** the comment model needs about 8 seconds to load on this machine and the three TF-IDF models hold large vocabularies; free tiers have limited memory. Measure on the host.
- **Abuse:** the in-app rate limit is per browser session only (see `docs/security.md`); public traffic needs the host's limits too. Consider keeping the app private or link-only at first.
- **Honest labelling:** keep the "Predictions can be wrong" banner and the model cards linked from the app's README. The comment model must not be presented as a moderation tool.
- **Licence of trained models:** see `docs/dataset_cards/README.md` before sharing the model files publicly.

---

## Self-check
1. Why do we tick none of the three boxes when creating the GitHub repository?
2. What does `git add -n .` do, and why run it before `git add .`?
3. You committed `data/private/my_sms/messages.csv` and pushed. Why is deleting it in the next commit not enough?
4. Why can the app safely download model files from the internet at start-up?

## Done when
The repository is on GitHub with one commit, no data or model binaries, CI has run at least once and its result is understood; and the model-delivery option for deployment has been chosen.
