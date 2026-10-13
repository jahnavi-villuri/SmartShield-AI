# Phase 7 README: Phishing URL Detection

> Numbers come from real runs of `python -m scripts.prepare_phishing`, `python -m scripts.split_phiusiil` and `python -m src.training.train_phishing --with-phiusiil`. Tables: `reports/metrics/phishing_*.csv` (rounds 1-3 archived next to them); data report: `reports/data_quality/phishing.md`. **No URL in this document or in the tests points to a real malicious site, and no URL was ever visited.**

## 1. Result in one paragraph

The URL model reads only the text of an address (no page, no certificate, no domain age) and gives a score. On a **temporal test** of 162,718 URLs from Sep-Dec 2025 (newer than all training data) it catches **85% of phishing URLs while flagging 1.8% of legitimate ones** (precision 0.974). On real phishing from outside its training sources it catches **80-88%**: 80% of a live OpenPhish feed (2026-10-12), 88% of PhishBench, 86% of CompPhish. The cost is false alarms on the kinds of legitimate URLs the public data rarely contains: **25% of CompPhish's legitimate URLs** (small blogs and business sites), **16% of PhishBench's** (bare homepages), **56% of the legitimate pages on free hosting platforms** (github.io, vercel.app, Google Sites) and 3.8% of domains it has not seen. It never claims a site is safe, never opens a link, and gives "Cannot judge" for shortened links.

The main lesson of this phase is not the model but the **data**: every public phishing dataset has a structural shortcut, and each one pushes in a different direction.

## 2. What we built

| Piece | File |
|---|---|
| Column-only reader for huge remote parquet files | `src/data/range_file.py`, `scripts/fetch_phreshphish_urls.py` |
| URL features (the ONE function used by training and the app) and URL validation | `src/features/url_features.py` |
| Data pipeline + report (labels unified, deduplicated, grouped split) | `scripts/prepare_phishing.py`, `scripts/split_phiusiil.py` |
| Training, thresholds, ablations, evaluation | `src/training/train_phishing.py` |
| Detector (validation, checksum load, score, notes) | `src/detection/phishing_detector.py` |
| URL tab | `app.py` |
| Hand-written challenge URLs (68: 56 legitimate, 12 phishing-style on `.example` domains) | `tests/data/url_challenge.csv` |
| Tests | `test_url_features.py`, `test_phishing_detector.py`, `test_phishing_real_model.py`, `test_model_files_portable.py` |

## 3. Safety rules (enforced by tests)

- A URL is **text**. It is never fetched, resolved, opened, previewed or rendered, and the app shows it in a code block, never as a link. Tests block `socket.connect` and DNS lookups while predicting.
- Only `http://` and `https://` addresses are accepted. `javascript:`, `data:`, `file:`, `ftp:`, spaces, control characters, over-long text (2,000 characters) and missing hosts get a clear message.
- The result is never "safe": the strongest positive wording is "Likely legitimate", with a standing caption that the model cannot confirm a site is safe.
- Shortened links (bit.ly and similar) get **"Cannot judge"**, because the destination is unknowable without opening them.
- Hostile data handling: phishing URLs were printed only "defanged" (`hxxps`), PhreshPhish's HTML column was never requested, CompPhish's saved pages were never downloaded, and code archives that came with CompPhish were not run.

## 4. The data

| Dataset | Role | Period | Size used | Label convention |
|---|---|---|---|---|
| **PhreshPhish** (CC BY 4.0, "for anti-phishing research") | train / validation, and the newer split as the **temporal test** | Jul 2024 - Dec 2025 | 400,114 train, 82,682 val, 162,718 test | benign / phish |
| **PhiUSIIL** (UCI 967) | Round 1: external; Rounds 2-4: training | ~2023 | 196,069 train, 31,968 val | **1 = legitimate (opposite of this project): flipped** |
| CompPhish | external | Sep 2024 - Aug 2025 | 12,148 | 1 = phishing |
| PhishBench-External | external | Jul 2025 | 856 | `Results` undocumented; verified by reading URLs: 1 = phishing |
| OpenPhish community feed | external, phishing only | 2026-10-12 | 286 | all phishing |
| Hand-written URLs | smoke test and guard | n/a | 68 | written for this project |

### 4.1 Getting the data without downloading the dangerous parts
- **PhreshPhish** is 36.6 GB (77 files of about 500 MB), mostly the HTML of phishing pages. Parquet stores columns separately, so I read only `url`, `label`, `target`, `date` and `lang` through HTTP range requests: **126 MB transferred in 2 minutes** instead of 36.6 GB.
- **CompPhish** is a 711 MB zip. Reading the zip's directory remotely and extracting only the small URL/label spreadsheet took **24 requests and 636 KB**.
- A dropped connection killed one run after minutes of work, so the fetch script now retries each file, saves each finished file and resumes.

### 4.2 What inspection found (before any training)
| Finding | Evidence |
|---|---|
| **PhiUSIIL: every legitimate URL is a bare `https://www.<domain>` homepage** | 100% https, 100% `www.`, 0% with a path or query. A one-line rule ("legitimate if it is a bare https://www homepage") is **99.68% accurate** with no machine learning |
| PhiUSIIL: about half of its 54 features need the web page; four use the authors' hidden lookup statistics | Not usable by an app that never opens a page; only its raw URL column is used |
| **PhreshPhish: no benign URL has a query string** (phishing: 5-22%) | A model would learn "has a `?` means phishing". The query and fragment are ignored everywhere |
| PhreshPhish benign URLs are mostly deep pages (91-93% have a path); phishing is often a bare domain | The mirror image of PhiUSIIL: a model learns "no path means phishing" |
| `www.`: legitimate sites 57-61% vs phishing 4-16% (PhreshPhish) and 100% vs 43% (PhiUSIIL) | Looks like a signal; it is an accident of how the legitimate URLs were sampled |
| PhreshPhish: **0%** of benign URLs are on free hosting platforms; 33% of training phishing is | The model will call every `github.io` page phishing (a base rate, not evidence) |
| Only 51.5% of test URLs come from a domain group seen in training (phishing 29%) | Test results are reported overall and on unseen groups |
| Label conventions differ (PhiUSIIL flipped), PhishBench undocumented | Each mapping written down and tested |
| 15,432 duplicates in the training file; 25 invalid URLs | Removed, keyed on host + path |

### 4.3 Splitting
- Train/validation: `StratifiedGroupKFold`, 20 folds (17 train, 3 validation), **grouped by tenant host on free-hosting platforms and by registered domain elsewhere**, so one domain never straddles splits. Leakage checks: 0 shared URLs and 0 shared groups.
- Test: PhreshPhish's own **newer** split, minus 35 URLs also in training: a real temporal hold-out ("train on the past, test on the future").
- **Cross-source group consistency (a leak I found and fixed).** PhreshPhish and PhiUSIIL were first split independently, so a domain could be in one source's training part and the other's validation part: **37.9% of validation rows (44,342) were in a domain group that training also contained**, and validation recall at the 2% budget looked about 10 points better than it was (0.839, truly 0.741). A PhiUSIIL URL now follows its domain group's PhreshPhish assignment, and only new groups are split fresh.
- **External sets filtered against all training data.** They had been cleaned against the PhreshPhish pool only, so **9.3% of CompPhish's and 5.9% of PhishBench's legitimate URLs were also in PhiUSIIL's training data** (1,251 and 54 URLs). Those are now removed, and the training script **stops with an error** if train/validation share a URL or a group, if the test shares a URL with training, or if an external set does.
- The temporal test never shared URLs with training, so its numbers were unaffected (recall 0.842 before, 0.846 after).

## 5. Features (the one function, 44 numbers)

`extract_features(url)` returns the same 44 numbers in the same order for training and for the app (a test proves the sklearn wrapper equals it exactly). Families: length and character counts; host structure (subdomain levels, hyphens, digits, entropy, longest label, punycode, IP host, port); domain ending (common / throw-away list); free-hosting platform flag; URL shortener flag; login-type words and well-known brand names in the host and path; path depth, length, entropy, file extension; `@`, `//`, percent-encoding, long hex strings. The character model instead reads `host/path` as text (character 2-5-grams).

Deliberately **not** features, each learned from a failure:
| Removed | Why |
|---|---|
| query string and fragment | no legitimate URL had one in the training data |
| `www.` (stripped from the host before every feature) | a sampling accident; Round 2 made PhishBench legitimate homepages flagged 99.8% |
| `https` vs `http` | the largest feature in Round 3 (importance 0.16); it made `dell.com` typed without a scheme "phishing" at 0.99, and most phishing now uses https anyway |

## 6. Choosing the algorithm

| Candidate | Why it is on the list |
|---|---|
| Dummy | the floor (PR-AUC = phishing share) |
| Logistic Regression on scaled features | simple, explainable weights |
| Random Forest | handles mixed numeric features with no scaling |
| Gradient boosting (`HistGradientBoosting`) | usually the strongest on tabular features |
| **Logistic Regression on character n-grams of `host/path`** | needs no hand-made features; sees patterns like random-looking subdomains |

**Rules fixed before looking at results:** (1) validation PR-AUC with phishing as the positive class; (2) within 0.003 prefer the simpler model; (3) **guard**: skip any candidate that flags more than 20% of the hand-written legitimate URLs; (4) threshold = the highest validation recall with **at most 2% of legitimate URLs flagged** (missing phishing is the costly error, but a tool that flags many real links is unusable, so the false-alarm budget is fixed first); (5) the temporal test and external sets are read once, after the choice; (6) ablations are diagnostics and never change the choice.

## 7. Four rounds (each fixes a failure that the previous round exposed)

| | R1: PhreshPhish only | R2: + PhiUSIIL | R3: `www` ignored | **R4: https not a feature (final)** |
|---|---|---|---|---|
| Chosen model | char LR | gradient boosting | gradient boosting | **char LR** |
| Temporal test recall / false alarms | 0.904 / 1.6% | 0.862 / 3.0% | 0.782 / 3.6% | **0.846 / 1.8%** |
| Temporal test PR-AUC | 0.988 | 0.968 | 0.947 | **0.979** |
| PhiUSIIL legitimate wrongly flagged | **76%** | trained on | trained on | trained on |
| PhishBench legitimate wrongly flagged | **89%** | **99.8%** | 1.9%† | **15.6%** |
| CompPhish legitimate wrongly flagged | 27% | 18%† | 19%† | **25.4%** |
| OpenPhish 2026 phishing caught | 77% | 86% | 83% | **80%** |
| CompPhish phishing caught | 84% | 87% | 86% | **86%** |
| Hand-written legit URLs flagged | 4 of 52* | 7 of 56 | 6 of 56 | **5 of 56** |
| Hand-written phishing-style caught | 7 of 12 | 12 of 12 | 12 of 12 | **10 of 12** |

†**Contaminated:** Rounds 2-3 were run before the overlap fix, so their PhishBench and CompPhish legitimate false-alarm rates are too good (the overlapping URLs were in PhiUSIIL's training data). Only the Round 4 column is clean, and it is the one to trust; the earlier columns show the trend of each fix, not exact values. Round 1 did not train on PhiUSIIL, so its numbers are clean.

*Round 1 used the original 52 legitimate URLs. The 12 bare homepages were added after Round 1's external results exposed the gap (a blind spot in my own challenge set, as with the email "automated mail" gap); 8 hand-written URLs that also occur in the training data are excluded each time.

How to read it: no round is best everywhere. **Removing a shortcut changes which errors you make, not just how many.** Round 3's gradient boosting handled PhishBench's bare homepages well but flagged 3.6% of legitimate temporal-test URLs; Round 4 is steadier over time but flags more bare no-`www` homepages. Round 4 is the final model because the pre-agreed rule picked it and because its features contain no unreliable signals; I did not keep switching to whichever model scored best on the latest test, which would have turned test sets into tuning sets.

## 8. Final results (Round 4, leak-free: character n-gram Logistic Regression, threshold 0.644)

| Set | URLs | Precision | Recall | Legitimate flagged |
|---|---|---|---|---|
| Validation (PhreshPhish + PhiUSIIL) | 114,650 | 0.970 | 0.741 | 2.0% (by design) |
| **Temporal test, Sep-Dec 2025** | 162,718 | **0.974** | **0.846** | **1.8%** |
| Temporal test, domain groups not seen in training | 75,680 | 0.975 | 0.805 | 3.8% |
| Temporal test, free-hosting URLs only (12,085; only 108 legitimate) | | 0.995 | 0.996 | **55.6%** of the legitimate ones |
| CompPhish (external) | 12,148 | 0.755 | 0.860 | 25.4% |
| PhishBench (external) | 856 | 0.852 | 0.875 | 15.6% |
| OpenPhish live feed (phishing only) | 286 | n/a | **80.4%** detected | n/a |

Validation by source: PhreshPhish recall 0.748 at 2.1% false alarms; PhiUSIIL recall 0.716 at 1.7%. By month (temporal test): Sep recall 0.72 / precision 0.88, Oct 0.86 / 0.98, Nov 0.85 / 1.00. (December has 987 URLs and almost no phishing, so it says nothing.) Model: 11.5 MB, about 3 ms per URL.

**Shortcut ablations** (gradient boosting, diagnostics only): all 44 features: test PR-AUC 0.958; without the free-hosting and shortener flags: 0.956; host only (no path features): 0.951. The path adds little: most of the signal is in the domain name itself.

## 9. Understanding the numbers

- **Recall at a fixed false-alarm budget** is the honest way to report this task. A model can always catch more phishing by flagging more links; the question is how many real links it hurts. We fixed 2% on validation and report what the other sets show.
- **Why the validation recall (0.84) and false alarms differ from external sets:** every dataset has its own idea of "legitimate". PhreshPhish's are popular sites' pages; PhishBench's are the top-ranked domains; CompPhish's include personal blogs and small business sites (the hardest for any URL model, because an unfamiliar legitimate domain looks like a throw-away one).
- **Why free-hosting pages are flagged:** in the data, 99% of URLs on those platforms are phishing (11,977 vs 108 in the test split). The model learned a base rate, not evidence. In real life most pages on github.io are fine. The app says so whenever the platform flag is set.
- **Why a URL model has a ceiling:** an address contains no page content, no domain age, no certificate, no blocklist, no redirect target. Real systems combine this with reputation data and page analysis; this project deliberately does not.

## 10. What the app does

The URL tab takes an address, validates it, shows it as plain text, and returns Potentially phishing / Likely legitimate / Cannot judge with the score and the threshold. Below it are **plain-language notes computed from the address itself** ("numeric IP instead of a domain name", "a well-known brand appears but is not the real domain", "free hosting platform...") and a table of the features read. The notes describe the address; they are not the model's reasoning, and legitimate URLs can trigger some. It explains when it ignored a query or an absent scheme. It shows a "Known weakness" warning built from the model's own challenge results whenever the guard failed (not the case now).

## 11. Difficulties faced

| # | Difficulty | Lesson |
|---|---|---|
| 1 | Both public data styles have a structural shortcut, in opposite directions | A near-perfect score on one dataset proves nothing; train on mixed sources and test on held-out sources |
| 2 | Huge remote files with dangerous content | Read only the columns you need; verify nothing is rendered |
| 3 | A dropped connection lost a long run | Make long jobs resumable (save each unit of work) |
| 4 | Mendeley rejected Python's default user-agent | Servers can treat clients differently; curl's address worked |
| 5 | My challenge set had no bare homepages, so the guard passed misleadingly | A guard set must cover the common cases; extend it when a gap is found and say so |
| 6 | **The saved model pickled a helper class as `__main__.UrlText`, so no other program could load it** | Define everything a model needs in importable modules; test loading in a fresh process (`test_model_files_portable.py` now does this for all five models) |
| 7 | A NumPy string function does not accept a tuple of prefixes (my diagnostics crashed after 15 minutes) | Test the diagnostics code on a tiny input before the long run |
| 8 | Transient "Access is denied" when starting Python | Re-run before debugging code that did not change |
| 9 | Typing `dell.com` without a scheme gave 0.99 phishing | Test the app with real, careless user input, not only tidy URLs |
| 11 | **Train/validation leakage across two data sources, and external tests overlapping training** (found only when I wrote a health check that compared the final data files) | Split once across all sources; compare the *final* training data with every test set; make the training script fail on any overlap |
| 10 | Four training rounds, each ~15 minutes | Say plainly how many times you iterated and why; avoid tuning on the final test |

## 12. Concepts introduced

URL anatomy (scheme, host, registered domain, tenant host, path, query); structural vs semantic features; train/serve parity through one shared function; dataset-specific shortcuts and how to detect them (one-line rules, per-class shape tables, ablations); cross-source and temporal evaluation; recall at a fixed false-alarm rate; grouped splitting by tenant host; column-only remote reading; model files must load in a fresh process.

## 13. Self-check questions

1. Why is a rule that is 99.7% accurate on PhiUSIIL a reason to distrust PhiUSIIL?
2. PhreshPhish and PhiUSIIL have opposite structural biases. Describe each, and why training on both helps.
3. Why did removing `www.` and `https` make the model better even though some scores dropped?
4. Why do we fix a false-alarm budget before looking at recall?
5. Why must the split group by tenant host for URLs on `github.io` or `weebly.com`?
6. Why does the app show "Cannot judge" for bit.ly links?
7. What does "Likely legitimate" mean, and what does it not mean?
8. Why did the model file fail to load outside the training script, and how does the new test prevent it?
9. Why does the temporal test matter more than a random split for a phishing detector?

## 14. Done when

- [ ] `python -m scripts.prepare_phishing`, `python -m scripts.split_phiusiil` and `python -m src.training.train_phishing --with-phiusiil` reproduce the tables
- [ ] You tried a normal link, a bare `dell.com`, a login page, a `github.io` page, a shortened link, `javascript:alert(1)` and a fake phishing-style address in the URL tab
- [ ] You can explain the four rounds table in your own words
- [ ] You answered the self-check questions

## 15. Next

Phase 8: bring all four modules into one finished application (Home page, navigation, shared styling, consistent wording, error handling).
