# SmartShield AI: Decision Log

Format: **context -> options -> decision -> why -> consequences.** Add an entry whenever you choose between real alternatives. Future you (and reviewers) will thank you.

## D-001: Drop the Kaggle 3,000-column email dataset
- **Options:** use as an extra experiment; drop.
- **Decision:** drop.
- **Why:** 5,172 rows matches the Enron1 subset, so it is likely not independent; its vocabulary is unknown so the app could never feed it raw email; it adds a data format with little to learn.

## D-002: Build SMS first, then email, toxicity, phishing
- **Why:** SMS data is small and clean, so the whole workflow is learned cheaply. Phishing has the most traps, so it comes last.

## D-003: Use only the raw URL column of PhiUSIIL
- **Why:** about half of its 54 features need the page HTML or hidden lookup tables the app cannot reproduce. One shared feature function guarantees training/serving parity.

## D-004: Group-split phishing data by domain
- **Why:** many URLs share a host. A random split would put the same domain in train and test and reward memorising.

## D-005: One label convention, 1 = bad, 0 = good
- **Why:** datasets disagree (PhiUSIIL 1 = legitimate, CompPhish 1 = phishing). Converting at load time plus a unit test prevents silent inversion.

## D-006: Train on consistent-era data, test on modern data
- **Why:** mixing 2026 spam with 2002 ham teaches "modern = spam". External validation measures real-world drift honestly.

## D-007: Traditional ML first, no deep learning or LLM in v1
- **Why:** text and URL tasks here are well handled by linear models; they train in seconds on CPU, are explainable and small. Transformers are a Phase 16 comparison.

## D-008: Skip SQLite history
- **Why:** adds privacy responsibility and teaches no ML.

## D-009: Python 3.11 or 3.12
- **Why:** best wheel availability for scikit-learn and Streamlit on Windows.

## D-011: Use Python 3.14 (installed) instead of 3.11/3.12
- **Context:** D-009 preferred 3.11/3.12, but only 3.14.3 is installed.
- **Options:** install 3.12 via winget; use 3.14.
- **Decision:** use 3.14 for now.
- **Why:** a dry run resolved every required package, and all 10 smoke tests and the Streamlit check passed. It avoids changing the system.
- **Consequences:** if a later library or the hosting platform lacks 3.14 support, install 3.12 and recreate `.venv` (`pip install -r requirements-dev.txt`). Pinned versions make this safe.

## D-012: Pool UCI + Mendeley + ScamShield `Smishing_Dataset` (74,364 unique messages)
- **Context:** inspection showed 84% of Mendeley's unique messages are also in UCI. A search for newer data found no public 2025-2026 ham+spam SMS set; the largest real, English, labelled option was the ScamShield `Smishing_Dataset` part (69,070 rows).
- **Options:** (a) UCI + Mendeley only; (b) pool all three after deduplication; (c) use the large set only.
- **Decision:** (b). Synthetic rows and non-English rows are excluded.
- **Why:** more and more varied data; deduplication removes the overlap. The cost is that about 92% of the pool is one source whose per-row origin is unknown.
- **Consequences:** pooled spam share is 38% (unrealistic), label noise exists, and its licence chain was not verified.

## D-014: Hold out a different-source external test set
- **Decision:** `Indian_Telecom_SMS` + `Benchmark_Ground_Truth` (English), deduplicated against the pool, are never used for training or tuning.
- **Why:** an in-domain test cannot show generalization; this set differs in region and period. Its "spam" is mostly telecom promos, so it is a stress test, not a verdict.

## D-013: Split with StratifiedGroupKFold on masked-text groups
- **Why:** 519 near-duplicate template groups (1,646 rows, largest 28) exist; a plain stratified split could place siblings in train and test.

## D-015: Add artifact-removing text normalization (Round 2)
- **Context:** Round 1 diagnostics showed the model learning dataset artifacts: HTML-code messages were 0% spam, `XX`-masked messages 98% spam.
- **Decision:** `normalize_sms` (decode HTML, map number-like tokens and `<#>` to `0`, lowercase) is run inside the pipeline; both raw and normalized variants were trained and compared.
- **Honest disclosure:** the candidate set was extended *after* seeing Round 1 results; the selection rule did not change. The external set was therefore looked at in both rounds, so its Round 2 numbers are not a clean, single-look test.
- **Result:** external PR-AUC 0.9205 -> 0.9364, in-domain unchanged. Style shortcuts remain (see Phase 3 README).

## D-016: Model selection and threshold rules (fixed before Round 1)
- Select by validation PR-AUC; within 0.003 prefer simple models; threshold = highest validation recall at validation precision >= 0.98 (false positives are the worse error).
- **Selected (Round 3):** `logreg_char_norm` (C=10), threshold 0.6503.

## D-018: Template-written training augmentation for bank/transactional messages (Round 3)
- **Context:** the user noticed that legitimate messages like "Rs 300 credited to your account" are not spam. A 33-message hand-written challenge set confirmed 18 of 23 legitimate messages were flagged.
- **Options:** (a) ignore it and document; (b) add rules (e.g. whitelist "credited"); (c) add balanced synthetic training examples; (d) find real transactional ham.
- **Decision:** (c). 1,500 legitimate + 1,500 scam template messages sharing the same vocabulary, **training split only**. Real transactional ham was not available; a whitelist rule would be trivially exploited by scammers.
- **Consequences:** fixes most of the blind spot (18 -> 4 of 23) without hurting the real test (precision 0.984, recall 0.951). The fix is only as realistic as my templates, and the challenge set is small and written by me.

## D-019: Show three categories (Legitimate / Promotional / Scam) with a two-stage detector
- **Context:** the project owner reported a genuine Taco Bell offer flagged as "Spam". By the training data's definition, marketing is spam; for the app, annoying and dangerous are different things.
- **Options:** keep binary spam (explain it); add a Promotional category; flag only scams.
- **Decision (owner's choice):** add a Promotional category. Stage 1 = legitimate vs unwanted (existing model); Stage 2 = promotional vs scam, trained on hand-labelled data.
- **Why two stages and hand labels:** the dataset's `head2_scam_intent` labels are unreliable (phishing tagged telemarketing, promos tagged UPI fraud), so a 3-class model trained on them would mislabel promotions as fraud.
- **Consequences:** Stage 2 rests on 451 labels from one AI labeller, not human-verified; its test set has 19 scams. Documented in the app and `docs/sms_type_labeling_guide.md`.

## D-020: Keep the Round 4 main model although it is not clearly better than Round 3
- **Context:** Round 4 added more varied bank-alert templates. In-domain unchanged, 3 of 23 challenge legitimate messages flagged (was 4), but external recall fell from 0.863 to 0.806, and the threshold moved from 0.65 to 0.78.
- **Decision:** keep Round 4 (latest; differences partly noise). Round 3's model file was not archived, only its metrics.
- **Open problem:** the owner's real SBI debit alert is still flagged. It cannot be honestly fixed with more templates written from that one example; real legitimate messages are needed (see Phase 3b README, section 10).

## D-021: Use the owner's real private messages as scarce, hash-split training and test data
- **Context:** all SBI/YONO messages in the public data are phishing; genuine ones were flagged as scams. Synthetic templates could not fix it honestly.
- **Decision:** a private CSV (ignored by Git) of the owner's own received messages. A hash of the text assigns each message to a train or test half; the train half is repeated 30 times.
- **Result:** held-out messages right 8 -> 10 of 13; the reported SBI alert is no longer flagged (0.63 vs threshold 0.70, thin margin).
- **Caveat:** only 7 messages were in the training half; more are needed.

## D-022: Email: train on Enron only; 2026 spam is a test set, never training data
- **Why:** there is no modern legitimate email; mixing 2026 spam with 2002 ham would teach "modern = spam".
- **Consequence:** the 2026 set measures detection only; false alarms on modern mail rest on a 34-email hand-written set and the owner's future private emails.

## D-023: Email "visible text" drops CSS-hidden elements
- **Context:** 73% of the 2026 spam hides a harmless filler paragraph with inline CSS. Reading it made the model look much worse (32% detection) than it is (77%).
- **Decision:** the parser drops elements hidden by inline styles (`display:none`, `font-size:0/1px`, `opacity:0`, `mso-hide:all`, zero-size overflow-hidden boxes, `hidden`). Stylesheet-based hiding is not covered.

## D-024: Selection guard for email, and its failure
- **Rule fixed before results:** skip a candidate that flags more than 25% of 24 hand-written modern legitimate emails.
- **Outcome:** every candidate failed (29-42%). The pre-agreed fallback chose `logreg_word_masked`; the app shows a standing "known weakness" warning driven by `guard_passed = false` in the model metadata.
- **Next step that actually helps:** the owner's real legitimate emails (private pipeline ready in `data/private/my_email/`).

## D-025: For the email model, promotional mail is not spam
- Mapping for private emails: legitimate and promotional -> 0, spam -> 1. (The SMS app shows Promotional as its own category; the email model has only Spam / Not spam.)

## D-026: Toxicity: one binary classifier per label with per-label thresholds
- **Why:** the six labels overlap, so "multi-class" would lose meaning; each label has a very different frequency (toxic 10%, threat 0.3%), so each needs its own threshold.
- **Evidence:** macro-F1 0.518 with the default 0.5, 0.642 with per-label thresholds (internal test); `threat` F1 0.23 to 0.56.
- **Threshold rule (fixed before results):** maximise F1 on pooled validation + 3-fold out-of-fold training predictions, because `threat` has only ~70 positives per split.

## D-027: Toxicity: word + character TF-IDF over word-only
- Validation macro PR-AUC 0.674 vs 0.647 (tolerance 0.003). Cost: 450,000 features, 36.8 MB, about 10 ms per comment. Characters help with disguised spellings.

## D-028: Official Jigsaw test set is rebuilt by joining two blocks by id; only scored rows are used
- The Hugging Face "test" split stacks a text-only block and a label-only block, and uses -1 for unscored rows. Only the 63,978 scored rows are a valid test set (63,908 after removing duplicates and overlap with training).

## D-029: Identity-bias mitigation by counterfactual augmentation, and its limits (3 rounds)
- **Rule fixed before results:** no identity term may have more than 5% of its probe sentences flagged. The probe uses templates and identity words that the augmentation never uses.
- **Outcome:** terms over the limit 12 to 5 to 3; worst term "gay" 83% to 67% to 50%; no accuracy cost (official macro-F1 0.569 in every round). **Still fails the rule**, so the app shows a standing "Known weakness" warning built from the model's metadata.
- **Ethics decision:** this model must only queue comments for human review, never remove them automatically.

## D-030: Offensive text stays out of committed files
- Reports contain counts only. Error-analysis files with comment text are written to the Git-ignored data folder. Tests and probes contain no slurs or hate speech.

## D-031: Phishing: PhreshPhish is the main data source; PhiUSIIL moves from "primary" to a mixed training source
- **Context (changes D-003):** PhiUSIIL's legitimate URLs are all bare `https://www.<domain>` homepages (a one-line rule scores 99.68%), and half of its features need the web page. PhreshPhish (2024-25, benign URLs with real paths, newer test split) has the opposite shape.
- **Decision:** train on both so that no URL shape identifies the class; test on held-out sources (PhishBench, CompPhish, OpenPhish) and on PhreshPhish's newer split.

## D-032: Read huge datasets column-by-column with HTTP range requests; never download HTML
- PhreshPhish: 126 MB transferred instead of 36.6 GB; CompPhish: 636 KB instead of 711 MB. No phishing page content is ever stored.

## D-033: The query string, `www.` and the http/https scheme are not model features
- **Evidence:** no benign URL had a query string; `www.` made PhishBench's legitimate homepages 99.8% flagged; `https` was the biggest feature (importance 0.16) and made a bare `dell.com` "phishing" at 0.99.
- **Consequence:** 44 features; the detector reports the scheme as a note only when the user typed `http://`.

## D-034: Phishing threshold = highest validation recall with at most 2% of legitimate URLs flagged
- Fixed before results. Missing phishing is the costly error, but a tool that flags many real links is unusable.

## D-035: Selection guard on hand-written legitimate URLs, and its gap
- Candidates flagging more than 20% of 56 hand-written legitimate URLs are skipped. The first version had no bare homepages (a blind spot exposed by external evaluation); 12 were added and any hand-written URL also in the training data is excluded.

## D-036: Four phishing rounds; Round 4 chosen by the pre-agreed rule, not by the latest test
- Round 4 (char n-gram LR): temporal test recall 0.842 at 1.7% false alarms; PhishBench legitimate false alarms 14.9% (Round 3 had 1.9%). No round wins everywhere; the pre-agreed rules, not the latest external set, decide.

## D-037: Every saved model must load in a fresh process (test added)
- A helper class defined in the training script pickled as `__main__.UrlText`, so the app could not load the model. Helpers live in importable modules; `tests/test_model_files_portable.py` loads all five real models in a fresh interpreter.

## D-038: Leakage fix across data sources (phishing), and permanent guards against it
- **Found by:** a health check comparing the final training/validation/test/external files. 37.9% of validation rows were in a domain group also present in training (two sources split independently); 9.3% of CompPhish's and 5.9% of PhishBench's legitimate URLs were in PhiUSIIL training.
- **Fix:** PhiUSIIL URLs follow the group assignment already made for PhreshPhish; externals are filtered against all training data. The training script asserts no URL/group overlap between train and validation, no test URL in training, and no external URL in any training data.
- **Effect:** validation recall at the 2% budget 0.839 to 0.741; the temporal test is unchanged (0.842 to 0.846); external legitimate false-alarm rates rose by 2-3 points. Round 2-3 external numbers in the docs are marked as contaminated.

## D-017: List pyarrow explicitly
- It is needed to read the ScamShield parquet files (already installed as a Streamlit dependency); now listed in `requirements.in` so the data loader does not depend on that by accident.

## D-010: Always include a Dummy baseline
- **Why:** gives the "zero-skill" score every real model must beat. On imbalanced data, accuracy alone is misleading.

## D-039: Home and About pages read model facts from metadata.json
- **Decision:** tabs are Home, SMS, Email, Comments, URL, About. `src/detection/model_info.py` summarises each model from its `metadata.json` without loading the `.joblib`; a missing or corrupt model is shown as "not available", never an exception.
- **Why:** numbers typed into the UI go stale after retraining (this already happened in the docs). The warning "Predictions can be wrong" stays above all tabs, and the privacy statement (not stored, links never opened) is on Home.
- **Rejected:** `st.navigation` multi-page app (more moving parts, the page-level warning would need repeating, and tabs keep a single cached model loader).

## D-040: Security rules are enforced by tests, not only documented
- **Decision:** static (AST, config, .gitignore, requirements) and hostile-input tests live in `tests/test_security.py`; threats and residual risks are in `docs/security.md`.
- **Why:** documented rules drift; a failing test stops a later change from logging input or opening links.
- **Not done:** rate limiting and signed model files (need infrastructure; listed as residual risk).

## D-041: Comment-bias round 4 rejected; email gets low-weight synthetic modern data
- **Comments:** short-form identity sentences gave no gain, so the previous model stays. Rule fixed before training: accept only if fewer identity terms exceed the 5% limit and official-test macro-F1 falls by no more than 0.01.
- **Email:** darkknight25 data (synthetic, one-sentence bodies) is trained at weight 5x on half of 200 rows; the other half is a held-out test. The app warns whenever more than 15% of the modern challenge emails are flagged, not only when the guard fails.
- **Rate limit:** per-session sliding window; reduces the D-040 residual risk, does not remove it.

## D-042: CI gates and what they cover
- **Gates:** `ruff check` and `ruff format --check`; repo-hygiene script; pytest on Ubuntu and Windows with `--cov-fail-under=90`; `pip-audit` on every push and weekly.
- **Coverage scope:** app + serving code + data loaders. Training scripts, plots, big-file download and the inspection helper are omitted on purpose: they are exercised by running the pipeline, and unit-testing them would mostly test pandas.
- **Real models:** `.joblib` files are not in Git, so tests that need them skip. A local CI simulation (copy without data and models) gave 272 passed, 47 skipped, 94% coverage.
- **Hygiene:** error-analysis CSVs quote third-party text, so they are git-ignored and the hygiene script rejects them, along with model binaries, datasets, secrets and files over 2 MB.

## D-043: Interface built with Streamlit's own HTML/CSS channel, no external assets
- **Decision:** one stylesheet in `src/ui/theme.py` sent through `st.html`; building blocks in `src/ui/components.py` that escape every dynamic string; illustrations as SVG documents embedded as base64 `<img>`; Altair for charts; dark theme in `config.toml`.
- **Why:** `st.html` keeps the sanitiser on, so we never need `unsafe_allow_html`. A first version with inline `<svg>` rendered blank because the sanitiser removes it, which was found only by screenshotting the running app in a browser (AppTest cannot see layout).
- **Rejected:** web fonts and CDN images (third-party requests, privacy), custom JavaScript components (larger attack surface), a different front-end framework (out of scope).
- **Accessibility:** alt text on images, native Streamlit alerts kept beside the custom cards, `prefers-reduced-motion` support.

## D-044: Dashboard honesty rules and performance
- **Rules:** every number is read from `metadata.json` or the benchmark CSV; the same-source vs different-source charts are first-class (they show where the models are weaker); known weaknesses appear as measured numbers; session tally holds counts only.
- **Performance:** the Dashboard renders only when its tab is open, because Streamlit otherwise reruns every tab's code on every click (about 2.5 s per click before this fix).
- **Reproducibility:** `scripts/rebuild_all.py` lists the pipeline in order, checks raw inputs and refuses to run if they are missing; `scripts/benchmark_models.py` writes a measured speed/size table.

## D-045: Crimson Noir is the app's theme
- **Decision:** the owner chose Crimson Noir from six rendered previews (the original blue plus five alternatives). Colours now come from one palette (`PALETTES` in `src/ui/theme.py`); the other palettes were removed so the app has one look and no dead options.
- **Details:** accents are crimson `#f43f5e`, silver `#d4d4d8` and rose `#fda4af` on `#09090b`. Status colours (green, amber, red) are unchanged so meaning never depends on the theme. Streamlit's own blue info boxes are restyled to neutral silver, and the primary button is crimson to deep crimson for contrast with its white label.
- **Guard:** a test fails if any of the old blue colours reappear in `src/ui/`.
- **Method:** previews were real browser screenshots of the running app, not mock-ups, which also exposed two theme bugs (navy text boxes, blue info boxes).

## D-046: Model cards are generated; dataset cards and concepts are written
- **Decision:** `scripts/make_model_cards.py` writes `docs/model_cards/*.md` from each `metadata.json`; `tests/test_docs.py` fails if a committed card differs from what the script would write. Dataset cards and `docs/concepts.md` are written by hand because they describe sources and ideas, not model files. A link-checking test covers every Markdown file.
- **Why:** numbers typed into documents drift (it happened with the README before the health check existed). Generated cards cannot disagree with the model they describe.
- **Licence:** MIT for the code and docs. Datasets are not redistributed and keep their own licences; the dataset cards list them and flag the unverified ScamShield chain and the CC BY-SA text in Jigsaw. The copyright holder named in `LICENSE` is the project owner and can be changed.
- **Honesty rules for cards:** every card has Intended use, Out of scope, Known limitations and Ethical considerations; weaknesses found in earlier phases are stated, not softened.
