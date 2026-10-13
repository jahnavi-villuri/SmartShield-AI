# SmartShield AI: Risk Register

Likelihood and impact: L(ow) / M(edium) / H(igh). Review at the end of every phase.

| # | Risk | Likelihood | Impact | Mitigation | Phase |
|---|---|---|---|---|---|
| R1 | Old datasets do not reflect 2026 messages, so the model misses modern spam | H | H | External modern test sets (untroubled.org 2026, SmishTank, CompPhish); report both numbers; label model age in the UI | 3, 5, 7 |
| R2 | Data leakage (duplicates or related items across splits) inflates scores | H | H | Dedup before split; group split for URLs; tests that check splits share no rows | 2, 5, 7 |
| R3 | Source bias: model learns where data came from, not what spam is (Enron names; PhiUSIIL homepages vs paths) | H | H | Inspect top features; mask-and-retrain experiment; deep-URL sanity list | 5, 7 |
| R4 | Label convention mixed up (PhiUSIIL 1=legit, CompPhish 1=phishing) | M | H | Convert at load; unit test per loader | 2, 7 |
| R5 | Toxicity model biased against identity terms | H | H | Bias template test set; document results; UI warning | 6, 9 |
| R6 | Users over-trust "Likely legitimate" | M | H | Careful wording, never "safe"; disclaimers on every result | 8, 9 |
| R7 | Rare classes (e.g. `threat`) perform poorly | H | M | Per-label thresholds; report honestly; avoid claiming coverage | 6 |
| R8 | Training/serving mismatch (URL features computed differently in app) | M | H | One shared feature function; parity test | 7 |
| R9 | Loading an untrusted model file executes code | L | H | Only load self-built models; store and verify SHA-256 | 9 |
| R10 | Model file too large to deploy | M | M | Cap `max_features`; measure size early | 6, 11 |
| R11 | scikit-learn version change breaks saved models | M | M | Pin versions; record version in metadata; rebuild script | 1, 11 |
| R12 | Handling malicious samples (spam emails, phishing URLs) | M | H | Treat as text only; never open links or attachments; never fetch URLs; parse only subject and text body | 5, 7 |
| R13 | Private data or datasets committed to Git | M | H | `.gitignore` first; check `git status` before every commit | 1 |
| R14 | Scope creep stalls the project | H | M | Out-of-scope list; one module at a time; milestone tags | all |
| R15 | Synthetic data (e.g. ChatGPT-extended sets) mistaken for real | L | M | Never use synthetic data as a test set; document provenance | 2 |
| R16 | Dataset license misunderstood | M | M | Record license in each dataset card; cite as required | 2, 12 |
| R17 | Offensive text in Jigsaw affects the developer | M | L | Warn in docs; sample in small batches; take breaks | 6 |

## Updates from Phase 6 (harmful comments)

| # | Risk | Status |
|---|---|---|
| R5 | Toxicity model biased against identity terms | **Realised.** Bias probe: "gay" flagged in 50% of neutral sentences after 3 mitigation rounds (was 83%). Mitigation partial; warning shown in the app; human review only |
| R7 | Rare labels perform poorly | **Realised.** severe_toxic F1 0.36-0.49, threat 0.50-0.56, identity_hate 0.47-0.56; per-label thresholds help but do not solve it |
| R10 | Model file too large | 36.8 MB; fine locally, to be checked in Phase 11/13 |
| R17 | Offensive text affects the developer or leaks into the repo | Reports contain counts only; error files are Git-ignored; probes contain no slurs |
| R18 | Download truncated or corrupted | **Realised once** (train file 27.7 of 44.1 MB); caught on load; fixed by verifying exact byte size |
| R19 | Mis-structured public data (stacked blocks, -1 sentinels) silently corrupts metrics | **Realised.** Test split rebuilt by joining on id; only scored rows used |

## Updates from Phase 7 (phishing URLs)

| # | Risk | Status |
|---|---|---|
| R4 | Label convention mixed up (PhiUSIIL 1 = legitimate) | Flipped explicitly and documented; PhishBench's undocumented label verified by reading URLs |
| R3 | Source bias: model learns the dataset's URL shape | **Realised twice, in opposite directions** (PhiUSIIL: bare www homepages are legitimate; PhreshPhish: deep pages are legitimate). Mitigation: mixed sources, held-out external sets, `www`/scheme/query removed from features |
| R8 | Training/serving mismatch | One shared feature function; a test proves the sklearn wrapper equals it exactly |
| R9 | Loading an untrusted/unportable model file | SHA-256 check before load; **a fresh-process load test now exists after a `__main__` pickling bug** |
| R12 | Handling malicious URLs/pages | URLs are text only; defanged when printed; HTML never downloaded; no network access during prediction (tested) |
| R20 | False reassurance ("this site is safe") | Wording never says safe; standing caption; free-hosting and shortener caveats; a URL model cannot see page content |
| R21 | Free-hosting platforms: training base rate (99% phishing) mistaken for evidence | Known weakness: 55% of legitimate hosted pages flagged; caption and warning in the app |

## Phase 8-9 updates
| # | Risk | Status |
|---|---|---|
| R5 | Toxicity identity bias | **Still realised.** Round 4 (short-form neutral sentences) rejected: no change (gay 50%). Warning stays; do not use for automatic moderation |
| R6 | Users over-trust a result | Responsible-use text on the About tab; standing warning on every tab |
| R9 | Untrusted model file | Checksum verified before load; documented that it does not stop an attacker who can change both files |
| R22 | Email false alarms on modern mail | 33% to 25% after adding modern data; detection of real 2026 spam fell 77% to 69%; warning shown above 15% |
| R23 | Request flooding | Per-session rate limit added; multi-session abuse needs host-level limits |
| R24 | Broken dependency file hid the vulnerability scan | Stray byte-order mark fixed; `pip-audit` clean on both files |
