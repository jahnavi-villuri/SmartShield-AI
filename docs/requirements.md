# SmartShield AI: Requirements (v1)

Status: Phase 0 draft. Numeric targets marked **TBD** are set in the modelling phases, after we see validation results, never before.

## 1. Purpose

A web app where a user pastes a message, URL or comment and gets a machine-learning prediction with a score and a plain-language caveat. Built as a learning project covering the full ML lifecycle.

## 2. Users

- **Primary:** the project author (learning), plus reviewers and recruiters who read the repository.
- **Secondary:** curious visitors trying the public demo.
- Not a security product. Nobody should rely on it to protect them.

## 3. Functional requirements

| ID | Requirement |
|---|---|
| F1 | User selects Email or SMS, enters text, and gets **Spam / Not spam** plus a model score. |
| F2 | Email input has a subject field and a body field. SMS input has one message field. |
| F3 | User enters a URL and gets **Potentially phishing / Likely legitimate**, a score, and the URL features that were checked. |
| F4 | User enters a comment and gets scores for 6 labels (toxic, severe_toxic, obscene, threat, insult, identity_hate). Labels above their threshold are highlighted. |
| F5 | Every result page shows a short explanation and a "predictions can be wrong" warning. |
| F6 | Empty, too-long or malformed input produces a clear message, never a crash. |
| F7 | A missing model file produces a clear message, never a crash. |
| F8 | A Home page explains the system and links to the three detectors. |
| F9 | An About page shows each model's dataset, date, test results and known limitations, read from saved metadata (not typed by hand). |

## 4. Non-functional requirements

| ID | Requirement | Target |
|---|---|---|
| N1 | Prediction latency per request | < 1 s (confirm in Phase 11) |
| N2 | Hardware | CPU only, no GPU, no paid API, no LLM |
| N3 | Privacy | User input is processed in memory and never stored or logged |
| N4 | URL safety | The app never fetches, opens or links a submitted URL |
| N5 | Reproducibility | Models can be rebuilt from raw data with one script per module; fixed seeds; pinned package versions |
| N6 | Honesty | Every reported metric comes from an actual run; no hard-coded results |
| N7 | Model size | Each model file small enough to deploy on a free host (limit **TBD** in Phase 11) |
| N8 | Separation | Training code, inference code and UI code live in separate modules |

## 5. ML problem definitions

| Module | Task type | Input | Output | Label convention |
|---|---|---|---|---|
| SMS spam | Binary classification | message text | score 0..1 | 1 = spam/smishing, 0 = legitimate |
| Email spam | Binary classification | subject + body | score 0..1 | 1 = spam, 0 = legitimate |
| Harmful content | Multi-label classification (6 binary tasks) | comment text | 6 scores | 1 = label applies |
| Phishing URL | Binary classification | URL string | score 0..1 | 1 = phishing, 0 = legitimate |

**Project-wide rule: 1 = bad, 0 = good.** Every loader converts its dataset to this immediately.

## 6. Success criteria

Measured on held-out data, reported honestly:

- Beats the Dummy baseline by a clear margin on every module.
- Spam: precision on the spam class >= **TBD** (false positives are the worse error) with recall reported.
- Phishing: recall on phishing at a false-alarm rate <= **TBD** on legitimate URLs.
- Toxicity: per-label precision/recall/F1 and macro-F1 reported, including weak labels such as `threat`.
- External (modern) test results reported next to in-domain results, even when worse.
- Full test suite passes in CI.

## 7. Out of scope for v1

User accounts, SQLite history, transformer/LLM models, mobile app, browser extension, real-time monitoring of inboxes, fetching or scanning web pages, automatic retraining in production.

## 8. Constraints

Windows 11, Python 3.11 or 3.12, free tools only, one developer, beginner level.

## 9. Assumptions

- Public datasets are old or limited. Results on them overestimate real-world performance.
- Modern legitimate messages are not publicly available, so false-alarm measurement on modern ham relies on the author's own private samples (kept out of Git).
