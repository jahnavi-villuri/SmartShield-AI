# Phase 9: Security and privacy hardening

## What we did
- Wrote a threat table (`docs/security.md`) with a control and a test for each threat.
- Added `tests/test_security.py` (44 checks): serving code has no network/process imports and no print/logging/eval/open; no `unsafe_allow_html`; hostile text (control characters, script tags, SQL/template strings, huge emoji strings) never crashes a detector; dangerous URLs are rejected; URLs render as plain code; git ignore rules, config and pinned requirements are verified.
- Added `.streamlit/config.toml`: no telemetry, XSRF protection on, error details hidden.
- Ran `pip-audit` on both requirement files: no known vulnerabilities.
- Added a "Responsible use" section to the About tab.
- Fixed `requirements.txt`/`requirements-dev.txt` (a stray byte-order mark in the middle of the file made `pip-audit` fail; both are now clean sorted UTF-8).
- Added `data/labels/` to `.gitignore`: it holds third-party message text and private labels.

## Concepts
- **Threat modelling:** list what an attacker or accident could do, then map each to a control and a test.
- **Defence in depth:** limits in the detector and in the text box, checksum plus trusted build.
- **Pickle risk:** loading a joblib file can execute code, so only load files we trained.
- **Static checks as tests:** an AST scan fails the build if someone later adds `print(user_text)`.

## Difficulties
- My first import scan flagged `urllib.parse`, which only splits strings. The rule now forbids `urllib.request` but allows `parse`.
- The audit tool crashed on the file, which revealed the stray BOM: the checks found a real bug.

## Self-check
1. Why does a checksum stored next to the model not stop a determined attacker?
2. Why is a URL displayed in a code block instead of a link?
3. What would the AST test catch that a code review might miss?

## Done when
All tests pass, pip-audit is clean, docs/security.md lists residual risks honestly.


## Follow-up: acting on risks that had already happened
| Risk (realised) | What we tried | Result |
|---|---|---|
| Email flags modern automated mail | Added the darkknight25 modern email dataset (half trained at low weight, half held out) | False alarms on my 24 modern legitimate emails 33% to 25%; **real 2026 spam detection fell 77% to 69%**; guard passes only on its limit; app warning kept |
| Comment model reads identity words as toxic | **Round 4:** 25 extra short-form neutral sentence shapes per identity term (4,375 synthetic rows in total) | **No improvement** (gay still 50%, same 3 terms over the 5% limit; macro-F1 0.569 to 0.564). Rejected under the rule fixed beforehand (accept only if fewer terms are over the limit and macro-F1 drops by at most 0.01). Old model restored and checksum verified; round-4 metadata kept in `models/toxicity/round4_rejected/` |
| Rapid repeated requests (residual security risk) | Per-session sliding-window rate limit | Added and tested; does not stop multi-session abuse |
| Untrusted model file | Checksum already verified before load | Unchanged; it detects corruption, not an attacker who can change both files |
| Free-hosting URL false alarms; negation in comments | Not attempted | Need legitimate hosted-page data or a different model type; left as documented limits |

Lesson: more synthetic examples of the same kind stopped helping. The bias comes from how real comments use those words, which counter-examples written by me cannot undo. Files in `reports/metrics/toxicity_*` were overwritten by the rejected round-4 run (nearly identical numbers); the model and `metadata.json` in use are the earlier ones.
