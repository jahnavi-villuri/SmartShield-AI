# Security and privacy notes

SmartShield AI processes text that may be hostile (scams, phishing links, abusive comments). This page lists what could go wrong and what the project does about it. Each control is checked by `tests/test_security.py` unless stated otherwise.

## Assets and threats
| Threat | Control | Checked by |
|---|---|---|
| A pasted link is opened or fetched (SSRF, malware) | URLs are only parsed as text; serving code imports no network or process modules; URL shown in a code block, never a link | import scan, AppTest |
| Pasted HTML/script runs in the page | `unsafe_allow_html` is never used; HTML in emails is reduced to visible text | static check, hostile-input tests |
| Oversized or malformed input exhausts memory or crashes | Per-field limits (SMS 1,000, subject 300, body 20,000, comment 5,000, URL length) plus the text boxes' own `max_chars`; control characters, empty input, `javascript:`/`file:` URLs and bad hosts get a friendly error | hostile-input tests |
| User text is logged or stored | No `print`, logging, file writes or databases in serving code; Streamlit telemetry off; error details hidden | AST scan, config test |
| A tampered model file is loaded (pickle can run code) | SHA-256 of each `model.joblib` is stored in `metadata.json` and verified before loading; models are produced only by our training scripts | detector tests |
| Private data leaks through Git | `data/private`, `data/raw`, `data/processed`, `data/external_test`, `data/labels`, `models/**/*.joblib`, `.env` are ignored | .gitignore test |
| Vulnerable dependencies | Pinned `requirements.txt`; `pip-audit` run in Phase 9: no known vulnerabilities in either requirements file | manual, repeat before release |
| Secrets in source | None used; a scan test guards against adding some | source scan |
| Rapid repeated requests from one session | Sliding-window limit (30 analyses per 60 s per session, `src/detection/rate_limit.py`); the app shows a wait message | `tests/test_rate_limit.py` |
| Injection through the new HTML/SVG blocks | Only static markup and escaped text reach `st.html` (`src/ui/components.py`); SVG art is embedded as `<img>` data URIs, which cannot run scripts; no external fonts, scripts or images; tests parse the output for script tags and `on*` attributes | `tests/test_ui_components.py`, `tests/test_security.py` |
| Cross-site request forgery | Streamlit XSRF protection enabled in `.streamlit/config.toml` | config test |

## Residual risks (not solved)
- **Model files are trusted:** the checksum is stored beside the file, so it detects accidental corruption, not an attacker who can change both. Deployment must ship models from a trusted release.
- **Adversarial text:** anyone can rewrite a message to evade the models. The app says so.
- **Denial of service:** the rate limit is per browser session, so a script that opens many sessions bypasses it. Hosting provider or reverse-proxy limits are still needed when the app is public.
- **Hostile data in the repo history:** raw datasets contain real spam and phishing text and are never committed.
- **Bias:** the comment model misreads identity terms (see the About tab); it must not moderate people automatically.

## Responsible use
A flag is a prompt to look closer, never proof. Do not use scores to punish or report people. A "safe" result is not a guarantee.

## Dashboard tally (Phase 11)
The Dashboard keeps counts of verdicts per detector in the browser session's memory (for example `SMS / Scam: 2`). It never stores the analysed text, it is not written to disk and it ends with the session. A test checks that a secret input does not appear in it.
