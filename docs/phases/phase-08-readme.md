# Phase 8: Application and user experience

## What we built
- **Home tab:** what the app does, a plain-language "how it works", the privacy statement, and a live status line per detector.
- **About tab:** for each model, the algorithm, training date, training size, headline result and limitations, all read from `models/*/metadata.json`.
- **Unchanged:** the four detector tabs and their per-tab error handling. A missing model breaks only its own tab.
- **New code:** `src/detection/model_info.py` (UI-free, unit-tested); `tests/test_app_home.py`.

## Why (concepts)
- **Separation of UI and prediction code:** `app.py` only draws; facts come from `src/detection/`. That is why the page can be tested with AppTest and no browser.
- **Single source of truth:** a figure shown to users must come from the file that produced it. If it is retyped, it will drift. A test changes the training size in metadata and checks the page follows.
- **Honest UX:** the "predictions can be wrong" warning is always visible; limitations are shown, not hidden in a README.
- **Graceful failure:** unreadable metadata means "model not available", not a stack trace.

## Difficulties
- Existing tests assumed the tab order SMS/Email/Comments/URL; three were updated to the new order.
- A quoting slip (escaped newlines written as real ones) broke app.py once; ruff caught it immediately.

## Results
260 tests pass, 4 expected failures (known weaknesses), linter clean.

## Self-check
1. Why read numbers from metadata.json instead of writing them in app.py?
2. What does the app show if one model file is missing? Which tabs still work?
3. Why is the Home tab not a separate Streamlit page?

## Done when
Home and About render with all models and with none; all tests pass; README and decisions updated.
