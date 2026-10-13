# Phase 4 README: The Thin Slice (first working app)

## 1. What we built

A complete path from a typed message to a prediction on screen, for SMS only:

```text
saved model files -> load_sms_detector() [checksum check] -> SpamDetector.predict() [validation, score, threshold, explanation] -> Streamlit page
```

| File | Role |
|---|---|
| `src/detection/spam_detector.py` | All prediction logic: validation, loading, checksum, scoring, explanation. **No Streamlit inside.** |
| `app.py` | Page layout only. Calls the detector. |
| `tests/conftest.py` | A tiny synthetic model so tests never need real data |
| `tests/test_spam_detector.py` | 16 detector tests |
| `tests/test_app_sms.py` | 6 page tests using Streamlit's `AppTest` |

Whole suite: **65 tests pass**, ruff is clean. Checked with the real model: spam example scored 0.99 (Spam), a lunch message 0.00 (Not spam), the page rendered with no exceptions, and a live `streamlit run` server answered HTTP 200.

## 2. Why a "thin slice" this early

Training a model is only half the problem. The other half is using it safely. Building the whole path early exposes problems while they are cheap:

- Phase 3 had already shown one: the model pipeline raises an error on an empty list, so **input validation belongs in the detector, before the model**.
- It confirms the saved file loads in a different process from the one that trained it.
- It forces the separation of training code, inference code and UI code.

**Rule: the app never trains.** Training is a separate script; the app only loads a finished file.

## 3. Design decisions and why

### 3.1 Detector separate from UI
`SpamDetector` can be tested without a browser, reused by a command-line tool or API later, and changed without touching the page. The page only handles buttons and text.

### 3.2 Errors written for users
Every failure raises a `DetectorError` subclass whose message is safe to show:
- `InvalidInputError`: empty, blank, not text, or longer than 1,000 characters.
- `ModelNotAvailableError`: files missing, metadata unreadable, checksum mismatch, load failure.

The page catches `DetectorError` and shows a message; it never shows a stack trace. Missing model = a clear instruction ("Run ... train_sms"), and the form is hidden.

### 3.3 The checksum check (a security habit)
`joblib` files can **execute code when loaded**. Training stores the file's SHA-256 in `metadata.json`; the detector recomputes it and refuses to load on mismatch (tested with a deliberately tampered file). This protects against a swapped or corrupted file. It does not protect against someone who can edit both files, so the model folder must still be trusted.

### 3.4 Load the model once
`@st.cache_resource` loads the model once per server process. Streamlit reruns the whole script on every click, so without caching the model would reload each time.

### 3.5 Input limits
1,000 characters maximum (training messages top out near 900). Limits stop slow requests and abuse. Input is stripped and NUL characters removed. Nothing the user types is stored or logged.

### 3.6 Score wording
The page shows a "Model score", explains that scores at or above the threshold are called spam, and states that real-world spam is rarer than in the training data. It does not say "probability of spam in your inbox", because the training data is 38% spam. The "About this model" panel reads its numbers from `metadata.json`, so the page can never drift away from the real evaluation.

### 3.7 Explanation by occlusion
For messages called spam, the detector removes one word at a time and measures how far the score drops; words with a drop of at least 0.02 are listed (top 3). Why this method: it works for **any** model, including our character n-gram model, whose internal features (2-5 letter chunks) are not readable as words. Its limits are real: a message with several spam signals shows little drop for any single word (our test message showed only "Call"), words are checked one at a time, and it describes sensitivity, not a proof of the model's reasoning. Messages over 40 words are not explained (cost).

## 4. How the tests protect the app

| Risk | Test |
|---|---|
| Empty/blank/None/number/too long input | `test_empty_or_blank_input_is_rejected`, `..._non_text_...`, `..._too_long_...` |
| Missing model crashes the app | `test_missing_model_raises_friendly_error`, `test_app_shows_error_not_crash_when_model_missing` |
| Tampered or broken model files | `test_tampered_model_file_is_refused`, `test_broken_metadata_is_refused` |
| Wrong predictions | spam/ham separation, determinism, unknown words |
| Explanations wrong or too slow | names spam words, none for ham, skipped for long messages |
| Page logic | spam flagged, normal accepted, errors shown, warning always present |

Tests use a **synthetic toy model**, so they are fast and run anywhere, including future automatic checks on GitHub where the real data is not available.

## 5. Difficulties faced

| Difficulty | Resolution |
|---|---|
| `AppTest.from_file("app.py")` looked for `tests/app.py` | Streamlit resolves relative paths against the test file. Use an absolute path from `settings.PROJECT_ROOT` |
| Cached model leaking between tests | Clear `st.cache_resource` before and after each test |
| Testing "model missing" without deleting real files | Monkeypatch `settings.MODELS_DIR` to an empty temp folder |
| Explanation reveals little for character models | Documented as a limit; a word-level model or probability-contribution method could improve it later |
| Streamlit prints a harmless "missing ScriptRunContext" warning when run outside the server | Ignored; it is expected in bare mode |

## 6. Honest limits of this slice

- SMS only. Email, phishing and toxicity come later.
- The model was trained on roughly 2011-2021 data. It missed a "Mum, I lost my phone" scam in our probes. The page says so in "About this model".
- No user feedback, no history, no logging (by design).
- Running locally: `streamlit run app.py` also lists a "Network URL" others on your network could open; keep it local unless you intend to share.

## 7. Concepts introduced

Inference vs training; separation of concerns; caching; input validation; defensive error handling; checksum verification; occlusion explanation; synthetic test fixtures; UI testing with `AppTest`.

## 8. How to select the algorithm for a *serving* problem (extension of Phase 0's lesson)

Accuracy is not the only thing that decides a model. Ask also:
1. **Size and load time:** 5 MB loads instantly; a 500 MB model changes hosting options.
2. **Latency:** about 2.6 ms per message here; fine for interactive use.
3. **Can I explain it?** Linear models can; the occlusion trick helps for others.
4. **Does it fail safely?** Unknown input must give a valid score, not a crash (tested).
5. **Can I verify the file?** Checksums and pinned versions.

## 9. Self-check questions

1. Why must input validation live in the detector and not only in the model?
2. Why does the page refuse to run when the checksum fails?
3. Why is the model loaded with `st.cache_resource`?
4. Why does the page say "model score" instead of "probability of spam"?
5. What are two limits of occlusion explanations?
6. Why do the tests use a toy model instead of the real one?

## 10. Done when

- [ ] `python -m pytest` shows 65 passing
- [ ] `streamlit run app.py` opens in your browser and you can analyse a message
- [ ] You tried an empty message and a very long one
- [ ] You temporarily renamed `models/sms_spam/model.joblib` and saw the friendly error (then renamed it back)
- [ ] You answered the self-check questions

## 11. Next: Phase 5 (Email module)

Parse raw email safely (subject + text body, no attachments or links opened), train on the Enron-Spam data, and measure drift against recent spam collected from untroubled.org.
