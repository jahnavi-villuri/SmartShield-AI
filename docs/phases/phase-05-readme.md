# Phase 5 README: The Email Module

> All numbers come from real runs of `python -m scripts.prepare_email` and `python -m src.training.train_email` (seed 42). Tables: `reports/metrics/email_*.csv`. Data report: `reports/data_quality/email.md`.

## 1. Result in one paragraph

On held-out data of the same kind as its training data, the email model is almost perfect (precision **0.984**, recall **0.989**). On **real spam from June to October 2026** it catches **77%** of 10,228 emails. On 24 hand-written, ordinary modern emails (statements, verification codes, receipts, newsletters) it wrongly flags **8 of 24 (33%)**. The near-perfect test score is mostly the model recognising "Enron staff email vs everything else", not a general ability to spot spam. The app therefore shows a standing warning on the Email tab. This module is a strong demonstration of *why a high test score is not enough*, not a usable spam filter.

## 2. What we built

| Piece | File |
|---|---|
| Safe raw-email parser (text only, hidden-text aware) | `src/data/email_parse.py` |
| Text preparation shared by training and the app | `src/features/email_text.py` (`join_email`, `normalize_email`, `SignatureMasker`) |
| Data pipeline + data-quality report | `scripts/prepare_email.py` |
| Training (compare, select with a guard, evaluate, save) | `src/training/train_email.py` |
| Detector (validation, checksum load, score, explanation) | `src/detection/email_detector.py` |
| Shared detector code (errors, verified loader) | `src/detection/common.py` |
| Private-email loader + report | `src/data/private_email.py`, `scripts/evaluate_my_email.py` |
| Modern hand-written challenge set | `tests/data/email_challenge.csv` (24 legitimate, 10 spam) |
| UI | Email tab in `app.py` (subject + body) |
| Tests | `tests/test_email_detector.py`, `test_app_email.py`, `test_private_email.py` |

## 3. Choosing the data (and what we refused to do)

| Dataset | Role | Period |
|---|---|---|
| Enron-Spam (33,716 emails; Metsis, Androutsopoulos, Paliouras; CSV by M. Wiechmann; repo licence GPL-3.0) | train / validation / test | ham 1999-2002, spam 2001-2005 |
| untroubled.org spam archive, June-October 2026 (11,251 files) | **external test, spam only** | 2026 |
| Hand-written modern emails (24 legitimate, 10 spam) | smoke test of false alarms | written for this project |
| Owner's own emails (`data/private/my_email/`, none yet) | train/test halves when supplied | modern |

**What we refused to do:** train on the 2026 spam together with the Enron ham. That is the "modern text = spam" mistake from the start of this project: the model would learn the *era*, not spam. Without modern legitimate email, 2026 spam can only be a test set.

## 4. What the inspection found

1. **Dates give the label away.** Ham: 1999-12 to 2002-03. Spam: 2001-05 to 2005-09. 15,796 spam emails are dated 2003 or later; no ham email is. Model features never include the date.
2. **Source words identify the class.** "enron" is in 45% of ham and 0.01% of spam; "ect", "hou", "vince", "kaminski" behave the same. The unmasked model's top ham words are `enron`, `vince`, `louise`, `houston`, `energy`, `gas`, `meter`, `deal`.
3. **Duplicates.** 3,527 removed (after dropping 51 rows with empty subject and body): 33,716 to 30,138. Spam campaigns repeat heavily; one near-duplicate group has 449 emails. Groups are kept together in one split.
4. **Tokenized text.** The data is lowercased with spaces before punctuation ("vince , thank you"); real pasted emails are not. `normalize_email` makes both look alike, and `join_email` builds the identical string at training and prediction time.
5. **Label noise.** Some "ham" emails read like spam ("get a $ 25 certificate just for responding to this e - mail", dated 2000) and some "spam" has empty text. Public labels are not ground truth.
6. **Perfect scores are a warning.** Every model reached validation PR-AUC 0.998-0.9995 within the first minute.

## 5. What the 2026 spam taught us (read this part)

Real spam in 2026 uses tricks that did not exist in the training data:

| Trick | How common | What it does | What we did |
|---|---|---|---|
| **Hidden filler text** (inline CSS: `display:none`, `font-size:1px`, `opacity:0`, `mso-hide:all`, zero-size boxes) | **7,468 of 10,228 emails (73%)**, June 83% | A harmless paragraph ("I tried that new ramen place...") hidden from readers but visible to word-based filters | The parser drops CSS-hidden elements |
| HTML-only emails | about a third | No plain-text part | Use the HTML's visible text |
| Look-alike characters ("Sams 1OO", "pIan", "1,343.OO", "C0STC0") | common in the missed spam | Defeats word matching | **Not handled yet** (see section 10) |
| Non-English | 113 emails (1.1%), mostly Japanese | English-only model cannot read them | Reported separately |

**This changed the headline number.** My first version of the parser read the hidden filler as the email's text, and the model caught only **32%** of the 2026 spam. After dropping hidden text it catches **77%**. The model had not changed; my *measurement* had been wrong. Test your data pipeline as hard as your model.

The filler also explains why the old text/plain part could not be trusted, and why the app asks users to paste what they can *see*.

## 6. Safe handling of hostile data

The 2026 files are real spam and phishing. Rules followed:
- Archives were listed before extraction (no unsafe paths) and extracted into their own folders.
- Python never ran from inside a downloaded folder.
- Emails are parsed with the standard library as **text only**: attachments are counted, never decoded; HTML is reduced to visible text (script/style/head dropped); nothing is rendered, fetched or opened.
- Links stay plain text. The app shows no clickable URL.
- Parsing is capped (20,000 characters of text; the model reads 6,000).
- A broken file produces an empty result, not a crash (tested with garbage bytes).

## 7. Training approach, with rules fixed before the results

1. Compare by **validation PR-AUC** (threshold-free, suits imbalance).
2. Models within 0.003 of the best are ordered by preference. Variants that **mask source-signature words** come first, on principle.
3. **Guard (new):** skip any candidate that flags more than 25% of the 24 modern legitimate challenge emails. It exists because a model can ace old office email and still treat today's normal email as spam.
4. Threshold = highest validation recall at validation precision of at least 0.98 (a false alarm hides a real email).
5. Test and the 2026 set are looked at once, after the choice.

The "source-signature words" are learned from the **training** split only: words in at least 2% of ham and at most 0.4% of spam (120 words, for example `afternoon`, `allen`, `analyst`, `capacity`, `ect`, `draft`, `doc`). In masked variants those words are removed inside the saved pipeline.

Candidates: Dummy, Naive Bayes (word counts), Logistic Regression (word and bigram TF-IDF), Linear SVM, each plain and masked. Text is the subject plus the first 6,000 characters of the body.

## 8. Results

### 8.1 Comparison (validation + 3-fold grouped CV + checks)

| Model | Val PR-AUC | CV PR-AUC | Modern legit flagged (of 24) | 2026 spam detected |
|---|---|---|---|---|
| dummy | 0.4829 | 0.4826 | n/a | n/a |
| naive_bayes | 0.9984 | 0.9950 ± 0.0016 | 33% | 80% |
| naive_bayes_masked | 0.9979 | 0.9952 ± 0.0011 | 29% | 74% |
| logreg_word | 0.9990 | 0.9985 ± 0.0009 | 42% | 83% |
| **logreg_word_masked** (chosen) | 0.9987 | 0.9987 ± 0.0004 | 33% | 77% |
| linear_svm_word | 0.9995 | 0.9989 ± 0.0006 | 42% | 86% |
| linear_svm_word_masked | 0.9991 | 0.9987 ± 0.0006 | 42% | 81% |

The last two columns are diagnostics, not selection criteria (except the 25% guard).

### 8.2 The masking experiment
Removing the source-signature words **lowers false alarms on modern legitimate email** (Logistic Regression 42% to 33%; Naive Bayes 33% to 29%) and **costs some detection of 2026 spam** (83% to 77%; 80% to 74%). In-domain scores barely move. That trade-off is exactly what removing a shortcut should look like: the model gets less good at its easy, fake task and somewhat better at the real one.

### 8.3 Final model (`logreg_word_masked`, C = 100, threshold 0.443)

| Set | Rows | Precision | Recall | False alarms | Missed |
|---|---|---|---|---|---|
| Validation | 4,521 | 0.980 | 0.990 | 44 | 22 |
| **Test (Enron-era)** | 4,521 | **0.984** | **0.989** | 35 | 23 |
| Modern challenge set | 34 | 0.529 | 0.900 | 8 | 1 |
| Dummy baseline (test) | 4,521 | 0.000 | 0.000 | 0 | 2,182 |

**Guard result: FAILED.** No candidate flagged 25% or fewer, so the rule picks the first model within the PR-AUC tolerance and records `guard_passed = false`; the app reads that flag and shows the warning.

### 8.4 Real 2026 spam (10,228 emails; spam only)

| Month | Emails | Detected |
|---|---|---|
| June | 5,531 | 73% |
| July | 2,024 | 81% |
| August | 1,416 | 80% |
| September | 1,124 | 79% |
| October (partial) | 133 | 93% |
| **All** | **10,228** | **77%** |
| English-like text | 10,115 | 76% |
| Mostly non-ASCII text | 113 | 90% |

There is no steady decline over time, so there is no clean "drift" signal; the months differ mainly in which campaigns dominate (June is mostly the "points about to expire" family). The set has no legitimate email, so it can only measure detection, never false alarms.

### 8.5 What it misses, and what it wrongly flags
- **Missed 2026 spam:** mostly retail reward-point and gift lures (Kroger, Lowe's, Marriott, Costco style) with look-alike characters ("1OO.OO", "pIan", "C0STC0").
- **Missed Enron-era spam:** very short or empty emails, some non-English.
- **Wrongly flagged modern legitimate emails (8 of 24):** weekly newsletter digest, bank statement notice, flight itinerary, verification code, storage warning, payment receipt, forum welcome, library notice. The words behind the scores are "your", "account", "online", "verification". In the training data, "your account" language almost only appears in spam, and legitimate mail is internal office chatter.
- **Wrongly flagged Enron-era "ham":** real label noise (for example a "$25 certificate" offer).

## 9. Why it fails on modern legitimate email (same lesson as the SMS bank alerts)

The training data contains **almost no automated, legitimate email**: no receipts, statements, verification codes, order updates or newsletters. The model never learned what they look like, so it assumes "automated, commercial-sounding, mentions your account" means spam. In the SMS module the same gap showed up for bank messages and was fixed by adding the owner's real messages. Email has the same remedy: real examples. A private pipeline is ready (section 11).

## 10. Difficulties faced

| # | Difficulty | What happened | Lesson |
|---|---|---|---|
| 1 | Perfect scores (PR-AUC 0.999) | Source words and dates identify the class | Treat perfect scores as a warning; check what the model reads |
| 2 | **My measurement was wrong** | The parser read hidden filler text; 2026 detection looked like 32% | Verify the data pipeline against raw examples before judging the model (true value 77%) |
| 3 | No modern legitimate email | None exists publicly | 2026 spam can only be a test set; a hand-written set and the owner's own emails are the only checks |
| 4 | Guard failed for every model | All flagged 29%-42% of modern legitimate emails | A pre-agreed gate is useful because it makes the failure impossible to hide; it is reported in the app |
| 5 | Look-alike character obfuscation | "1OO.OO", "pIan", "0nly" | A homoglyph normalizer (map `0` to `o` and `1`/`I` to `l` inside words) or character n-grams would help; not done yet |
| 6 | Label noise in Enron | Ham that reads like spam | Public labels are not ground truth |
| 7 | Tokenized training text | Pasted emails look different | One shared normalization at training and prediction time |
| 8 | A full test run took 27 minutes once | Individual files and the whole suite run in 11 seconds on rerun; not reproduced | Likely a machine stall; noted rather than ignored |
| 9 | Windows shell quoting for long commands | Heredocs with quotes failed repeatedly | Put multi-line scripts in files |

## 11. How to supply your own emails (the fix that works)

The owner's private emails can be added the same way as SMS: `data/private/my_email/emails.csv` (ignored by Git; instructions in the folder's `README.txt`). Columns: `subject`, `body`, `expected_category` (`legitimate`, `promotional` or `spam`), `note`.

- **Most valuable:** legitimate *automated* emails from many senders (bank statements, receipts, OTPs and verification codes, password resets, order and travel confirmations, newsletters you asked for, university notices). Redact names, numbers and tracking links.
- Mapping: legitimate and promotional both count as "not spam" for the email model; only spam and scams count as spam.
- A hash of subject and body assigns each email to a train half (repeated 30 times) or a held-out test half, so new emails never move old ones. `python -m scripts.evaluate_my_email` reports the held-out half separately, and counts false alarms on legitimate mail.
- Then rerun `python -m src.training.train_email` (about 20 minutes).

## 12. Concepts introduced

Source bias and signature words; data pipeline verification; CSS-hidden text and word-filter evasion; look-alike (homoglyph) obfuscation; train/serve parity for text; pre-registered selection guard; external validation with spam only; detection rate vs false-alarm rate; hash-based deterministic splitting.

## 13. How to choose the algorithm here (applying Phase 0)

- Task: binary classification of long text. Features: word and bigram TF-IDF (200,000 columns) feeding a linear model.
- Dummy proves learning; Naive Bayes is the fast baseline (and gives unstable probabilities: its best threshold was 0.0007); Logistic Regression gives usable scores and readable weights; Linear SVM was marginally best in-domain but has no probabilities.
- The choice was *not* decided by the 0.0005 PR-AUC difference between linear models. It was decided by the pre-agreed rules: tie, then masked variants, then the guard.
- If real modern data arrives, the first thing to try is not a bigger model but more diverse data.

## 14. Self-check questions

1. Why is a PR-AUC of 0.999 on this data a warning, not a success?
2. Why did we refuse to train on 2026 spam plus Enron ham?
3. What did hidden-text spam do to my first measurement, and how was it caught?
4. Why does masking source words lower the 2026 detection rate but lower false alarms on modern email?
5. Why can the 2026 set never tell us the false-alarm rate?
6. Why did the guard fail, and why is it still useful that it did?
7. Which emails would you add first to the private file, and why?
8. Why is the date column in the dataset never a feature?

## 15. Done when

- [ ] `python -m scripts.prepare_email` and `python -m src.training.train_email` reproduce the tables above
- [ ] The Email tab shows the "Known weakness" warning
- [ ] You tried a normal email, a spam email and an automated email (receipt, OTP) in the Email tab
- [ ] You decided when to supply private emails (section 11)
- [ ] You answered the self-check questions

## 16. Next

Phase 6: harmful-content detection (Jigsaw, six labels, per-label thresholds, bias template tests).


## Update (Phase 8 follow-up): adding the darkknight25 dataset
Source: [darkknight25/phishing_benign_email_dataset](https://huggingface.co/datasets/darkknight25/phishing_benign_email_dataset) (MIT; 200 one-sentence emails, 100 benign / 100 phishing, modern topics such as invoices, receipts and 2FA notices; looks synthetic).
- Before training, the old model flagged **71 of its 100 benign emails**.
- We split it by hash: about half joined training (repeated 5 times, far less than real mail), the other half stayed as an untouched test.
- After retraining (same selection rules): Enron-era test precision 0.986 / recall 0.991; held-out darkknight25 benign flagged **15 of 52 (29%)**, phishing caught 53 of 56; hand-written modern legitimate emails flagged **6 of 24 (25%)**, down from 8 (33%); **2026 spam detection fell from 77% to 69%**.
- The guard now passes, but exactly on its 25% limit, so the app keeps its warning whenever more than 15% of the challenge emails are flagged.
- Read this honestly: false alarms fell, detection of real 2026 spam fell too, and the new data is synthetic. The tables above are the pre-update run.
