# Phase 3 README: SMS Model Training and Evaluation

> All numbers are from real runs of `python -m src.training.train_sms` (seed 42). Full tables: `reports/metrics/`. Charts: `reports/figures/`. The model and its metadata: `models/sms_spam/`. Round 1 (before the artifact fix) is archived in `reports/metrics/round1/`.

## 1. Result in one paragraph

The selected model is **Logistic Regression on normalized character n-gram TF-IDF** (`logreg_char_norm`, C = 10, threshold 0.6503), trained in **three rounds**. On the held-out in-domain test set it reaches **precision 0.984 and recall 0.951** (67 real messages wrongly flagged, 207 spam missed, out of 11,152). On a **different-source external set** it drops to **precision 0.874 and recall 0.863**. On a small hand-written set of legitimate bank and service messages (33 messages, 23 of them legitimate) it wrongly flags 4 of the 23 legitimate ones; the first model flagged 18 of 23. The honest summary: very good on data that looks like its training data, clearly weaker on other regions and periods, and it carries leftover dataset biases. It is a learning-project model, not a security product.

### The three rounds

| Round | What changed | Why |
|---|---|---|
| 1 | Raw text, 5 models | First comparison |
| 2 | Added artifact-removing normalization (`_norm` variants) | Round 1 diagnostics showed shortcuts (HTML codes only in ham, `XX` masks 98% spam) |
| 3 | Added 3,000 template-written training examples (legit bank alerts + scams sharing the same vocabulary) | **You pointed out that "Rs 300 credited to your account" is a normal bank message**; a hand-written check showed the model flagged 18 of 23 legitimate messages |

> **Update (Round 4 and the three-category redesign).** After this README's Round 3 numbers, the main model was retrained once more (Round 4) and a second model was added so the app can say Legitimate / Promotional / Scam. Sections 5.1-5.4 below describe **Round 3**. The current main model's numbers (Round 4: test precision 0.982, recall 0.948; external 0.860 / 0.806; threshold 0.778; 3 of 23 challenge legitimate messages flagged) and the redesign are in `docs/phases/phase-03b-readme.md`.

## 2. What we built

| Piece | File |
|---|---|
| Metrics + threshold selection | `src/evaluation/metrics.py` |
| Charts (confusion matrix, PR curves, comparison) | `src/evaluation/plots.py` |
| Artifact-removing text normalization | `src/features/text_normalize.py` |
| Training script (compare, select, evaluate, save) | `src/training/train_sms.py` |
| Saved pipeline + metadata | `models/sms_spam/model.joblib` (5.2 MB), `metadata.json` |
| Tests | `tests/test_sms_training.py`, `tests/test_text_normalize.py` |

## 3. Choosing the approach from the requirements

1. **Classification or regression?** The answer is a category (spam / not spam), so **classification**. The model outputs a score between 0 and 1; a threshold turns it into a decision.
2. **What kind of data?** Short text. Turn it into numbers (CountVectorizer or TF-IDF), which gives tens of thousands of sparse columns. **Linear models** (Naive Bayes, Logistic Regression, Linear SVM) are the standard fit.
3. **Which error is worse?** A **false positive** (a real message hidden). So we require high precision.
4. **How much data?** 52,048 training messages: enough for stable comparison, small enough to train in seconds to minutes on a CPU. A transformer is unnecessary for a first version.
5. **What does the app need?** A score, speed, a small file, and an explanation. Logistic Regression satisfies all four.

### The candidates and what each is

| Model | Idea | Why included |
|---|---|---|
| **Dummy** (always predicts the prior) | Zero skill | The floor every model must beat |
| **Naive Bayes** + word counts | Multiplies per-word spam/ham evidence | Classic fast baseline |
| **Logistic Regression** + word TF-IDF | Learns a weight per word | Probabilities and explanations |
| **Linear SVM** + word TF-IDF | Widest-margin straight boundary | Often strongest on text; no probabilities |
| **Logistic Regression** + **character n-gram** TF-IDF | Weights for 2-5 letter chunks | Catches obfuscation ("fr33", "claim-now") |

Each ran in two versions: raw text, and `_norm` (after artifact-removing normalization). That gives 9 models.

### Feature extraction, in plain words
- **CountVectorizer:** counts how many times each word (or two-word phrase) appears.
- **TF-IDF:** like counts, but down-weights words that appear everywhere ("the") and up-weights distinctive ones; `sublinear_tf` stops a word repeated 20 times from dominating.
- **Character n-grams:** chop text into overlapping chunks of 2-5 characters, so misspellings and number tricks still share chunks with the real word.
- **Rule:** the vectorizer is **fitted on training text only**. Validation, test and live messages are only *transformed*.

## 4. The training workflow, step by step

```text
load splits -> fit vectorizers on TRAIN only -> tune each model's setting on VALIDATION
-> cross-validate on TRAIN -> pick the model by a pre-agreed rule -> pick the threshold on VALIDATION
-> evaluate ONCE on test and external -> error analysis -> save pipeline + metadata
```

### Rules fixed before looking at results
1. Compare models by **PR-AUC (average precision) on validation**. It needs no threshold and suits imbalanced data.
2. If models are within **0.003** PR-AUC of the best, prefer the simpler, more explainable one.
3. Threshold = the one with the **highest validation recall subject to validation precision >= 0.98**.
4. Test and external sets are evaluated once, after the choice.

Fixing rules first matters because, after seeing scores, it is very easy to pick the metric that flatters your favourite model.

### Hyperparameters
A hyperparameter is a setting you choose (not learned). We tried a few values per model (e.g. Logistic Regression's `C` in 1/3/10/30/100; larger `C` = trust the training data more) and kept the best on **validation**. Note: some best values sat at the edge of the grid (e.g. char model without normalization, C = 30 of 3/10/30/100), so a wider grid might gain a little.

### Cross-validation
We also split the **training** data into 5 folds (grouped, so templates stay together), trained on 4 and scored on 1, five times. The standard deviation (about 0.001) shows the differences between top models are tiny and stable, not luck.

## 5. Results

### 5.1 Model comparison (Round 3, all nine models; validation + 5-fold CV on training data)

| Model | Validation PR-AUC | CV PR-AUC (mean ± std) | Val recall at precision 0.98 |
|---|---|---|---|
| dummy | 0.381 | 0.387 ± 0.000 | n/a |
| naive_bayes | 0.979 | 0.979 ± 0.001 | 0.915 |
| naive_bayes_norm | 0.977 | 0.978 ± 0.001 | 0.908 |
| logreg_word | 0.992 | 0.992 ± 0.001 | 0.938 |
| logreg_word_norm | 0.991 | 0.992 ± 0.001 | 0.926 |
| linear_svm_word | 0.991 | 0.992 ± 0.001 | 0.946 |
| linear_svm_word_norm | 0.991 | 0.992 ± 0.001 | 0.935 |
| logreg_char | **0.995** | **0.995 ± 0.001** | **0.961** |
| **logreg_char_norm** (selected) | 0.995 | 0.993 ± 0.001 | 0.954 |

Reading it: the Dummy's PR-AUC equals the spam share, so it learns nothing. Naive Bayes is the weakest real model. The three linear models are close; character n-grams are best. The pre-agreed rule picked `logreg_char_norm` because it was within 0.003 of the best and normalized models are preferred. Caveat: the best `C` for `logreg_word` and `logreg_char` sat at the grid edge (C = 100), so a wider search might gain slightly. Chart: `reports/figures/sms_model_comparison.png`.

### 5.2 Final evaluation of the selected model (Round 3)

| Set | Rows | Precision | Recall | F1 | PR-AUC | ROC-AUC | False alarms (FP) | Missed spam (FN) |
|---|---|---|---|---|---|---|---|---|
| Validation | 11,164 | 0.980 | 0.954 | 0.967 | 0.9945 | 0.9965 | 82 | 197 |
| **Test (in-domain, looked at once)** | 11,152 | **0.984** | **0.951** | 0.967 | 0.9952 | 0.9967 | **67** | **207** |
| **External (India telecom)** | 2,049 | 0.874 | 0.863 | 0.868 | 0.9356 | 0.9601 | 91 | 100 |
| **Hand-written challenge set** (33 msgs) | 33 | 0.692 | 0.900 | 0.783 | n/a | n/a | 4 | 1 |
| Test, older-source messages only* | 955 | 0.978 | 0.936 | 0.957 | 0.9957 | 0.9989 | 4 | 12 |
| Test, Dummy baseline | 11,152 | 0.000 | 0.000 | 0.000 | 0.3809 | 0.5000 | 0 | 4,248 |

*The part of the test split whose messages appear in UCI or Mendeley (the rest come only from the large ScamShield set). Only 188 of these are spam, so its numbers are noisy. All figures: `reports/metrics/sms_final_metrics.csv`.

Inference time: about 1.7 ms per message; model file 5.2 MB.

**The challenge set is tiny and written by me**, so treat its numbers as a smoke test of one known blind spot, not as a measured error rate. Its precision looks low (0.69) only because it contains few spam messages (10), so 4 false alarms weigh heavily.

### 5.3 Why the threshold matters (actual numbers, test set)

| Threshold | Precision | Recall | False alarms | Missed spam |
|---|---|---|---|---|
| Default 0.5 | 0.974 | 0.965 | 111 | 150 |
| **Chosen 0.6503** | **0.984** | 0.951 | **67** | 207 |

Raising the threshold removed 44 false alarms and cost 57 extra missed spam. That is the trade-off we chose on purpose, because a false alarm hides a real message. There is no free lunch; the right setting depends on how costly each error is.

### 5.4 Round by round (what each fix did)

| | Round 1 `logreg_char` | Round 2 `logreg_char_norm` | Round 3 `logreg_char_norm` + augmentation |
|---|---|---|---|
| Test precision / recall | 0.981 / 0.954 | 0.983 / 0.953 | 0.984 / 0.951 |
| External precision / recall | 0.855 / 0.841 | 0.868 / 0.885 | 0.874 / 0.863 |
| External PR-AUC | 0.9205 | 0.9364 | 0.9356 |
| Legitimate challenge messages wrongly flagged | 18 of 23 | 18 of 23 | **4 of 23** |

Reading it: the in-domain score barely moved in any round, which is exactly why in-domain tests cannot reveal these problems. Normalization helped the external set (Round 2). Augmentation fixed the bank-alert blind spot (Round 3) at no real cost on the real test data. **Caveat:** the external set was looked at in all three rounds, so its numbers are slightly optimistic for the choices we made; a fresh, untouched modern set is still needed. (Round 1 and 2 files: `reports/metrics/round1/`, `round2/`.)

### 5.4b The "Rs" lesson (reported by you)

In Round 1 I wrote that messages containing "Rs" were 99% spam, and treated it as a data finding. **That was really a warning, not a rule.** In the public training data, almost every message with a rupee amount was a promotion or a scam, so the model learned "rupee amount = spam" and flagged ordinary bank alerts such as "Rs 300.00 has been credited to your account". Measured on 23 hand-written legitimate messages: 18 were flagged (13 of 14 bank alerts). No test split could show this, because the test data has the same blind spot.

What fixed it was not a better algorithm but **better data**: 1,500 legitimate transaction alerts and 1,500 bank-themed scams, written from templates so that rupee amounts, account numbers, "credited" and "debited" appear in **both** classes. Then only the *request* (click a link, call, share an OTP) separates them. After training, "Rs :300/- is credited to your account" scores 0.12 (Not spam), while "Rs 300 credited to your account. Click http://claim-now.example to withdraw your bonus" scores 0.98 (Spam).

Limits of this fix: the augmentation is synthetic (my templates, labelled `synthetic_aug`, training only, links use the reserved `.example` domain), real-world bank alerts vary far more, and 4 legitimate challenge messages are still flagged (a NEFT transfer, a wallet refund, a clinic fee and a prepaid-parcel notice). The model also still misses "Your SIM will be blocked... share the OTP" (score 0.20).

### 5.5 Understanding the confusion matrix with real examples
(From `reports/metrics/sms_test_false_positives.csv` and `..._false_negatives.csv`.)

- **True positive:** "WINNER! You have won a 1000 pound prize. Call 09061701461" -> flagged, correct.
- **True negative:** "Hey, are we still meeting for lunch tomorrow at 1?" -> not flagged, correct.
- **False positive (real message flagged):** typically carrier notices, promotions and OTP texts that are legitimate but look like marketing, e.g. "OTP for verification of your DigiLocker account is ...".
- **False negative (spam missed):** very short or odd spam, plus mislabelled junk such as `SAMSON` or `2.00E+12` that the dataset calls spam. Many "errors" are actually label noise.

Hand-typed probes of the saved model: prize scams, a fake bank suspension, a parcel-fee scam and a "free iPhone" text were flagged; a lunch message and "haha ok see you later" were not. **A "Mum, I lost my phone, this is my new number" scam scored 0.01 and was missed**, and an OTP message scored 0.23: the training data contains almost no conversational or modern scams.

## 6. Difficulties faced (and what each taught)

| # | Difficulty | What happened | Lesson |
|---|---|---|---|
| 1 | **No 2025-2026 data exists publicly** | Searches found nothing recent with both spam and legitimate English SMS; SmishTank has no public file link; a GitHub candidate returned 404 | Say plainly what you could not get; never claim "latest" without proof |
| 2 | **Second dataset was a copy of the first** | 84% of Mendeley's unique messages are in UCI | Measure overlap before pooling data |
| 3 | **Dataset artifacts became shortcuts** | HTML-code messages were 0% spam; `XX`-masked numbers 98% spam; "Rs" 99% spam (a data bias, not a rule: real bank alerts contain Rs too, see 5.4b); chat words ("haha", "lol") 99% ham | A 99% score can come from how the data was produced. Always inspect top weighted words |
| 4 | **Label noise** | Junk strings labelled spam, promos labelled ham | Public labels are not ground truth; read your errors |
| 5 | **Source bias remains** | Even after normalization (Round 2), top "spam" words include `rs`, `bhk`, `lac`, `uk` and top "ham" words `haha`, `lol`, `lor`, `wat`: region and chat style | Cleaning artifacts does not remove sampling bias; only more diverse data can |
| 6 | **Impressive in-domain, weaker external** | 0.953 recall in-domain vs 0.885 external | In-domain tests overestimate; always keep a different-source test |
| 7 | **I changed the plan after seeing results** | Extended the candidate set after Round 1 | Allowed if disclosed. The rule was unchanged, but the external set was seen twice |
| 8 | **Own bug in the report** | The first overlap figure double-counted rows (9,772 instead of 4,886 messages) | Re-read generated reports critically; test outputs, not only code |
| 9 | **Character model is slow** | Cross-validating the character models took several minutes | Run long jobs in the background; tune cheaply first |
| 10 | **Environment friction** | Jupyter install hung once (Phase 1); Python 3.14 instead of 3.12; Windows console encoding errors when printing odd characters | Pin versions, use `-X utf8`, install heavy tools separately |
| 11 | **Pipeline rejects an empty list** | `predict([])` raises an error | Input validation must happen in the detector layer (Phase 4) before the model |
| 13 | **Legitimate bank alerts flagged as spam** (reported by you) | "Rs" and bank words occurred almost only in promos/scams, so the model learned currency = spam; 18 of 23 legitimate challenge messages flagged | Subgroup failures hide behind great overall scores. Test a hand-written set for each important kind of legitimate message, and fix with balanced data, not just a different algorithm |
| 12 | **Pooled spam share is 38%** | Real inboxes have far less spam | Precision on this test overstates real-world precision |

## 7. Why this model, and what would change the decision

- Logistic Regression gives real probabilities and an explanation path, and is small (5 MB) and fast.
- Character n-grams beat word features by roughly 0.003 PR-AUC, a small but consistent gain (CV std 0.001).
- If a wider hyperparameter search or new data narrows the gap, the rule would prefer the word model (smaller, more interpretable).
- The score is a **model score**, not a validated real-world probability.

## 8. Concepts introduced

Pipeline; fitting on training data only; TF-IDF; n-grams; hyperparameter; cross-validation; PR-AUC; ROC-AUC; precision/recall trade-off; threshold; confusion matrix; baseline; external validation; shortcut learning; label noise; model card basics (metadata).

## 9. Self-check questions

1. Why is this classification, not regression?
2. Why do we fit the vectorizer on training data only?
3. Why choose by PR-AUC instead of accuracy? What would a model that always says "not spam" score on accuracy here?
4. Why is the threshold chosen on validation and not on test?
5. What does it mean that external recall (0.885) is lower than test recall (0.953)?
6. Why did normalization not change the in-domain score but help externally?
7. What two kinds of shortcut did we find, and which one did normalization fix?
8. Why is it dangerous that I looked at the external set twice?
9. Why would real-world precision be lower than 0.984?
10. Why did a 95% recall, 98% precision model flag 18 of 23 legitimate bank alerts, and why could the test set not show it?
11. Why does the augmentation include scams that use the same rupee/bank vocabulary, not only legitimate alerts?

## 10. Done when

- [ ] `python -m src.training.train_sms` reproduces the tables above
- [ ] `models/sms_spam/model.joblib` and `metadata.json` exist
- [ ] You can explain Section 5.3 (the threshold trade-off) in your own words
- [ ] You answered the self-check questions

## 11. Next: Phase 4 (thin slice)

Load this saved model in a small detector module and a minimal Streamlit page, with input validation, a checksum check and tests.
