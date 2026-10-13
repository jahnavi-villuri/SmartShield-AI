# Phase 2 README: SMS Data Preparation (no training in this phase)

> Numbers come from the real run of `python -m scripts.prepare_sms` (seed 42). The regenerated report is `reports/data_quality/sms.md`; if the two disagree, the report is right.
> **Revision note:** the first version of Phase 2 used only UCI + Mendeley (6,104 messages). After searching for newer data it was redone with a larger third source and an external test set. This README describes the current version.

## 1. Which datasets, and how "recent" are they? (honest answer)

I searched for 2025-2026 SMS spam data. **No public English dataset with both spam and legitimate messages from 2025-2026 exists** that I could find or download. The newest usable material is from about 2021-2024.

| Dataset | Role | Period | Rows used | Notes |
|---|---|---|---|---|
| UCI SMS Spam Collection | train/val/test pool | ~2011 | 5,574 | UK spam forum vs Singapore student ham |
| Mendeley SMS Phishing | train/val/test pool | published 2022 | 5,971 | **84% of it re-uses UCI messages** |
| ScamShield `Smishing_Dataset` (Hugging Face, MIT card), English only, synthetic rows excluded | train/val/test pool | claimed 2012-2024 | 69,070 | Largest; origin of each row not given; possibly the "Super SMS Dataset" from papers (not confirmed) |
| ScamShield `Indian_Telecom_SMS` + `Benchmark_Ground_Truth`, English | **external test only, never trained on** | ~2021 | 2,051 | Different region and period; its "spam" is mostly telecom promo alerts |
| SmishTank (about 900 real smishing messages, 2023-24) | wanted as a modern test | 2023-24 | not obtained | No public file link; needs a manual request |

Rejected on purpose: ChatGPT-extended SMS set (synthetic), ScamShield's `Synthetic_Tier_C` and `synth_*` rows (machine-generated), Hindi/Hinglish rows (a word-based model cannot use them).

## 2. What we did

| Step | Result |
|---|---|
| Downloaded and listed each archive **before** extracting; never ran code inside them | UCI zip, Mendeley zip (only the data file extracted), ScamShield parquet files |
| Loaded with explicit options, inspected all three | `reports/data_quality/sms.md` |
| Mapped labels to the project convention (1 = bad) | ham -> 0; spam/smishing -> 1 |
| Deduplicated within each dataset, then across them | 79,989 rows -> **74,364 unique messages** |
| Removed conflicting labels | 8 rows |
| Built near-duplicate groups | 519 groups with more than one message (1,646 rows; largest 28) |
| Split train / validation / test | **52,048 / 11,164 / 11,152** rows, spam share 0.381 in every split |
| Leakage checks | 0 shared messages, 0 shared groups between any two splits |
| Built the external test set (deduplicated against the pool) | 2,049 rows (1,319 ham, 730 spam) |
| Tests | 24 passing (data) |

Code: `src/data/` (`text_utils.py`, `sms_load.py`, `dedup.py`, `splitting.py`, `inspect_data.py`), `scripts/prepare_sms.py`.

## 3. What the inspection found

1. **Mendeley mostly re-uses UCI** (4,886 shared messages = 84% of Mendeley's unique messages). Treating it as independent would have double-counted data. The ScamShield set shares almost nothing with UCI (4 messages) and 1% with Mendeley.
2. **One source dominates:** about 92% of the pool comes from the ScamShield `Smishing_Dataset`, so the model mostly learns that source's world.
3. **Duplicates:** UCI 415, Mendeley 140, ScamShield 71 within-dataset, then 5,617 across datasets.
4. **Inconsistent labels in Mendeley** (`Smishing`/`smishing`, `Spam`/`spam`); its own `URL` column misses 22 messages with links and flags 35 without. We lowercase labels and ignore that column.
5. **Lost characters:** 145 Mendeley rows contain `U+FFFD` where `£` was destroyed.
6. **Export artifacts:** HTML codes like `&lt;#&gt;` (309 UCI, 255 Mendeley, 23 ScamShield rows) and `<#>` / `XXXX` anonymisation markers. *Phase 3 later showed these are a shortcut* (see the Phase 3 README).
7. **Label noise in the large set** (seen later in the error analysis): meaningless strings such as `SAMSON` or `2.00E+12` labelled spam, and promotional texts labelled ham. Public data is not clean ground truth.
8. **Source bias is built in:** UCI ham is Singapore student chat, UCI spam is UK promotions. For the large set the per-row origin is unknown, so the same risk cannot be ruled out.
9. **Spam share is 38%** in the pool, far above a real inbox. Precision measured on this data will overstate real-world precision.

> **Lesson:** inspection is not a formality. It changed which datasets we use and exposed shortcuts that scores alone would hide.

## 4. How to handle a dataset: the reusable workflow

```text
Download -> list archive contents -> load with explicit options -> inspect (10 questions)
-> map labels -> normalized keys -> dedup within -> combine -> dedup across
-> near-duplicate groups -> split -> leakage checks -> save -> report -> tests
```

Loading options that matter: `sep="\t"`, `header=None` (UCI has no header); `quoting=3` (quote characters inside messages are text); `keep_default_na=False` (a message that is literally `NA` must not become a missing value); `pd.read_parquet` for ScamShield.

The 10 inspection questions: rows/columns, dtypes, missing values, empty text, class counts, text length range, odd characters, duplicates, **reading real examples from each class**, and "is anything here a giveaway the app won't have?".

## 5. Removing duplicates: the five levels with real numbers

| Level | What | How handled | Result |
|---|---|---|---|
| 1. Exact | identical text | keep one | part of 415 + 140 + 71 |
| 2. Normalized | case / spacing differ | key = lowercase + collapsed spaces, original text kept | same step |
| 3. Conflicting | same key, different labels | drop all copies | 8 rows |
| 4. Near-duplicate templates | same spam, different number/link | **kept, but grouped** so siblings never straddle splits | 519 groups |
| 5. Cross-dataset | same message in several sources | dedup after combining; record every source | 5,617 removed |

Deduplicate **before** splitting, never after. Log every step so any reader can follow `79,989 -> 74,364`.

## 6. Splitting

Train = textbook exercises; validation = mock exams (choose models, settings, thresholds); test = the final exam (looked at once); external = a different school's exam.

Method: `StratifiedGroupKFold` with 20 folds (3 test, 3 validation, 14 train): stratified (same spam share everywhere) and grouped (templates stay together). Result: spam share 0.381 in all three. We test for leakage rather than assume it (the script stops on any overlap; a pytest test re-checks the saved files).

The test and external sets stay untouched until a model and threshold are chosen on validation.

## 7. Choosing the training approach from the requirements (preview)

1. Binary classification (spam / not).
2. Short text -> CountVectorizer / TF-IDF -> linear models (Naive Bayes, Logistic Regression, Linear SVM).
3. Imbalance and a high pooled spam share -> precision, recall, PR-AUC, never accuracy alone; include a Dummy baseline.
4. Worse error: **false positive** (real message hidden) -> choose the threshold for high precision.
5. Plenty of data now (52k rows) -> validation for tuning, grouped cross-validation to check stability.
6. The app needs a score and an explanation -> Logistic Regression is favoured if scores are close.

## 8. Still open: truly recent data

Nothing from 2025-2026 is in the project. Options: request SmishTank (<https://smishtank.com>, scams only, measures recall on modern scams); collect your own consented samples in `data/private/` (never committed). Until then every SMS result must be described as "tested on data up to about 2021-2024".

## 9. Concepts introduced

Data-quality report, normalization key, label mapping, cross-dataset duplicate, group split, stratified split, leakage, source bias, external test set, label noise.

## 10. Self-check questions

1. Why deduplicate before splitting?
2. Why is Mendeley not an independent dataset here, and how did we measure that?
3. Exact vs near-duplicate: how and why are they handled differently?
4. Why a group split rather than a random one?
5. Why keep the original text and use a separate normalization key?
6. Why is the external set never used for training or tuning?
7. Why can't a 38%-spam test set tell us real-world precision?
8. Why did we refuse to use the synthetic rows?

## 11. Done when

- [ ] `python -m scripts.prepare_sms` runs cleanly and `reports/data_quality/sms.md` exists
- [ ] `python -m pytest` passes
- [ ] You can explain each row of the cleaning table (79,989 -> 74,364)
- [ ] You answered the self-check questions
- [ ] You decided what to do about truly recent data (section 8)

## 12. Next: Phase 3 (SMS training and evaluation)
