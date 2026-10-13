# Phase 6 README: Harmful-Comment Detection (Jigsaw, six labels)

> All numbers come from real runs of `python -m scripts.prepare_toxicity` and `python -m src.training.train_toxicity --augment` (seed 42). Tables: `reports/metrics/toxicity_*.csv`; data report: `reports/data_quality/toxicity.md`. **No offensive comments appear in this document or in any committed file.** Error-analysis files that contain offensive text are written only to the Git-ignored data folder.

## 1. Result in one paragraph

The model scores a comment on six labels (toxic, severe_toxic, obscene, threat, insult, identity_hate) and flags each label that reaches its own threshold. On held-out data like its training data it reaches **macro-F1 0.64**; on the official Jigsaw test set **0.57**. Common labels (toxic, obscene) are decent (F1 0.66-0.82); rare labels are weak (severe_toxic 0.36-0.49, threat 0.50-0.56, identity_hate 0.47-0.56). A deliberate **bias test fails**: neutral sentences that mention an identity are flagged, e.g. 50% of the "gay" sentences. Three rounds of mitigation cut the identity terms over my 5% limit from 12 to 3 without hurting accuracy, but did not remove the problem. The model also cannot read context ("You are *not* an idiot" is flagged). It is a learning model and a prompt for human review, never an automatic verdict on a person.

## 2. What we built

| Piece | File |
|---|---|
| Data pipeline + data-quality report | `scripts/prepare_toxicity.py` |
| Six-label model (one classifier per label, shared features) | `src/models/toxicity_model.py` |
| Text clipping used inside the saved model | `src/features/comment_text.py` |
| Identity-sentence augmentation (training) and bias-probe words | `src/data/augment_toxicity.py` |
| Training, thresholds, evaluation, probes | `src/training/train_toxicity.py` |
| Detector (validation, checksum load, scores, explanation) | `src/detection/toxicity_detector.py` |
| Comments tab | `app.py` |
| Hand-written context probe (36 sentences) | `tests/data/toxicity_probe.csv` |
| Tests | `tests/test_toxicity.py`, `tests/test_toxicity_real_model.py` |

## 3. Concepts: multi-label classification

This is **classification**, but not the usual kind. In *multi-class* classification a comment has exactly one category. Here a comment can have **several labels at once**: of 16,225 toxic training comments, 9,865 have more than one label. `severe_toxic` is *always* also `toxic`; 94% of `threat` comments are also toxic; 86% of `severe_toxic` are insults.

How a multi-label prediction is produced: we train **six independent yes/no classifiers** ("binary relevance"), one per label, on the same text features. Each gives a probability, and each is compared with **its own threshold**. That preserves the original meaning ("this comment is toxic *and* an insult *and* obscene") and lets each label have a threshold that suits how rare it is.

Choosing the task type (applying Phase 0): the answer is a set of categories, so **classification**. If we had wanted "how toxic is it, from 0 to 10", that would be regression. We output scores between 0 and 1, but they are confidences in categories, not a measured quantity.

## 4. The data, and the traps in it

Source: Jigsaw Toxic Comment Classification Challenge via Hugging Face (`thesofakillers/jigsaw-toxic-comment-classification-challenge`). Wikipedia talk-page comments (about 2017) rated by crowd workers. Licence per the card: labels CC0, comment text CC BY-SA 3.0. The text is offensive.

| Finding | Detail | What we did |
|---|---|---|
| **Truncated download** | The first train file was 27.7 MB instead of 44.1 MB; the parquet footer was missing | Detected on load; re-downloaded with retries; verified the exact byte size |
| **Test split is two stacked blocks** | 153,164 rows with text but no labels, plus 153,164 rows with labels but no text, same ids | Joined by `id`. Naive loading gives either no labels or no text |
| **-1 labels** | 89,186 test rows were not scored by the competition | Dropped; only 63,978 scored rows are usable (63,908 after cleaning) |
| Duplicates and conflicts | After normalising case and spacing: 215 duplicate rows and 84 rows whose copies disagree on labels | Duplicates removed, conflicting copies dropped entirely; 159,571 to 159,272 |
| Severe imbalance | threat 478 positives (0.3%), identity_hate 1,405, severe_toxic 1,595; 10.2% of comments have any label | Stratified, grouped split on the rare-label pattern |
| Overlapping labels | see section 3 | Per-label models and thresholds |
| Identity-word bias already in the data | comments mentioning "gay" are 57.6% toxic against a 9.6% base rate | Bias probe + augmentation (section 8) |
| `astype(str)` on missing values | pandas 3 keeps missing text as missing, which crashed `normalize_key` | Fill missing text before casting |

Splits (14/3/3 of 20 folds, stratified on the rare-label pattern and grouped by near-duplicate text): train 111,488; validation 23,892; internal test 23,892; plus the official test set of 63,908. Positives per split for the rarest labels: threat 327 / 71 / 71 / 211 (train / validation / internal test / official); leakage checks between splits are all zero.

## 5. Choosing the algorithm

| Candidate | Features | Why it is on the list |
|---|---|---|
| Dummy | none | The floor: PR-AUC equals the label's frequency (macro 0.037) |
| `logreg_word` | word and bigram TF-IDF, 250,000 columns | Fast, explainable baseline |
| `logreg_wordchar` | the above + character 3-5-grams, 450,000 columns total | Character chunks survive misspellings and disguised words ("stuuupid", "f*ck") |

Logistic Regression per label is the right first model for sparse text (Phase 0 rule). A transformer would read context and probably fix negation, but needs a GPU or long CPU training and is a Phase 16 comparison, not part of this learning version.

**Rules fixed before looking at results:**
1. Compare by macro-average **PR-AUC** over the six labels on validation. Within 0.003, prefer the smaller word-only model.
2. Thresholds per label = value maximising **F1** on pooled predictions: validation plus 3-fold out-of-fold predictions on the training set. (Pooling gives `threat` ~400 positives instead of 71, so its threshold is not decided by a handful of comments.)
3. Look at the internal and official test sets once, after the choice.
4. Bias is acceptable only if **no identity term has more than 5% of its probe sentences flagged**.

## 6. Results

### 6.1 Candidate comparison (validation, macro PR-AUC)

| Model | C = 2 | C = 8 |
|---|---|---|
| word | 0.634 | 0.647 |
| **word + character** | **0.674** | 0.670 |

Characters add 0.027 macro PR-AUC, well above the 0.003 tie tolerance, so `logreg_wordchar` (C = 2) was chosen. Per label (validation PR-AUC, C = 2): toxic 0.887, obscene 0.896, insult 0.804, severe_toxic 0.506, threat 0.443, identity_hate 0.508. The model is 36.8 MB and takes about 10 ms per comment.

### 6.2 Per label, with each label's own threshold

| Label | Threshold | Internal test P / R / F1 | Official test P / R / F1 | Official positives |
|---|---|---|---|---|
| toxic | 0.274 | 0.81 / 0.79 / **0.80** | 0.54 / 0.85 / **0.66** | 6,085 |
| severe_toxic | 0.114 | 0.42 / 0.60 / **0.49** | 0.24 / 0.67 / **0.36** | 367 |
| obscene | 0.189 | 0.82 / 0.82 / **0.82** | 0.61 / 0.80 / **0.69** | 3,687 |
| threat | 0.082 | 0.56 / 0.56 / **0.56** | 0.42 / 0.60 / **0.50** | 211 |
| insult | 0.184 | 0.67 / 0.77 / **0.72** | 0.57 / 0.76 / **0.65** | 3,423 |
| identity_hate | 0.137 | 0.44 / 0.50 / **0.47** | 0.50 / 0.62 / **0.56** | 712 |

| Aggregate | Internal test | Official test |
|---|---|---|
| Macro-F1 (each label counts equally) | **0.642** | **0.569** |
| Micro-F1 (every decision counts equally) | 0.750 | 0.648 |
| Macro PR-AUC | 0.664 | 0.607 |
| Macro ROC-AUC | 0.985 | 0.979 |
| Exact match (all six labels right) | 91.4% | 86.7% |
| "Anything harmful?" precision / recall | 0.84 / 0.78 | 0.55 / 0.86 |

### 6.3 Lessons visible in these tables

1. **ROC-AUC flatters imbalanced data.** Every label has ROC-AUC of 0.96-0.99, yet the rare labels have PR-AUC of only 0.32-0.49. When a label is 0.3% of the data, a model can rank almost everything correctly and still be wrong about half of what it flags. Judge rare labels by precision, recall and PR-AUC.
2. **Thresholds are decisions, and they matter.** With the default 0.5, macro-F1 is 0.518 (internal) / 0.521 (official); with per-label thresholds it is **0.642 / 0.569**. For `threat` the default catches 14% of threats (F1 0.23) against 56% with its own threshold (F1 0.56). Thresholds are low (0.08-0.27) because the rare labels' probabilities are small. The cost is more false alarms: precision falls. Choose by the cost of each error; for human-review queues, high recall with low precision is often acceptable, for automatic removal it is not.
3. **The official test set is harder.** At the same thresholds, recall is similar but precision falls (toxic 0.81 to 0.54) and macro-F1 drops from 0.64 to 0.57. That is a generalization gap; I do not know its exact cause (possible: different rater behaviour or sampling in the held-out competition set). Report both numbers.
4. **Rare labels are not solved.** severe_toxic (a subset of toxic, hard to separate), threat (few examples) and identity_hate (varied wording) are the weakest. A threat flag from this model is a hint, not evidence.

## 7. Context: the model reads words, not meaning

The 36-sentence probe (24 benign, 12 mildly toxic, all hand-written, no slurs):

- **Benign sentences wrongly flagged: 5 of 24**, all of the same kind: sentences that *mention* toxic words or negate them. "You are **not** an idiot, you simply misread the policy" is flagged as toxic, obscene and insult; "The word stupid is considered an insult in many contexts" and "He called me an idiot yesterday, so I reported it" are flagged too.
- **Mildly toxic sentences caught: 9 of 12.**
- Quotes, negation, figurative language and discussions *about* toxicity are known failures of word-based models. A transformer would handle some of these; context would still need a human.

## 8. Bias: the model can mistake an identity for an insult

**Why it happens:** in the training data, comments that mention some identity words are labelled toxic much more often than average ("gay" 57.6% vs 9.6% overall). A model that learns from that treats the word itself as a toxicity signal.

**The test:** 6 neutral sentence templates ("I am a {x} person.", "She is happy to be {x}.", ...) filled with 37 words: 25 identity terms seen during augmentation, 12 identity terms never seen, and 8 non-identity control words (tall, teacher, ...). One sentence is 17% of a term's score, so a single flagged sentence already exceeds the 5% limit; the probe is small and strict.

**The mitigation (one round at a time, rule unchanged):** add harmless synthetic sentences that mention identity terms, labelled clean. Training uses *different* templates and a *subset* of the identity words, so the probe measures generalization and not memorisation.

| | Round 1: none | Round 2: 1,000 sentences (10 templates) | Round 3: 3,750 sentences (150 per identity, 300 templates) |
|---|---|---|---|
| Identity terms over the 5% limit | **12** | 5 | **3** |
| Worst term | gay 83% | gay 67% | **gay 50%** |
| Others over the limit | lesbian, Muslim, atheist, Jewish, black, deaf, male, bisexual, old, American, Mexican (17% or more each) | Muslim, Jewish, black, bisexual (17%) | black 17%, bisexual 17% |
| Mean flagged: identity terms seen / never seen / controls | 10.0% / 1.4% / 0% | 4.7% / 1.4% / 0% | **2.7% / 1.4% / 0%** |
| Internal test macro-F1 | 0.643 | 0.639 | 0.642 |
| Official test macro-F1 | 0.569 | 0.569 | 0.569 |
| Benign context sentences flagged / mild toxic caught | 5 / 10 | 5 / 10 | 5 / 9 |

Reading it: more varied clean examples steadily reduced the bias at essentially **no accuracy cost** (one fewer mild-toxic sentence caught). Control words are never flagged, so the effect is about identity words specifically. **It still fails my acceptance rule.** "gay" is still flagged in half the sentences because the real training data pairs it with toxicity so strongly (about 500 toxic comments mention it) that 150 synthetic sentences cannot cancel it. In a spot-check, a sentence with other wording ("I am a gay man and I love gardening.") is not flagged, which shows the fix generalizes only partly.

**What the app does:** the Comments tab shows a standing "Known weakness" warning built from the model's own probe results.

**What would help more:** more and more varied counterfactual sentences; training on the newer Jigsaw "unintended bias" data with identity annotations; sample re-weighting; or a model that reads context. Each is a bigger project than this phase.

**Ethics note:** this system can disadvantage people who are discussing or describing themselves with identity words. Using it for automatic moderation would hurt exactly those users. It should only ever queue content for human review.

## 9. Error analysis (no offensive text quoted here)

- **severe_toxic** is flagged for many comments that are only toxic: the model finds the "very toxic" boundary hard (official precision 0.24).
- **identity_hate** misses contain varied wording the model has not seen, and its false alarms often contain an identity word without hatred.
- **Disguised and creative spellings** are partly caught by character n-grams (the reason the word + character model won).
- **Context errors:** mentions, quotes, negation (section 7).
- **Label noise:** the raters disagree in places; some "toxic" training comments are mild and some clearly toxic ones are unlabelled. This limits every metric.
- Offensive examples with scores are in `data/processed/toxicity/errors_official_test.csv` (Git-ignored; 40 worst errors per label and type; read in small batches).

## 10. Difficulties faced

| # | Difficulty | Lesson |
|---|---|---|
| 1 | Truncated download | Verify file size or integrity after every download |
| 2 | Test split was two stacked blocks | Read the schema, not only the column names; check the row counts per column |
| 3 | `-1` labels | Never treat a sentinel value as a real label |
| 4 | Missing text became `NaN` after `astype(str)` in pandas 3 | New library versions change old habits |
| 5 | The preparation script (grouped, stratified splitting of about 150,000 rows) ran for about 10 minutes | Measure before waiting; run long jobs in the background |
| 6 | Character n-grams: 59 million non-zeros and about 6 minutes just to build the features | Cap text length and vocabulary; use float32 |
| 7 | Rare labels: `threat` has about 70 positives per split | Pool predictions to choose thresholds; report the uncertainty |
| 8 | Official test scored lower than internal test | Keep more than one test set |
| 9 | Bias persisted through two rounds | Report the trend and the unresolved remainder instead of declaring success |
| 10 | My first augmentation repeated only 10 distinct sentences per identity | Check what your synthetic data actually contains (unique count) |
| 11 | Offensive text in data | Aggregate statistics in reports; error files only in ignored folders; no slurs in tests |
| 12 | Windows shell quoting for long scripts | Put multi-line scripts in files |

## 11. How the app uses it

The Comments tab takes a comment (up to 5,000 characters; empty and over-long input are rejected with a message), shows all six scores with a bar and the label's threshold, marks the labels at or above their thresholds, and lists the words that pushed each flagged label up (word-level weights). It always shows a note that the model reads words, not meaning, and a bias warning whenever the model's own probe failed. The comment is analysed in memory and is not stored or logged; the model file is verified by checksum before loading.

## 12. Concepts introduced

Multi-label classification; binary relevance; per-label thresholds; macro vs micro averaging; exact-match accuracy; PR-AUC vs ROC-AUC under imbalance; out-of-fold predictions; counterfactual augmentation for bias; bias probes; held-out templates; content-moderation ethics.

## 13. Self-check questions

1. Why is toxicity classification *multi-label* and not multi-class? Give an example comment with three labels.
2. Why does each label get its own threshold? What happened to `threat` at the default 0.5?
3. Why are ROC-AUC values near 0.99 not evidence that the rare labels work?
4. What is the difference between macro-F1 and micro-F1, and which one hides the weakness on rare labels?
5. Why did we choose the thresholds on pooled out-of-fold and validation predictions?
6. Why must the bias probe use different templates and identity words from the augmentation?
7. Why is it dangerous to use this model to remove comments automatically?
8. Why did the official test score lower than the internal test, and what should you report?

## 14. Done when

- [ ] `python -m scripts.prepare_toxicity` and `python -m src.training.train_toxicity --augment` reproduce the tables
- [ ] You tried a polite comment, an insult, a threat and a negated sentence in the Comments tab
- [ ] You can explain why "I am a gay man" is a fair test of bias
- [ ] You answered the self-check questions

## 15. Next

Phase 7: phishing URLs (URL-only features, domain-grouped split, external tests).
