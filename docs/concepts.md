# Machine-learning concepts, explained with this project

Each entry says what an idea means, then shows where it appeared in SmartShield AI so you can see it in real code and real numbers. Terms in **bold** have their own entry.

## 1. What kind of problem is it?
**Classification vs regression.** Classification picks a category (spam or not); regression predicts a number (a house price). Every detector here is classification, because the answer is a label. The scores (0 to 1) are only an intermediate step before a **threshold** turns them into a label.

**Binary, multi-class and multi-label.** The SMS main model is binary (ordinary or unwanted). The SMS pipeline then adds a second binary model (promotional or scam), which together gives three outcomes. The comment model is **multi-label**: six independent yes/no questions, because one comment can be toxic, obscene and an insult at the same time.

**Why traditional ML and not deep learning?** The data is text with simple signals, we need to explain every decision, the models must run in a few milliseconds on a laptop (see the Dashboard's speed table), and a beginner should be able to read the model's weights. Logistic regression on TF-IDF features is a strong, honest baseline. The cost: no understanding of meaning (see **negation** below).

## 2. Turning text into numbers
**Feature.** A number the model can use. For text: how often each word, or each short run of characters, appears.

**TF-IDF.** "Term frequency x inverse document frequency": a word counts more when it is frequent in this message and rare across all messages. Words like "the" fade; "urgent" and "verify" stand out.

**n-gram (word and character).** A word n-gram is a run of n words ("click here"); a character n-gram is a run of n letters ("verif"). Character n-grams survive misspellings and look-alike letters ("C0STC0").

**URL features.** For addresses we measure instead of counting words: length, subdomain levels, hyphens, whether a brand name appears in the wrong place. 44 measurements, computed by one shared function used in training and in the app (**train/serve parity**).

## 3. The models
**Logistic regression.** Learns one weight per feature; the score is a squashed (0 to 1) weighted sum. Easy to inspect: the largest weights are the words that push towards "unwanted". Chosen for the SMS, email, comment and URL models after comparing alternatives.

**Naive Bayes, linear SVM, random forest, gradient boosting.** Compared in the training scripts. Naive Bayes is fast and simple; linear SVM is similar to logistic regression; trees can capture interactions but are heavier. The comparison, not a hunch, picked the winner.

**Per-label classifiers.** The comment model trains six logistic regressions on shared features, each with its own threshold.

**Pipeline.** The saved file contains every step (normalise, vectorise, classify) so that new text goes through exactly what training text went through. The detector refuses to load a file whose SHA-256 differs from `metadata.json`.

## 4. Handling data
**Deduplication.** Remove repeated rows. Duplicates across train and test inflate scores, because the model is tested on what it memorised. SMS: 79,989 rows became 74,364 unique.

**Train / validation / test.** Train to learn weights; validation to choose settings and thresholds; test touched once at the end. Never choose anything by looking at test.

**Leakage.** Information from the test set sneaking into training. Examples met here: the same message in two splits, the same website domain in train and validation (37.9% of validation URLs at one point), and external test sets that overlapped training data.

**Grouped and stratified splitting.** Keep related items together (same domain) so a split really separates sites; keep the class ratio the same in every split.

**Temporal and external tests.** Test on data from a later period (URLs collected after training) or a different source (Indian telecom SMS, CompPhish). They measure what the in-distribution test cannot.

**Class imbalance.** Real spam is rarer than ordinary mail. The data here is 38% spam, so measured precision is higher than real-world precision.

## 5. Measuring quality
**Confusion matrix.** Four counts: true positives, false positives (false alarms), false negatives (misses), true negatives.

**Precision.** Of everything flagged, how much was really bad. **Recall.** Of everything bad, how much was caught. **F1** combines them. Which matters more depends on harm: a false alarm hides a real bank alert, so the SMS and email thresholds favour precision.

**Threshold.** The score at which we flag. Chosen on validation data by a rule written before looking at results (for example "highest recall with precision at least 0.98"). Changing it trades false alarms for misses.

**PR-AUC vs ROC-AUC.** Both summarise quality over all thresholds. PR-AUC is more honest when the bad class is rare, because ROC-AUC can look excellent while precision is poor.

**Macro vs micro averaging.** Macro-F1 averages labels equally (so rare labels count); micro pools all decisions (so common labels dominate). The comment model reports both.

**Baseline.** A trivial model (always predict the majority) that every real model must beat.

## 6. When a good score is not good
**Shortcut learning.** The model finds an easy clue that is not the real signal. SMS: "Rs" (rupee amounts) meant spam in the data, so genuine bank alerts were flagged. Email: dates and company words identified "Enron versus not". URLs: `www`, `https` and an empty path identified the data source.

**Source bias.** The labels differ because the data came from different places, not because of what the text means.

**Generalisation gap.** The drop from same-source test to different-source test. On the Dashboard: email recall is 99% on Enron-era data and about 69% on real 2026 spam.

**Concept drift.** The world changes (new scams, new platforms), so a model slowly gets worse. Hence the maintenance plan: re-measure on fresh data, retrain when it drops.

**Guard set.** A small hand-written test of cases we care about (modern legitimate email, bank alerts, ordinary links). Too small to measure, useful to catch a bad model. A model that fails the guard is reported as failing; the app then shows a "Known weakness" warning.

**Pre-registration.** Writing the rule for choosing a model, or accepting a change, before seeing the results. The rejected comment-bias round in Phase 9 followed a rule fixed beforehand.

**Negation and context.** A word-based model cannot tell "you are not an idiot" from "you are an idiot". It is a documented limit.

**Bias probe.** Neutral sentences ("I am a ... person") filled with identity words, to see whether the word alone raises the score. For the worst term the comment model flags 50% of them.

## 7. Explaining a prediction
**Influential words.** For linear models the weights say which words pushed the score up; for the two-stage SMS model we also remove each word and see how much the score drops (**occlusion**). Explanations are approximate: they describe the model, not the truth about the message.

## 8. Responsible use
**A flag is a prompt to look, never proof.** The app never says "safe", and the comment model must not moderate people automatically (see its model card).

**Hostile data.** Spam and phishing are handled as inert text: links are never opened, HTML is reduced to visible text, error tables quoting real messages are git-ignored.

**Privacy.** Nothing a user types is stored or logged; the Dashboard keeps only counts of verdicts for the current session; private data is excluded from Git and checked by `scripts/check_repo_hygiene.py`.

## 9. Engineering practice
**Reproducibility.** Pinned requirements, a fixed seed, `python -m scripts.rebuild_all` listing the pipeline in order, and metadata (library versions, data sizes, checksum) stored beside each model.

**Tests as documentation.** 389 tests, including tests that check the leakage rules, that every model loads in a fresh process, that no log or storage code reaches user text, and that the model cards match the model files.

**Continuous integration.** The same checks run automatically in a clean machine on every push.

## Where to practise
- Change a threshold in a training script and watch precision and recall move (`reports/metrics/`).
- Open `reports/metrics/*_logreg_top_words.csv` and ask which top words are real signals and which are shortcuts.
- Compare a model's same-source and different-source numbers on the Dashboard and explain the gap in one sentence.
- Read the Phase READMEs in order: each lists what went wrong and how it was found.
