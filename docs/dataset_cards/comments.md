# Dataset card: Comments (Jigsaw Toxic Comment Classification)

> A dataset card says where data came from, what it contains, how we used it and where it can mislead. Figures come from the generated reports in `reports/data_quality/` and the phase READMEs. The datasets themselves are **not** in this repository (they are downloaded by hand and git-ignored).

## Source
Jigsaw Toxic Comment Classification Challenge, as redistributed on Hugging Face (`thesofakillers/jigsaw-toxic-comment-classification-challenge`). Wikipedia talk-page comments (about 2017) rated by crowd workers. Licence per the card: labels CC0, comment text CC BY-SA 3.0.

## Labels
Six independent binary labels (multi-label): toxic, severe_toxic, obscene, threat, insult, identity_hate. A comment can have several or none.

## Sizes
159,272 unique comments after cleaning (train 115,238 / validation 23,892 / internal test 23,892 after the model's own split) plus the **official test set of 63,908 scored comments**. Rare labels have few positives (threat 211 and severe_toxic 367 in the official test).

## What we did to the data
- Removed duplicates and split by hash. The Hugging Face "test" split came as two stacked blocks (text without labels, labels without text); we joined them by id and kept only scored rows (-1 means unscored).
- Detected and re-downloaded a truncated training file by checking its exact size.

## Known problems
- **Identity bias:** comments that mention some identity words are labelled toxic far more often than average (for example 'gay' about 58% against a 10% base rate), so a model learns that the word itself signals toxicity. A bias probe measures this; mitigation by synthetic neutral sentences did not fix it.
- Crowd labels are noisy and rater-dependent; rare labels are unstable.
- Wikipedia talk pages in 2017 are not today's comments sections.
- English only.

## Risks
The text contains slurs, threats and harassment. Reports contain counts only, error-analysis files are git-ignored, and probe sentences are hand-written and mild. Reading raw examples can be upsetting; take breaks.

Further reading: `reports/data_quality/toxicity.md`, `docs/phases/phase-06-readme.md`.
