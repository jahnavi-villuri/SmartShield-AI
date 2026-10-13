# Dataset card: Email

> A dataset card says where data came from, what it contains, how we used it and where it can mislead. Figures come from the generated reports in `reports/data_quality/` and the phase READMEs. The datasets themselves are **not** in this repository (they are downloaded by hand and git-ignored).

## Sources
| Source | Period | Size | Role |
|---|---|---|---|
| Enron-Spam | ham 1999-2002, spam 2001-2005 | 33,716 emails, 30,138 unique after cleaning | train / validation / test |
| untroubled.org spam archive, June-October 2026 | 2026 | 11,251 files, **10,228 unique** | external test, **spam only** |

## Labels
1 = spam or phishing, 0 = legitimate. The 2026 archive has only spam, so it measures the detection rate, never false alarms.

## What we did to the data
- Parsed each email safely as text (never executing or opening anything). CSS-hidden filler text, which 73% of 2026 spam uses to hide a harmless paragraph, is dropped; this changed measured detection from 32% to 77% when first fixed.
- Removed duplicates and any text identical to an Enron email.
- Masked words that merely identify the Enron source ("signature words"), because dates and company terms give the class away.
- Split by groups so related emails never straddle splits.

## Known problems
- **Era and source bias:** Enron ham is staff mail; spam is from other sources and years. The model can learn "Enron versus not", which is why a near-perfect test score is misleading.
- **No modern legitimate email** exists publicly. A small hand-written set (24 legitimate, 10 spam) and a 200-row synthetic dataset are the only modern checks.
- Part of the 2026 spam is not English; the model reads English.
- Forged headers: sender fields are not trusted and not used.

## Risks
The archive contains real spam and phishing. Files are read as data; links are never followed; the raw files are git-ignored.

Further reading: `reports/data_quality/email.md`, `docs/phases/phase-05-readme.md`.
