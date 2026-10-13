# Dataset card: SMS

> A dataset card says where data came from, what it contains, how we used it and where it can mislead. Figures come from the generated reports in `reports/data_quality/` and the phase READMEs. The datasets themselves are **not** in this repository (they are downloaded by hand and git-ignored).

## Sources
| Source | Period | Raw size | Role |
|---|---|---|---|
| UCI SMS Spam Collection | about 2011 | 5,574 messages | training pool |
| Mendeley SMS Phishing Dataset | published 2022 | 5,971 messages (84% re-use UCI) | training pool |
| ScamShield `Smishing_Dataset` (Hugging Face), English, non-synthetic rows | claimed 2012-2024 | 69,070 messages | training pool (about 92% of it) |
| ScamShield `Indian_Telecom_SMS` and benchmark rows | about 2021 | 2,049 messages | **external test only** |

## Labels
Binary (1 = unwanted, 0 = ordinary) for the main model. A second, small model separates promotional from scam using 451 hand-labelled messages (labelled by one AI labeller following `docs/sms_type_labeling_guide.md`; not human-verified).

## What we did to the data
- Removed within-dataset duplicates, 5,617 cross-dataset duplicates and 8 conflicting-label rows: 79,989 rows to **74,364 unique**.
- Split by hash into **52,048 train / 11,164 validation / 11,152 test**, spam share 38.1% in each split.
- Kept HTML-code artefacts out of the features after finding they were a shortcut.
- Added 3,000 template-written bank messages (training only) and the owner's own messages (see the synthetic-and-private card).

## Known problems
- **Source bias:** UCI ham is Singapore student chat and UCI spam is UK promotions, so region and style leak into the label.
- **Mendeley overlaps UCI** heavily; treating it as independent would double count.
- The spam share (38%) is far above a real inbox, so measured precision overstates real precision.
- ScamShield rows have unknown origin and unverified underlying licences.
- The external set labels mostly telecom promotions as spam, which is a different definition of spam.
- English-focused; Hindi/Hinglish rows were dropped.

## Risks
Spam and scam text can contain real phone numbers and links. They are used as text only and never opened or called. Reports quote no raw messages (error-analysis tables are git-ignored).

Further reading: `reports/data_quality/sms.md`, `docs/phases/phase-02-readme.md`.
