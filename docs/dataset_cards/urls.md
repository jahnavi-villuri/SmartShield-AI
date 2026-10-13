# Dataset card: URLs

> A dataset card says where data came from, what it contains, how we used it and where it can mislead. Figures come from the generated reports in `reports/data_quality/` and the phase READMEs. The datasets themselves are **not** in this repository (they are downloaded by hand and git-ignored).

## Sources
| Source | Period | Size used | Role |
|---|---|---|---|
| PhreshPhish (CC BY 4.0, anti-phishing research; only the URL columns were read, remotely) | Jul 2024 - Dec 2025 | 400,114 train / 82,682 val / 162,718 test | train / validation / **newer temporal test** |
| PhiUSIIL (UCI 967) | about 2023 | 228,037 | training (**label flipped**: 1 = legitimate there) |
| CompPhish | 2024-2026 | 12,148 after filtering | **external test, never trained on** |
| PhishBench-External | - | 856 after filtering | external test |
| OpenPhish community feed, 2026-10-12 | 2026 | 286 | external test (phishing only) |

## Labels
1 = phishing, 0 = legitimate (PhiUSIIL and CompPhish were converted at load; a unit test checks each loader).

## What we did to the data
- Features come from the URL text only. The query string and fragment are ignored.
- Split by registered-domain groups so one site never sits in both train and validation; consistent group assignment across PhreshPhish and PhiUSIIL after a health check found 37.9% of validation rows sharing a domain with training.
- External sets are filtered against every training URL so they stay truly external.

## Known problems
- **Structural shortcuts** found and removed as features: PhiUSIIL's legitimate URLs are all bare `https://www.<domain>` homepages; `https`, `www`, a query string and an empty path were each a shortcut.
- **Free hosting platforms:** in the training data almost every URL on platforms such as github.io or docs.google.com is phishing, so legitimate pages there are flagged.
- Legitimate URLs in public data skew towards popular sites; small legitimate sites are flagged more (25% of CompPhish's legitimate URLs).
- No page content, certificate, domain age or reputation data.

## Risks
Some URLs are live phishing sites. They are handled as text only, printed defanged when listed, and never fetched.

Further reading: `reports/data_quality/phishing.md`, `docs/phases/phase-07-readme.md`.
