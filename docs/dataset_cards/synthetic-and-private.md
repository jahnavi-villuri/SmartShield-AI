# Dataset card: synthetic, hand-written and private data

> A dataset card says where data came from, what it contains, how we used it and where it can mislead. Figures come from the generated reports in `reports/data_quality/` and the phase READMEs. The datasets themselves are **not** in this repository (they are downloaded by hand and git-ignored).

| Data | What it is | Used for | Rule |
|---|---|---|---|
| Template SMS (`src/data/augment_sms.py`) | 3,000 bank/transaction messages (1,500 legitimate, 1,500 scams), links on the reserved `.example` domain | training only | never evaluated on; teaches that "Rs" and amounts are not spam signals |
| Identity sentences (`src/data/augment_toxicity.py`) | neutral sentences mentioning identity terms | training only (one round) | probe uses different templates and held-out identity words |
| Challenge sets (`tests/data/*.csv`) | 33 SMS, 34 emails, 36 comment sentences, 68 URLs, written by hand | smoke tests and guards | too small for measurement; never trained on |
| darkknight25 email dataset (Hugging Face, MIT) | 200 one-sentence emails (100 benign, 100 phishing), looks synthetic | half trained at low weight, half held out | does not replace real modern email |
| Owner's private SMS (`data/private/my_sms/`) | 20 real messages | hash-split: half trains (repeated), half held out | git-ignored; never published |
| Owner's private emails (`data/private/my_email/`) | header-only file waiting for rows | same rule | git-ignored |

## Why this matters
Synthetic data can make a model look good by teaching it the generator's style, so it is kept out of every test set. Private data is real and personal: it stays on this machine, is excluded from Git by `.gitignore` and checked by `scripts/check_repo_hygiene.py`.
