# Dataset cards

> A dataset card says where data came from, what it contains, how we used it and where it can mislead. Figures come from the generated reports in `reports/data_quality/` and the phase READMEs. The datasets themselves are **not** in this repository (they are downloaded by hand and git-ignored).

| Card | Datasets | Used for |
|---|---|---|
| [SMS](sms.md) | UCI SMS Spam Collection, Mendeley SMS Phishing, ScamShield `Smishing_Dataset`, ScamShield Indian telecom SMS | SMS training pool and external test |
| [Email](email.md) | Enron-Spam, untroubled.org spam archive (2026) | Email training pool and a 2026 spam-only test |
| [Comments](comments.md) | Jigsaw Toxic Comment Classification | Comment training, validation and official test |
| [URLs](urls.md) | PhreshPhish, PhiUSIIL, CompPhish, PhishBench, OpenPhish feed | URL training, temporal test, external tests |
| [Synthetic, hand-written and private data](synthetic-and-private.md) | template messages, challenge sets, darkknight25, the owner's own messages | augmentation, smoke tests, real-world checks |

## Licences at a glance
| Data | Licence (as recorded in the project) | Redistributed here? |
|---|---|---|
| UCI SMS Spam Collection | CC BY 4.0 | No |
| Mendeley SMS Phishing Dataset | CC BY 4.0 | No |
| ScamShield (Hugging Face) | MIT per its card; licences of underlying sources **not verified** | No |
| Enron-Spam | repository licence GPL-3.0 (as recorded in this project) | No |
| untroubled.org spam archive | free use "without restriction", citation requested | No |
| Jigsaw Toxic Comment | labels CC0, comment text CC BY-SA 3.0 (Wikipedia) | No |
| PhreshPhish | CC BY 4.0, for anti-phishing research | No (URL columns read remotely) |
| PhiUSIIL | UCI repository terms (check the dataset page) | No |
| CompPhish, PhishBench, OpenPhish feed | each source's own terms; OpenPhish community feed is for research | No |
| darkknight25 phishing/benign emails | MIT (Hugging Face card) | No |

Please cite the original authors if you reuse any of these. If a licence above is marked "not verified" or you are unsure, check the source before redistributing anything derived from it. **Models trained on share-alike data (Jigsaw text, CC BY-SA) may carry obligations; seek advice before commercial use.**

## Provenance gaps (honest list)
- The exact download URLs of some sources were not recorded at download time (only the Hugging Face id of Jigsaw and the PhreshPhish listing are in the code). Record the URL and download date next to each raw folder when you refresh data.
- Row-level origin inside the ScamShield `Smishing_Dataset` is unknown.
- SmishTank (real 2023-24 smishing) was not obtained; it needs a manual request.
