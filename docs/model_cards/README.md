# Model cards

One card per detector, generated from each model's `metadata.json` (`python -m scripts.make_model_cards`). A model card says what a model is for, what it was trained and tested on, how well it did, and where it fails.

| Detector | Algorithm | Trained | Data sizes |
|---|---|---|---|
| [SMS](sms.md) | `logreg_char_norm` | 2026-10-12 | train 55,258, val 11,164, test 11,152, external 2,049, challenge 33 |
| [Email](email.md) | `logreg_word_masked` | 2026-10-12 | train 21,554, val 4,521, test 4,521, external_2026_spam 10,228, challenge 34 |
| [Comments](comments.md) | `logreg_wordchar` | 2026-10-12 | train 115,238, val 23,892, test 23,892, official_test 63,908 |
| [URL](url.md) | `logreg_char` | 2026-10-12 | train 596,183, val 114,650, test 162,718, compphish 12,148, phishbench 856, openphish_2026-10-12 286, challenge 68 |

Read the 'Known limitations' and 'Ethical considerations' sections before using any score.
