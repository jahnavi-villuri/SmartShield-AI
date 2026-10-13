# Phase 12: Documentation, model cards, dataset cards, concepts and licence

## What we built
| Document | Purpose | How it stays true |
|---|---|---|
| `docs/model_cards/` (5 files) | What each model is for, what it learned from, how it scored, where it fails | **Generated** from `metadata.json`; a test fails if a card is stale |
| `docs/dataset_cards/` (6 files) | Source, licence, size, labels, cleaning, biases and risks per data family | Written by hand from the data-quality reports; a test checks every data source is covered |
| `docs/concepts.md` | Every ML idea in the project with a real example | Written once; links checked |
| `LICENSE` | MIT for the code and docs; data keeps its own licences | Tested to exist and to separate code from data |
| `tests/test_docs.py` | Link checker for README and all docs, card freshness, licence, index coverage, phase numbering | Runs in CI |

## Why these documents (concepts)
- **Model card.** A short, standard document that says what a model is for, what it must not be used for, how it was evaluated and where it fails. Reading it should stop someone from misusing the model. Ours are generated, so the numbers cannot drift.
- **Dataset card.** The same idea for data: provenance, licence, biases, risks. Many ML failures start in the data.
- **Provenance.** Where data came from, when, under what terms. We found real gaps (exact download URLs were not all recorded; the origin of individual ScamShield rows is unknown) and wrote them down instead of hiding them.
- **Licences.** Code, data and trained models can have different licences. A model trained on share-alike text may inherit obligations. We state this and recommend checking before commercial use.
- **Docs as tests.** A link checker turns "the docs are probably fine" into something CI can fail.

## Difficulties
1. **Numbers in prose go stale.** The first cards were going to be hand-written. After the Phase 9 retraining showed how quickly numbers change, we generated them from metadata.
2. **Unverifiable facts.** Some licences and source URLs were not recorded when data was downloaded. The cards say "not verified" rather than guessing; adding the URL and download date to each raw folder is a to-do for anyone refreshing data.
3. **Phase numbering.** The link/phase test first failed because Phase 3b shares number 3; the test now compares the set of numbers.

## Not done
- The copyright holder in `LICENSE` is the project owner's name; the licence choice (MIT) is a default, so change it if you prefer another.
- No translated documentation.
- Citations (`CITATION.cff`) belong to the portfolio phase.

## Self-check
1. Why generate model cards instead of writing them?
2. What is a provenance gap, and why list it?
3. Which ML concept explains why email recall is 99% on Enron-era data and far lower on 2026 spam? Where is it explained?
4. Why can the code be MIT while a model trained on CC BY-SA text still needs care?

## Done when
Every detector has a generated model card, every data source has a dataset card, the concepts page exists, the licence is present, the link test passes and the README indexes them.
