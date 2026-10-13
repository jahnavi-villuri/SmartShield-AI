# Phase 3b README: From "spam / not spam" to Legitimate / Promotional / Scam

> Trigger: the project owner tested two real messages and both were flagged "Spam": a genuine SBI debit alert and a genuine Taco Bell offer. This phase explains what each failure was, what we changed, what it fixed and what it did not.

## 1. Two messages, two different problems

| Message | What the model said | What it really is | Problem type |
|---|---|---|---|
| SBI: "Dear UPI user A/C XXXX debited by 320.00 on date 08Oct26 trf to ... Refno ... If not u? call-1800111109 for other services-18001234-SBI" | Spam (0.77 on the first model) | A legitimate bank alert | **A model error** (false alarm) |
| Taco Bell: "FLAT 50% OFF at Taco Bell! DINE-IN and pay via District by Zomato..." | Spam (0.99) | A real marketing message from a real brand | **A definition problem**, not a bug |

The model was behaving consistently with its training data. In that data, "spam" means *any unsolicited marketing or scam*: about 22,400 of the roughly 27,600 spam messages in the biggest source are tagged "General Spam / Telemarketing". So an honest brand promotion is "spam" by the data's own definition. The question is whether that is the right definition for *this app*.

**Lesson: before blaming the model, check the label definition.** A binary label ("spam") hides at least two different things the user cares about differently: *annoying* and *dangerous*.

## 2. The decision: three categories

The owner chose to show **Legitimate / Promotional / Scam**:

| Category | Meaning | Example |
|---|---|---|
| **Legitimate (Not spam)** | Ordinary or transactional message | bank alert, OTP, personal text |
| **Promotional** | Marketing from a business; unsolicited but not deceptive | "FLAT 50% OFF at Taco Bell" |
| **Scam** | Tries to deceive or take something | "Your KYC expired, click here", "You have won Rs 25 lakh" |

## 3. A discovery that changed the plan: the dataset's sub-labels are unreliable

The big dataset has a column `head2_scam_intent` that looks like exactly what we need (Telemarketing vs Fake KYC vs Lottery...). Reading samples showed it is **not trustworthy**:

- A bank phishing link ("We have identify some unusual activity in your Internet banking. Please login via http://ib.nab.com.au.card-verify.ga") was tagged *General Spam / Telemarketing*.
- Ordinary cashback and restaurant promotions ("get flat rs cashback at ... pay via ...") were tagged *UPI Fraud*.

The category looks keyword-derived. Training a 3-way model on it would have taught the model that a Taco Bell-style offer is a "UPI fraud". So we did **not** use it.

> **General lesson:** a label column existing does not mean it is correct. Read samples from every category before using it.

## 4. What we built: a two-stage detector

```text
message -> [Stage 1: legitimate vs unwanted]  --legit-->  "Not spam"
                    |unwanted
                    v
           [Stage 2: promotional vs scam]  -->  "Promotional"  or  "Scam"
```

- **Stage 1** is the existing model (`models/sms_spam/`), retrained in Round 4.
- **Stage 2** is new (`models/sms_type/`), `src/training/train_sms_type.py`: a Logistic Regression on word n-gram TF-IDF with the shared normalization, trained only on messages Stage 1 would call unwanted.
- The detector (`src/detection/spam_detector.py`) combines them. If Stage 2 is missing, unwanted messages are shown as plain "Spam", and the app says so. A tampered Stage 2 file is refused like any other (checksum).
- The page shows a green / amber / red result, an explanation of what each category means, and two scores: "Unwanted-message score" and "Scam likelihood".

### Why two stages instead of one 3-class model
Stage 1 already has 50,000+ examples and a tested threshold for the costly error (hiding a real message). Stage 2 only needs to learn one narrower distinction and can be small. A single 3-class model would have needed reliable 3-way labels for all 74,000 messages, which we do not have.

## 5. How Stage 2 got its labels (honest account)

1. Sampled 500 messages that the data labels as spam: 340 from the training split, 80 validation, 80 test (fixed seed).
2. The **AI assistant** labelled each one **S (scam)**, **P (promotional)** or **U (unclear)** following written rules (`docs/sms_type_labeling_guide.md`).
3. Result: 309 P, 142 S, 49 U. The U messages are excluded, leaving **451 usable labels**.

**These labels have not been checked by a human.** One labeller, written rules, judgement calls on contests and premium-rate services (rule: "enter a competition" is P; "you have won" is S). A person spot-checking 50 of them in `data/labels/sms_type_labels.csv` would be the best next improvement.

## 6. Results of Stage 2 (small data, read with care)

| Check | Result |
|---|---|
| Training set | 380 messages (123 scams) |
| Cross-validated accuracy (out-of-fold, 5-fold) | 0.924; scam precision 0.905, recall 0.854 |
| Held-out hand-labelled test | 71 messages (19 scams): accuracy 0.958; scam precision 0.900, recall 0.947 (always saying "promotional" scores 0.732) |
| Taco Bell message | **Promotional**, scam likelihood 0.03 |
| 10 scam/promo messages in the challenge set | 6 of 10 typed as I had labelled them |

Model choice by the pre-agreed rule: three models (word LR, character LR, Naive Bayes) all had cross-validated PR-AUC of 0.963-0.967, so the rule (within 0.01 of the best, prefer the simplest) picked word-level Logistic Regression. The test set has only 19 scams, so its 0.958 could easily be 0.90 or 0.99 on other draws; the cross-validated 0.92 is the better estimate.

The 4 challenge "errors": a loan lure, a "free gift, reply YES" message, a "FREE entry to win, text WIN" message and a reward-points phishing message. Two of those follow my own labelling rule (contest entry = promotional), which contradicts how I labelled that probe file; this is exactly the kind of ambiguity that one labeller creates.

## 7. Round 4: retraining Stage 1 with more bank-alert styles

The SBI message showed that my earlier synthetic bank alerts were too narrow ("Rs 300 credited..."). Real alerts use abbreviations (`trf`, `w/d`, `Refno`), account formats (`A/C XXXX`), and helpline text. I added templates in those styles, and gave the scam templates toll-free-looking numbers too, so "has a helpline number" does not become a shortcut for "legitimate". I did **not** copy the owner's message wording ("trf to", "Refno", "for other services", "UPI user"), so that message stays a fair test.

| | Round 3 | Round 4 |
|---|---|---|
| Test precision / recall | 0.984 / 0.951 | 0.982 / 0.948 |
| External precision / recall | 0.874 / 0.863 | 0.860 / 0.806 |
| Legitimate challenge messages wrongly flagged | 4 of 23 | 3 of 23 |
| Threshold | 0.650 | 0.778 |

**Honest reading: Round 4 is not clearly better.** In-domain is unchanged, one more challenge message is correct, but external recall fell by about 6 points. I kept it because it is the latest and the differences are partly noise (the threshold also moved), but I did not keep a copy of the Round 3 model file, only its metrics, so I could not restore it without retraining.

## 8. The SBI message: still not fixed

| Version | Unwanted score (threshold 0.78) |
|---|---|
| Original message (de-identified copy; the version with the real name scored 0.87) | 0.93 -> flagged (shown as "Promotional", scam likelihood 0.30) |
| Without "for other services-18001234-SBI" | 0.70 -> would pass |
| Without "If not u? call-1800111109" | 0.88 |
| Without both | 0.51 -> would pass |
| "trf to" replaced by "transferred to" | 0.99 |

So the model takes the helpline text and the abbreviated "bank SMS" wording as promotion signals. **This failure is real and still open.** It cannot be fixed honestly with more of my own templates, because I would be writing the answer from the one example I have seen. It needs real examples. See next steps.

A small improvement is that it is no longer labelled "Scam" (0.30), only "Promotional", but a legitimate bank alert shown as promotional is still wrong.

## 9. Difficulties faced in this phase

| Difficulty | What we learned |
|---|---|
| Two "wrong" messages had different causes | Separate model error from definition error before fixing |
| Dataset sub-labels were wrong | Read samples before trusting a label column |
| No ground truth for promotional vs scam | Built a hand-labelled set; recorded who labelled it and how; limited the claims |
| My own probe set contradicted my labelling rule | Two documents written by one person can disagree; keep rules and probes consistent |
| Round 4 slightly worse externally | More training data is not automatically better; we report it |
| The reported failure survived a fix | A fix that targets "the style of one example" is overfitting to the example; get independent data |
| Long shell commands with quotes failed | Put multi-line scripts in files |

## 10. What would actually fix it (next steps, in order of value)

1. **Real legitimate messages** from your own phone (bank alerts, OTPs, delivery updates, utility bills), 30-50 of them, with names, last digits of accounts and reference numbers replaced. Put them in `data/private/sms_my_legit.csv` (never committed). Use half to train and half as a clean test. This is the only reliable fix for the SBI-style failure.
2. A human spot-check of 50 labels in `data/labels/sms_type_labels.csv`, and label 300 more messages.
3. Request SmishTank for modern scams.
4. If real legitimate data stays scarce, consider treating the unwanted score as a *warning level* instead of a hard decision.

## 11. Concepts introduced

Label definition vs model error; weak/unreliable labels; hand labelling and its limits; two-stage classification; cascaded thresholds; honest reporting of a fix that did not work; overfitting to a single example.

## 12. Self-check questions

1. Why was the Taco Bell result "consistent" with the training data, and why might it still be the wrong product behaviour?
2. Why did we not train a 3-class model on the dataset's `head2_scam_intent` column?
3. Why use two stages rather than one 3-class model?
4. What are three weaknesses of the Stage 2 labels, and how does each affect the numbers?
5. Why is 0.958 accuracy on 71 test messages weaker evidence than 0.924 cross-validated on 380?
6. Why can I not fix the SBI message by writing more templates in its style?
7. What would you check to decide whether Round 4 or Round 3 is better?

## 13. Done when

- [ ] You ran the app and tried a bank alert, a promotion, a scam and an ordinary message
- [ ] You read `docs/sms_type_labeling_guide.md` and spot-checked some labels
- [ ] You decided whether to provide real legitimate messages (section 10)
- [ ] You answered the self-check questions


---

## 14. Round 5: adding the owner's real messages (a clean before/after)

After the SBI failure the owner supplied 20 real messages from their own phone (13 legitimate: OTPs, SBI notices, a government quiz, a regulator advisory; 7 promotional). **None was spam or a scam**; the owner stated promotional messages are not spam. They live in `data/private/my_sms/messages.csv` (ignored by Git).

### 14.1 Why the model got them wrong (a data finding)
In the 74,000-message training pool:

| Pattern | Messages | Labelled spam |
|---|---|---|
| mention SBI | 831 | **100%** |
| mention YONO | 314 | **100%** |
| contain "OTP" | 74 | 89% (only 8 genuine OTPs exist) |
| contain "do not share" | 60 | 88% |

All of the SBI/YONO messages in the data are phishing templates, so the model learned "SBI, YONO, OTP, 'do not share' means scam". Genuine messages from those same senders look like scams to it. **This is a data gap, not a modelling bug, and a rule or threshold tweak would only hide it.**

### 14.2 How the real messages are used
- Each message is assigned to a **train half or a test half by a hash of its text** (`src/data/private_sms.py`). Adding more messages later never moves an existing one, so a message held out for testing can never silently become training data.
- The train half is **repeated 30 times** in the training set so it carries weight against about 55,000 other rows. The test half is never trained on.
- Mapping for the main model: legitimate -> 0, promotional or scam -> 1.
- `python -m scripts.evaluate_my_sms` prints the honest report: it separates the held-out half (a fair test) from the trained half (not a test) and counts "Scam" labels on non-scam messages separately, because that is the worst mistake.

### 14.3 Result on the 13 held-out messages (never trained on)

| | Before (Round 4) | After (Round 5) |
|---|---|---|
| Exactly right | 8 of 13 | **10 of 13** |
| Legitimate messages flagged as promotional or scam | 4 of 8 | **2 of 8** |
| False "Scam" labels | 2 | **1** |

Only **7 of the owner's messages** were in the training half (5 legitimate), so this is a small experiment, but it moved the result in the right direction and cost nothing elsewhere:

| Stage 1 model (threshold) | Round 4 (0.778) | Round 5 (0.700) |
|---|---|---|
| Test precision / recall | 0.982 / 0.948 | 0.982 / 0.949 |
| External precision / recall | 0.860 / 0.806 | 0.884 / 0.845 |
| Challenge legitimate messages flagged | 3 of 23 | 2 of 23 |

### 14.4 The original SBI alert, re-tested
| Message | Round 4 | Round 5 (threshold 0.70) |
|---|---|---|
| SBI debit alert, original wording | 0.93 -> flagged | **0.63 -> Not spam** |
| Same style, other amount and name | n/a | 0.28 -> Not spam |
| Same style, a credit alert | n/a | 0.13 -> Not spam |
| Taco Bell offer | Promotional | Promotional |
| Bank phishing ("SBI YONO account will be blocked...") | Scam | Scam |

**The reported failure is fixed, but the original message passes with a thin margin (0.63 against 0.70).** The held-out set still contains one genuine bank notice labelled "Scam" (the SBI branch-feedback message with a link) and one OTP labelled "Promotional".

### 14.5 What this proves, and what it does not
- It proves the diagnosis: **real examples of the missing kind fix the problem**, while more synthetic templates did not.
- It does not prove generalization to all banks and senders. Seven training messages cannot cover the variety of real alerts.
- The owner's Telugu messages remain unreadable to an English-only model; it would be better to detect the language and say so.

### 14.6 What the owner should do next
Send more messages (50-100 more). The loader picks them up automatically; rerun `python -m src.training.train_sms` (about 15 minutes) and `python -m scripts.evaluate_my_sms`.

### 14.7 Lessons
1. When a real-world failure persists, look at **what the training data contains about that sender or pattern** before touching the model.
2. Use a deterministic split for scarce, precious data so that new data never contaminates an existing test.
3. Count mistakes by cost: a "Scam" label on a genuine bank notice is worse than a "Promotional" label.
