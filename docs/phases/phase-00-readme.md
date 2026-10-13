# Phase 0 README: Requirements, Scope and Risks

> Purpose of this document: explain **what we did, why we did it, and how you would make the same decisions on a different ML project.** A copy of this format is written after every phase.

---

## 1. What Phase 0 produced

| File | What it is |
|---|---|
| `docs/requirements.md` | What the app must do (functional), how well (non-functional), what is out of scope, how success is measured |
| `docs/risks.md` | 17 things that could go wrong, with mitigations and the phase that handles each |
| `docs/wireframes.md` | Text sketches of every page and every error state |
| `docs/decisions.md` | The decision log: 10 decisions so far, each with its reason |

No code and no data yet. That is intentional.

## 2. Why plan before coding?

Beginner ML projects usually fail in one of three ways:

1. **They never end**, because nobody wrote down what "done" means.
2. **They produce a number nobody trusts**, because success was defined after seeing the results.
3. **They work in a notebook and not in an app**, because input and output formats were never defined.

Phase 0 prevents all three. Its most important rule: **decide how you will judge the project before you see any result.** If you pick the success metric after seeing scores, you will (without meaning to) pick the one that looks best.

---

## 3. The first big decision: what kind of ML problem is this?

Every ML project starts by naming the **task type**. It decides the algorithms, the metrics and the data you need.

### 3.1 The main task types

| Task type | The question it answers | Output | Examples |
|---|---|---|---|
| **Classification** | Which category? | A label (or a score for each label) | Spam / not spam; phishing / legitimate |
| **Regression** | How much? | A number on a continuous scale | House price; delivery time; temperature |
| **Clustering** | Which items group together? (no labels) | Group IDs | Customer segments |
| **Anomaly detection** | Is this unusual? | Normal / unusual score | Fraud with few or no labeled examples |
| **Ranking / recommendation** | Which items first? | Ordered list | Search results; movies |
| **Generation** | Create new content | Text, image, etc. | Chatbots |

### 3.2 A 3-question test to choose

1. **Do I have labeled examples?**
   - Yes -> *supervised learning* (classification or regression).
   - No -> *unsupervised* (clustering, anomaly detection).
2. **What is the answer I want?**
   - A category (finite list) -> **classification**.
   - A quantity on a scale (where 2.7 is meaningful) -> **regression**.
3. **Could it be both?** Often yes. Then ask what decision follows. A "spam score" of 0.93 is a number, but the user's decision is "spam or not", so it is **classification** that outputs a score.

### 3.3 Why every SmartShield module is classification

| Module | Why classification, not regression |
|---|---|
| SMS / Email spam | The answer is a category: spam or not. There is no "amount of spam" in the data, only labels. |
| Phishing URL | Category again: phishing or legitimate. |
| Harmful content | Several yes/no questions at once (toxic? threat? ...), so **multi-label classification**. |

**A common beginner confusion:** the model output is a *probability-like score* between 0 and 1. That does not make it regression. Regression predicts a quantity that exists in the real world (a price). Our score is the model's confidence in a category. We then choose a **threshold** to turn the score into a decision.

### 3.4 Three flavours of classification

| Flavour | Meaning | In this project |
|---|---|---|
| **Binary** | Exactly 2 classes | SMS, Email, Phishing |
| **Multi-class** | One of 3+ classes (exactly one) | Possible later: ham / spam / smishing |
| **Multi-label** | Several labels, each yes/no, can be true together | Toxicity (a comment can be toxic + insult + obscene) |

Choosing between multi-class and multi-label: *"Can an item have two labels at once?"* Yes -> multi-label. No -> multi-class.

---

## 4. How to choose a suitable algorithm (the method you will reuse)

### 4.1 The decision flow

```text
1. Task type?            -> classification (here)
2. What is the data?     -> text / tabular numbers / images / audio
3. How much data?        -> hundreds / thousands / millions
4. What does the app need? -> probabilities? explanations? speed? small file?
5. Pick 1 dummy + 1 simple + 1-2 stronger candidates
6. Compare on validation data with the SAME split
7. Choose the simplest model that is nearly as good as the best
```

### 4.2 Match the algorithm family to the data

| Data type | Good starting algorithms | Why |
|---|---|---|
| **Text** turned into word counts/TF-IDF (thousands of sparse columns) | Naive Bayes, Logistic Regression, Linear SVM | Linear models are fast and strong when there are many features and few "interactions" between them |
| **Tabular numbers** (URL length, dot count, ...) | Random Forest, Gradient Boosting, Logistic Regression (with scaling) | Tree models handle mixed scales and non-linear rules without scaling |
| **Images / audio** | Convolutional neural networks (deep learning) | Learn patterns in pixels and waveforms |
| **Text needing real meaning/context** (sarcasm, long documents) | Transformers (deep learning) | Understand word order and context, but need more compute |

### 4.3 The algorithms in this project, in plain language

| Algorithm | One-sentence idea | Strengths | Weaknesses | Where we use it |
|---|---|---|---|---|
| **Dummy classifier** | Always predicts the most common class | Gives the zero-skill baseline | Useless as a model | Every module, as the floor |
| **Multinomial Naive Bayes** | Counts how often each word appears in spam vs ham, then multiplies the evidence | Very fast, good on short text, works with little data | Assumes words are independent; scores are poorly calibrated | SMS/Email baseline |
| **Logistic Regression** | Learns a weight per feature; sums weights; squashes to 0..1 | Gives probabilities; weights are explainable; strong on text | Can only draw straight (linear) boundaries | SMS/Email main candidate, toxicity, URL |
| **Linear SVM** | Finds the straight boundary with the widest safety margin | Often the best accuracy on text | No native probabilities (needs calibration) | SMS/Email comparison |
| **Random Forest** | Many decision trees vote | Handles non-linear patterns, no scaling needed | Bigger files, slower, less explainable | Phishing URL features |
| **Char n-gram TF-IDF + Logistic Regression** | Treats URL/text as overlapping chunks of characters | Catches disguised spellings and URL patterns with no hand-made features | Large feature space | Toxicity and URL candidate |

### 4.4 Rules for picking among candidates

1. **Always include a Dummy baseline.** If your model barely beats it, it learned nothing useful.
2. **Start simple.** A simple model that works is easier to debug, explain and ship.
3. **Add complexity only for a meaningful gain.** If a bigger model gains 0.2% on validation, the added size, time and opacity are rarely worth it.
4. **Let the app's needs break ties.** Logistic Regression often wins here even if not top-scoring, because the app needs probabilities and explanations.
5. **Do not choose by the test set.** Compare on validation; look at test once.
6. **Expect near-ties.** On clean text data the three spam models usually land within about 1%. The comparison is for learning and for evidence, not because we expect a dramatic winner.

---

## 5. Why each technology was chosen

| Technology | Role | Why this one | Alternatives and why not (yet) |
|---|---|---|---|
| **Python** | Language | Dominant ML ecosystem, readable for beginners | R (fewer app tools), Java (heavier) |
| **Pandas / NumPy** | Load and manipulate tables / arrays | The standard; every tutorial and library speaks them | Polars (faster but smaller ecosystem) |
| **scikit-learn** | Traditional ML algorithms, pipelines, metrics | One consistent API; ideal for text and tabular data on CPU | PyTorch / TensorFlow (deep learning, needs more compute, overkill here) |
| **CountVectorizer / TF-IDF** | Turn text into numbers | Simple, fast, explainable; the baseline every text project should beat | Word embeddings / transformers (later) |
| **Joblib** | Save and load fitted pipelines | Efficient for scikit-learn objects (big arrays) | ONNX (portable, extra complexity); pickle (same risk, less efficient) |
| **Streamlit** | Web interface in pure Python | A beginner can build a working UI in hours; no JavaScript | Flask / FastAPI + frontend (more work, separate backend) |
| **Matplotlib / Seaborn** | Charts and evaluation plots | Standard, scriptable, save to files for the README | Plotly (interactive, optional) |
| **pytest** | Automated tests | Simple, readable, works with CI | unittest (more boilerplate) |
| **Git / GitHub** | Version control, portfolio, CI | Industry standard | None sensible |
| **Jupyter notebooks** | Exploring data interactively | Seeing samples and charts live is how beginners learn | Scripts for reproducible training; notebooks only for exploration |

**Security note carried forward:** Joblib files can execute code on load. We only load files we built ourselves and will verify a checksum (Phase 9).

---

## 6. Requirements: how to write them

### 6.1 Functional vs non-functional
- **Functional** = *what* the system does. "Returns Spam or Not spam."
- **Non-functional** = *how well*. "Under 1 second. Never stores input."
- A requirement is good if you can **test** it. "Fast" is not testable. "Under 1 second" is.

### 6.2 Out of scope is a feature
Writing what you will *not* build protects the schedule. SQLite history, accounts, transformers and a mobile app are all out of v1 because each would add weeks and teach little new ML.

### 6.3 Success criteria with "TBD"
You may notice targets like "precision >= TBD". That is deliberate. A good target depends on the validation results and the cost of mistakes, so we fix the *rule* now (e.g. "false positives are worse for spam") and the *number* in the modelling phase, **using validation data, never test data**.

### 6.4 Which error is worse? (decide now, not later)

| Module | Worse error | Reason | Metric emphasis |
|---|---|---|---|
| Spam | False positive | A real message gets hidden | Precision on the spam class |
| Phishing | False negative | A user may be tricked | Recall on phishing at a capped false-alarm rate |
| Toxicity | Both matter | Missing harm vs over-silencing | Per-label F1, with tuned thresholds |

This one decision shapes the metric, the threshold and the UI wording for the entire module.

---

## 7. Risk thinking (a beginner-friendly version)

For each thing that could go wrong ask: **how likely? how bad? what will I do about it, and when?** The risk register answers all three. The most important ML-specific risks, which you should check on *every* future project:

1. **Leakage:** test data secretly appears in training.
2. **Source bias:** the model learns where examples came from, not what the class means.
3. **Distribution shift:** real inputs differ from training data (old data, new world).
4. **Label mistakes and mixed conventions:** 1 means different things in different files.
5. **Training/serving mismatch:** the app computes features differently from training.
6. **Harmful errors and bias:** who is hurt when the model is wrong?

---

## 8. Wireframes: why sketch first?

Sketching each page forces answers to questions like "what does the user see when the model file is missing?" before they become bugs. It also shows which functions the inference layer must provide (`predict_spam(...)`, `predict_url(...)`, `predict_toxicity(...)`), which keeps the UI separate from the ML code.

---

## 9. The decision log

`docs/decisions.md` records each choice with its reason. Writing the reason down is the skill. On your next project, a decision log is the best way to learn from your own judgement and defend it in an interview.

---

## 10. Common mistakes in Phase 0 (and how we avoided them)

| Mistake | How we avoided it |
|---|---|
| Starting with an algorithm ("I'll use a neural network") | Started with the task and data |
| No definition of done | Success criteria written before results |
| Treating accuracy as the goal | Chose the error that matters first |
| Ignoring data origin | Risk register lists source bias, drift and licensing |
| Too ambitious | Explicit out-of-scope list |

---

## 11. Self-check questions

Answer in your own words before Phase 1:

1. Why is spam detection classification and not regression?
2. Why is the toxicity module multi-label rather than multi-class?
3. Why must the Dummy baseline be included?
4. For spam, which error is worse, and what does that mean for the metric?
5. Why can't we mix 2026 spam with 2002 ham and trust the score?
6. Why do some success targets say TBD?

## 12. Done when (Phase 0 checklist)

- [ ] You read `requirements.md` and agree with it (edit anything you disagree with)
- [ ] You can explain the app in 60 seconds, including what it does *not* do
- [ ] You answered the six self-check questions
- [ ] You added at least one decision or risk of your own

## 13. Next: Phase 1

Environment, repository and project skeleton: Python 3.11/3.12, virtual environment, pinned packages, folder structure, `.gitignore`, ruff, Git/GitHub, and a hello-world Streamlit page.
