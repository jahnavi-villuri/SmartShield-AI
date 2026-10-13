"""Phase 3: train and compare SMS spam models, then save the selected pipeline.

Run from the project root (after `python -m scripts.prepare_sms`):
    python -m src.training.train_sms

RULES FIXED BEFORE LOOKING AT ANY RESULT (so we cannot bend them to flatter a model):
  1. Models are compared by PR-AUC (average precision) on the VALIDATION set. It does not
     depend on a threshold and suits imbalanced data.
  2. If several models are within TIE_TOLERANCE of the best PR-AUC, choose the first one in
     PREFERENCE_ORDER (simple, explainable models first).
  3. The decision threshold is the one with the highest validation recall whose validation
     precision is at least PRECISION_TARGET (a false alarm hides a real message, so it is
     the worse error).
  4. The test set and the external set are evaluated once, after the model is chosen.
"""

import hashlib
import json
import platform
import time
from datetime import UTC, datetime
from functools import partial

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from config import settings
from src.data.augment_sms import generate_augmentation
from src.data.private_sms import load_private_sms
from src.data.text_utils import normalize_key
from src.evaluation import plots
from src.evaluation.metrics import (
    get_scores,
    score_type,
    threshold_for_precision,
    threshold_metrics,
)
from src.features.text_normalize import normalize_sms

DATA_DIR = settings.DATA_PROCESSED / "sms"
EXTERNAL_FILE = settings.DATA_EXTERNAL_TEST / "sms" / "india_telecom.csv"
MODEL_DIR = settings.MODELS_DIR / "sms_spam"
METRICS_DIR = settings.REPORTS_DIR / "metrics"
FIGURES_DIR = settings.REPORTS_DIR / "figures"

CHALLENGE_FILE = settings.PROJECT_ROOT / "tests" / "data" / "sms_challenge.csv"
PRIVATE_REPEAT = 30  # each private TRAIN-half message is repeated this many times
AUGMENT_PER_CLASS = 1500  # template-written legit alerts + scams, TRAINING split only
PRECISION_TARGET = 0.98
TIE_TOLERANCE = 0.003
# Round 2 added normalized variants ("_norm") that remove dataset artifacts (see text_normalize.py).
# The selection rule is unchanged; ties prefer normalized, simple, explainable models first.
BASE_ORDER = ["logreg_word", "naive_bayes", "linear_svm_word", "logreg_char"]
PREFERENCE_ORDER = [f"{m}_norm" for m in BASE_ORDER] + BASE_ORDER
CV_FOLDS = 5

_FEATURE_FACTORIES = {
    "count_word": partial(CountVectorizer, ngram_range=(1, 2), min_df=2),
    "tfidf_word": partial(TfidfVectorizer, ngram_range=(1, 2), min_df=2, sublinear_tf=True),
    "tfidf_char": partial(
        TfidfVectorizer,
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=3,
        sublinear_tf=True,
        max_features=300_000,
    ),
}
# Feature extractors (each is FIT ON TRAINING TEXT ONLY). `_norm` = with normalize_sms().
FEATURES = {}
for _name, _factory in _FEATURE_FACTORIES.items():
    FEATURES[_name] = _factory
    FEATURES[f"{_name}_norm"] = partial(_factory, preprocessor=normalize_sms)

_BASE_CANDIDATES = {
    "naive_bayes": (
        "count_word",
        MultinomialNB,
        [{"alpha": a} for a in (0.03, 0.1, 0.3, 1.0, 3.0)],
    ),
    "logreg_word": (
        "tfidf_word",
        partial(LogisticRegression, max_iter=2000),
        [{"C": c} for c in (1, 3, 10, 30, 100)],
    ),
    "linear_svm_word": (
        "tfidf_word",
        partial(LinearSVC, dual="auto"),
        [{"C": c} for c in (0.1, 0.3, 1, 3)],
    ),
    "logreg_char": (
        "tfidf_char",
        partial(LogisticRegression, max_iter=2000),
        [{"C": c} for c in (3, 10, 30, 100)],
    ),
}
# name -> (feature set, classifier factory, hyperparameter grid chosen on validation)
CANDIDATES = {"dummy": ("count_word", partial(DummyClassifier, strategy="prior"), [{}])}
for _name, (_feat, _clf, _grid) in _BASE_CANDIDATES.items():
    CANDIDATES[_name] = (_feat, _clf, _grid)
    CANDIDATES[f"{_name}_norm"] = (f"{_feat}_norm", _clf, _grid)


def load_split(path) -> pd.DataFrame:
    # keep_default_na=False: a message that is literally "NA" or "null" must stay text.
    return pd.read_csv(path, keep_default_na=False, dtype={"text": str})


def select_model(results: pd.DataFrame) -> str:
    """Apply selection rules 1 and 2 to the per-model best validation PR-AUC."""
    real = results[results["model"] != "dummy"].set_index("model")["val_pr_auc"]
    best = real.max()
    close = [m for m in PREFERENCE_ORDER if m in real.index and real[m] >= best - TIE_TOLERANCE]
    return close[0]


def main() -> None:
    for folder in (MODEL_DIR, METRICS_DIR, FIGURES_DIR):
        folder.mkdir(parents=True, exist_ok=True)

    train = load_split(DATA_DIR / "train.csv")
    val = load_split(DATA_DIR / "val.csv")
    test = load_split(DATA_DIR / "test.csv")
    external = load_split(EXTERNAL_FILE)
    challenge = load_split(CHALLENGE_FILE)

    # Synthetic examples fix a known blind spot (legitimate bank alerts). TRAIN ONLY.
    augment = generate_augmentation(AUGMENT_PER_CLASS, settings.RANDOM_SEED)
    known = set(train["text"].map(normalize_key)) | set(challenge["text"].map(normalize_key))
    augment = augment[~augment["text"].map(normalize_key).isin(known)]
    train = pd.concat([train, augment], ignore_index=True)

    # The owner's real messages: half go into training (repeated, so they carry weight against
    # ~55,000 other rows), half stay untouched as an honest test. See src/data/private_sms.py.
    private = load_private_sms()
    private_test = None
    private_info = {"messages": 0}
    if private is not None and len(private):
        private_train = private[private["split"] == "train"]
        private_test = private[private["split"] == "test"].reset_index(drop=True)
        repeated = pd.concat([private_train] * PRIVATE_REPEAT, ignore_index=True)
        columns = ["text", "label", "source", "original_label", "group"]
        train = pd.concat([train, repeated[columns]], ignore_index=True)
        private_info = {
            "messages": len(private),
            "train_half": len(private_train),
            "test_half": len(private_test),
            "train_half_legit": int((private_train["label"] == 0).sum()),
            "repeat": PRIVATE_REPEAT,
        }
        print(f"private messages: {private_info}")
    y_train, y_val = train["label"].to_numpy(), val["label"].to_numpy()
    print(
        f"train(incl. {len(augment)} synthetic) {len(train)} | val {len(val)} | test {len(test)} | external {len(external)}"
    )

    # ---- 1. Fit each feature extractor on TRAIN only; transform train and validation ----
    fitted_features, x_train, x_val = {}, {}, {}
    for name, factory in FEATURES.items():
        vec = factory().fit(train["text"])
        fitted_features[name] = vec
        x_train[name], x_val[name] = vec.transform(train["text"]), vec.transform(val["text"])
        print(f"features {name}: {x_train[name].shape[1]:,} columns")

    # ---- 2. Hyperparameter search on VALIDATION (threshold-free metric) ----
    grid_rows, best = [], {}
    for model, (feat, factory, grid) in CANDIDATES.items():
        for params in grid:
            clf = factory(**params).fit(x_train[feat], y_train)
            scores = get_scores(clf, x_val[feat])
            row = {
                "model": model,
                "features": feat,
                "params": json.dumps(params),
                "val_pr_auc": average_precision_score(y_val, scores),
                "val_roc_auc": roc_auc_score(y_val, scores),
            }
            grid_rows.append(row)
            if model not in best or row["val_pr_auc"] > best[model]["row"]["val_pr_auc"]:
                best[model] = {"row": row, "params": params, "clf": clf, "feat": feat}
        print(f"tuned {model}: best {best[model]['params']}")
    pd.DataFrame(grid_rows).to_csv(METRICS_DIR / "sms_hyperparameter_search.csv", index=False)

    # ---- 3. Cross-validation on TRAIN (grouped) for stability of each best config ----
    cv = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=settings.RANDOM_SEED)
    summary = []
    for model, info in best.items():
        feat, factory, _ = CANDIDATES[model]
        pipe = Pipeline([("features", FEATURES[feat]()), ("clf", factory(**info["params"]))])
        cv_scores = cross_val_score(
            pipe,
            train["text"],
            y_train,
            groups=train["group"],
            cv=cv,
            scoring="average_precision",
        )
        val_scores = get_scores(info["clf"], x_val[feat])
        pick = threshold_for_precision(y_val, val_scores, PRECISION_TARGET)
        at_target = threshold_metrics(y_val, val_scores, pick["threshold"])
        summary.append(
            {
                "model": model,
                "params": json.dumps(info["params"]),
                "val_pr_auc": info["row"]["val_pr_auc"],
                "val_roc_auc": info["row"]["val_roc_auc"],
                "cv_pr_auc_mean": float(cv_scores.mean()),
                "cv_pr_auc_std": float(cv_scores.std()),
                "val_recall_at_precision_target": at_target["recall"],
                "val_precision_at_threshold": at_target["precision"],
                "target_reached": pick["reached_target"],
            }
        )
        print(f"cv {model}: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    results = pd.DataFrame(summary)
    results.to_csv(METRICS_DIR / "sms_model_comparison.csv", index=False)

    # ---- 4. Select the model by the pre-registered rules ----
    chosen = select_model(results)
    info = best[chosen]
    feat = info["feat"]
    pipeline = Pipeline([("features", fitted_features[feat]), ("clf", info["clf"])])
    val_scores = get_scores(pipeline, val["text"])
    pick = threshold_for_precision(y_val, val_scores, PRECISION_TARGET)
    threshold = pick["threshold"]
    print(f"\nCHOSEN: {chosen} {info['params']} | threshold {threshold:.4f}")

    # ---- 5. Evaluate ONCE: validation (for the record), test, external ----
    metrics = {"validation": threshold_metrics(y_val, val_scores, threshold)}
    test_scores = get_scores(pipeline, test["text"])
    metrics["test"] = threshold_metrics(test["label"], test_scores, threshold)
    ext_scores = get_scores(pipeline, external["text"])
    metrics["external_india"] = threshold_metrics(external["label"], ext_scores, threshold)

    challenge_scores = get_scores(pipeline, challenge["text"])
    metrics["challenge_transactional"] = threshold_metrics(
        challenge["label"], challenge_scores, threshold
    )

    if private_test is not None and len(private_test):
        priv_scores = get_scores(pipeline, private_test["text"])
        priv_pred = priv_scores >= threshold
        metrics["private_heldout"] = {
            "n": len(private_test),
            "legit": int((private_test["label"] == 0).sum()),
            "legit_flagged": int(((private_test["label"] == 0) & priv_pred).sum()),
            "unwanted": int((private_test["label"] == 1).sum()),
            "unwanted_caught": int(((private_test["label"] == 1) & priv_pred).sum()),
            "threshold": float(threshold),
        }
        print("private held-out:", metrics["private_heldout"])

    mask_super = (test["source"] == "super").to_numpy()
    for tag, mask in (("test_super_only", mask_super), ("test_older_sources", ~mask_super)):
        metrics[tag] = threshold_metrics(
            test["label"].to_numpy()[mask], test_scores[mask], threshold
        )

    # Baseline context on the same test set (the Dummy gets no say in selection)
    dummy = best["dummy"]["clf"]
    dummy_scores = get_scores(dummy, fitted_features["count_word"].transform(test["text"]))
    metrics["test_dummy_baseline"] = threshold_metrics(test["label"], dummy_scores, 0.5)

    # Same chosen model, but with the default 0.5 threshold, to show why thresholds matter.
    if score_type(pipeline) == "probability":
        metrics["test_at_default_0.5"] = threshold_metrics(test["label"], test_scores, 0.5)

    # ---- 6. Error analysis files + inference time ----
    pred = (test_scores >= threshold).astype(int)
    errors = test.assign(score=test_scores, predicted=pred)
    errors[(errors["label"] == 0) & (pred == 1)].sort_values("score", ascending=False).to_csv(
        METRICS_DIR / "sms_test_false_positives.csv", index=False
    )
    errors[(errors["label"] == 1) & (pred == 0)].sort_values("score").to_csv(
        METRICS_DIR / "sms_test_false_negatives.csv", index=False
    )
    sample = test["text"].sample(300, random_state=settings.RANDOM_SEED).tolist()
    start = time.perf_counter()
    for text in sample:
        get_scores(pipeline, [text])
    latency_ms = (time.perf_counter() - start) / len(sample) * 1000

    # ---- 7. Diagnostics: is the model leaning on shortcuts? (Logistic Regression weights) ----
    diag = {}
    for tag, model_name, feat_name in (
        ("raw", "logreg_word", "tfidf_word"),
        ("norm", "logreg_word_norm", "tfidf_word_norm"),
    ):
        lr = best[model_name]["clf"]
        names = np.array(fitted_features[feat_name].get_feature_names_out())
        order = np.argsort(lr.coef_[0])
        pd.DataFrame(
            {
                "word": np.concatenate([names[order[-40:]][::-1], names[order[:40]]]),
                "weight": np.concatenate([lr.coef_[0][order[-40:]][::-1], lr.coef_[0][order[:40]]]),
                "pushes_toward": ["spam"] * 40 + ["not spam"] * 40,
            }
        ).to_csv(METRICS_DIR / f"sms_logreg_{tag}_top_words.csv", index=False)
    has_entity = train["text"].str.contains(r"&(?:lt|gt|amp|quot);", regex=True)
    diag["train_html_entity_rows"] = int(has_entity.sum())
    diag["train_spam_share_with_html_entity"] = float(train.loc[has_entity, "label"].mean())
    diag["train_spam_share_without_html_entity"] = float(train.loc[~has_entity, "label"].mean())
    has_fffd = train["text"].str.contains("�", regex=False)
    diag["train_replacement_char_rows"] = int(has_fffd.sum())
    diag["train_spam_share_with_replacement_char"] = (
        float(train.loc[has_fffd, "label"].mean()) if has_fffd.any() else None
    )

    # ---- 8. Plots ----
    plots.plot_confusion(
        metrics["test"],
        f"Test set: {chosen} at threshold {threshold:.3f}",
        FIGURES_DIR / "sms_confusion_test.png",
    )
    plots.plot_confusion(
        metrics["external_india"],
        "External set (India telecom): same model and threshold",
        FIGURES_DIR / "sms_confusion_external.png",
    )
    plots.plot_model_comparison(
        results["model"].tolist(),
        results["val_pr_auc"].tolist(),
        "Validation PR-AUC by model (higher is better)",
        "PR-AUC (average precision)",
        FIGURES_DIR / "sms_model_comparison.png",
    )
    curves = {
        m: (y_val, get_scores(best[m]["clf"], x_val[best[m]["feat"]]))
        for m in ("naive_bayes", "logreg_word", "linear_svm_word")
    }
    plots.plot_pr_curves(
        curves, "Validation precision-recall curves", FIGURES_DIR / "sms_pr_curves.png"
    )

    # ---- 9. Save model + metadata (+ reload check) ----
    model_path = MODEL_DIR / "model.joblib"
    joblib.dump(pipeline, model_path)
    reloaded = joblib.load(model_path)
    assert np.allclose(get_scores(reloaded, sample), get_scores(pipeline, sample))
    sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    metadata = {
        "module": "sms_spam",
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "label_convention": "1 = spam/smishing (bad), 0 = ham (good)",
        "model": chosen,
        "model_params": info["params"],
        "features": feat,
        "score_type": score_type(pipeline),
        "threshold": threshold,
        "threshold_rule": f"highest validation recall with validation precision >= {PRECISION_TARGET}",
        "threshold_target_reached": pick["reached_target"],
        "selection_rule": f"best validation PR-AUC; ties (within {TIE_TOLERANCE}) -> {PREFERENCE_ORDER}",
        "seed": settings.RANDOM_SEED,
        "data_sizes": {
            "train": len(train),
            "val": len(val),
            "test": len(test),
            "external": len(external),
            "challenge": len(challenge),
        },
        "train_spam_share": float(y_train.mean()),
        "private_data": private_info,
        "augmentation": {
            "synthetic_rows_added_to_train": int(len(augment)),
            "per_class_requested": AUGMENT_PER_CLASS,
            "description": "template-written legitimate bank/transaction alerts and scams sharing "
            "the same vocabulary; training split only; see src/data/augment_sms.py",
        },
        "metrics": metrics,
        "inference_ms_per_message": latency_ms,
        "diagnostics": diag,
        "model_file_mb": round(model_path.stat().st_size / 1e6, 2),
        "model_sha256": sha,
        "limitations": [
            "Trained on data from about 2011-2021; no 2025-2026 data was available.",
            "Training spam share is ~38%, far above real inboxes: real-world precision will be lower.",
            "Ham and spam in the training sources differ by region and style; part of the score may reflect source.",
            "Legitimate bank alerts were flagged as spam until synthetic template examples were added; a 33-message hand-written challenge set still shows 4 of 23 legitimate messages flagged.",
            "Conversational and impersonation scams (e.g. 'Mum, I lost my phone') are largely missed.",
        ],
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    pd.DataFrame(
        [{"set": k, **{m: v for m, v in d.items() if m not in ("n",)}} for k, d in metrics.items()]
    ).to_csv(METRICS_DIR / "sms_final_metrics.csv", index=False)

    print("\n=== final metrics (selected model) ===")
    show = ["precision", "recall", "f1", "pr_auc", "roc_auc", "false_positive_rate", "fp", "fn"]
    print(pd.DataFrame(metrics).T[show].round(4).to_string())
    print(f"\ninference {latency_ms:.2f} ms/message | model file {metadata['model_file_mb']} MB")
    print(results.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
