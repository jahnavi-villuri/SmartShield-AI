"""Phase 5: train and compare email spam models, then save the selected pipeline.

Run from the project root (after `python -m scripts.prepare_email`):
    python -m src.training.train_email

RULES FIXED BEFORE LOOKING AT ANY RESULT
  1. Models are compared by PR-AUC on the VALIDATION set (threshold-free, suits imbalance).
  2. Models within TIE_TOLERANCE of the best are ordered by PREFERENCE_ORDER. Variants that mask
     data-source words ("_masked") come first, on principle: a spam filter must not rely on
     words that merely identify Enron staff mail.
  3. GUARD: a candidate that flags more than MAX_LEGIT_FLAGGED_SHARE of the hand-written modern
     LEGITIMATE emails (tests/data/email_challenge.csv) is skipped, because it is likely to treat
     normal modern email as spam. The challenge set is a smoke test used only for this gate.
  4. Threshold: the highest validation recall whose validation precision is >= PRECISION_TARGET
     (a false alarm hides a real email, so it is the worse error).
  5. Test and the 2026 external set are evaluated once, after the model is chosen. The 2026 set is
     spam only: it measures the detection rate, never false alarms.
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
from src.data.private_email import load_private_email
from src.data.synthetic_email import load_synthetic_email
from src.evaluation import plots
from src.evaluation.metrics import (
    get_scores,
    score_type,
    threshold_for_precision,
    threshold_metrics,
)
from src.features.email_text import SignatureMasker, join_email, normalize_email

DATA_DIR = settings.DATA_PROCESSED / "email"
EXTERNAL_FILE = settings.DATA_EXTERNAL_TEST / "email" / "untroubled_2026.csv"
CHALLENGE_FILE = settings.PROJECT_ROOT / "tests" / "data" / "email_challenge.csv"
MODEL_DIR = settings.MODELS_DIR / "email_spam"
METRICS_DIR = settings.REPORTS_DIR / "metrics"
FIGURES_DIR = settings.REPORTS_DIR / "figures"

PRECISION_TARGET = 0.98
TIE_TOLERANCE = 0.003
MAX_LEGIT_FLAGGED_SHARE = 0.25
CV_FOLDS = 3
SYNTHETIC_REPEAT = (
    5  # darkknight25 train half: synthetic, so it gets far less weight than real mail
)
PRIVATE_REPEAT = 30  # each private TRAIN-half email is repeated this many times
BASE_ORDER = ["logreg_word", "naive_bayes", "linear_svm_word"]
PREFERENCE_ORDER = [f"{m}_masked" for m in BASE_ORDER] + BASE_ORDER

# Source-signature words: common in ham, nearly absent in spam (computed from TRAIN only)
SIG_MIN_HAM_SHARE = 0.02
SIG_MAX_SPAM_SHARE = 0.004

_FEATURES = {
    "count_word": partial(CountVectorizer, ngram_range=(1, 2), min_df=3, max_features=200_000),
    "tfidf_word": partial(
        TfidfVectorizer, ngram_range=(1, 2), min_df=3, max_features=200_000, sublinear_tf=True
    ),
}
_BASE_CANDIDATES = {
    "naive_bayes": ("count_word", MultinomialNB, [{"alpha": a} for a in (0.1, 0.3, 1.0)]),
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
}


def load_split(path) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    frame["text"] = [join_email(s, b) for s, b in zip(frame["subject"], frame["body"], strict=True)]
    return frame


def find_signature_words(train: pd.DataFrame) -> list[str]:
    """Words typical of the ham source: frequent in ham, almost absent in spam (train only)."""
    vec = CountVectorizer(
        binary=True, min_df=5, preprocessor=normalize_email, token_pattern=r"\w\w+"
    )
    matrix = vec.fit_transform(train["text"])
    is_spam = train["label"].to_numpy() == 1
    ham_share = np.asarray(matrix[~is_spam].mean(axis=0)).ravel()
    spam_share = np.asarray(matrix[is_spam].mean(axis=0)).ravel()
    names = np.array(vec.get_feature_names_out())
    keep = (ham_share >= SIG_MIN_HAM_SHARE) & (spam_share <= SIG_MAX_SPAM_SHARE)
    return sorted(names[keep].tolist())


def select_email_model(results: pd.DataFrame) -> tuple[str, bool]:
    """Apply rules 1-3. Returns (model name, whether the guard passed for it)."""
    real = results[results["model"] != "dummy"].set_index("model")
    best = real["val_pr_auc"].max()
    close = [
        m
        for m in PREFERENCE_ORDER
        if m in real.index and real.loc[m, "val_pr_auc"] >= best - TIE_TOLERANCE
    ]
    for model in close:
        if real.loc[model, "challenge_legit_flagged_share"] <= MAX_LEGIT_FLAGGED_SHARE:
            return model, True
    return close[0], False


def per_group_recall(frame: pd.DataFrame, flagged: np.ndarray, column: str) -> dict:
    out = {}
    for key, idx in frame.groupby(column).groups.items():
        out[str(key)] = {
            "n": int(len(idx)),
            "detected": float(flagged[frame.index.get_indexer(idx)].mean()),
        }
    return out


def main() -> None:
    for folder in (MODEL_DIR, METRICS_DIR, FIGURES_DIR):
        folder.mkdir(parents=True, exist_ok=True)
    train, val, test = (load_split(DATA_DIR / f"{n}.csv") for n in ("train", "val", "test"))
    external = load_split(EXTERNAL_FILE)
    challenge = pd.read_csv(CHALLENGE_FILE, keep_default_na=False)
    challenge["text"] = [
        join_email(s, b) for s, b in zip(challenge["subject"], challenge["body"], strict=True)
    ]
    print(f"train {len(train)} | val {len(val)} | test {len(test)} | external 2026 {len(external)}")

    signature = find_signature_words(train)
    pd.DataFrame({"word": signature}).to_csv(METRICS_DIR / "email_signature_words.csv", index=False)
    print(
        f"source-signature words masked in '_masked' variants: {len(signature)}; e.g. {signature[:12]}"
    )

    # The owner's real emails (if any): half join the training set (repeated so they carry
    # weight), half stay untouched as an honest test. Signature words above come from Enron only.
    private = load_private_email()
    private_test, private_info = None, {"emails": 0}
    if private is not None and len(private):
        private_train = private[private["split"] == "train"]
        private_test = private[private["split"] == "test"].reset_index(drop=True)
        repeated = pd.concat([private_train] * PRIVATE_REPEAT, ignore_index=True)
        cols = ["subject", "body", "label", "text", "group"]
        train = pd.concat([train, repeated[cols]], ignore_index=True)
        private_info = {
            "emails": len(private),
            "train_half": len(private_train),
            "test_half": len(private_test),
            "repeat": PRIVATE_REPEAT,
        }
        print(f"private emails: {private_info}")
    # darkknight25 (synthetic modern mail): train half repeated a few times, test half held out.
    synthetic = load_synthetic_email()
    synthetic_test, synthetic_info = None, {"emails": 0}
    if synthetic is not None and len(synthetic):
        syn_train = synthetic[synthetic["split"] == "train"]
        synthetic_test = synthetic[synthetic["split"] == "test"].reset_index(drop=True)
        cols = ["subject", "body", "label", "text", "group"]
        train = pd.concat([train] + [syn_train[cols]] * SYNTHETIC_REPEAT, ignore_index=True)
        synthetic_info = {
            "emails": len(synthetic),
            "train_half": len(syn_train),
            "test_half": len(synthetic_test),
            "repeat": SYNTHETIC_REPEAT,
        }
        print(f"synthetic emails: {synthetic_info}")
    y_train, y_val = train["label"].to_numpy(), val["label"].to_numpy()

    features = {}
    for name, factory in _FEATURES.items():
        features[name] = partial(factory, preprocessor=normalize_email)
        features[f"{name}_masked"] = partial(factory, preprocessor=SignatureMasker(signature))
    candidates = {"dummy": ("count_word", partial(DummyClassifier, strategy="prior"), [{}])}
    for name, (feat, clf, grid) in _BASE_CANDIDATES.items():
        candidates[name] = (feat, clf, grid)
        candidates[f"{name}_masked"] = (f"{feat}_masked", clf, grid)

    fitted, x_train, x_val = {}, {}, {}
    for name, factory in features.items():
        vec = factory().fit(train["text"])
        fitted[name] = vec
        x_train[name], x_val[name] = vec.transform(train["text"]), vec.transform(val["text"])
        print(f"features {name}: {x_train[name].shape[1]:,} columns")

    # ---- tune on validation ----
    best = {}
    for model, (feat, factory, grid) in candidates.items():
        for params in grid:
            clf = factory(**params).fit(x_train[feat], y_train)
            scores = get_scores(clf, x_val[feat])
            pr = average_precision_score(y_val, scores)
            if model not in best or pr > best[model]["val_pr_auc"]:
                best[model] = {
                    "params": params,
                    "clf": clf,
                    "feat": feat,
                    "val_pr_auc": pr,
                    "val_roc_auc": roc_auc_score(y_val, scores),
                }
        print(f"tuned {model}: {best[model]['params']} val PR-AUC {best[model]['val_pr_auc']:.4f}")

    # ---- per-candidate: threshold, grouped CV, guard on the modern-legit challenge emails ----
    cv = StratifiedGroupKFold(n_splits=CV_FOLDS, shuffle=True, random_state=settings.RANDOM_SEED)
    rows, pipelines = [], {}
    for model, info in best.items():
        feat, factory, _ = candidates[model]
        pipe = Pipeline([("features", fitted[feat]), ("clf", info["clf"])])
        pipelines[model] = pipe
        val_scores = get_scores(pipe, val["text"])
        pick = threshold_for_precision(y_val, val_scores, PRECISION_TARGET)
        at_target = threshold_metrics(y_val, val_scores, pick["threshold"])
        ch_scores = get_scores(pipe, challenge["text"])
        ch_flag = ch_scores >= pick["threshold"]
        legit = (challenge["label"] == 0).to_numpy()
        ext_flag = get_scores(pipe, external["text"]) >= pick["threshold"]
        cv_scores = cross_val_score(
            Pipeline([("features", features[feat]()), ("clf", factory(**info["params"]))]),
            train["text"],
            y_train,
            groups=train["group"],
            cv=cv,
            scoring="average_precision",
        )
        rows.append(
            {
                "model": model,
                "params": json.dumps(info["params"]),
                "val_pr_auc": info["val_pr_auc"],
                "val_roc_auc": info["val_roc_auc"],
                "cv_pr_auc_mean": float(cv_scores.mean()),
                "cv_pr_auc_std": float(cv_scores.std()),
                "val_recall_at_precision_target": at_target["recall"],
                "threshold": pick["threshold"],
                "target_reached": pick["reached_target"],
                "challenge_legit_flagged_share": float(ch_flag[legit].mean()),
                "challenge_spam_caught_share": float(ch_flag[~legit].mean()),
                "external_2026_detection_rate": float(ext_flag.mean()),
            }
        )
        print(
            f"{model}: CV {cv_scores.mean():.4f} | modern legit flagged {ch_flag[legit].mean():.2f} "
            f"| 2026 spam detected {ext_flag.mean():.2f}"
        )
    results = pd.DataFrame(rows)
    results.to_csv(METRICS_DIR / "email_model_comparison.csv", index=False)

    chosen, guard_ok = select_email_model(results)
    pipeline, info = pipelines[chosen], best[chosen]
    threshold = float(results.set_index("model").loc[chosen, "threshold"])
    print(
        f"\nCHOSEN: {chosen} {info['params']} | threshold {threshold:.4f} | guard passed: {guard_ok}"
    )

    # ---- evaluate once ----
    metrics = {"validation": threshold_metrics(y_val, get_scores(pipeline, val["text"]), threshold)}
    test_scores = get_scores(pipeline, test["text"])
    metrics["test"] = threshold_metrics(test["label"], test_scores, threshold)
    ch_scores = get_scores(pipeline, challenge["text"])
    metrics["challenge_modern"] = threshold_metrics(challenge["label"], ch_scores, threshold)
    dummy_scores = get_scores(pipelines["dummy"], test["text"])
    metrics["test_dummy_baseline"] = threshold_metrics(test["label"], dummy_scores, 0.5)
    if score_type(pipeline) == "probability":
        metrics["test_at_default_0.5"] = threshold_metrics(test["label"], test_scores, 0.5)

    if private_test is not None and len(private_test):
        flagged = get_scores(pipeline, private_test["text"]) >= threshold
        is_spam = private_test["label"].to_numpy() == 1
        metrics["private_heldout"] = {
            "n": len(private_test),
            "legit": int((~is_spam).sum()),
            "legit_flagged": int((flagged & ~is_spam).sum()),
            "spam": int(is_spam.sum()),
            "spam_caught": int((flagged & is_spam).sum()),
            "threshold": float(threshold),
        }
        print("private held-out:", metrics["private_heldout"])

    if synthetic_test is not None and len(synthetic_test):
        flagged = get_scores(pipeline, synthetic_test["text"]) >= threshold
        is_spam = synthetic_test["label"].to_numpy() == 1
        metrics["synthetic_heldout"] = {
            "n": len(synthetic_test),
            "legit": int((~is_spam).sum()),
            "legit_flagged": int((flagged & ~is_spam).sum()),
            "spam": int(is_spam.sum()),
            "spam_caught": int((flagged & is_spam).sum()),
        }
        print("synthetic held-out:", metrics["synthetic_heldout"])

    ext_flag = get_scores(pipeline, external["text"]) >= threshold
    external = external.reset_index(drop=True)
    external["flagged"] = ext_flag
    external["language"] = np.where(
        external["mostly_non_ascii"].astype(str).str.lower() == "true",
        "mostly non-ASCII",
        "ASCII/English-like",
    )
    external_report = {
        "n": int(len(external)),
        "detection_rate": float(ext_flag.mean()),
        "by_month": per_group_recall(external, ext_flag, "month"),
        "by_language": per_group_recall(external, ext_flag, "language"),
    }
    print("2026 spam detection rate:", round(external_report["detection_rate"], 3))
    for month, d in external_report["by_month"].items():
        print(f"   {month}: {d['detected']:.3f} of {d['n']}")

    # ---- error analysis ----
    pred = (test_scores >= threshold).astype(int)
    errors = test.assign(score=test_scores, predicted=pred)
    keep = ["id", "subject", "label", "score", "date"]
    errors[(errors["label"] == 0) & (pred == 1)].sort_values("score", ascending=False)[keep].to_csv(
        METRICS_DIR / "email_test_false_positives.csv", index=False
    )
    errors[(errors["label"] == 1) & (pred == 0)].sort_values("score")[keep].to_csv(
        METRICS_DIR / "email_test_false_negatives.csv", index=False
    )
    external[~external["flagged"]][["id", "month", "subject", "language"]].head(300).to_csv(
        METRICS_DIR / "email_2026_missed_spam.csv", index=False
    )

    sample = test["text"].sample(200, random_state=settings.RANDOM_SEED).tolist()
    start = time.perf_counter()
    for text in sample:
        get_scores(pipeline, [text])
    latency_ms = (time.perf_counter() - start) / len(sample) * 1000

    # ---- diagnostics: what does the unmasked logistic regression lean on? ----
    lr = best["logreg_word"]["clf"]
    names = np.array(fitted["tfidf_word"].get_feature_names_out())
    order = np.argsort(lr.coef_[0])
    pd.DataFrame(
        {
            "word": np.concatenate([names[order[-40:]][::-1], names[order[:40]]]),
            "weight": np.concatenate([lr.coef_[0][order[-40:]][::-1], lr.coef_[0][order[:40]]]),
            "pushes_toward": ["spam"] * 40 + ["not spam"] * 40,
        }
    ).to_csv(METRICS_DIR / "email_logreg_top_words.csv", index=False)

    # ---- plots ----
    plots.plot_confusion(
        metrics["test"], f"Test set: {chosen}", FIGURES_DIR / "email_confusion_test.png"
    )
    plots.plot_model_comparison(
        results["model"].tolist(),
        results["val_pr_auc"].tolist(),
        "Email: validation PR-AUC by model",
        "PR-AUC",
        FIGURES_DIR / "email_model_comparison.png",
    )
    plots.plot_pr_curves(
        {
            m: (y_val, get_scores(pipelines[m], val["text"]))
            for m in ("naive_bayes", "logreg_word", "logreg_word_masked")
        },
        "Email validation precision-recall curves",
        FIGURES_DIR / "email_pr_curves.png",
    )

    # ---- save ----
    model_path = MODEL_DIR / "model.joblib"
    joblib.dump(pipeline, model_path)
    reloaded = joblib.load(model_path)
    assert np.allclose(get_scores(reloaded, sample), get_scores(pipeline, sample))
    metadata = {
        "module": "email_spam",
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "label_convention": "1 = spam (bad), 0 = ham (good)",
        "model": chosen,
        "model_params": info["params"],
        "features": info["feat"],
        "score_type": score_type(pipeline),
        "threshold": threshold,
        "threshold_rule": f"highest validation recall with validation precision >= {PRECISION_TARGET}",
        "selection_rule": f"best validation PR-AUC; ties (within {TIE_TOLERANCE}) -> {PREFERENCE_ORDER}; "
        f"guard: skip candidates flagging > {MAX_LEGIT_FLAGGED_SHARE:.0%} of modern legitimate challenge emails",
        "guard_passed": guard_ok,
        "private_data": private_info,
        "synthetic_data": synthetic_info,
        "signature_words_masked": len(signature) if "masked" in chosen else 0,
        "seed": settings.RANDOM_SEED,
        "data_sizes": {
            "train": len(train),
            "val": len(val),
            "test": len(test),
            "external_2026_spam": len(external),
            "challenge": len(challenge),
        },
        "metrics": metrics,
        "external_2026": external_report,
        "inference_ms_per_email": latency_ms,
        "model_file_mb": round(model_path.stat().st_size / 1e6, 2),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "limitations": [
            "Trained on Enron-Spam (ham: Enron staff email 1999-2002; spam: 2001-2005). It has never seen a modern legitimate email in training.",
            "The 2026 spam test has no legitimate email, so it measures detection rate only, not false alarms.",
            "Only 34 hand-written modern emails (24 legitimate) check false alarms on modern mail; that is a smoke test, not a measurement.",
            "100 of the extra modern training emails come from a synthetic one-sentence dataset (darkknight25); they teach modern topics but not realistic email structure.",
            "Part of 2026 spam is not English; the model reads English only.",
            "Reads only the first 6,000 characters of the body; attachments and images are ignored.",
        ],
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    pd.DataFrame([{"set": k, **v} for k, v in metrics.items()]).to_csv(
        METRICS_DIR / "email_final_metrics.csv", index=False
    )

    show = ["precision", "recall", "f1", "pr_auc", "roc_auc", "false_positive_rate", "fp", "fn"]
    print("\n=== final metrics (selected model) ===")
    print(pd.DataFrame(metrics).T[show].round(4).to_string())
    print(f"\ninference {latency_ms:.1f} ms/email | model {metadata['model_file_mb']} MB")
    print(results.drop(columns=["params"]).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
