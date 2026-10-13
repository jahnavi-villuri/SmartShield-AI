"""Stage 2 of the SMS detector: promotional vs scam, for messages stage 1 calls "unwanted".

Run from the project root:
    python -m src.training.train_sms_type

Why a second model: the main model separates legitimate from unwanted, but "unwanted" mixes
harmless marketing (a Taco Bell offer) with harmful scams. The dataset's own sub-category
labels proved unreliable, so this model learns from ~450 messages hand-labelled by the AI
assistant (data/labels/sms_type_labels.csv, rules in docs/sms_type_labeling_guide.md).

RULES FIXED BEFORE LOOKING AT RESULTS
  1. Train on the labelled `train` + `val` messages together (the set is small), choose the
     model and its C by repeated stratified 5-fold cross-validation PR-AUC for the scam class.
  2. Models within 0.01 of the best prefer the order in PREFERENCE_ORDER.
  3. Decision threshold for "scam" is fixed at 0.5 (no tuning: too few examples to tune safely).
  4. The labelled `test` messages and the probe sets are evaluated once, after the choice.
"""

import hashlib
import json
import platform
from datetime import UTC, datetime
from functools import partial

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from config import settings
from src.evaluation.metrics import get_scores, threshold_metrics
from src.features.text_normalize import normalize_sms

LABELS_FILE = settings.DATA_DIR / "labels" / "sms_type_labels.csv"
CHALLENGE_FILE = settings.PROJECT_ROOT / "tests" / "data" / "sms_challenge.csv"
REPORTED_FILE = settings.PROJECT_ROOT / "tests" / "data" / "sms_reported_failures.csv"
MODEL_DIR = settings.MODELS_DIR / "sms_type"
METRICS_DIR = settings.REPORTS_DIR / "metrics"

SCAM_THRESHOLD = 0.5
TIE_TOLERANCE = 0.01
PREFERENCE_ORDER = ["logreg_word_norm", "naive_bayes_norm", "logreg_char_norm"]
CV_SPLITS, CV_REPEATS = 5, 3

FEATURES = {
    "word": partial(
        TfidfVectorizer, ngram_range=(1, 2), min_df=1, sublinear_tf=True, preprocessor=normalize_sms
    ),
    "char": partial(
        TfidfVectorizer,
        analyzer="char_wb",
        ngram_range=(2, 5),
        min_df=2,
        sublinear_tf=True,
        preprocessor=normalize_sms,
    ),
    "count": partial(CountVectorizer, ngram_range=(1, 2), min_df=1, preprocessor=normalize_sms),
}
CANDIDATES = {
    "logreg_word_norm": (
        "word",
        partial(LogisticRegression, max_iter=2000),
        [{"C": c} for c in (1, 3, 10, 30, 100)],
    ),
    "logreg_char_norm": (
        "char",
        partial(LogisticRegression, max_iter=2000),
        [{"C": c} for c in (1, 3, 10, 30, 100)],
    ),
    "naive_bayes_norm": ("count", MultinomialNB, [{"alpha": a} for a in (0.1, 0.3, 1.0)]),
}


def make_pipeline(feature: str, factory, params: dict) -> Pipeline:
    return Pipeline([("features", FEATURES[feature]()), ("clf", factory(**params))])


def select_type_model(results: pd.DataFrame) -> str:
    best = results["cv_pr_auc"].max()
    close = results.set_index("model")["cv_pr_auc"]
    return next(m for m in PREFERENCE_ORDER if close[m] >= best - TIE_TOLERANCE)


def probe(pipeline, path, mapping) -> pd.DataFrame:
    """Score a hand-written probe file; `mapping` turns its columns into 'S'/'P' or None."""
    frame = pd.read_csv(path, keep_default_na=False)
    frame["expected"] = [mapping(r) for _, r in frame.iterrows()]
    frame = frame[frame["expected"].notna()].copy()
    frame["p_scam"] = get_scores(pipeline, frame["text"])
    frame["predicted"] = np.where(frame["p_scam"] >= SCAM_THRESHOLD, "S", "P")
    return frame


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(LABELS_FILE, keep_default_na=False)
    labels = labels[labels["type"].isin(["S", "P"])].copy()
    labels["y"] = (labels["type"] == "S").astype(int)
    dev = labels[labels["split"].isin(["train", "val"])].reset_index(drop=True)
    test = labels[labels["split"] == "test"].reset_index(drop=True)
    print(f"dev {len(dev)} (scam {dev.y.sum()}) | test {len(test)} (scam {test.y.sum()})")

    cv = RepeatedStratifiedKFold(
        n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=settings.RANDOM_SEED
    )
    rows = []
    for model, (feat, factory, grid) in CANDIDATES.items():
        best = None
        for params in grid:
            scores = cross_val_score(
                make_pipeline(feat, factory, params),
                dev["text"],
                dev["y"],
                cv=cv,
                scoring="average_precision",
            )
            if best is None or scores.mean() > best["cv_pr_auc"]:
                best = {
                    "model": model,
                    "params": params,
                    "feature": feat,
                    "cv_pr_auc": float(scores.mean()),
                    "cv_std": float(scores.std()),
                }
        rows.append(best)
        print(
            f"{model}: CV PR-AUC {best['cv_pr_auc']:.3f} +/- {best['cv_std']:.3f} {best['params']}"
        )
    results = pd.DataFrame(rows)
    results.assign(params=results["params"].map(json.dumps)).to_csv(
        METRICS_DIR / "sms_type_model_comparison.csv", index=False
    )

    chosen = select_type_model(results)
    spec = results.set_index("model").loc[chosen]
    factory = CANDIDATES[chosen][1]
    pipeline = make_pipeline(spec["feature"], factory, spec["params"]).fit(dev["text"], dev["y"])
    print(f"CHOSEN {chosen} {spec['params']}")

    # ---- evaluate once ----
    test_scores = get_scores(pipeline, test["text"])
    test_metrics = threshold_metrics(test["y"], test_scores, SCAM_THRESHOLD)
    baseline_acc = float(max(test["y"].mean(), 1 - test["y"].mean()))
    challenge = probe(
        pipeline,
        CHALLENGE_FILE,
        lambda r: ("P" if r["kind"] == "promo" else "S") if r["label"] == 1 else None,
    )
    reported = probe(
        pipeline, REPORTED_FILE, lambda r: "P" if r["expected"] == "promotional" else None
    )
    challenge_acc = float((challenge["predicted"] == challenge["expected"]).mean())
    reported_ok = bool((reported["predicted"] == reported["expected"]).all())

    # out-of-fold estimate on dev (honest accuracy that did not see its own label)
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    oof = cross_val_predict(
        clone(pipeline),
        dev["text"],
        dev["y"],
        cv=StratifiedKFold(5, shuffle=True, random_state=settings.RANDOM_SEED),
        method="predict_proba",
    )[:, 1]
    oof_metrics = threshold_metrics(dev["y"], oof, SCAM_THRESHOLD)

    metrics = {"cross_validated_dev": oof_metrics, "test": test_metrics}
    print(
        "\nout-of-fold dev :",
        {k: round(v, 3) for k, v in oof_metrics.items() if isinstance(v, float)},
    )
    print(
        "hand-labelled test:",
        {k: round(v, 3) if isinstance(v, float) else v for k, v in test_metrics.items()},
    )
    print(f"majority-class accuracy on test: {baseline_acc:.3f}")
    print(
        f"challenge-set (10 spam msgs) type accuracy: {challenge_acc:.2f}; Taco Bell probe ok: {reported_ok}"
    )
    print(reported[["expected", "predicted", "p_scam"]].to_string(index=False))

    joblib.dump(pipeline, MODEL_DIR / "model.joblib")
    sha = hashlib.sha256((MODEL_DIR / "model.joblib").read_bytes()).hexdigest()
    metadata = {
        "module": "sms_type",
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
        "classes": {"0": "promotional", "1": "scam"},
        "model": chosen,
        "model_params": spec["params"],
        "score_type": "probability",
        "scam_threshold": SCAM_THRESHOLD,
        "data_sizes": {
            "dev": len(dev),
            "dev_scam": int(dev["y"].sum()),
            "test": len(test),
            "test_scam": int(test["y"].sum()),
        },
        "labels": "hand-labelled by the AI assistant using docs/sms_type_labeling_guide.md; "
        "not verified by a human; unclear messages excluded",
        "metrics": metrics,
        "majority_class_accuracy_test": baseline_acc,
        "challenge_type_accuracy": challenge_acc,
        "model_sha256": sha,
        "limitations": [
            f"Trained on only {len(dev)} hand-labelled messages; the test set has just "
            f"{len(test)} messages ({int(test['y'].sum())} scams), so its numbers are very noisy.",
            "Labels come from one labeller (an AI) with written rules; contests and "
            "premium-rate services are judgement calls.",
            "Only meaningful for messages already judged unwanted by the main model.",
        ],
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("saved", MODEL_DIR)


if __name__ == "__main__":
    main()
