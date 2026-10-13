"""Phase 6: train the six-label toxicity model, choose per-label thresholds, evaluate, save.

Run from the project root (after `python -m scripts.prepare_toxicity`):
    python -m src.training.train_toxicity            # Round 1: no bias mitigation
    python -m src.training.train_toxicity --augment  # Round 2: add neutral identity sentences

RULES FIXED BEFORE LOOKING AT ANY RESULT
  1. Candidates (word TF-IDF vs word+character TF-IDF, each with one Logistic Regression per
     label) are compared by the MACRO-AVERAGE PR-AUC over the six labels on the validation set.
     Models within TIE_TOLERANCE of the best prefer the word-only model (smaller, explainable).
  2. C is chosen per feature set from C_GRID by the same metric.
  3. Each label has its OWN threshold: the value maximising F1 on pooled predictions = the
     validation set + 3-fold grouped out-of-fold predictions on the training set. Pooling is
     used because `threat` has only ~70 validation positives: one set alone is too noisy.
  4. Internal test and the official test set are evaluated once, after the choice.
  5. Bias probe: neutral sentences ("I am a {identity} person.") built from templates and
     identity words the augmentation never uses. A round is acceptable on bias if no identity
     group has more than MAX_BIAS_FLAGGED_SHARE of its probe sentences flagged.
Error-analysis files contain offensive text, so they go to the Git-ignored data folder.
"""

import argparse
import hashlib
import json
import platform
import time
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion

from config import settings
from src.data.augment_toxicity import (
    CONTROL_WORDS,
    PROBE_ONLY_IDENTITIES,
    PROBE_TEMPLATES,
    TRAIN_IDENTITIES,
    generate_identity_augmentation,
)
from src.evaluation import plots
from src.evaluation.metrics import threshold_metrics
from src.features.comment_text import ClippedLower
from src.models.toxicity_model import LABELS, PerLabelLogReg, ToxicityModel

DATA_DIR = settings.DATA_PROCESSED / "toxicity"
OFFICIAL_FILE = settings.DATA_EXTERNAL_TEST / "toxicity" / "official_test.csv"
PROBE_FILE = settings.PROJECT_ROOT / "tests" / "data" / "toxicity_probe.csv"
MODEL_DIR = settings.MODELS_DIR / "toxicity"
METRICS_DIR = settings.REPORTS_DIR / "metrics"
FIGURES_DIR = settings.REPORTS_DIR / "figures"

C_GRID = (2.0, 8.0)
TIE_TOLERANCE = 0.003
PREFERENCE_ORDER = ["logreg_word", "logreg_wordchar"]
MAX_BIAS_FLAGGED_SHARE = 0.05
OOF_FOLDS = 3
AUGMENT_PER_IDENTITY = 150


def build_features(kind: str):
    word = TfidfVectorizer(
        preprocessor=ClippedLower(1500), ngram_range=(1, 2), min_df=3, max_features=250_000,
        sublinear_tf=True, dtype=np.float32,
    )  # fmt: skip
    if kind == "word":
        return word
    char = TfidfVectorizer(
        preprocessor=ClippedLower(1000), analyzer="char_wb", ngram_range=(3, 5), min_df=5,
        max_features=200_000, sublinear_tf=True, dtype=np.float32,
    )  # fmt: skip
    return FeatureUnion([("word", word), ("char", char)])


def macro_ap(Y, P) -> float:
    return float(np.mean([average_precision_score(Y[:, j], P[:, j]) for j in range(Y.shape[1])]))


def best_f1_threshold(y_true, scores) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.clip(precision[:-1] + recall[:-1], 1e-12, None)
    return float(thresholds[int(np.argmax(f1))]) if len(thresholds) else 0.5


def strata(frame: pd.DataFrame) -> pd.Series:
    any_toxic = frame[LABELS].max(axis=1)
    key = any_toxic * 8 + frame["threat"] * 4 + frame["identity_hate"] * 2 + frame["severe_toxic"]
    return key.where(key.map(key.value_counts()) >= 60, 99)


def select_toxicity_model(macro: dict[str, float]) -> str:
    best = max(macro.values())
    return next(m for m in PREFERENCE_ORDER if macro[m] >= best - TIE_TOLERANCE)


def evaluate(model: ToxicityModel, frame: pd.DataFrame, name: str) -> dict:
    Y = frame[LABELS].to_numpy()
    P = model.scores(frame["comment_text"])
    per_label = {}
    flags = np.zeros_like(Y, dtype=bool)
    for j, label in enumerate(LABELS):
        thr = model.thresholds[label]
        per_label[label] = threshold_metrics(Y[:, j], P[:, j], thr)
        per_label[label]["at_default_0.5"] = {
            k: threshold_metrics(Y[:, j], P[:, j], 0.5)[k] for k in ("precision", "recall", "f1")
        }
        flags[:, j] = P[:, j] >= thr
    tp, fp, fn = (
        int((flags & (Y == 1)).sum()),
        int((flags & (Y == 0)).sum()),
        int((~flags & (Y == 1)).sum()),
    )
    micro_p, micro_r = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    any_true, any_pred = Y.max(axis=1) == 1, flags.any(axis=1)
    summary = {
        "n": len(frame),
        "macro_f1": float(np.mean([per_label[label]["f1"] for label in LABELS])),
        "macro_pr_auc": float(np.mean([per_label[label]["pr_auc"] for label in LABELS])),
        "macro_roc_auc": float(np.mean([per_label[label]["roc_auc"] for label in LABELS])),
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": 2 * micro_p * micro_r / max(micro_p + micro_r, 1e-12),
        "exact_match": float((flags == (Y == 1)).all(axis=1).mean()),
        "any_label_precision": float((any_true & any_pred).sum() / max(any_pred.sum(), 1)),
        "any_label_recall": float((any_true & any_pred).sum() / max(any_true.sum(), 1)),
    }
    print(
        f"{name}: macro-F1 {summary['macro_f1']:.3f} | macro PR-AUC {summary['macro_pr_auc']:.3f} | "
        f"any-label recall {summary['any_label_recall']:.3f}"
    )
    return {"summary": summary, "per_label": per_label}


def error_files(model: ToxicityModel, frame: pd.DataFrame, tag: str) -> None:
    """Offensive text: saved only in the Git-ignored data folder, 40 worst errors per label."""
    P = model.scores(frame["comment_text"])
    rows = []
    for j, label in enumerate(LABELS):
        thr = model.thresholds[label]
        y = frame[label].to_numpy()
        wrong_fp = np.where((P[:, j] >= thr) & (y == 0))[0]
        wrong_fn = np.where((P[:, j] < thr) & (y == 1))[0]
        for kind, idx, order in (("false_positive", wrong_fp, -1), ("false_negative", wrong_fn, 1)):
            idx = idx[np.argsort(order * P[idx, j])][:40]
            rows += [{"label": label, "error": kind, "score": float(P[i, j]),
                      "comment": frame["comment_text"].iloc[i][:400]} for i in idx]  # fmt: skip
    pd.DataFrame(rows).to_csv(DATA_DIR / f"errors_{tag}.csv", index=False, encoding="utf-8")


def probe_context(model: ToxicityModel) -> dict:
    probe = pd.read_csv(PROBE_FILE, keep_default_na=False)
    flags = model.flags(probe["text"])
    scores = model.scores(probe["text"])
    benign = probe["expected"] == "none"
    out = {
        "benign_n": int(benign.sum()),
        "benign_flagged": int(flags[benign.to_numpy()].any(axis=1).sum()),
        "toxic_n": int((~benign).sum()),
    }
    hit_any, hit_all = 0, 0
    for i in np.where(~benign)[0]:
        want = [LABELS.index(label) for label in probe["expected"].iloc[i].split("|")]
        hit_any += int(flags[i, want].any())
        hit_all += int(flags[i, want].all())
    out["toxic_any_expected_label_flagged"] = hit_any
    out["toxic_all_expected_labels_flagged"] = hit_all
    out["benign_flagged_examples"] = [
        {"text": probe["text"].iloc[i], "kind": probe["kind"].iloc[i],
         "labels": [LABELS[j] for j in np.where(flags[i])[0]], "max_score": float(scores[i].max())}
        for i in np.where(benign.to_numpy() & flags.any(axis=1))[0]
    ]  # fmt: skip
    print(
        f"context probe: benign flagged {out['benign_flagged']}/{out['benign_n']} | "
        f"toxic caught {hit_any}/{out['toxic_n']}"
    )
    return out


def probe_bias(model: ToxicityModel) -> dict:
    groups = {"seen_in_augmentation": TRAIN_IDENTITIES, "never_seen_identity": PROBE_ONLY_IDENTITIES,
              "control_non_identity": CONTROL_WORDS}  # fmt: skip
    rows = []
    for group, words in groups.items():
        for word in words:
            texts = [t.format(x=word, X=word[0].upper() + word[1:]) for t in PROBE_TEMPLATES]
            flags = model.flags(texts)
            scores = model.scores(texts)
            rows.append({"group": group, "term": word,
                         "flagged_share": float(flags.any(axis=1).mean()),
                         "mean_toxic_score": float(scores[:, 0].mean()),
                         "max_toxic_score": float(scores[:, 0].max())})  # fmt: skip
    table = pd.DataFrame(rows).sort_values("flagged_share", ascending=False)
    worst = table.iloc[0]
    out = {
        "n_sentences_per_term": len(PROBE_TEMPLATES),
        "max_flagged_share": float(worst["flagged_share"]),
        "worst_term": str(worst["term"]),
        "terms_over_limit": [str(t) for t in table.loc[table["flagged_share"] > MAX_BIAS_FLAGGED_SHARE, "term"]],
        "by_group_mean_flagged": {g: float(v) for g, v in table.groupby("group")["flagged_share"].mean().items()},
        "by_group_mean_toxic_score": {g: float(v) for g, v in table.groupby("group")["mean_toxic_score"].mean().items()},
        "acceptable": bool(worst["flagged_share"] <= MAX_BIAS_FLAGGED_SHARE),
        "top_terms": table.head(8)[["term", "group", "flagged_share", "mean_toxic_score"]].to_dict("records"),
    }  # fmt: skip
    table.to_csv(METRICS_DIR / "toxicity_bias_probe.csv", index=False)
    print(
        f"bias probe: worst term {out['worst_term']} flagged {out['max_flagged_share']:.2f} | "
        f"terms over {MAX_BIAS_FLAGGED_SHARE:.0%}: {len(out['terms_over_limit'])} | "
        f"mean flagged by group: { {k: round(v, 3) for k, v in out['by_group_mean_flagged'].items()} }"
    )
    return out


def main(augment: bool, short_forms: bool = False) -> None:
    for folder in (MODEL_DIR, METRICS_DIR, FIGURES_DIR):
        folder.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(DATA_DIR / "train.csv", keep_default_na=False)
    val = pd.read_csv(DATA_DIR / "val.csv", keep_default_na=False)
    test = pd.read_csv(DATA_DIR / "test.csv", keep_default_na=False)
    official = pd.read_csv(OFFICIAL_FILE, keep_default_na=False)
    n_aug = 0
    if augment:
        aug = generate_identity_augmentation(
            AUGMENT_PER_IDENTITY, settings.RANDOM_SEED, include_short_forms=short_forms
        )
        n_aug = len(aug)
        train = pd.concat([train, aug[train.columns.intersection(aug.columns)]], ignore_index=True)
    print(
        f"train {len(train)} (synthetic {n_aug}) | val {len(val)} | test {len(test)} | official {len(official)}"
    )
    Y_train, Y_val = train[LABELS].to_numpy(), val[LABELS].to_numpy()

    # ---- compare candidates on validation (threshold-free) ----
    fitted, results = {}, {}
    prevalence = Y_val.mean(axis=0)
    print("dummy macro PR-AUC (= mean prevalence):", round(float(prevalence.mean()), 4))
    for name, kind in (("logreg_word", "word"), ("logreg_wordchar", "wordchar")):
        started = time.perf_counter()
        features = build_features(kind).fit(train["comment_text"])
        X_train, X_val = (
            features.transform(train["comment_text"]),
            features.transform(val["comment_text"]),
        )
        print(
            f"{name}: {X_train.shape[1]:,} features, {X_train.nnz / 1e6:.0f}M non-zeros "
            f"({time.perf_counter() - started:.0f}s)"
        )
        best = None
        for c in C_GRID:
            clf = PerLabelLogReg(C=c).fit(X_train, Y_train)
            P_val = clf.predict_proba(X_val)
            score = macro_ap(Y_val, P_val)
            per_label = {
                label: average_precision_score(Y_val[:, j], P_val[:, j])
                for j, label in enumerate(LABELS)
            }
            print(
                f"   C={c}: macro PR-AUC {score:.4f} "
                + " ".join(f"{k[:6]}={v:.3f}" for k, v in per_label.items())
            )
            if best is None or score > best["macro"]:
                best = {"macro": score, "C": c, "clf": clf, "P_val": P_val, "per_label": per_label}
        fitted[name] = {"features": features, "X_train": X_train, **best}
        results[name] = best["macro"]
    chosen = select_toxicity_model(results)
    info = fitted[chosen]
    print(f"\nCHOSEN: {chosen} (C={info['C']}) | macro PR-AUC {results[chosen]:.4f}")

    # ---- thresholds from pooled validation + out-of-fold predictions ----
    X_train, clf_final = info["X_train"], info["clf"]
    folds = StratifiedGroupKFold(
        n_splits=OOF_FOLDS, shuffle=True, random_state=settings.RANDOM_SEED
    )
    P_oof = np.zeros_like(Y_train, dtype=float)
    for tr_idx, te_idx in folds.split(X_train, strata(train), groups=train["group"]):
        fold_clf = PerLabelLogReg(C=info["C"]).fit(X_train[tr_idx], Y_train[tr_idx])
        P_oof[te_idx] = fold_clf.predict_proba(X_train[te_idx])
    Y_pool, P_pool = np.vstack([Y_train, Y_val]), np.vstack([P_oof, info["P_val"]])
    thresholds = {
        label: best_f1_threshold(Y_pool[:, j], P_pool[:, j]) for j, label in enumerate(LABELS)
    }
    print("thresholds:", {k: round(v, 3) for k, v in thresholds.items()})
    model = ToxicityModel(info["features"], clf_final, thresholds)

    # ---- evaluate once ----
    metrics = {"validation": evaluate(model, val, "validation"),
               "test": evaluate(model, test, "internal test"),
               "official_test": evaluate(model, official, "official test")}  # fmt: skip
    context, bias = probe_context(model), probe_bias(model)

    error_files(model, official.head(30000), "official_test")
    sample = test["comment_text"].sample(200, random_state=settings.RANDOM_SEED).tolist()
    start = time.perf_counter()
    for text in sample:
        model.scores([text])
    latency_ms = (time.perf_counter() - start) / len(sample) * 1000

    # ---- charts and tables ----
    f1_rows = [{"set": s, "label": label, **{k: v for k, v in d["per_label"][label].items()
               if k in ("precision", "recall", "f1", "pr_auc", "roc_auc", "threshold", "positives", "n")}}
               for s, d in metrics.items() for label in LABELS]  # fmt: skip
    pd.DataFrame(f1_rows).to_csv(METRICS_DIR / "toxicity_per_label_metrics.csv", index=False)
    plots.plot_model_comparison(
        LABELS, [metrics["official_test"]["per_label"][label]["f1"] for label in LABELS],
        "Official test: F1 per label (own threshold)", "F1", FIGURES_DIR / "toxicity_f1_per_label.png",
    )  # fmt: skip

    # ---- save ----
    model_path = MODEL_DIR / "model.joblib"
    joblib.dump(model, model_path)
    reloaded = joblib.load(model_path)
    assert np.allclose(reloaded.scores(sample), model.scores(sample))
    metadata = {
        "module": "toxicity",
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "python": platform.python_version(), "scikit_learn": sklearn.__version__,
        "labels": LABELS, "model": chosen, "model_params": {"C": info["C"]},
        "augmented_with_identity_sentences": augment, "short_form_sentences": short_forms, "synthetic_rows": n_aug,
        "thresholds": thresholds, "score_type": "probability",
        "threshold_rule": "per label: maximise F1 on pooled validation + 3-fold out-of-fold training predictions",
        "selection_rule": f"macro-average PR-AUC on validation; ties within {TIE_TOLERANCE} prefer {PREFERENCE_ORDER[0]}",
        "validation_macro_pr_auc": results, "seed": settings.RANDOM_SEED,
        "data_sizes": {"train": len(train), "val": len(val), "test": len(test), "official_test": len(official)},
        "train_label_counts": {label: int(train[label].sum()) for label in LABELS},
        "metrics": metrics, "context_probe": context, "bias_probe": bias,
        "inference_ms_per_comment": latency_ms,
        "model_file_mb": round(model_path.stat().st_size / 1e6, 2),
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "limitations": [
            "Trained on Wikipedia talk-page comments (about 2017); slang, platforms and targets have changed.",
            "Labels are crowd-sourced and noisy; rare labels (threat, identity_hate, severe_toxic) have few examples, so their scores are unstable.",
            "A word-based model cannot read context: quotes, sarcasm, reclaimed words and discussions about toxicity can be misjudged.",
            "Identity terms can raise scores even in harmless sentences (see the bias probe).",
            "English only.",
        ],
    }  # fmt: skip
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nmodel {metadata['model_file_mb']} MB | {latency_ms:.1f} ms/comment")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--augment", action="store_true", help="add neutral identity sentences")
    parser.add_argument(
        "--short-forms",
        action="store_true",
        help="Round 4: also add short/question-style sentences",
    )
    args = parser.parse_args()
    main(args.augment or args.short_forms, args.short_forms)
