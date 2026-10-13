"""Metrics and threshold selection for binary classifiers (label 1 = bad)."""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def get_scores(model, texts) -> np.ndarray:
    """Return a 'bad-ness' score per input: probability if available, else decision value.

    Higher = more likely class 1. Logistic Regression / Naive Bayes give probabilities;
    Linear SVM gives a margin (NOT a probability), so its scale is arbitrary.
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(texts)[:, 1]
    return model.decision_function(texts)


def score_type(model) -> str:
    return "probability" if hasattr(model, "predict_proba") else "decision_score"


def threshold_metrics(y_true, scores, threshold: float) -> dict:
    """All metrics at one threshold. ROC-AUC / PR-AUC are threshold-free and use raw scores."""
    y_true = np.asarray(y_true)
    pred = (np.asarray(scores) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float((tp + tn) / len(y_true)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else 0.0,
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "pr_auc": float(average_precision_score(y_true, scores)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
    }


def threshold_for_precision(y_true, scores, min_precision: float) -> dict:
    """Pick the threshold with the highest recall whose precision is >= `min_precision`.

    Chosen on VALIDATION data only. If no threshold reaches the target, fall back to the
    threshold with the best F1 and say so (`reached_target = False`).
    """
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    precision, recall = precision[:-1], recall[:-1]  # align with `thresholds`
    ok = precision >= min_precision
    if ok.any():
        best = int(np.argmax(np.where(ok, recall, -1.0)))
        return {"threshold": float(thresholds[best]), "reached_target": True}
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    return {"threshold": float(thresholds[int(np.argmax(f1))]), "reached_target": False}


def threshold_for_fpr(y_true, scores, max_fpr: float) -> dict:
    """Highest-recall threshold whose false-positive rate (legitimate flagged) is <= `max_fpr`.

    Used when missing a bad item is the costly error but flagging too many good items makes the
    tool unusable, so the false-alarm budget is fixed first and recall is maximised inside it.
    Chosen on VALIDATION data only.
    """
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y_true, scores)
    allowed = fpr <= max_fpr
    best = int(np.argmax(np.where(allowed, tpr, -1.0)))
    threshold = float(thresholds[best])
    if not np.isfinite(threshold):  # roc_curve's first point is +inf: nothing flagged
        threshold = float(np.max(scores)) + 1e-9
    return {"threshold": threshold, "reached_target": bool(allowed.any() and fpr[best] <= max_fpr)}
