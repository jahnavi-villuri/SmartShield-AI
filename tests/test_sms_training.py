"""Phase 3 tests: metrics, threshold selection, model selection, save/load."""

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.evaluation.metrics import get_scores, threshold_for_precision, threshold_metrics
from src.training.train_sms import select_model

SPAM = ["win free prize now", "free cash claim now", "urgent win cash prize", "claim free prize"]
HAM = ["see you at lunch", "call me later", "running late for dinner", "thanks for the help"]


def _toy_pipeline():
    pipe = Pipeline([("features", TfidfVectorizer()), ("clf", LogisticRegression())])
    return pipe.fit(SPAM + HAM, [1] * 4 + [0] * 4)


def test_threshold_metrics_counts_confusion_correctly():
    y = [1, 1, 0, 0]
    scores = [0.9, 0.4, 0.6, 0.1]
    m = threshold_metrics(y, scores, threshold=0.5)
    assert (m["tp"], m["fn"], m["fp"], m["tn"]) == (1, 1, 1, 1)
    assert m["precision"] == 0.5
    assert m["recall"] == 0.5


def test_threshold_metrics_perfect_ranking_has_auc_one():
    m = threshold_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 0.5)
    assert m["roc_auc"] == 1.0
    assert m["pr_auc"] == 1.0


def test_threshold_for_precision_meets_target_with_max_recall():
    y = np.array([0, 0, 0, 1, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.6, 0.5, 0.7, 0.8, 0.9])
    pick = threshold_for_precision(y, scores, min_precision=1.0)
    assert pick["reached_target"]
    m = threshold_metrics(y, scores, pick["threshold"])
    assert m["precision"] == 1.0
    assert m["recall"] == 0.75  # the 0.5 spam sits below a ham at 0.6, so it is given up


def test_threshold_for_precision_reports_when_target_unreachable():
    y = np.array([0, 1, 0, 1])
    scores = np.array([0.9, 0.8, 0.2, 0.1])  # ranking is worse than random
    pick = threshold_for_precision(y, scores, min_precision=1.0)
    assert pick["reached_target"] is False


def test_select_model_prefers_simple_model_within_tolerance():
    results = pd.DataFrame(
        {
            "model": ["dummy", "naive_bayes", "logreg_word", "linear_svm_word", "logreg_char"],
            "val_pr_auc": [0.38, 0.990, 0.988, 0.991, 0.980],
        }
    )
    assert select_model(results) == "logreg_word"  # within 0.003 of the best, and preferred


def test_select_model_picks_clear_winner():
    results = pd.DataFrame(
        {
            "model": ["dummy", "naive_bayes", "logreg_word", "linear_svm_word", "logreg_char"],
            "val_pr_auc": [0.38, 0.90, 0.92, 0.99, 0.95],
        }
    )
    assert select_model(results) == "linear_svm_word"


def test_get_scores_returns_probabilities_for_logistic_regression():
    scores = get_scores(_toy_pipeline(), ["win a free prize", "see you at lunch"])
    assert scores.shape == (2,)
    assert (scores >= 0).all() and (scores <= 1).all()
    assert scores[0] > scores[1]


def test_saved_pipeline_gives_identical_scores_after_reload(tmp_path):
    pipe = _toy_pipeline()
    path = tmp_path / "model.joblib"
    joblib.dump(pipe, path)
    texts = ["free cash prize", "lunch tomorrow?"]
    assert np.allclose(get_scores(joblib.load(path), texts), get_scores(pipe, texts))


def test_vectorizer_is_fit_on_training_text_only():
    pipe = _toy_pipeline()
    vocab = set(pipe.named_steps["features"].vocabulary_)
    assert "lunch" in vocab
    assert "zebra" not in vocab  # a word never seen in training cannot be in the vocabulary
    # a message made only of unseen words still gets a valid score, it is not an error
    assert 0 <= get_scores(pipe, ["zebra quokka"])[0] <= 1


def test_empty_string_gets_a_valid_score_but_zero_messages_is_an_error():
    """An empty STRING scores fine, but a LIST with no messages is invalid input.

    So the detector (Phase 4) must reject empty input itself, before calling the model.
    """
    pipe = _toy_pipeline()
    assert get_scores(pipe, [""]).shape == (1,)
    with pytest.raises(ValueError):
        get_scores(pipe, [])
