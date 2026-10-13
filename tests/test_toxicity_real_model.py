"""Regression tests on the REAL trained toxicity model (skipped when it has not been trained)."""

import pandas as pd
import pytest

from config import settings
from src.detection.toxicity_detector import ModelNotAvailableError, load_toxicity_detector
from src.models.toxicity_model import LABELS

PROBE = settings.PROJECT_ROOT / "tests" / "data" / "toxicity_probe.csv"


@pytest.fixture(scope="module")
def detector():
    try:
        return load_toxicity_detector()
    except ModelNotAvailableError:
        pytest.skip("real comment model not trained")


@pytest.fixture(scope="module")
def probe():
    return pd.read_csv(PROBE, keep_default_na=False)


def test_six_scores_in_order_between_zero_and_one(detector):
    result = detector.predict("Thanks for the correction.")
    assert list(result.scores) == LABELS
    assert all(0.0 <= v <= 1.0 for v in result.scores.values())


def test_metadata_thresholds_match_the_detector(detector):
    assert set(detector.metadata["thresholds"]) == set(LABELS)
    assert detector.thresholds == {k: float(v) for k, v in detector.metadata["thresholds"].items()}


def test_a_plain_insult_and_a_plain_threat_are_flagged(detector):
    assert "insult" in detector.predict("You are such an idiot.").flagged
    assert "threat" in detector.predict("I will find you and hurt you.").flagged


def test_plain_neutral_sentences_are_not_flagged(detector, probe):
    neutral = probe[probe["kind"].isin(["neutral", "polite", "news"])]
    flagged = sum(detector.predict(t, explain=False).any_flagged for t in neutral["text"])
    assert flagged <= 1, f"{flagged} of {len(neutral)} neutral sentences flagged"


def test_most_mild_toxic_probe_sentences_are_caught(detector, probe):
    toxic = probe[probe["expected"] != "none"]
    caught = 0
    for row in toxic.itertuples():
        flagged = set(detector.predict(row.text, explain=False).flagged)
        caught += bool(flagged & set(row.expected.split("|")))
    assert caught >= 0.75 * len(toxic)


def test_overall_quality_has_not_regressed(detector):
    """Guard rails, not goals: official-test macro-F1 and every label's ROC-AUC."""
    official = detector.metadata["metrics"]["official_test"]
    assert official["summary"]["macro_f1"] >= 0.50
    for label in LABELS:
        assert official["per_label"][label]["roc_auc"] >= 0.95


def test_metadata_reports_the_bias_probe_honestly(detector):
    bias = detector.metadata["bias_probe"]
    assert bias["acceptable"] == (bias["max_flagged_share"] <= 0.05)


@pytest.mark.xfail(
    reason="KNOWN WEAKNESS: a word-based model cannot read negation or mentions, so 'You are "
    "not an idiot' is flagged as an insult.",
    strict=False,
)
def test_negated_insult_is_not_flagged(detector):
    assert not detector.predict("You are not an idiot, you simply misread the policy.").any_flagged


@pytest.mark.xfail(
    reason="KNOWN WEAKNESS (documented in the app): neutral sentences that mention an identity "
    "term such as 'gay' are flagged because the training data pairs those words with toxicity.",
    strict=False,
)
def test_neutral_sentences_with_identity_terms_are_not_flagged(detector):
    for term in ("gay", "Muslim", "Jewish", "black", "bisexual"):
        assert not detector.predict(f"I am a {term} person.", explain=False).any_flagged


def test_predicting_never_touches_the_network(detector, monkeypatch):
    import socket

    def blocked(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    assert detector.predict("Visit http://x.example for details", explain=False).scores
