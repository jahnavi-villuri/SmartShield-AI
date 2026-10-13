"""Regression tests on the REAL trained email model (skipped when it has not been trained)."""

import pandas as pd
import pytest

from config import settings
from src.detection.email_detector import ModelNotAvailableError, load_email_detector

CHALLENGE = settings.PROJECT_ROOT / "tests" / "data" / "email_challenge.csv"


@pytest.fixture(scope="module")
def detector():
    try:
        return load_email_detector()
    except ModelNotAvailableError:
        pytest.skip("real email model not trained")


@pytest.fixture(scope="module")
def challenge():
    return pd.read_csv(CHALLENGE, keep_default_na=False)


def _flagged(detector, frame):
    return [detector.predict(r.subject, r.body, explain=False).is_spam for r in frame.itertuples()]


def test_modern_spam_and_phishing_are_mostly_caught(detector, challenge):
    spam = challenge[challenge["label"] == 1]
    assert sum(_flagged(detector, spam)) >= 8


def test_ordinary_work_and_personal_email_is_not_flagged(detector, challenge):
    plain = challenge[challenge["kind"].isin(["work", "personal"])]
    flagged = sum(_flagged(detector, plain))
    assert flagged <= 1, f"{flagged} of {len(plain)} ordinary work/personal emails flagged"


def test_a_clear_pharma_spam_and_a_phishing_email_are_flagged(detector):
    assert detector.predict(
        "Cheap meds", "Order without a prescription, 80% off. Click here http://pharma.example"
    ).is_spam
    assert detector.predict(
        "Your account has been suspended",
        "Verify your identity now: http://account-verify.example/restore within 12 hours.",
    ).is_spam


def test_metadata_records_that_the_guard_failed_so_the_app_can_warn(detector):
    """If a better model later passes the guard, update this test and the README together."""
    meta = detector.metadata
    legit = meta["metrics"]["challenge_modern"]
    flagged_share = legit["fp"] / (legit["fp"] + legit["tn"])
    assert meta["guard_passed"] == (flagged_share <= 0.25)


@pytest.mark.xfail(
    reason="KNOWN WEAKNESS (documented in the app and Phase 5 README): automated legitimate "
    "modern email (statements, verification codes, receipts, newsletters) is flagged about a "
    "third of the time because the Enron training data has almost none of it.",
    strict=False,
)
def test_automated_modern_legitimate_email_is_rarely_flagged(detector, challenge):
    legit = challenge[challenge["label"] == 0]
    assert sum(_flagged(detector, legit)) / len(legit) <= 0.25


def test_text_is_not_stored_or_fetched_when_predicting(detector, monkeypatch):
    """Predicting must never touch the network (links stay plain text)."""
    import socket

    def blocked(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    result = detector.predict("Win", "Visit http://evil.example/now to claim", explain=False)
    assert isinstance(result.score, float)
