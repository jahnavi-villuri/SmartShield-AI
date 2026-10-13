"""Regression tests on the REAL trained model (skipped when it has not been trained yet)."""

import pandas as pd
import pytest

from config import settings
from src.detection.spam_detector import ModelNotAvailableError, load_sms_detector


@pytest.fixture(scope="module")
def detector():
    try:
        return load_sms_detector()
    except ModelNotAvailableError:
        pytest.skip("real SMS model not trained")


def test_legitimate_bank_alerts_are_mostly_not_flagged(detector):
    """Regression for the 'Rs means spam' shortcut (18 of 23 legitimate messages were flagged)."""
    challenge = pd.read_csv(settings.PROJECT_ROOT / "tests" / "data" / "sms_challenge.csv")
    legit = challenge[challenge["label"] == 0]
    flagged = sum(detector.predict(t, explain=False).is_spam for t in legit["text"])
    assert flagged <= 6, f"{flagged} of {len(legit)} legitimate messages flagged"


def test_scams_in_challenge_set_are_mostly_caught(detector):
    challenge = pd.read_csv(settings.PROJECT_ROOT / "tests" / "data" / "sms_challenge.csv")
    scams = challenge[challenge["label"] == 1]
    caught = sum(detector.predict(t, explain=False).is_spam for t in scams["text"])
    assert caught >= 8, f"only {caught} of {len(scams)} scams caught"


@pytest.mark.parametrize(
    "text", ["Rs :300/- is credited to your account", "Rs:300/- debited from your a/c XX1234"]
)
def test_simple_credit_and_debit_alerts_are_not_spam(detector, text):
    assert not detector.predict(text, explain=False).is_spam


def test_credit_alert_with_a_link_to_claim_money_is_spam(detector):
    text = "Rs 300 credited to your account. Click http://claim-now.example to withdraw your bonus"
    assert detector.predict(text, explain=False).is_spam


def test_reported_brand_promotion_is_promotional_not_scam(detector):
    """Reported by the project owner: a Taco Bell offer is marketing, not a scam."""
    reported = pd.read_csv(settings.PROJECT_ROOT / "tests" / "data" / "sms_reported_failures.csv")
    text = reported.loc[reported["expected"] == "promotional", "text"].iloc[0]
    result = detector.predict(text, explain=False)
    if detector.type_pipeline is None:
        pytest.skip("promotional-vs-scam model not trained")
    assert result.label == "Promotional"


def test_phishing_message_is_labelled_scam(detector):
    text = "Your account has been suspended. Verify now at http://secure-bank-login.example"
    result = detector.predict(text, explain=False)
    assert result.is_spam
    if detector.type_pipeline is not None:
        assert result.label == "Scam"


def test_reported_real_bank_alert_is_not_spam(detector):
    """Reported by the project owner (de-identified copy). It was wrongly flagged until real
    private messages were added to training; the margin is thin, so variants are tested too."""
    reported = pd.read_csv(settings.PROJECT_ROOT / "tests" / "data" / "sms_reported_failures.csv")
    text = reported.loc[reported["expected"] == "legitimate", "text"].iloc[0]
    assert not detector.predict(text, explain=False).is_spam


@pytest.mark.parametrize(
    "text",
    [
        "Dear UPI user A/C X7782 debited by 1500.00 on date 02Nov26 trf to S Reddy Refno "
        "771204488121 If not u? call-1800222333 for other services-18004455-SBI",
        "Dear UPI user A/C X7782 credited by 900.00 on date 02Nov26 trf from A Rao Refno "
        "771204488999 -SBI",
    ],
)
def test_similar_bank_alerts_are_not_spam(detector, text):
    assert not detector.predict(text, explain=False).is_spam


def test_bank_phishing_is_still_a_scam(detector):
    text = "Dear SBI user your SBI YONO account will be blocked today. Update your PAN CARD at http://sbi-kyc.example"
    result = detector.predict(text, explain=False)
    assert result.is_spam
    if detector.type_pipeline is not None:
        assert result.label == "Scam"
