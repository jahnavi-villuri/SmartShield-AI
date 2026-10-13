"""Regression tests on the REAL trained phishing model (skipped when it has not been trained)."""

import pandas as pd
import pytest

from config import settings
from src.detection.phishing_detector import (
    LABEL_UNKNOWN,
    ModelNotAvailableError,
    load_phishing_detector,
)

CHALLENGE = settings.PROJECT_ROOT / "tests" / "data" / "url_challenge.csv"
ORDINARY_KINDS = [
    "reference",
    "code",
    "news",
    "shopping",
    "video_social",
    "government_education",
    "homepage",
]


@pytest.fixture(scope="module")
def detector():
    try:
        return load_phishing_detector()
    except ModelNotAvailableError:
        pytest.skip("real URL model not trained")


@pytest.fixture(scope="module")
def challenge():
    return pd.read_csv(CHALLENGE, keep_default_na=False)


def _flagged(detector, urls):
    return [detector.predict(u).is_phishing for u in urls]


def test_phishing_style_addresses_are_caught(detector, challenge):
    phish = challenge[challenge["kind"] == "phishing_style"]
    assert sum(_flagged(detector, phish["url"])) >= 10


def test_ordinary_links_and_homepages_are_not_flagged(detector, challenge):
    """Regression for both structural shortcuts found in this phase: 'no path = phishing' and
    'no www = phishing'. Deep links AND bare homepages must pass."""
    ordinary = challenge[challenge["kind"].isin(ORDINARY_KINDS)]
    flagged = sum(_flagged(detector, ordinary["url"]))
    assert flagged <= 3, f"{flagged} of {len(ordinary)} ordinary links flagged"


@pytest.mark.parametrize(
    "url", ["https://dell.com", "https://github.com", "https://www.github.com", "https://bbc.co.uk"]
)
def test_bare_homepages_with_or_without_www_are_not_flagged(detector, url):
    assert not detector.predict(url).is_phishing


def test_scheme_does_not_change_the_verdict(detector):
    for host in ("dell.com", "example.org/docs"):
        assert detector.predict(f"http://{host}").score == pytest.approx(
            detector.predict(f"https://{host}").score
        )
    assert not detector.predict("dell.com").is_phishing  # typed without a scheme


def test_genuine_login_pages_are_mostly_not_flagged(detector, challenge):
    logins = challenge[challenge["kind"] == "login_and_banking_pages"]
    assert sum(_flagged(detector, logins["url"])) <= 3


def test_www_and_query_do_not_change_the_score(detector):
    base = detector.predict("https://example.org/docs/page").score
    assert detector.predict("https://www.example.org/docs/page").score == pytest.approx(base)
    assert detector.predict("https://example.org/docs/page?id=1&x=2#top").score == pytest.approx(
        base
    )


def test_shortened_links_get_no_verdict(detector):
    assert detector.predict("https://bit.ly/3AbCdEf").label == LABEL_UNKNOWN


def test_metadata_is_consistent_with_the_behaviour_it_reports(detector, challenge):
    meta = detector.metadata
    challenge_stats = meta["metrics"]["challenge"]
    share = challenge_stats["legit_flagged"] / challenge_stats["legit_n"]
    assert meta["guard_passed"] == (share <= 0.20)
    assert meta["query_and_fragment_ignored"] is True
    assert "phiusiil" in meta["trained_on"] and "phreshphish" in meta["trained_on"]
    assert "has_www" not in meta["feature_names"]
    assert "is_https" not in meta["feature_names"]


def test_overall_quality_has_not_regressed(detector):
    """Guard rails, not goals: temporal-test PR-AUC and the two cross-source sets."""
    m = detector.metadata["metrics"]
    assert m["test_temporal"]["pr_auc"] >= 0.90
    assert m["external"]["phishbench"]["false_positive_rate"] <= 0.25  # Round 3 had 0.02
    assert m["external"]["openphish_2026-10-12"]["detection_rate"] >= 0.70


@pytest.mark.xfail(
    reason="KNOWN WEAKNESS (documented in the app): legitimate pages on free hosting platforms "
    "(github.io, docs.google.com, sites.google.com, ...) are flagged, because in the training data "
    "almost every URL on those platforms is phishing.",
    strict=False,
)
def test_legitimate_pages_on_hosting_platforms_are_not_flagged(detector, challenge):
    hosted = challenge[challenge["kind"] == "hosting_platform_legit"]
    assert sum(_flagged(detector, hosted["url"])) <= 1


def test_predicting_never_touches_the_network(detector, monkeypatch):
    import socket

    def blocked(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    assert detector.predict("https://does-not-exist.example/login").score >= 0
