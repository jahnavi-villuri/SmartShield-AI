"""URL feature extraction and validation (the code shared by training and the app)."""

import math

import numpy as np
import pytest

from src.evaluation.metrics import threshold_for_fpr, threshold_metrics
from src.features.url_features import (
    FEATURE_NAMES,
    MAX_URL_CHARS,
    InvalidUrlError,
    UrlFeatureTransformer,
    UrlText,
    clean_url,
    extract_features,
    features_frame,
    parse_url,
)
from src.training.train_phishing import select_url_model

# ---------- validation ----------


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "   ",
        "javascript:alert(1)",
        "data:text/html;base64,AAAA",
        "file:///etc/passwd",
        "ftp://example.com/file",
        "http://",
        "http:///path",
        "not a url",
        "http://exa mple.com",
        "http://example.com/a b",
        "http://[::1",
        "http://exam\x00ple.com",
        "a" * (MAX_URL_CHARS + 1),
    ],
)
def test_unusable_input_is_rejected_with_a_message(bad):
    with pytest.raises(InvalidUrlError) as info:
        clean_url(bad)
    assert str(info.value)


def test_non_text_input_is_rejected():
    for bad in (None, 123, ["http://example.com"]):
        with pytest.raises(InvalidUrlError):
            clean_url(bad)


@pytest.mark.parametrize(
    ("typed", "expected_start"),
    [
        ("https://example.com/a", "https://example.com"),
        ("HTTP://EXAMPLE.COM/A", "HTTP://EXAMPLE.COM"),
        ("example.com/page", "http://example.com"),
        ("example.com:8080/x", "http://example.com:8080"),
        ("  https://example.com  ", "https://example.com"),
        ("http://192.168.1.5/login", "http://192.168.1.5"),
    ],
)
def test_reasonable_input_is_accepted_and_normalised(typed, expected_start):
    assert clean_url(typed).startswith(expected_start)


# ---------- parsing ----------


def test_registered_domain_handles_two_part_suffixes():
    assert parse_url("https://www.bbc.co.uk/news").registered_domain == "bbc.co.uk"
    assert parse_url("https://a.b.example.com/x").registered_domain == "example.com"
    assert parse_url("https://a.b.example.com/x").subdomain_labels == ("a", "b")


def test_ip_hosts_are_detected():
    assert extract_features("http://192.168.1.5/login")["is_ip_host"] == 1.0
    assert extract_features("https://example.com/")["is_ip_host"] == 0.0


# ---------- features ----------


def test_feature_names_are_fixed_and_unique():
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES)) == 44
    assert list(extract_features("https://example.com/x")) == FEATURE_NAMES


def test_every_feature_is_a_finite_number():
    for url in [
        "https://example.com",
        "http://a.b.c.d.example.xyz/x/y/z.php",
        "http://1.2.3.4:81/p",
    ]:
        for value in extract_features(url).values():
            assert isinstance(value, int | float) and math.isfinite(value)


def test_www_prefix_is_ignored():
    assert extract_features("https://www.dell.com/support") == extract_features(
        "https://dell.com/support"
    )
    assert "has_www" not in FEATURE_NAMES


def test_scheme_is_not_a_model_feature():
    assert "is_https" not in FEATURE_NAMES
    assert extract_features("http://example.com/a") == extract_features("https://example.com/a")


def test_query_string_and_fragment_are_ignored():
    plain = extract_features("https://www.youtube.com/watch")
    assert extract_features("https://www.youtube.com/watch?v=abc&list=xyz#t=3") == plain


def test_phishing_style_address_shows_the_expected_signals():
    f = extract_features("http://paypal-secure-login.verify-account.example.xyz/signin.php")
    assert f["brand_not_registered_domain"] == 1.0
    assert f["n_login_words_host"] >= 2
    assert f["tld_suspicious"] == 1.0
    assert f["ends_with_script_ext"] == 1.0


def test_known_platforms_and_shorteners_are_flagged():
    assert extract_features("https://someone.github.io/page")["is_hosting_platform"] == 1.0
    assert extract_features("https://bit.ly/abc123")["is_shortener"] == 1.0
    assert extract_features("https://www.wikipedia.org/")["is_hosting_platform"] == 0.0


def test_punycode_and_userinfo_are_flagged():
    assert extract_features("https://xn--pypal-4ve.com/")["has_punycode"] == 1.0
    assert extract_features("https://paypal.com@evil.example/login")["has_at"] == 1.0


# ---------- bulk use (training) and the sklearn wrapper ----------


def test_features_frame_nan_mode_keeps_row_count_and_order():
    frame = features_frame(["https://a.com", "javascript:x", "https://b.com/p"], on_error="nan")
    assert list(frame.columns) == FEATURE_NAMES
    assert len(frame) == 3 and frame.iloc[1].isna().all() and not frame.iloc[0].isna().any()


def test_features_frame_raise_mode_propagates_the_error():
    with pytest.raises(InvalidUrlError):
        features_frame(["https://a.com", "javascript:x"], on_error="raise")


def test_transformer_output_equals_the_shared_function_exactly():
    """Train/serve parity: the pipeline's feature step is the same code as the app's."""
    urls = ["https://www.example.com/a/b", "http://paypal-login.example.xyz/x.php"]
    matrix = UrlFeatureTransformer().transform(urls)
    expected = features_frame(urls).to_numpy()
    assert matrix.shape == (2, len(FEATURE_NAMES))
    assert np.array_equal(matrix, expected)


def test_transformer_turns_invalid_urls_into_zero_rows_instead_of_crashing():
    matrix = UrlFeatureTransformer().transform(["javascript:alert(1)"])
    assert matrix.shape == (1, len(FEATURE_NAMES)) and not matrix.any()


def test_extraction_never_touches_the_network(monkeypatch):
    import socket

    def blocked(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    extract_features("https://does-not-exist.example/login?x=1")


# ---------- the char-model text, thresholds, selection ----------


def test_url_text_drops_scheme_www_and_query():
    assert UrlText()("https://WWW.Example.com/Path/Page?id=7#top") == "example.com/path/page"
    assert UrlText()("javascript:alert(1)") == ""


def test_false_positive_budget_threshold_respects_the_budget():
    rng = np.random.default_rng(0)
    y = np.array([0] * 500 + [1] * 500)
    scores = np.concatenate([rng.uniform(0, 0.7, 500), rng.uniform(0.3, 1, 500)])
    pick = threshold_for_fpr(y, scores, max_fpr=0.02)
    result = threshold_metrics(y, scores, pick["threshold"])
    assert pick["reached_target"] and result["false_positive_rate"] <= 0.02


def test_selection_skips_a_candidate_that_fails_the_guard():
    import pandas as pd

    results = pd.DataFrame(
        {
            "model": ["dummy", "logreg_features", "random_forest", "hist_gb", "logreg_char"],
            "val_pr_auc": [0.5, 0.958, 0.959, 0.960, 0.90],
            "challenge_legit_flagged_share": [0.0, 0.5, 0.1, 0.1, 0.0],
        }
    )
    # logreg_features is preferred but fails the guard, so the next preferred candidate wins
    assert select_url_model(results) == ("random_forest", True)


def test_selection_reports_when_no_candidate_passes_the_guard():
    import pandas as pd

    results = pd.DataFrame(
        {
            "model": ["dummy", "logreg_features", "random_forest", "hist_gb", "logreg_char"],
            "val_pr_auc": [0.5, 0.95, 0.95, 0.95, 0.95],
            "challenge_legit_flagged_share": [0.0, 0.6, 0.6, 0.6, 0.6],
        }
    )
    name, passed = select_url_model(results)
    assert name == "logreg_features" and passed is False
