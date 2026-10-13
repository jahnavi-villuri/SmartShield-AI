"""Phishing detector and the URL tab: safety rules, loading, prediction, honesty of wording."""

import json

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from config import settings
from src.detection.phishing_detector import (
    LABEL_LEGIT,
    LABEL_PHISHING,
    LABEL_UNKNOWN,
    InvalidInputError,
    ModelNotAvailableError,
    load_phishing_detector,
)

PHISH = "http://paypal-secure-login-42.xyz/verify/account/login7.php"
LEGIT = "https://www.gardenrecipes.com/news/travel-12"


# ---------- loading ----------


def test_missing_model_gives_a_friendly_error(tmp_path):
    with pytest.raises(ModelNotAvailableError, match="not been trained"):
        load_phishing_detector(tmp_path)


def test_tampered_model_is_refused(tmp_path):
    from tests.conftest import write_fake_model

    write_fake_model(tmp_path, with_type=False, with_email=False, with_toxicity=False)
    model = tmp_path / "phishing" / "model.joblib"
    model.write_bytes(model.read_bytes() + b"tampered")
    with pytest.raises(ModelNotAvailableError, match="checksum"):
        load_phishing_detector(tmp_path)


# ---------- input safety ----------


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "   ",
        "javascript:alert(1)",
        "data:text/html,hi",
        "file:///etc/passwd",
        "ftp://x.example/f",
        "http://exa mple.com",
        "a" * 3000,
    ],
)
def test_unsafe_or_malformed_input_is_refused_with_a_message(fake_models_dir, bad):
    with pytest.raises(InvalidInputError) as info:
        load_phishing_detector(fake_models_dir).predict(bad)
    assert str(info.value)


def test_non_text_input_is_refused(fake_models_dir):
    with pytest.raises(InvalidInputError):
        load_phishing_detector(fake_models_dir).predict(None)


def test_predicting_never_touches_the_network(fake_models_dir, monkeypatch):
    import socket

    detector = load_phishing_detector(fake_models_dir)

    def blocked(*args, **kwargs):
        raise AssertionError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    assert detector.predict("https://does-not-exist.example/login").score >= 0


# ---------- predictions and wording ----------


def test_phishing_style_and_ordinary_urls_are_told_apart(fake_models_dir):
    detector = load_phishing_detector(fake_models_dir)
    bad, good = detector.predict(PHISH), detector.predict(LEGIT)
    assert bad.is_phishing and bad.label == LABEL_PHISHING
    assert not good.is_phishing and good.label == LABEL_LEGIT
    assert good.score < bad.score


def test_wording_never_claims_a_site_is_safe(fake_models_dir):
    detector = load_phishing_detector(fake_models_dir)
    for url in (PHISH, LEGIT, "https://bit.ly/abc"):
        label = detector.predict(url).label.lower()
        assert "safe" not in label and "secure" not in label and "trusted" not in label


def test_shortened_links_get_no_verdict(fake_models_dir):
    result = load_phishing_detector(fake_models_dir).predict("https://bit.ly/3AbCdEf")
    assert result.label == LABEL_UNKNOWN and not result.judged and not result.is_phishing
    assert "shortened link" in result.observations[0] and "never does" in result.observations[0]


def test_query_and_missing_scheme_are_reported_not_hidden(fake_models_dir):
    detector = load_phishing_detector(fake_models_dir)
    with_query = detector.predict("https://www.gardenrecipes.com/news?id=7#top")
    assert with_query.ignored_query and not with_query.assumed_scheme
    bare = detector.predict("www.gardenrecipes.com/news")
    assert bare.assumed_scheme and bare.url.startswith("http://")
    assert (
        detector.predict("https://x.example/a?z=1").score
        == detector.predict("https://x.example/a").score
    )


def test_observations_describe_the_address(fake_models_dir):
    detector = load_phishing_detector(fake_models_dir)
    ip = detector.predict("http://192.0.2.7/login/index.php")
    assert any("numeric IP" in n for n in ip.observations)
    brand = detector.predict(PHISH)
    assert any("brand" in n for n in brand.observations)
    hosted = detector.predict("https://someone.github.io/portfolio")
    assert any("free hosting platform" in n for n in hosted.observations)


# ---------- the URL tab ----------


@pytest.fixture(autouse=True)
def _fresh_cache():
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


def _run(models_dir, monkeypatch):
    monkeypatch.setattr(settings, "MODELS_DIR", models_dir)
    return AppTest.from_file(str(settings.PROJECT_ROOT / "app.py"), default_timeout=30).run()


def _check(app, url):
    app.text_input(key="url_input").set_value(url).run()
    app.button(key="url_analyze").click().run()
    return app


def test_page_has_the_url_tab(fake_models_dir, monkeypatch):
    app = _run(fake_models_dir, monkeypatch)
    assert not app.exception and "URL" in [t.label for t in app.tabs]


def test_url_tab_flags_phishing_and_shows_the_url_as_plain_text(fake_models_dir, monkeypatch):
    app = _check(_run(fake_models_dir, monkeypatch), PHISH)
    assert not app.exception
    assert any(LABEL_PHISHING in e.value for e in app.error)
    assert any(PHISH in c.value for c in app.code)  # shown in a code block, not as a link
    assert not any(f"]({PHISH}" in m.value or f"<{PHISH}>" in m.value for m in app.markdown)
    assert "Model score" in [m.label for m in app.metric]


def test_url_tab_accepts_an_ordinary_url_but_never_says_safe(fake_models_dir, monkeypatch):
    app = _check(_run(fake_models_dir, monkeypatch), LEGIT)
    assert any(LABEL_LEGIT in s.value for s in app.success)
    shown = " ".join(
        [s.value for s in app.success]
        + [c.value for c in app.caption]
        + [e.value for e in app.error]
    )
    assert "cannot confirm that a site is safe" in shown
    assert not any("**Safe" in s.value for s in app.success)


def test_url_tab_rejects_bad_input_with_a_message(fake_models_dir, monkeypatch):
    for bad in ("", "javascript:alert(1)", "http://exa mple.com"):
        app = _check(_run(fake_models_dir, monkeypatch), bad)
        assert not app.exception
        assert app.error, f"no message shown for {bad!r}"


def test_shortened_link_shows_cannot_judge(fake_models_dir, monkeypatch):
    app = _check(_run(fake_models_dir, monkeypatch), "https://bit.ly/3AbCdEf")
    assert any(LABEL_UNKNOWN in i.value for i in app.info)


def test_missing_url_model_does_not_break_the_other_tabs(tmp_path, monkeypatch):
    from tests.conftest import write_fake_model

    write_fake_model(tmp_path, with_phishing=False)
    app = _run(tmp_path, monkeypatch)
    assert not app.exception
    assert any("URL model has not been trained" in e.value for e in app.error)
    keys = [b.key for b in app.button]
    assert "url_analyze" not in keys and "sms_analyze" in keys


def test_weakness_warning_only_when_the_guard_failed(fake_models_dir, monkeypatch):
    assert not any("Known weakness" in w.value for w in _run(fake_models_dir, monkeypatch).warning)
    meta_path = fake_models_dir / "phishing" / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["guard_passed"] = False
    meta["metrics"]["challenge"] = {"legit_flagged": 15, "legit_n": 52}
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    st.cache_resource.clear()
    text = next(
        w.value for w in _run(fake_models_dir, monkeypatch).warning if "Known weakness" in w.value
    )
    assert "15 of 52" in text
