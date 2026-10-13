"""The Streamlit page, driven by Streamlit's AppTest (no browser needed)."""

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from config import settings


@pytest.fixture(autouse=True)
def _fresh_cache():
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


def _run_app(models_dir, monkeypatch):
    monkeypatch.setattr(settings, "MODELS_DIR", models_dir)
    return AppTest.from_file(str(settings.PROJECT_ROOT / "app.py"), default_timeout=30).run()


def _analyze(app, text):
    app.text_area(key="sms_message").set_value(text).run()
    app.button(key="sms_analyze").click().run()
    return app


def test_app_shows_error_not_crash_when_model_missing(tmp_path, monkeypatch):
    app = _run_app(tmp_path, monkeypatch)
    assert not app.exception
    assert any("not been trained" in e.value for e in app.error)
    assert len(app.button) == 0  # the form is not offered without a model


def test_app_always_shows_the_accuracy_warning(fake_models_dir, monkeypatch):
    app = _run_app(fake_models_dir, monkeypatch)
    assert any("can be wrong" in w.value for w in app.warning)


def test_app_flags_a_scam_in_red_with_both_scores(fake_models_dir, monkeypatch):
    app = _analyze(
        _run_app(fake_models_dir, monkeypatch), "your account is blocked verify click link"
    )
    assert not app.exception
    assert any("Scam" in e.value for e in app.error)
    assert [m.label for m in app.metric] == ["Unwanted-message score", "Scam likelihood"]


def test_app_shows_promotional_as_a_warning_not_an_error(fake_models_dir, monkeypatch):
    app = _analyze(_run_app(fake_models_dir, monkeypatch), "big sale discount offer buy now")
    assert not app.exception
    assert any("Promotional" in w.value for w in app.warning)
    assert not any("Promotional" in e.value for e in app.error)


def test_app_accepts_normal_message(fake_models_dir, monkeypatch):
    app = _analyze(_run_app(fake_models_dir, monkeypatch), "see you at lunch")
    assert not app.exception
    assert any("Not spam" in s.value for s in app.success)
    assert [m.label for m in app.metric] == ["Unwanted-message score"]


def test_app_without_type_model_still_works_and_says_so(fake_models_no_type, monkeypatch):
    app = _analyze(_run_app(fake_models_no_type, monkeypatch), "win a free prize now")
    assert not app.exception
    assert any("not available" in i.value for i in app.info)
    assert any("Spam" in e.value for e in app.error)


def test_app_rejects_empty_input_with_message(fake_models_dir, monkeypatch):
    app = _run_app(fake_models_dir, monkeypatch)
    app.button(key="sms_analyze").click().run()
    assert not app.exception
    assert any("enter a message" in e.value for e in app.error)


def test_app_rejects_overlong_input_with_message(fake_models_dir, monkeypatch):
    app = _analyze(_run_app(fake_models_dir, monkeypatch), "a" * 1100)
    assert not app.exception
    assert any("too long" in e.value for e in app.error)
