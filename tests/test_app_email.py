"""The Email tab of the Streamlit page (AppTest, no browser)."""

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from config import settings


@pytest.fixture(autouse=True)
def _fresh_cache():
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


def _run(models_dir, monkeypatch):
    monkeypatch.setattr(settings, "MODELS_DIR", models_dir)
    return AppTest.from_file(str(settings.PROJECT_ROOT / "app.py"), default_timeout=30).run()


def _analyze(app, subject, body):
    app.text_input(key="email_subject").set_value(subject)
    app.text_area(key="email_body").set_value(body).run()
    app.button(key="email_analyze").click().run()
    return app


def test_page_has_all_tabs(fake_models_dir, monkeypatch):
    app = _run(fake_models_dir, monkeypatch)
    assert not app.exception
    assert [t.label for t in app.tabs] == [
        "Home",
        "SMS",
        "Email",
        "Comments",
        "URL",
        "Dashboard",
        "About",
    ]


def test_email_spam_is_flagged_with_a_score(fake_models_dir, monkeypatch):
    app = _analyze(
        _run(fake_models_dir, monkeypatch),
        "Win a prize",
        "Click the link now to claim your free cash prize",
    )
    assert not app.exception
    assert any("Spam" in e.value for e in app.error)
    assert "Spam score" in [m.label for m in app.metric]


def test_normal_email_is_accepted(fake_models_dir, monkeypatch):
    app = _analyze(
        _run(fake_models_dir, monkeypatch),
        "Report draft",
        "I attached the draft report, comments by Friday",
    )
    assert not app.exception
    assert any("Not spam" in s.value for s in app.success)


def test_empty_email_is_rejected_with_a_message(fake_models_dir, monkeypatch):
    app = _run(fake_models_dir, monkeypatch)
    app.button(key="email_analyze").click().run()
    assert not app.exception
    assert any("subject or body" in e.value for e in app.error)


def test_overlong_body_is_rejected_with_a_message(fake_models_dir, monkeypatch):
    app = _analyze(_run(fake_models_dir, monkeypatch), "s", "b" * 20_001)
    assert not app.exception
    assert any("too long" in e.value for e in app.error)


def test_missing_email_model_does_not_break_the_sms_tab(fake_models_no_email, monkeypatch):
    app = _run(fake_models_no_email, monkeypatch)
    assert not app.exception
    assert any("email model has not been trained" in e.value for e in app.error)
    app.text_area(key="sms_message").set_value("see you at lunch").run()
    app.button(key="sms_analyze").click().run()
    assert any("Not spam" in s.value for s in app.success)
    assert "email_analyze" not in [b.key for b in app.button]  # no email form offered
    assert "sms_analyze" in [b.key for b in app.button]


def test_weakness_warning_is_shown_only_when_the_guard_failed(fake_models_dir, monkeypatch):
    import json

    app = _run(fake_models_dir, monkeypatch)
    assert not any("Known weakness" in w.value for w in app.warning)

    meta_path = fake_models_dir / "email_spam" / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["guard_passed"] = False
    meta["metrics"]["challenge_modern"] = {"fp": 8, "tn": 16}
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    st.cache_resource.clear()
    app = _run(fake_models_dir, monkeypatch)
    warning = next(w.value for w in app.warning if "Known weakness" in w.value)
    assert "8 of 24" in warning
