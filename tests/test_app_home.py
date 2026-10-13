"""Home and About pages, and the model summaries behind them (AppTest, no browser)."""

import json

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from config import settings
from src.detection.model_info import summarize_models


@pytest.fixture(autouse=True)
def _fresh_cache():
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


def _run(models_dir, monkeypatch):
    monkeypatch.setattr(settings, "MODELS_DIR", models_dir)
    return AppTest.from_file(str(settings.PROJECT_ROOT / "app.py"), default_timeout=30).run()


def test_summaries_read_metadata_without_loading_models(fake_models_dir):
    items = summarize_models(fake_models_dir)
    assert [i.key for i in items] == ["sms", "email", "comments", "url"]
    assert all(i.available and i.headline for i in items)


def test_missing_and_corrupt_metadata_are_reported_not_raised(tmp_path):
    (tmp_path / "sms_spam").mkdir()
    (tmp_path / "sms_spam" / "metadata.json").write_text("{not json", encoding="utf-8")
    items = summarize_models(tmp_path)
    assert not any(i.available for i in items)


def test_home_shows_warning_privacy_and_status(fake_models_dir, monkeypatch):
    app = _run(fake_models_dir, monkeypatch)
    assert not app.exception
    assert any("Predictions can be wrong" in w.value for w in app.warning)
    text = " ".join(m.value for m in app.markdown)
    assert "not stored" in text and "never opened" in text
    assert "model not available" not in text


def test_home_and_about_survive_with_no_models(tmp_path, monkeypatch):
    app = _run(tmp_path, monkeypatch)
    assert not app.exception
    text = " ".join(m.value for m in app.markdown)
    assert text.count("model not available") == 4
    assert len([e for e in app.error if "not available" in e.value]) >= 4


def test_about_numbers_follow_the_metadata_file(fake_models_dir, monkeypatch):
    meta_path = fake_models_dir / "sms_spam" / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["data_sizes"]["train"] = 12345
    meta_path.write_text(json.dumps(meta), encoding="utf-8")
    app = _run(fake_models_dir, monkeypatch)
    assert any("12,345" in m.value for m in app.markdown)
