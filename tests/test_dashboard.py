"""Dashboard: tables are built from metadata only; the page renders with all, some or no models."""

import json
from collections import Counter

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from config import settings
from src.ui import dashboard


def test_collect_with_no_models_is_empty_but_safe(tmp_path):
    data = dashboard.collect(tmp_path)
    assert data["meta"] == {} and data["totals"]["ready"] == 0
    assert data["gap_recall"].empty and data["weaknesses"].empty


def test_collect_reads_real_metadata_and_nothing_else():
    data = dashboard.collect(settings.MODELS_DIR)
    if not data["meta"]:
        pytest.skip("models not available")
    totals = data["totals"]
    assert totals["ready"] == len(data["meta"])
    assert totals["examples"] == sum(m["data_sizes"]["train"] for m in data["meta"].values())
    for frame in (data["gap_recall"], data["gap_false"]):
        assert frame["Value"].between(0, 1).all()
    assert set(data["gap_recall"]["Test"]) <= {"Same-source test", "Different source"}


def test_numbers_follow_the_metadata_file(fake_models_dir):
    path = fake_models_dir / "sms_spam" / "metadata.json"
    meta = json.loads(path.read_text(encoding="utf-8"))
    meta["data_sizes"]["train"] = 4321
    path.write_text(json.dumps(meta), encoding="utf-8")
    data = dashboard.collect(fake_models_dir)
    sms = data["sizes"]
    assert (
        int(sms[(sms["Detector"] == "SMS") & (sms["Split"] == "train")]["Examples"].iloc[0]) == 4321
    )


def test_corrupt_metadata_is_skipped(tmp_path):
    (tmp_path / "sms_spam").mkdir()
    (tmp_path / "sms_spam" / "metadata.json").write_text("{broken", encoding="utf-8")
    assert dashboard.collect(tmp_path)["meta"] == {}


def _render_app():
    from src.ui import dashboard as d

    d.render()


@pytest.fixture(autouse=True)
def _fresh_cache():
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


def test_dashboard_renders_with_fake_models_and_with_none(fake_models_dir, tmp_path, monkeypatch):
    for folder in (fake_models_dir, tmp_path):
        monkeypatch.setattr(settings, "MODELS_DIR", folder)
        app = AppTest.from_function(_render_app, default_timeout=30).run()
        assert not app.exception


def test_activity_keeps_counts_only_never_text():
    def script():
        import streamlit as st

        from src.ui import dashboard as d

        d.record_activity("SMS", "Scam")
        d.record_activity("SMS", "Scam")
        st.session_state["frame"] = d.activity_frame()

    app = AppTest.from_function(script, default_timeout=30).run()
    frame = app.session_state["frame"]
    assert frame.to_dict("records") == [{"Detector": "SMS", "Verdict": "Scam", "Count": 2}]
    assert isinstance(app.session_state["activity"], Counter)


def test_analysing_in_the_app_records_a_verdict_but_not_the_text(fake_models_dir, monkeypatch):
    monkeypatch.setattr(settings, "MODELS_DIR", fake_models_dir)
    secret = "private-words-xyz"
    app = AppTest.from_file(str(settings.PROJECT_ROOT / "app.py"), default_timeout=30).run()
    app.text_area(key="sms_message").set_value(secret).run()
    app.button(key="sms_analyze").click().run()
    activity = app.session_state["activity"]
    assert sum(activity.values()) == 1
    assert secret not in repr(dict(activity))


def test_example_buttons_fill_the_input_box(fake_models_dir, monkeypatch):
    monkeypatch.setattr(settings, "MODELS_DIR", fake_models_dir)
    app = AppTest.from_file(str(settings.PROJECT_ROOT / "app.py"), default_timeout=30).run()
    app.button(key="sms_example_2").click().run()
    assert "KYC" in app.text_area(key="sms_message").value
    app.button(key="url_example_1").click().run()
    assert "secure-login-verify" in app.text_input(key="url_input").value
