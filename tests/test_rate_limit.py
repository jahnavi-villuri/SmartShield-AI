"""The per-session throttle (unit tests with a fake clock, plus the app)."""

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from config import settings
from src.detection.rate_limit import RateLimiter


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def test_allows_up_to_the_limit_then_blocks():
    clock = Clock()
    limiter = RateLimiter(3, 10, clock)
    assert [limiter.allow() for _ in range(4)] == [True, True, True, False]
    assert limiter.retry_after() == pytest.approx(10)


def test_window_slides_and_calls_are_allowed_again():
    clock = Clock()
    limiter = RateLimiter(2, 10, clock)
    limiter.allow()
    clock.now = 5
    limiter.allow()
    assert not limiter.allow()
    clock.now = 10.5  # the first call has left the window
    assert limiter.allow()
    assert not limiter.allow()


def test_invalid_settings_are_rejected():
    for args in ((0, 10), (1, 0)):
        with pytest.raises(ValueError):
            RateLimiter(*args)


def test_app_blocks_rapid_clicks_with_a_message(fake_models_dir, monkeypatch):
    st.cache_resource.clear()
    monkeypatch.setattr(settings, "MODELS_DIR", fake_models_dir)
    monkeypatch.setattr(settings, "RATE_LIMIT_CALLS", 2)
    app = AppTest.from_file(str(settings.PROJECT_ROOT / "app.py"), default_timeout=30).run()
    app.text_area(key="sms_message").set_value("hello there").run()
    for _ in range(3):
        app.button(key="sms_analyze").click().run()
    assert any("Too many requests" in w.value for w in app.warning)
    assert not app.exception
    st.cache_resource.clear()
