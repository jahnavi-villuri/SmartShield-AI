"""Security and privacy guard rails (Phase 9). Static checks plus hostile-input checks."""

import ast
import re
import tomllib
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from config import settings
from src.detection.common import DetectorError

ROOT = settings.PROJECT_ROOT
CODE_FILES = [ROOT / "app.py", *sorted((ROOT / "src" / "detection").glob("*.py"))]
CODE_FILES += sorted((ROOT / "src" / "features").glob("*.py"))
UI_FILES = sorted((ROOT / "src" / "ui").glob("*.py"))
CODE_FILES += UI_FILES
NETWORK_MODULES = {"requests", "http", "socket", "httpx", "aiohttp", "subprocess"}
NETWORK_SUBMODULES = {"urllib.request", "urllib.error"}  # urllib.parse (string parsing) is fine


def _imports(path: Path) -> set[str]:
    """Top-level module names plus full dotted names, e.g. {'urllib', 'urllib.parse'}."""
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module, *[f"{node.module}.{a.name}" for a in node.names]]
        else:
            continue
        for name in names:
            found |= {name.split(".")[0], name}
    return found


@pytest.mark.parametrize("path", CODE_FILES, ids=lambda p: p.name)
def test_serving_code_has_no_network_or_process_imports(path):
    assert not (_imports(path) & (NETWORK_MODULES | NETWORK_SUBMODULES)), path.name


@pytest.mark.parametrize("path", CODE_FILES, ids=lambda p: p.name)
def test_serving_code_never_logs_prints_or_evals(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    calls = {
        n.func.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert not calls & {"print", "eval", "exec", "open"}, f"{path.name}: {calls}"
    assert "logging" not in _imports(path)


def test_app_never_allows_raw_html():
    for path in [ROOT / "app.py", *UI_FILES]:
        assert "unsafe_allow_html" not in path.read_text(encoding="utf-8"), path.name
        assert "unsafe_allow_javascript" not in path.read_text(encoding="utf-8"), path.name


def test_no_script_tags_or_external_resources_in_ui_code():
    for path in UI_FILES:
        text = path.read_text(encoding="utf-8").lower()
        assert "<script" not in text, path.name
        assert not re.search(r"(src|href)=[\"']https?://", text), path.name
        assert "@import" not in text and "url(http" not in text, path.name


def test_streamlit_config_is_privacy_and_security_friendly():
    config = tomllib.loads((ROOT / ".streamlit" / "config.toml").read_text(encoding="utf-8"))
    assert config["browser"]["gatherUsageStats"] is False
    assert config["server"]["enableXsrfProtection"] is True
    assert config["client"]["showErrorDetails"] is False


def test_private_data_models_and_secrets_are_git_ignored():
    ignored = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for pattern in ("data/private/*", "data/raw/*", "data/labels/*", "models/**/*.joblib", ".env"):
        assert pattern in ignored, pattern


def test_requirements_are_parseable_and_pinned():
    for name in ("requirements.txt", "requirements-dev.txt"):
        raw = (ROOT / name).read_bytes()
        assert not raw.startswith(b"\xef\xbb\xbf") and b"\xef\xbb\xbf" not in raw
        for line in raw.decode("utf-8").splitlines():
            if line.strip() and not line.startswith("-r "):
                assert re.fullmatch(r"[A-Za-z0-9_.\-\[\]]+==[^\s=]+", line.strip()), line


def test_no_secrets_in_source():
    pattern = re.compile(r"(api[_-]?key|secret|password|token)\s*=\s*['\"][^'\"]{8,}", re.I)
    for path in [*CODE_FILES, *(ROOT / "config").glob("*.py")]:
        assert not pattern.search(path.read_text(encoding="utf-8")), path.name


HOSTILE = [
    "a\x00b\x07 hello ‮ evil",
    "<script>alert(1)</script> win a free prize",
    "'; DROP TABLE users;-- ",
    "{{7*7}} ${jndi:ldap://x}",
    "\U0001f600" * 400,
    "x" * 999,
]


@pytest.fixture(autouse=True)
def _fresh_cache():
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


@pytest.mark.parametrize("text", HOSTILE)
def test_hostile_text_never_crashes_a_detector(fake_models_dir, monkeypatch, text):
    from src.detection.email_detector import load_email_detector
    from src.detection.spam_detector import load_sms_detector
    from src.detection.toxicity_detector import load_toxicity_detector

    monkeypatch.setattr(settings, "MODELS_DIR", fake_models_dir)
    for call in (
        lambda: load_sms_detector().predict(text),
        lambda: load_toxicity_detector().predict(text),
        lambda: load_email_detector().predict("subject", text),
    ):
        try:
            call()
        except DetectorError:
            pass  # a clean, user-readable rejection is fine; any other exception is a bug


@pytest.mark.parametrize(
    "url",
    ["javascript:alert(1)", "file:///etc/passwd", "http://[::1", "http://a\x00b.com", "a" * 5000],
)
def test_dangerous_urls_are_rejected_cleanly(fake_models_dir, monkeypatch, url):
    from src.detection.phishing_detector import load_phishing_detector

    monkeypatch.setattr(settings, "MODELS_DIR", fake_models_dir)
    with pytest.raises(DetectorError):
        load_phishing_detector().predict(url)


def test_script_text_is_shown_as_text_and_url_is_not_a_link(fake_models_dir, monkeypatch):
    monkeypatch.setattr(settings, "MODELS_DIR", fake_models_dir)
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    app.text_input(key="url_input").set_value("http://paypal.com.evil-login.example/x").run()
    app.button(key="url_analyze").click().run()
    assert not app.exception
    assert any("evil-login.example" in c.value for c in app.code)  # plain code block, no link
    assert not any("](http" in m.value for m in app.markdown)


def test_user_input_is_not_logged(fake_models_dir, monkeypatch, caplog):
    monkeypatch.setattr(settings, "MODELS_DIR", fake_models_dir)
    secret = "my-very-private-message-12345"
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    app.text_area(key="sms_message").set_value(secret).run()
    app.button(key="sms_analyze").click().run()
    assert secret not in caplog.text
