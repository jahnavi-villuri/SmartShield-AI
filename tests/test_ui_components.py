"""UI building blocks: untrusted text must be escaped; scores are clamped; SVG art is sealed."""

import base64
import re
from html.parser import HTMLParser

import pytest

from src.ui import components

HOSTILE = "<script>alert(1)</script><img src=x onerror=alert(2)>\"'&"


class _Collector(HTMLParser):
    """Records every real tag and attribute name the browser would see."""

    def __init__(self):
        super().__init__()
        self.tags, self.attrs = set(), set()

    def handle_starttag(self, tag, attrs):
        self.tags.add(tag)
        self.attrs |= {name for name, _ in attrs}


def _no_live_markup(html: str) -> None:
    parsed = _Collector()
    parsed.feed(html)
    assert "script" not in parsed.tags
    assert not [a for a in parsed.attrs if a.startswith("on")]
    assert "javascript:" not in html.lower()


def test_hero_and_section_escape_text():
    html = components.hero(HOSTILE, HOSTILE, HOSTILE) + components.section(HOSTILE)
    _no_live_markup(html)
    assert "&lt;script&gt;" in html


def test_result_card_escapes_heading_note_and_label():
    html = components.result_card("bad", HOSTILE, HOSTILE, 0.9, 0.5, HOSTILE)
    _no_live_markup(html)
    svg = base64.b64decode(re.search(r"base64,([^\"]+)", html).group(1)).decode()
    assert "<script" not in svg.lower()  # the label inside the sealed SVG is escaped too


def test_chips_cards_kpis_and_pipeline_escape_text():
    html = (
        components.chips(HOSTILE, [HOSTILE, "ok"], hot=True)
        + components.feature_cards([("sms", HOSTILE, HOSTILE)])
        + components.kpis([(HOSTILE, HOSTILE)])
        + components.pipeline([(HOSTILE, HOSTILE)])
        + components.score_bar(HOSTILE, 0.5, 0.3, True)
    )
    _no_live_markup(html)


@pytest.mark.parametrize("score", [-5, 0, 0.5, 1, 7, float("inf")])
def test_scores_are_clamped_into_the_ring_and_bar(score):
    ring = components.ring_svg(score, "#fff", 0.5, "score")
    offsets = [float(x) for x in re.findall(r"stroke-dashoffset:([0-9.]+)", ring)]
    assert offsets and all(0 <= o <= 326.7 for o in offsets)
    bar = components.score_bar("x", score, score, False)
    widths = [float(x) for x in re.findall(r"width:([0-9.]+)%", bar)]
    assert all(0 <= w <= 100 for w in widths)


def test_unknown_level_falls_back_to_a_safe_class():
    assert 'class="ss-result warn"' in components.result_card('evil"><b>', "h", "n", 0.5)


def test_images_are_sealed_data_uris_without_scripts():
    for html in (components.shield_art(), components.icon("url"), components.icon("nope")):
        assert html.startswith("<img ") and "data:image/svg+xml;base64," in html
        svg = base64.b64decode(re.search(r"base64,([^\"]+)", html).group(1)).decode()
        assert "<script" not in svg.lower() and "http" not in svg.replace("http://www.w3.org", "")


def test_empty_chip_list_renders_nothing():
    assert components.chips("label", []) == ""


def test_theme_is_crimson_noir_and_no_legacy_blue_remains():
    from pathlib import Path

    from src.ui import theme

    assert theme.DEFAULT_THEME == "noir" and list(theme.PALETTES) == ["noir"]
    root = Path(theme.__file__).parent
    legacy = ("#22d3ee", "#8b5cf6", "#070b17", "#0e1530", "34,211,238", "139,92,246")
    for path in root.glob("*.py"):
        text = path.read_text(encoding="utf-8").lower()
        assert not [c for c in legacy if c in text], (path.name, legacy)


def test_every_illustration_uses_the_active_palette():
    from src.ui import theme

    svg = base64.b64decode(re.search(r"base64,([^\"]+)", components.shield_art()).group(1)).decode()
    assert theme.active_palette()["a1"] in svg and "@A1@" not in svg
