"""Small HTML/SVG building blocks for the app. Every function returns a string.

SECURITY RULE: any text that did not come from this file (user input, model metadata, detector
output) is passed through `html.escape` before it is placed in markup. Tests in
tests/test_ui_components.py check this. No external resources are referenced.

Why SVG goes in an <img> data URI: Streamlit's HTML sanitiser removes inline <svg>. An SVG shown
through <img> is a sealed document: it keeps its own styles and animations but cannot run scripts
or touch the page, which is also the safer way to show vector art.
"""

import base64
from html import escape

from src.ui import theme

_FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"

_DEFS = (
    '<defs><linearGradient id="gcy" x1="0" y1="0" x2="1" y2="1">'
    '<stop offset="0" stop-color="@A1@"/><stop offset="1" stop-color="@A2@"/></linearGradient>'
    '<linearGradient id="gpk" x1="0" y1="0" x2="1" y2="1">'
    '<stop offset="0" stop-color="@A3@"/><stop offset="1" stop-color="@A2@"/></linearGradient></defs>'
)

_ICONS = {
    "sms": (
        '<path d="M10 14h44a6 6 0 0 1 6 6v22a6 6 0 0 1-6 6H30l-12 10V48h-8a6 6 0 0 1-6-6V20a6 6 0 0 1 6-6z" '
        'fill="none" stroke="url(#gcy)" stroke-width="3.5" stroke-linejoin="round"/>'
        '<circle cx="22" cy="31" r="3" fill="@A1@"><animate attributeName="opacity" values="1;.2;1" dur="1.6s" repeatCount="indefinite"/></circle>'
        '<circle cx="32" cy="31" r="3" fill="@A2@"><animate attributeName="opacity" values="1;.2;1" dur="1.6s" begin=".2s" repeatCount="indefinite"/></circle>'
        '<circle cx="42" cy="31" r="3" fill="@A3@"><animate attributeName="opacity" values="1;.2;1" dur="1.6s" begin=".4s" repeatCount="indefinite"/></circle>'
    ),
    "email": (
        '<rect x="6" y="14" width="52" height="38" rx="7" fill="none" stroke="url(#gcy)" stroke-width="3.5"/>'
        '<path d="M8 18l24 18 24-18" fill="none" stroke="url(#gpk)" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round"/>'
        '<circle cx="52" cy="14" r="6" fill="#fb7185"><animate attributeName="r" values="5;7;5" dur="1.8s" repeatCount="indefinite"/></circle>'
    ),
    "comment": (
        '<path d="M12 10h40a6 6 0 0 1 6 6v24a6 6 0 0 1-6 6H34l-12 10v-10h-10a6 6 0 0 1-6-6V16a6 6 0 0 1 6-6z" '
        'fill="none" stroke="url(#gpk)" stroke-width="3.5" stroke-linejoin="round"/>'
        '<path d="M22 31c3 5 17 5 20 0" fill="none" stroke="@A1@" stroke-width="3.5" stroke-linecap="round"/>'
        '<circle cx="24" cy="22" r="2.6" fill="@A3@"/><circle cx="40" cy="22" r="2.6" fill="@A3@"/>'
    ),
    "url": (
        '<circle cx="32" cy="32" r="22" fill="none" stroke="url(#gcy)" stroke-width="3.5"/>'
        '<ellipse cx="32" cy="32" rx="9" ry="22" fill="none" stroke="@A2@" stroke-width="3">'
        '<animate attributeName="rx" values="9;3;9" dur="3.4s" repeatCount="indefinite"/></ellipse>'
        '<path d="M10 32h44M14 20h36M14 44h36" stroke="@A1@" stroke-width="2.5" fill="none" stroke-linecap="round"/>'
    ),
    "dashboard": (
        '<rect x="8" y="34" width="10" height="20" rx="3" fill="url(#gcy)"/>'
        '<rect x="27" y="20" width="10" height="34" rx="3" fill="url(#gpk)"/>'
        '<rect x="46" y="10" width="10" height="44" rx="3" fill="url(#gcy)"/>'
    ),
}


def _paint(svg: str) -> str:
    """Fill the colour tokens of an illustration with the active palette."""
    p = theme.active_palette()
    for token, key in (
        ("@A1@", "a1"),
        ("@A2@", "a2"),
        ("@A3@", "a3"),
        ("@TEXT@", "text"),
        ("@MUTED@", "muted"),
        ("@SURFACE@", "surface"),
    ):
        svg = svg.replace(token, p[key])
    return svg


def svg_image(svg: str, alt: str, css_class: str = "", width: int | None = None) -> str:
    """Embed a standalone SVG document as an <img> (sanitiser-safe, self-animating)."""
    svg = _paint(svg)
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    size = f' width="{width}"' if width else ""
    return (
        f'<img class="{escape(css_class)}" alt="{escape(alt)}"{size} '
        f'src="data:image/svg+xml;base64,{encoded}"/>'
    )


def icon(kind: str) -> str:
    """A 64x64 line-art icon (kind: sms, email, comment, url, dashboard)."""
    body = _ICONS.get(kind, _ICONS["sms"])
    svg = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">{_DEFS}{body}</svg>'
    return svg_image(svg, f"{kind} icon", "ss-icon", 54)


def shield_art() -> str:
    """The hero illustration: a shield with orbiting rings and a glowing core."""
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 200">'
        "<style>"
        "@keyframes spin{to{transform:rotate(360deg)}}@keyframes spinr{to{transform:rotate(-360deg)}}"
        "@keyframes glow{0%,100%{opacity:.10}50%{opacity:.26}}"
        "@keyframes draw{from{stroke-dashoffset:90}to{stroke-dashoffset:0}}"
        ".a{transform-origin:100px 100px;animation:spin 14s linear infinite}"
        ".b{transform-origin:100px 100px;animation:spinr 22s linear infinite}"
        ".g{animation:glow 3.2s ease-in-out infinite}"
        ".t{stroke-dasharray:90;animation:draw 1.4s ease-out .3s both}"
        "@media (prefers-reduced-motion:reduce){*{animation:none!important}}"
        "</style>"
        f"{_DEFS}"
        '<circle class="g" cx="100" cy="100" r="82" fill="url(#gcy)"/>'
        '<g class="a"><ellipse cx="100" cy="100" rx="92" ry="34" fill="none" stroke="@A1@" '
        'stroke-opacity=".6" stroke-dasharray="3 7" stroke-width="1.6"/>'
        '<circle cx="192" cy="100" r="5" fill="@A1@"/></g>'
        '<g class="b"><ellipse cx="100" cy="100" rx="34" ry="92" fill="none" stroke="@A3@" '
        'stroke-opacity=".6" stroke-dasharray="3 7" stroke-width="1.6"/>'
        '<circle cx="100" cy="8" r="5" fill="@A3@"/></g>'
        '<path d="M100 30l52 18v40c0 34-22 56-52 72-30-16-52-38-52-72V48z" fill="@SURFACE@" '
        'stroke="url(#gcy)" stroke-width="4" stroke-linejoin="round"/>'
        '<path d="M100 44l38 13v31c0 25-16 42-38 55-22-13-38-30-38-55V57z" fill="url(#gcy)" opacity=".2"/>'
        '<path class="t" d="M78 100l16 16 30-34" fill="none" stroke="#34d399" stroke-width="9" '
        'stroke-linecap="round" stroke-linejoin="round"/></svg>'
    )
    return svg_image(svg, "Glowing shield with orbiting rings", "ss-art", 250)


def hero(title: str, tagline: str, status: str) -> str:
    return (
        '<div class="ss-hero"><div>'
        f'<span class="ss-eyebrow"><span class="ss-dot"></span>{escape(status)}</span>'
        f'<div class="ss-title">{escape(title)}</div>'
        f'<div class="ss-tag">{escape(tagline)}</div></div>'
        f'<div class="ss-hero-art">{shield_art()}</div></div>'
    )


def section(title: str) -> str:
    return f'<div class="ss-section">{escape(title)}</div>'


def feature_cards(cards: list[tuple[str, str, str]]) -> str:
    """cards: (icon kind, title, text)."""
    items = "".join(
        f'<div class="ss-card">{icon(kind)}<h4>{escape(title)}</h4><p>{escape(text)}</p></div>'
        for kind, title, text in cards
    )
    return f'<div class="ss-grid">{items}</div>'


def kpis(items: list[tuple[str, str]]) -> str:
    """items: (value, label)."""
    cells = "".join(
        f'<div class="ss-kpi"><div class="v">{escape(value)}</div><div class="l">{escape(label)}</div></div>'
        for value, label in items
    )
    return f'<div class="ss-kpis">{cells}</div>'


def pipeline(steps: list[tuple[str, str]]) -> str:
    """steps: (name, caption)."""
    cells = "".join(
        f'<div class="ss-step"><b>{escape(name)}</b><span>{escape(text)}</span></div>'
        for name, text in steps
    )
    return f'<div class="ss-flow">{cells}</div>'


def _clamp01(value: float) -> float:
    return min(max(float(value), 0.0), 1.0)


def ring_svg(score: float, color: str, threshold: float | None, label: str) -> str:
    """The animated score ring as a standalone SVG document (numbers only, no free text)."""
    score = _clamp01(score)
    offset = 326.7 * (1 - score)
    tick = ""
    if threshold is not None:
        angle = _clamp01(threshold) * 360 - 90
        tick = (
            f'<g transform="rotate({angle:.1f} 60 60)"><line x1="106" y1="60" x2="119" y2="60" '
            'stroke="@TEXT@" stroke-width="3" stroke-linecap="round"/></g>'
        )
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 120 120">'
        f"<style>@keyframes f{{from{{stroke-dashoffset:326.7}}}}"
        f".fg{{stroke-dasharray:326.7;stroke-dashoffset:{offset:.1f};animation:f 1.3s cubic-bezier(.2,.8,.2,1) both}}"
        "@media (prefers-reduced-motion:reduce){*{animation:none!important}}</style>"
        '<circle cx="60" cy="60" r="52" fill="none" stroke="rgba(255,255,255,.10)" stroke-width="10"/>'
        f'<g transform="rotate(-90 60 60)"><circle class="fg" cx="60" cy="60" r="52" fill="none" '
        f'stroke="{color}" stroke-width="10" stroke-linecap="round"/></g>{tick}'
        f'<text x="60" y="65" text-anchor="middle" font-family="{_FONT}" font-weight="800" '
        f'font-size="25" fill="@TEXT@">{score:.2f}</text>'
        f'<text x="60" y="81" text-anchor="middle" font-family="{_FONT}" font-size="7" '
        f'letter-spacing="1.2" fill="@MUTED@">{escape(label.upper())}</text></svg>'
    )


def result_card(
    level: str,
    heading: str,
    note: str,
    score: float,
    threshold: float | None = None,
    score_label: str = "score",
) -> str:
    """Animated score ring plus verdict heading. level: ok | warn | bad."""
    level = level if level in {"ok", "warn", "bad"} else "warn"
    color = theme.STATUS[level]
    ring = svg_image(
        ring_svg(score, color, threshold, score_label),
        f"{score_label} {_clamp01(score):.2f}",
        "ss-ring",
        140,
    )
    return (
        f'<div class="ss-result {level}">{ring}<div><div class="k">Result</div>'
        f'<div class="big" style="color:{color}">{escape(heading)}</div>'
        f'<div class="note">{escape(note)}</div></div></div>'
    )


def chips(label: str, items: list[str], hot: bool = False) -> str:
    if not items:
        return ""
    cls = "ss-chip hot" if hot else "ss-chip"
    body = "".join(f'<span class="{cls}">{escape(str(item))}</span>' for item in items)
    return f'<div class="ss-chiplabel">{escape(label)}</div><div class="ss-chips">{body}</div>'


def score_bar(name: str, score: float, threshold: float, flagged: bool) -> str:
    """One horizontal bar with a tick at the decision threshold."""
    score, threshold = _clamp01(score), _clamp01(threshold)
    hot = " hot" if flagged else ""
    mark = " • flagged" if flagged else ""
    return (
        f'<div class="ss-bar{hot}"><div class="row"><span class="n">{escape(name)}{mark}</span>'
        f'<span class="s">score {score:.2f} (flag at {threshold:.2f})</span></div>'
        f'<div class="track"><div class="fill" style="width:{score * 100:.1f}%"></div>'
        f'<div class="tick" style="left:{threshold * 100:.1f}%"></div></div></div>'
    )
