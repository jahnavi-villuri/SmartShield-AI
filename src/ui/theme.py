"""Visual theme: palettes (now only Crimson Noir; add another entry to PALETTES to offer more) plus one CSS block injected once per page run.

Only static text written in this file is ever sent as HTML. Anything that comes from a user or a
model file goes through `src/ui/components.py`, which escapes it. No external fonts, images or
scripts are loaded, so the page makes no third-party requests.

A palette has three accent colours (a1 main, a2 second, a3 third), page colours and a few helpers.
Status colours (green / amber / red for ok / warn / bad) are the same in every theme on purpose:
meaning must not depend on the chosen colour scheme.
"""

import streamlit as st

STATUS = {"ok": "#34d399", "warn": "#fbbf24", "bad": "#fb7185"}
DEFAULT_THEME = "noir"

PALETTES = {
    "noir": {
        "label": "Crimson Noir",
        "bg": "#09090b",
        "surface": "#18181b",
        "text": "#fafafa",
        "muted": "#a1a1aa",
        "a1": "#f43f5e",
        "a2": "#d4d4d8",
        "a3": "#fda4af",
        "on_accent": "#ffffff",
        "chip_text": "#f4d6dc",
    },
}


def _rgb(hex_color: str) -> str:
    h = hex_color.lstrip("#")
    return ",".join(str(int(h[i : i + 2], 16)) for i in (0, 2, 4))


def active_name() -> str:
    """Theme chosen in this session, else the `?theme=` link parameter, else the default."""
    try:
        name = st.session_state.get("theme") or st.query_params.get("theme")
    except Exception:  # no Streamlit script context (unit tests, scripts)
        name = None
    return name if name in PALETTES else DEFAULT_THEME


def active_palette() -> dict:
    return PALETTES[active_name()]


def _root(p: dict) -> str:
    return (
        ":root {"
        f" --bg:{p['bg']}; --surface:{p['surface']}; --text:{p['text']}; --muted:{p['muted']};"
        f" --a1:{p['a1']}; --a2:{p['a2']}; --a3:{p['a3']};"
        f" --a1-rgb:{_rgb(p['a1'])}; --a2-rgb:{_rgb(p['a2'])}; --a3-rgb:{_rgb(p['a3'])};"
        f" --on-accent:{p['on_accent']}; --chip-text:{p['chip_text']};"
        " --card:rgba(255,255,255,.045); --line:rgba(255,255,255,.10);"
        f" --green:{STATUS['ok']}; --amber:{STATUS['warn']}; --red:{STATUS['bad']}; }}"
    )


STATIC_CSS = """
<style>
@keyframes fadeUp { from {opacity:0; transform:translateY(14px);} to {opacity:1; transform:none;} }
@keyframes floaty { 0%,100% {transform:translateY(0);} 50% {transform:translateY(-10px);} }
@keyframes spin { to {transform:rotate(360deg);} }
@keyframes spinRev { to {transform:rotate(-360deg);} }
@keyframes pulseGlow { 0%,100% {opacity:.55; transform:scale(1);} 50% {opacity:1; transform:scale(1.06);} }
@keyframes gradientShift { 0% {background-position:0% 50%;} 50% {background-position:100% 50%;} 100% {background-position:0% 50%;} }
@keyframes ringFill { from {stroke-dashoffset:326.7;} }
@keyframes dash { to {stroke-dashoffset:-60;} }
@keyframes scan { 0% {top:0; opacity:0;} 15% {opacity:1;} 85% {opacity:1;} 100% {top:100%; opacity:0;} }
@keyframes blink { 0%,100% {opacity:1;} 50% {opacity:.25;} }

.stApp { background:
  radial-gradient(1200px 600px at 8% -10%, rgba(var(--a2-rgb),.22), transparent 60%),
  radial-gradient(900px 500px at 105% 5%, rgba(var(--a1-rgb),.16), transparent 55%),
  radial-gradient(800px 500px at 50% 115%, rgba(var(--a3-rgb),.12), transparent 60%),
  var(--bg); }
.block-container { padding-top: 1.6rem; max-width: 1180px; }
header[data-testid="stHeader"] { background: transparent; }
h1, h2, h3, h4 { letter-spacing: -0.01em; }
.stApp h1 { font-weight:800; background: linear-gradient(90deg, var(--text), var(--a1) 55%, var(--a3)); -webkit-background-clip:text; background-clip:text; color:transparent; }

/* ---------- hero ---------- */
.ss-hero { position:relative; display:grid; grid-template-columns: 1.5fr 1fr; gap:1.5rem; align-items:center;
  padding:2rem 2.2rem; border-radius:26px; overflow:hidden; animation: fadeUp .7s ease both;
  background: linear-gradient(135deg, rgba(var(--a2-rgb),.20), rgba(var(--a1-rgb),.10) 60%, rgba(var(--a3-rgb),.12));
  border:1px solid var(--line); box-shadow: 0 20px 60px rgba(0,0,0,.35); }
.ss-hero::before { content:""; position:absolute; inset:-2px; z-index:0; border-radius:28px; opacity:.5;
  background: linear-gradient(120deg, var(--a1), var(--a2), var(--a3), var(--a1)); background-size:300% 300%;
  animation: gradientShift 9s ease infinite; -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
  -webkit-mask-composite: xor; mask-composite: exclude; padding:1.5px; pointer-events:none; }
.ss-hero > * { position:relative; z-index:1; }
.ss-eyebrow { display:inline-flex; gap:.5rem; align-items:center; font-size:.78rem; letter-spacing:.14em; text-transform:uppercase;
  color: var(--a1); background: rgba(var(--a1-rgb),.10); border:1px solid rgba(var(--a1-rgb),.35); padding:.25rem .7rem; border-radius:999px; }
.ss-dot { width:8px; height:8px; border-radius:50%; background:var(--green); box-shadow:0 0 10px var(--green); animation: blink 1.8s infinite; }
.ss-title { font-size: clamp(2.2rem, 5vw, 3.6rem); line-height:1.05; font-weight:800; margin:.8rem 0 .5rem;
  background: linear-gradient(90deg, var(--text), var(--a1) 45%, var(--a3)); background-size:200% 100%;
  -webkit-background-clip:text; background-clip:text; color:transparent; animation: gradientShift 8s ease infinite; }
.ss-tag { color: var(--muted); font-size:1.08rem; max-width: 34rem; }
.ss-hero-art { display:flex; justify-content:center; animation: floaty 6s ease-in-out infinite; }
.ss-orbit { transform-origin:100px 100px; }
.ss-orbit.a { animation: spin 14s linear infinite; }
.ss-orbit.b { animation: spinRev 22s linear infinite; }
.ss-glow { transform-origin:100px 100px; animation: pulseGlow 3.2s ease-in-out infinite; }

/* ---------- cards ---------- */
.ss-grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap:1rem; margin:1.1rem 0; }
.ss-card { position:relative; padding:1.1rem 1.2rem; border-radius:20px; background: var(--card); border:1px solid var(--line);
  backdrop-filter: blur(8px); transition: transform .25s ease, border-color .25s ease, box-shadow .25s ease; animation: fadeUp .6s ease both; }
.ss-card:hover { transform: translateY(-5px); border-color: rgba(var(--a1-rgb),.55); box-shadow: 0 14px 40px rgba(var(--a1-rgb),.12); }
.ss-card h4 { margin:.5rem 0 .25rem; font-size:1.05rem; color: var(--text); }
.ss-card img.ss-icon { display:block; }
.ss-card p { color: var(--muted); font-size:.92rem; margin:0; }
.ss-card .ss-icon { width:54px; height:54px; }
.ss-grid .ss-card:nth-child(2) { animation-delay:.08s; } .ss-grid .ss-card:nth-child(3) { animation-delay:.16s; } .ss-grid .ss-card:nth-child(4) { animation-delay:.24s; }

.ss-kpis { display:grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap:.9rem; margin:1rem 0; }
.ss-kpi { padding:1rem 1.1rem; border-radius:18px; background: var(--card); border:1px solid var(--line); animation: fadeUp .6s ease both; }
.ss-kpi .v { font-size:1.9rem; font-weight:800; background: linear-gradient(90deg, var(--a1), var(--a2)); -webkit-background-clip:text; background-clip:text; color:transparent; }
.ss-kpi .l { color: var(--muted); font-size:.82rem; text-transform:uppercase; letter-spacing:.08em; }

.ss-section { display:flex; align-items:center; gap:.7rem; margin:1.6rem 0 .4rem; font-size:1.35rem; font-weight:700; }
.ss-section::after { content:""; flex:1; height:1px; background: linear-gradient(90deg, var(--line), transparent); }

/* ---------- pipeline diagram ---------- */
.ss-flow { display:flex; flex-wrap:wrap; gap:.5rem; align-items:stretch; margin:.6rem 0 1rem; }
.ss-step { flex:1 1 130px; padding:.8rem .9rem; border-radius:16px; background: var(--card); border:1px solid var(--line); text-align:center; position:relative; animation: fadeUp .6s ease both; }
.ss-step b { display:block; color: var(--a1); font-size:.78rem; letter-spacing:.1em; text-transform:uppercase; }
.ss-step span { color: var(--muted); font-size:.85rem; }
.ss-step:not(:last-child)::after { content:"\\2192"; position:absolute; right:-.62rem; top:50%; transform:translateY(-50%); color: var(--a2); font-weight:800; z-index:2; }

/* ---------- results ---------- */
.ss-result { position:relative; display:grid; grid-template-columns: 150px 1fr; gap:1.2rem; align-items:center; padding:1.1rem 1.3rem;
  border-radius:22px; border:1px solid var(--line); background: var(--card); overflow:hidden; animation: fadeUp .5s ease both; margin:.6rem 0 1rem; }
.ss-result::after { content:""; position:absolute; left:0; right:0; height:2px; background: linear-gradient(90deg, transparent, var(--a1), transparent);
  animation: scan 1.6s ease-out 1 both; }
.ss-result.bad { border-color: rgba(251,113,133,.5); box-shadow: 0 0 40px rgba(251,113,133,.12); }
.ss-result.ok { border-color: rgba(52,211,153,.45); box-shadow: 0 0 40px rgba(52,211,153,.10); }
.ss-result.warn { border-color: rgba(251,191,36,.45); }
.ss-result .k { color: var(--muted); font-size:.8rem; text-transform:uppercase; letter-spacing:.1em; }
.ss-result .big { font-size:1.6rem; font-weight:800; margin:.1rem 0; }
.ss-result .note { color: var(--muted); font-size:.9rem; }
.ss-result img.ss-ring { width:140px; height:140px; justify-self:center; }
.ss-ring .bg { fill:none; stroke: rgba(255,255,255,.08); stroke-width:10; }
.ss-ring .fg { fill:none; stroke-width:10; stroke-linecap:round; stroke-dasharray:326.7; transform: rotate(-90deg); transform-origin: 60px 60px;
  animation: ringFill 1.2s cubic-bezier(.2,.8,.2,1) both; }
.ss-ring text { fill: var(--text); font-weight:800; font-size:24px; text-anchor:middle; }
.ss-ring .sub { fill: var(--muted); font-size:8px; font-weight:500; letter-spacing:.1em; }

.ss-chips { display:flex; flex-wrap:wrap; gap:.4rem; margin:.4rem 0 .8rem; }
.ss-chip { padding:.22rem .65rem; border-radius:999px; font-size:.85rem; border:1px solid var(--line); background: rgba(var(--a2-rgb),.14); color:var(--chip-text); animation: fadeUp .5s ease both; }
.ss-chip.hot { background: rgba(251,113,133,.14); border-color: rgba(251,113,133,.45); color:#ffd0d8; }
.ss-chiplabel { color: var(--muted); font-size:.85rem; margin-top:.4rem; }

.ss-bar { margin:.35rem 0; animation: fadeUp .5s ease both; }
.ss-bar .row { display:flex; justify-content:space-between; font-size:.88rem; }
.ss-bar .row .n { font-weight:700; } .ss-bar .row .s { color: var(--muted); }
.ss-bar .track { position:relative; height:10px; border-radius:999px; background: rgba(255,255,255,.08); overflow:visible; }
.ss-bar .fill { height:100%; border-radius:999px; background: linear-gradient(90deg, var(--a1), var(--a2)); transform-origin:left; animation: barGrow 1s ease both; }
.ss-bar.hot .fill { background: linear-gradient(90deg, var(--amber), var(--red)); }
.ss-bar .tick { position:absolute; top:-4px; width:2px; height:18px; background: var(--text); opacity:.7; border-radius:2px; }
@keyframes barGrow { from {transform:scaleX(0);} }

/* ---------- native Streamlit widgets, restyled ---------- */
.stTabs [data-baseweb="tab-list"] { gap:.25rem; background: var(--card); border:1px solid var(--line); padding:.3rem; border-radius:16px; overflow-x:auto; }
.stTabs [data-baseweb="tab"] { border-radius:12px; padding:.45rem .95rem; height:auto; color: var(--muted); transition: all .2s ease; }
.stTabs [data-baseweb="tab"]:hover { color: var(--text); background: rgba(255,255,255,.06); }
.stTabs [aria-selected="true"] { color:#fff !important; background: linear-gradient(135deg, rgba(var(--a2-rgb),.55), rgba(var(--a1-rgb),.35)); }
.stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] { display:none; }
.stButton > button { border-radius:14px; border:1px solid var(--line); transition: transform .15s ease, box-shadow .2s ease; }
.stButton > button:hover { transform: translateY(-2px); border-color: var(--a1); }
.stButton > button[kind="primary"] { background: linear-gradient(135deg, var(--a1), color-mix(in srgb, var(--a1) 55%, black)); border:0; font-weight:700; color:var(--on-accent);
  box-shadow: 0 8px 24px rgba(var(--a1-rgb),.25); }
.stButton > button[kind="primary"]:hover { box-shadow: 0 12px 32px rgba(var(--a2-rgb),.45); }
div[data-baseweb="textarea"], div[data-baseweb="input"], div[data-baseweb="base-input"] { background: var(--surface) !important; border-radius:14px !important; }
.stTextArea textarea, .stTextInput input { border-radius:14px !important; background: var(--surface) !important; border:1px solid var(--line) !important; }
.stTextArea textarea:focus, .stTextInput input:focus { border-color: var(--a1) !important; box-shadow: 0 0 0 3px rgba(var(--a1-rgb),.18) !important; }
/* Streamlit's info boxes are blue by default; make them follow the theme (neutral silver) */
div[data-testid="stAlertContainer"]:has([data-testid="stAlertContentInfo"]) { background: rgba(var(--a2-rgb), .08) !important; border-left: 3px solid var(--a2); }
[data-testid="stAlertContentInfo"], [data-testid="stAlertContentInfo"] p, [data-testid="stAlertContentInfo"] a { color: var(--text) !important; }
div[data-testid="stAlert"] { border-radius:16px; border:1px solid var(--line); animation: fadeUp .45s ease both; }
div[data-testid="stExpander"] { border-radius:16px; border:1px solid var(--line); background: var(--card); }
div[data-testid="stMetric"] { background: var(--card); border:1px solid var(--line); border-radius:16px; padding:.7rem 1rem; }
div[data-testid="stVerticalBlockBorderWrapper"] { border-radius:18px; }

div[data-testid="stProgress"] div[role="progressbar"] > div > div { background: linear-gradient(90deg, var(--a1), var(--a2)) !important; }

@media (max-width: 760px) {
  .ss-hero { grid-template-columns: 1fr; padding:1.4rem; } .ss-result { grid-template-columns: 1fr; justify-items:center; text-align:center; }
  .ss-step:not(:last-child)::after { display:none; }
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation: none !important; transition: none !important; }
}
</style>
"""


def build_css(palette: dict) -> str:
    return (
        "<style>\n"
        + _root(palette)
        + STATIC_CSS.replace("<style>", "").replace("</style>", "")
        + "</style>"
    )


CSS = build_css(PALETTES[DEFAULT_THEME])


def inject_theme() -> None:
    """Send the stylesheet for the active theme to the page (call once per script run)."""
    st.html(build_css(active_palette()))
