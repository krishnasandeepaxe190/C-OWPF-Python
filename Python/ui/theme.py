"""Shared look-and-feel for the C-OWPF app: appearance modes, CSS, headers, cards."""
from __future__ import annotations

# Palette: water = teal/blue, power = amber, coupled = violet.
WATER = "#2274A5"
POWER = "#E8A13A"
COUPLED = "#7B5EA7"
GOOD = "#2E8B57"
BAD = "#C6413A"

APPEARANCE_MODES = ["System", "Light", "Dark"]

# Light / dark surface palettes used by the forced modes.
_LIGHT = dict(bg="#f5f7fb", surface="#ffffff", text="#1a1d24", dim="#5a6070",
              header="rgba(245,247,251,.9)", inputbg="#ffffff", border="rgba(120,120,140,.28)")
_DARK = dict(bg="#0e1117", surface="#161a23", text="#e7e8ee", dim="#aab0be",
             header="rgba(14,17,23,.9)", inputbg="#1c2029", border="rgba(140,140,160,.28)")


def _surface_css(p: dict) -> str:
    """CSS rules that force one palette on the app surfaces (no media query)."""
    return f"""
  .stApp, [data-testid="stAppViewContainer"], [data-testid="stMain"] {{
    background: {p['bg']} !important; color: {p['text']} !important; }}
  [data-testid="stHeader"] {{ background: {p['header']} !important; }}
  [data-testid="stSidebar"] {{ background: {p['surface']} !important; }}
  .stMarkdown, .stMarkdown p, .stMarkdown li, label, .stMetric, .stCaption,
  .stRadio label, .stSelectbox label, .stSlider label, .stNumberInput label,
  [data-testid="stWidgetLabel"] p, .stCheckbox label {{ color: {p['text']} !important; }}
  .small-note, [data-testid="stCaptionContainer"] {{ color: {p['dim']} !important; }}
  .stTabs [data-baseweb="tab"] p {{ color: {p['dim']} !important; }}
  .stTabs [aria-selected="true"] p {{ color: {COUPLED} !important; }}
  [data-baseweb="select"] > div, .stNumberInput input, .stTextInput input,
  [data-baseweb="input"] input {{ background: {p['inputbg']} !important; color: {p['text']} !important; }}
  [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {{ color: {p['text']} !important; }}
"""


# Structural CSS shared by every mode (tab contrast, hero, cards, pills).
_BASE_CSS = f"""
<style>
.block-container {{ padding-top: 1.1rem; max-width: 1280px; }}
h1, h2, h3 {{ letter-spacing: -0.01em; }}

/* ---- Appearance switcher: make it obvious ---- */
.appearance-strip + div [role="radiogroup"],
div:has(> .appearance-strip) [role="radiogroup"] {{ gap: .1rem; }}
[data-testid="stRadio"] > label p {{ font-weight: 700 !important; font-size: .92rem !important; }}
[data-testid="stRadio"] [role="radiogroup"] label {{
  border: 1px solid rgba(123,94,167,.35); border-radius: 8px;
  padding: .12rem .5rem !important; margin-right: .25rem;
  background: rgba(123,94,167,.08); }}
[data-testid="stRadio"] [role="radiogroup"] label:hover {{ background: rgba(123,94,167,.18); }}

/* ---- TABS: high-contrast, pill-style ---- */
.stTabs [data-baseweb="tab-list"] {{
  gap: .35rem; border-bottom: 2px solid rgba(130,130,150,.28); padding-bottom: 2px; }}
.stTabs [data-baseweb="tab"] {{
  height: auto; padding: .5rem 1rem; border-radius: 10px 10px 0 0;
  background: rgba(130,130,150,.08); }}
.stTabs [data-baseweb="tab"] p {{ font-size: 1.03rem !important; font-weight: 700 !important; margin: 0; }}
.stTabs [aria-selected="true"] {{
  background: linear-gradient(180deg, rgba(123,94,167,.22), rgba(123,94,167,.05));
  border-bottom: 3px solid {COUPLED}; }}
.stTabs [data-baseweb="tab-highlight"] {{ background-color: {COUPLED} !important; height: 3px; }}

.cowpf-hero {{
  background: linear-gradient(135deg, rgba(34,116,165,.16), rgba(123,94,167,.16));
  border: 1px solid rgba(120,120,140,.24); border-radius: 16px;
  padding: 1.4rem 1.7rem; margin-bottom: 1rem; }}
.cowpf-hero h1 {{ margin: 0 0 .3rem 0; font-size: 2.05rem; }}
.cowpf-hero p {{ margin: .15rem 0; opacity: .9; font-size: 1.02rem; }}
.pill {{
  display:inline-block; padding: .13rem .6rem; border-radius: 999px;
  font-size:.76rem; font-weight:600; margin-right:.35rem; border:1px solid transparent; }}
.pill-water   {{ background: rgba(34,116,165,.18); color: {WATER}; border-color: rgba(34,116,165,.4); }}
.pill-power   {{ background: rgba(232,161,58,.18);  color: #c07f1e;   border-color: rgba(232,161,58,.45); }}
.pill-coupled {{ background: rgba(123,94,167,.18);  color: {COUPLED}; border-color: rgba(123,94,167,.45); }}
.sec-head {{ display:flex; align-items:center; gap:.6rem; margin:.2rem 0 .1rem 0; }}
.sec-head .bar {{ width:6px; height:1.7rem; border-radius:3px; }}
.card {{ border:1px solid rgba(120,120,140,.24); border-radius:12px; padding:.8rem 1rem;
        background: rgba(130,130,150,.06); }}
.small-note {{ font-size:.85rem; opacity:.78; }}
</style>
"""

# System mode: follow the viewer's OS preference via media queries.
_SYSTEM_CSS = f"""<style>
@media (prefers-color-scheme: light) {{ {_surface_css(_LIGHT)} }}
@media (prefers-color-scheme: dark)  {{ {_surface_css(_DARK)} }}
</style>"""


def _register_plotly(mode: str) -> None:
    """Give every Plotly chart a theme-neutral font so axes read on any background."""
    try:
        import plotly.io as pio
        import plotly.graph_objects as go
        font = "#1a1d24" if mode == "Light" else "#dfe1e8" if mode == "Dark" else "#8a8f99"
        grid = "rgba(130,130,150,.20)"
        pio.templates["cowpf"] = go.layout.Template(layout=dict(
            font=dict(color=font),
            xaxis=dict(gridcolor=grid, zerolinecolor=grid),
            yaxis=dict(gridcolor=grid, zerolinecolor=grid)))
        pio.templates.default = "plotly+cowpf"
    except Exception:
        pass


def inject_theme(st, mode: str = "System") -> None:
    """Inject base CSS + the chosen appearance mode; tune Plotly fonts to match."""
    st.markdown(_BASE_CSS, unsafe_allow_html=True)
    if mode == "Light":
        st.markdown(f"<style>{_surface_css(_LIGHT)}</style>", unsafe_allow_html=True)
    elif mode == "Dark":
        st.markdown(f"<style>{_surface_css(_DARK)}</style>", unsafe_allow_html=True)
    else:
        st.markdown(_SYSTEM_CSS, unsafe_allow_html=True)
    _register_plotly(mode)


# Backwards-compatible alias.
def inject_css(st) -> None:
    inject_theme(st, "System")


def hero(st, title: str, subtitle: str, tagline: str = "") -> None:
    st.markdown(
        f"""<div class="cowpf-hero">
        <h1>{title}</h1>
        <p>{subtitle}</p>
        {'<p class="small-note">' + tagline + '</p>' if tagline else ''}
        <div style="margin-top:.6rem">
          <span class="pill pill-water">💧 water · OWF</span>
          <span class="pill pill-power">⚡ power · LinDistFlow + Z-bus</span>
          <span class="pill pill-coupled">🔗 coupled · C-OWPF</span>
        </div></div>""",
        unsafe_allow_html=True,
    )


def section_header(st, color: str, title: str, subtitle: str = "") -> None:
    st.markdown(
        f"""<div class="sec-head"><div class="bar" style="background:{color}"></div>
        <h3 style="margin:0">{title}</h3></div>
        {'<p class="small-note">' + subtitle + '</p>' if subtitle else ''}""",
        unsafe_allow_html=True,
    )
