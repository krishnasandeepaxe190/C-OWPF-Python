"""C-OWPF Streamlit app — coupled Optimal Water-Power Flow.

Five sections: Landing · Water (decoupled OWF) · Power (PDN reactive OPF) ·
Coupled (joint C-OWPF) · Guide.

Run with:  streamlit run app.py
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from ui.theme import inject_theme, APPEARANCE_MODES   # noqa: E402
from ui.landing import render_landing              # noqa: E402
from ui.water_section import render_water          # noqa: E402
from ui.power_section import render_power          # noqa: E402
from ui.coupled_section import render_coupled      # noqa: E402
from guide import render_guide                     # noqa: E402

st.set_page_config(page_title="C-OWPF | Optimal Water-Power Flow",
                   page_icon="🔗", layout="wide")

# Appearance selector (System / Light / Dark) — top-left, clear of the Deploy menu.
if "appearance" not in st.session_state:
    st.session_state.appearance = "System"
_ICONS = {"System": "🖥️ System", "Light": "☀️ Light", "Dark": "🌙 Dark"}
_appear, _spacer = st.columns([2.4, 5])
with _appear:
    st.markdown('<div class="appearance-strip">', unsafe_allow_html=True)
    st.session_state.appearance = st.radio(
        "🎨 Appearance", APPEARANCE_MODES, format_func=lambda m: _ICONS[m],
        index=APPEARANCE_MODES.index(st.session_state.appearance),
        horizontal=True, help="System follows your OS light/dark setting.")
    st.markdown('</div>', unsafe_allow_html=True)
inject_theme(st, st.session_state.appearance)

tab_home, tab_water, tab_power, tab_coupled, tab_guide = st.tabs(
    ["🏠 Home", "💧 Water", "⚡ Power", "🔗 Coupled", "📖 Guide"])

with tab_home:
    render_landing(st)
with tab_water:
    render_water(st)
with tab_power:
    render_power(st)
with tab_coupled:
    render_coupled(st)
with tab_guide:
    render_guide(st)
