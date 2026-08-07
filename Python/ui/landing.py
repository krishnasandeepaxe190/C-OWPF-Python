"""Landing page: paper intro, the C-OWPF solve flowchart, a successive-linearization
animation with a "why it's better" explainer, the networks, and key formulations."""
from __future__ import annotations

import numpy as np

from pathlib import Path

from .theme import WATER, POWER, COUPLED, GOOD, BAD, hero, section_header

_DOCS = Path(__file__).resolve().parent.parent / "docs"


def _method_note_download(st) -> None:
    """Offer the one-page 'how each solve mode is derived in code' transparency note."""
    pdf = _DOCS / "solve_modes.pdf"
    if not pdf.exists():
        return
    st.caption("📄 **Transparency note** — how each solve mode is derived in code "
               "(decoupled water modes, the coupled schedule search, and the "
               "warm-started trust-region MILP), with a flowchart and a code map.")
    st.download_button("⬇  Download the one-page method note (PDF)",
                       data=pdf.read_bytes(), file_name="C-OWPF_solve_modes.pdf",
                       mime="application/pdf", key="dl_method_note")


# ------------------------------------------------------------------ flowchart
def _flowchart_svg() -> str:
    """Detailed flowchart of how the coupled C-OWPF problem is solved."""
    return f"""
<svg viewBox="0 0 960 470" width="100%" xmlns="http://www.w3.org/2000/svg" font-family="system-ui">
  <defs>
    <marker id="a" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto">
      <path d="M0,0 L6,3 L0,6 Z" fill="#8a8a99"/></marker>
    <marker id="ar" markerWidth="9" markerHeight="9" refX="6" refY="3" orient="auto">
      <path d="M0,0 L6,3 L0,6 Z" fill="{COUPLED}"/></marker>
  </defs>

  <!-- 1. warm start -->
  <rect x="330" y="12" width="300" height="52" rx="10" fill="{WATER}22" stroke="{WATER}" stroke-width="1.5"/>
  <text x="480" y="34" text-anchor="middle" font-size="13.5" font-weight="700" fill="{WATER}">① EPANET warm start</text>
  <text x="480" y="52" text-anchor="middle" font-size="11" fill="#8a8f99">impose a pump schedule → consistent flows &amp; heads</text>
  <line x1="480" y1="64" x2="480" y2="92" stroke="#8a8a99" stroke-width="2" marker-end="url(#a)"/>

  <!-- successive-linearization loop box -->
  <rect x="70" y="96" width="820" height="150" rx="14" fill="{COUPLED}0d" stroke="{COUPLED}" stroke-width="1.4" stroke-dasharray="6 4"/>
  <text x="86" y="116" font-size="12" font-weight="700" fill="{COUPLED}">Successive-linearization loop  (repeat until ‖Δ[H;Q;z]‖ &lt; tol)</text>

  <rect x="92" y="128" width="240" height="98" rx="10" fill="#8a8a9914" stroke="#9aa0a6" stroke-width="1.2"/>
  <text x="212" y="150" text-anchor="middle" font-size="12.5" font-weight="700">② Linearize (convex)</text>
  <text x="212" y="170" text-anchor="middle" font-size="10.5" fill="#8a8f99">Hazen-Williams head loss</text>
  <text x="212" y="185" text-anchor="middle" font-size="10.5" fill="#8a8f99">Π̃H = Cp + Π′Q</text>
  <text x="212" y="203" text-anchor="middle" font-size="10.5" fill="#8a8f99">FSP power  P = A′f + B′z</text>

  <rect x="360" y="128" width="240" height="98" rx="10" fill="{POWER}18" stroke="{POWER}" stroke-width="1.2"/>
  <text x="480" y="148" text-anchor="middle" font-size="12.5" font-weight="700" fill="#b9781a">③ Solve one step</text>
  <text x="480" y="166" text-anchor="middle" font-size="10.5" fill="#8a8f99">MILP (free z) or LP (z fixed)</text>
  <text x="480" y="180" text-anchor="middle" font-size="10.5" fill="#8a8f99">min pump energy + loss cost</text>
  <text x="480" y="194" text-anchor="middle" font-size="10.5" fill="#8a8f99">+ voltage limits · PV reactive</text>
  <text x="480" y="208" text-anchor="middle" font-size="10.5" fill="#8a8f99">+ voltage-dependent caps</text>

  <rect x="628" y="128" width="240" height="98" rx="10" fill="#8a8a9914" stroke="#9aa0a6" stroke-width="1.2"/>
  <text x="748" y="150" text-anchor="middle" font-size="12.5" font-weight="700">④ Relinearize</text>
  <text x="748" y="170" text-anchor="middle" font-size="10.5" fill="#8a8f99">around the new flow field</text>
  <text x="748" y="185" text-anchor="middle" font-size="10.5" fill="#8a8f99">(optional damping / trust region)</text>
  <text x="748" y="203" text-anchor="middle" font-size="10.5" fill="#8a8f99">soft bounds → always feasible</text>

  <line x1="332" y1="177" x2="358" y2="177" stroke="#8a8a99" stroke-width="2" marker-end="url(#a)"/>
  <line x1="600" y1="177" x2="626" y2="177" stroke="#8a8a99" stroke-width="2" marker-end="url(#a)"/>
  <path d="M748,226 v14 h-536 v-14" fill="none" stroke="{COUPLED}" stroke-width="1.6" marker-end="url(#ar)"/>
  <text x="480" y="256" text-anchor="middle" font-size="10.5" fill="{COUPLED}">↺ relinearize &amp; resolve</text>

  <line x1="480" y1="246" x2="480" y2="286" stroke="#8a8a99" stroke-width="2" marker-end="url(#a)"/>
  <text x="512" y="272" font-size="10.5" fill="#8a8f99">converged</text>

  <!-- schedule search -->
  <rect x="250" y="290" width="460" height="52" rx="10" fill="{COUPLED}22" stroke="{COUPLED}" stroke-width="1.5"/>
  <text x="480" y="312" text-anchor="middle" font-size="13" font-weight="700" fill="{COUPLED}">⑤ Voltage-aware schedule search</text>
  <text x="480" y="330" text-anchor="middle" font-size="10.5" fill="#8a8f99">price candidates → warm-started trust-region MILP → 1-opt polish  (min total $ = pump + loss)</text>
  <line x1="480" y1="342" x2="480" y2="374" stroke="#8a8a99" stroke-width="2" marker-end="url(#a)"/>

  <!-- validate -->
  <rect x="250" y="378" width="460" height="70" rx="10" fill="{GOOD}18" stroke="{GOOD}" stroke-width="1.5"/>
  <text x="480" y="400" text-anchor="middle" font-size="13" font-weight="700" fill="{GOOD}">⑥ Validate against exact simulators</text>
  <text x="480" y="420" text-anchor="middle" font-size="10.5" fill="#8a8f99">water: re-run the schedule in EPANET → ΔH, Δflow</text>
  <text x="480" y="435" text-anchor="middle" font-size="10.5" fill="#8a8f99">power: replay injections in nonlinear Z-bus → true voltages &amp; loss</text>
</svg>"""


# ------------------------------------------------------------------ animation
def successive_approx_fig():
    """Animate successive linearization converging on the nonlinear head-loss curve."""
    import plotly.graph_objects as go

    n = 1.852                      # Hazen-Williams exponent
    H = 1.0                        # target head; operating point q* where q*^n = H
    q = np.linspace(0.05, 2.0, 200)
    curve = q ** n

    # Newton / successive-linearization iterates q_{k+1} from the tangent at q_k
    its = [0.6]
    for _ in range(5):
        qk = its[-1]
        qk1 = qk + (H - qk**n) / (n * qk**(n-1))
        its.append(max(qk1, 0.05))
        if abs(qk1 - qk) < 1e-4:
            break
    its = its[:5]

    base = [
        go.Scatter(x=q, y=curve, mode="lines", name="nonlinear head loss  r·qⁿ",
                   line=dict(color=WATER, width=3)),
        go.Scatter(x=[q.min(), q.max()], y=[H, H], mode="lines", name="target head H",
                   line=dict(color="#9aa0a6", width=1.5, dash="dot")),
        go.Scatter(x=[1.0], y=[H], mode="markers", name="operating point q*",
                   marker=dict(color=GOOD, size=12, symbol="star")),
    ]

    frames = []
    for k in range(len(its)):
        qk = its[k]
        # tangent line at q_k:  y = qk^n + n qk^{n-1}(x - qk)
        tx = np.array([max(qk - 0.6, 0.0), qk + 0.6])
        ty = qk**n + n * qk**(n-1) * (tx - qk)
        frames.append(go.Frame(name=str(k), data=[
            go.Scatter(x=tx, y=ty, mode="lines", line=dict(color=POWER, width=2),
                       name=f"linearization @ iter {k}"),
            go.Scatter(x=[qk], y=[qk**n], mode="markers+text",
                       marker=dict(color=BAD, size=11),
                       text=[f"q{k}"], textposition="top center", name="iterate"),
        ], traces=[3, 4]))

    fig = go.Figure(
        data=base + [go.Scatter(x=[], y=[], mode="lines", line=dict(color=POWER, width=2),
                                name="linearization"),
                     go.Scatter(x=[], y=[], mode="markers", name="iterate")],
        frames=frames,
    )
    play = dict(label="▶ Play", method="animate",
                args=[None, dict(frame=dict(duration=900, redraw=True), fromcurrent=True)])
    pause = dict(label="⏸ Pause", method="animate",
                 args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
    steps = [dict(method="animate", label=f"iter {k}",
                  args=[[str(k)], dict(mode="immediate", frame=dict(duration=0, redraw=True))])
             for k in range(len(its))]
    fig.update_layout(
        height=440, margin=dict(l=10, r=10, t=44, b=40),
        xaxis_title="pump / pipe flow  q", yaxis_title="head loss",
        # legend inside the empty upper-left triangle (the curve rises to the right)
        legend=dict(orientation="v", x=0.01, y=0.99, xanchor="left", yanchor="top",
                    bgcolor="rgba(130,130,150,0.10)", bordercolor="rgba(130,130,150,0.3)",
                    borderwidth=1, font=dict(size=10)),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        updatemenus=[dict(type="buttons", direction="left", showactive=False,
                          x=0.0, y=1.10, xanchor="left", yanchor="bottom",
                          pad=dict(b=2), buttons=[play, pause])],
        sliders=[dict(active=0, x=0.1, y=-0.02, len=0.82,
                      currentvalue=dict(prefix="step "), steps=steps)])
    return fig


# ------------------------------------------------------------------ networks
def _networks_table():
    import pandas as pd
    from owf import NETWORKS
    from main_owf import NET_BLURB
    from pdn import FEEDERS
    water = [{"side": "💧 water", "name": NETWORKS[n].name, "detail": NET_BLURB[n]}
             for n in NETWORKS]
    power = [{"side": "⚡ power", "name": FEEDERS[k]["label"],
              "detail": f"{FEEDERS[k]['N']} buses · {int(np.sum(FEEDERS[k]['pv']))} PV sites"}
             for k in FEEDERS]
    return pd.DataFrame(water + power)


# ------------------------------------------------------------------ page
def render_landing(st) -> None:
    hero(st, "C-OWPF · Optimal Water-Power Flow",
         "Co-optimizing interdependent water- and electric-distribution networks by "
         "successive linear approximation.",
         "Companion implementation to IEEE Access-2024-18604 — “Successive Linear "
         "Approximations for Optimal Decision Making in Interdependent Electric and "
         "Water Distribution Networks.”")

    section_header(st, COUPLED, "How the coupled problem is solved",
                   "Pumps couple the two grids: their electrical power is a bus load on "
                   "the feeder. The solver linearizes the nonlinear physics, solves a "
                   "tractable step, and relinearizes until it converges.")
    st.markdown(_flowchart_svg(), unsafe_allow_html=True)
    _method_note_download(st)

    section_header(st, POWER, "Why successive approximation")
    c1, c2 = st.columns([1.05, 1])
    with c1:
        st.markdown("**Successive linear approximation converging** — press ▶ Play")
        st.plotly_chart(successive_approx_fig(), use_container_width=True, key="succ_anim")
    with c2:
        st.markdown(
            "The physics is **nonlinear and non-convex**: Hazen-Williams head loss "
            "(q^1.852), FSP pump power, and the AC power flow. Solving that directly as "
            "a single MINLP is intractable and unreliable at this scale.\n\n"
            "**Successive linear approximation** replaces it with a short sequence of "
            "**convex** problems — each a MILP (or an LP once the pump schedule is "
            "fixed) that HiGHS solves reliably:\n"
            "- ✅ **Tractable & reliable** — each step is convex; no MINLP solver, no "
            "getting stuck in a bad local basin.\n"
            "- ✅ **Warm-started from EPANET** — starting at a feasible operating point "
            "puts the linearization in the right region; looped networks then converge.\n"
            "- ✅ **Fast convergence** — with the schedule fixed it converges in ~1–2 "
            "iterations (see the animation: the tangent steps home in quickly).\n"
            "- ✅ **Only pump on/off is integer** — the feeder (LinDistFlow + PV VAr) "
            "adds *no* new integers, so the coupling stays cheap.\n"
            "- ✅ **Validated** — the converged decisions are replayed through the exact "
            "EPANET and Z-bus simulators, so the approximation is checked, not trusted.")

    section_header(st, WATER, "Networks")
    st.dataframe(_networks_table(), use_container_width=True, hide_index=True)
    st.caption("Mix any water network with any feeder in the **Coupled** tab. IEEE-13/33 "
               "and the SCE feeders are the cleanest coupling demos; SB-128 is a stressed "
               "feeder for robustness.")

    section_header(st, COUPLED, "Key formulations")
    f1, f2 = st.columns(2)
    with f1:
        st.markdown("**Objective — pump energy + priced network loss** (paper 33d)")
        st.latex(r"\min\ \sum_{t}\ \pi_t\Big[\tfrac{1}{1000}\textstyle\sum_{p} P^{\text{pump}}_{p,t}"
                 r"\;+\;\hat C^{\text{loss}}_{t}\Big]")
        st.markdown("**Network loss** — LinDistFlow branch flows, linearized each iteration")
        st.latex(r"\hat C^{\text{loss}}_t=\sum_{\ell} r_\ell\big(P_{\ell,t}^2+Q_{\ell,t}^2\big),"
                 r"\quad P_{\ell}=\!\!\sum_{k\in\mathcal C_\ell}\!\! p_{\text{net},k}")
        st.markdown("**FSP head gain & power** (general exponent ν)")
        st.latex(r"P^{\text{pump}}_p = c_m\,(h_{0,p}-\sigma_p f_p^{\nu})\,f_p,\qquad "
                 r"\tilde{\Pi}\,H = C_p + \Pi'\,Q")
    with f2:
        st.markdown("**LinDistFlow / Kekatos voltage** (squared magnitudes)")
        st.latex(r"v^2 = R\,(-p_{\text{net}}) + X\,(-q_{\text{net}}) + V_k,\qquad F=-A^{-1}")
        st.markdown("**Voltage-dependent shunt caps** (eq. 1b/2b) — folded into $R,X,V_k$")
        st.latex(r"Q_n=\!\!\sum_{k\in\mathcal C_n}\!\!Q_k-q_n-q^{s}_n v_n,\quad "
                 r"(\mathbf I-X\,\mathrm{diag}(q^{s}))\,v = R(-p)+X(-q)+V_k")
        st.markdown("**Pump reactive coupling & PV capability**")
        st.latex(r"q^{\text{pump}} = P^{\text{pump}}_{\text{pu}}\sqrt{\tfrac{1}{PF^2}-1},"
                 r"\qquad |q^{pv}| \le \sqrt{S^2 - (p^{pv})^2}")
        st.latex(r"p_{\text{net}} = \Psi p + \Xi P^{\text{pump}}_{\text{pu}} - \Gamma p^{pv},"
                 r"\quad V_{\min}^2 \le v^2 \le V_{\max}^2")

    st.info("Open **💧 Water**, **⚡ Power**, or **🔗 Coupled** above to run a case. "
            "See **📖 Guide** for the full methodology. VSP & PRV modeling are upcoming phases.")
