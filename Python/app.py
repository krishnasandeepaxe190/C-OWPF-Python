"""C-OWPF Streamlit UI (Phase 0: FSP water engine).

Run with:  streamlit run app.py
"""
from __future__ import annotations

import contextlib
import io
import sys
import warnings
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from owf import NETWORKS  # noqa: E402
from main_owf import (  # noqa: E402
    MODES,
    MODE_SUGGESTION,
    NET_BLURB,
    run_case,
)

st.set_page_config(page_title="C-OWPF | Optimal Water Flow", page_icon="💧",
                   layout="wide")

st.title("💧 C-OWPF — Optimal Water Flow (FSP)")
st.caption(
    "Pump scheduling on water distribution networks by successive linear "
    "approximation — CVXPY + HiGHS MILP, validated against EPANET. "
    "Water-side FSP engine; VSPs, PRVs and the power network are upcoming phases."
)

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Case setup")

    net = st.selectbox(
        "Network",
        options=list(NETWORKS),
        format_func=lambda n: f"{NETWORKS[n].name}  ({NET_BLURB[n]})",
    )
    rec, alt = MODE_SUGGESTION[net]

    mode = st.selectbox(
        "Solve mode",
        options=list(MODES),
        index=list(MODES).index(rec),
        help="\n\n".join(f"**{m}** — {d}" for m, d in MODES.items()),
    )
    st.caption(MODES[mode])
    if mode == rec:
        st.success(f"Recommended mode for {NETWORKS[net].name}.")
    elif mode == alt:
        st.info("Worth trying: reports savings vs EPANET (slower).")
    elif mode == "direct" and net in (11, 36, 97):
        st.warning("Direct mode does not converge on looped networks — "
                   f"use '{rec}'.")
    if net == 97 and mode == "optimize":
        st.warning("Net3 optimize takes ~3 min and honestly reports ~0% "
                   "savings (its tank-level controls are already near-optimal).")
    if net == 97 and mode == "warmstart":
        st.warning("The multi-start search does not scale to Net3 (50 binaries); "
                   "'epanet' is the reliable mode here.")

    price = st.radio("Electricity price", [1, 0], horizontal=True,
                     format_func=lambda p: "Time-of-use" if p == 1 else "Flat")

    run_clicked = st.button("▶  Run case", type="primary", use_container_width=True)

    st.divider()
    if st.button("Clear session history", use_container_width=True):
        st.session_state.cases = []
        st.session_state.pop("last", None)
        st.rerun()

# ---------------------------------------------------------------- state
if "cases" not in st.session_state:
    st.session_state.cases = []

if run_clicked:
    eta = {"direct": "a few seconds", "warmstart": "up to a minute",
           "epanet": "~30-60 s", "optimize": "1-3 minutes"}[mode]
    with st.spinner(f"Solving {NETWORKS[net].name} in {mode} mode ({eta})..."):
        log_buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(log_buf):
                case, wdn, result = run_case(net, mode, price, None,
                                             plot=True, outdir="outputs",
                                             verbose=False)
        except Exception as exc:  # surface, don't crash the app
            st.error(f"Case failed: {exc}")
            case = None
        if case is not None:
            st.session_state.cases.append(case)
            st.session_state.last = (case, log_buf.getvalue())

# ---------------------------------------------------------------- results
if "last" in st.session_state:
    case, log = st.session_state.last

    st.subheader(f"Result — {case.label}")
    ok = case.converged and case.owf_cost == case.owf_cost  # not NaN
    if not ok:
        st.error(f"No feasible solution ({case.note}). "
                 "Try the recommended mode for this network.")
    else:
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("EPANET cost", f"{case.epanet_cost:.4f}")
        c2.metric("C-OWPF cost", f"{case.owf_cost:.4f}",
                  delta=f"{-case.savings_pct:.1f}%", delta_color="inverse")
        c3.metric("Max head error", f"{case.max_dhead:.3f} ft",
                  help="EPANET replay of the optimized schedule")
        c4.metric("Min junction pressure", f"{case.min_pressure:.1f} ft",
                  help="> 0 means hydraulically feasible in EPANET")
        c5.metric("Solve time", f"{case.elapsed:.0f} s",
                  help=f"{case.n_iter} iterations")
        if case.note:
            st.caption(f"note: {case.note}")

        if case.plots:
            names = [Path(p).stem.split("_")[-1] for p in case.plots]
            order = ["schedule", "flows", "heads", "convergence", "error"]
            pairs = sorted(zip(names, case.plots),
                           key=lambda x: order.index(x[0]) if x[0] in order else 9)
            tabs = st.tabs([n.capitalize() for n, _ in pairs] + ["Solver log"])
            for tab, (_, path) in zip(tabs, pairs):
                with tab:
                    if Path(path).exists():
                        st.image(str(path), use_container_width=True)
            with tabs[-1]:
                st.code(log or "(no output)", language="text")

# ---------------------------------------------------------------- session table
if st.session_state.cases:
    st.subheader("Session comparison")
    df = pd.DataFrame([{
        "case": c.label,
        "EPANET cost": round(c.epanet_cost, 5),
        "C-OWPF cost": round(c.owf_cost, 5) if c.owf_cost == c.owf_cost else None,
        "saving %": round(c.savings_pct, 1) if c.savings_pct == c.savings_pct else None,
        "max |dHead| ft": round(c.max_dhead, 3) if c.max_dhead == c.max_dhead else None,
        "max |dPumpQ| GPM": round(c.max_dpumpflow, 3) if c.max_dpumpflow == c.max_dpumpflow else None,
        "min press. ft": round(c.min_pressure, 1) if c.min_pressure == c.min_pressure else None,
        "feasible": "yes" if c.converged else "no",
        "time s": round(c.elapsed),
    } for c in st.session_state.cases])
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.caption("cost = true nonlinear energy cost; errors from the EPANET replay "
               "of each case's schedule.")
else:
    st.info("Set up a case in the sidebar and press **Run case**.")
