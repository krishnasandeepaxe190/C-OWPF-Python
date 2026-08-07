"""Water tab: the decoupled FSP Optimal Water Flow (direct / warmstart / optimize /
EPANET modes), schedules, EPANET validation, plus per-session comparison and diff.

Self-contained: controls live in-tab so each app section is independent.
"""
from __future__ import annotations

import contextlib
import io
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as _st

from owf import NETWORKS
from owf.config import (DEFAULT_FALLBACK, DEFAULT_SOLVER, SOLVER_CHOICES,
                        available_solvers)
from owf.epanet_io import read_controls_rules
from owf.netmap import build_animated_map_figure, build_map_figure, extract_map_data
from main_owf import MODES, MODE_SUGGESTION, NET_BLURB, run_case
from .theme import WATER, section_header


def _csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv().encode()


@_st.cache_data(show_spinner=False)
def _pump_ids(net: int) -> list:
    """Pump link ids for a water net (cached; used to pick variable-speed pumps)."""
    from owf.config import SolverConfig
    from owf.network import setup as _setup
    w = _setup(SolverConfig(net_num=net, time=24))
    return [str(w.raw.link_name_id[i]) for i in w.raw.link_pump_index]


@_st.cache_data(show_spinner=False)
def _prv_ids(net: int) -> list:
    """PRV link ids for a water net (cached)."""
    from owf.config import SolverConfig
    from owf.network import setup as _setup
    w = _setup(SolverConfig(net_num=net, time=24))
    return [str(w.raw.link_name_id[i]) for i in w.M.valve_index]


def _build_exports(case, wdn, result) -> dict:
    T = wdn.time
    hours = [f"h{t}" for t in range(T)]
    link_ids = list(wdn.raw.link_name_id)
    node_ids = list(wdn.raw.node_name_id)
    out = {}
    out["flows_gpm.csv"] = _csv_bytes(pd.DataFrame(result.flows[:, :T], index=link_ids, columns=hours))
    out["heads_ft.csv"] = _csv_bytes(pd.DataFrame(result.heads[:, :T], index=node_ids, columns=hours))
    pump_ids = [link_ids[i] for i in wdn.raw.link_pump_index]
    out["pump_schedule.csv"] = _csv_bytes(pd.DataFrame(result.onoff[:, :T], index=pump_ids, columns=hours).astype(int))
    out["pump_power_kw.csv"] = _csv_bytes(pd.DataFrame(result.ppump_true[:, :T], index=pump_ids, columns=hours).round(3))
    if getattr(result, "speed", None) is not None:
        out["pump_speed.csv"] = _csv_bytes(
            pd.DataFrame(result.speed[:, :T], index=pump_ids, columns=hours).round(3))
    return out


def _session_df(records) -> pd.DataFrame:
    rows = []
    for rec in records:
        c = rec["case"]
        rows.append({
            "run": rec["id"], "case": c.label,
            "EPANET cost (rules)": round(c.epanet_cost, 5),
            "C-OWPF cost": round(c.owf_cost, 5) if np.isfinite(c.owf_cost) else None,
            "saving %": round(c.savings_pct, 1) if np.isfinite(c.savings_pct) else None,
            "replay max |dHead| ft": round(c.max_dhead, 3) if np.isfinite(c.max_dhead) else None,
            "replay min press. ft": round(c.min_pressure, 1) if np.isfinite(c.min_pressure) else None,
            "feasible": "yes" if c.converged else "no", "time s": round(c.elapsed),
        })
    return pd.DataFrame(rows)


def _render_case(st, rec) -> None:
    case = rec["case"]
    if not case.converged and not np.isfinite(case.owf_cost):
        st.error(f"No feasible solution ({case.note}). Try the recommended mode.")
        return
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("EPANET cost (rules)", f"{case.epanet_cost:.4f}",
              help="EPANET's own tank-level rules — a DIFFERENT schedule; cost baseline only.")
    c2.metric("C-OWPF cost", f"{case.owf_cost:.4f}", delta=f"{-case.savings_pct:.1f}%",
              delta_color="inverse", help="Optimized schedule, true nonlinear energy cost.")
    c3.metric("Max head error", f"{case.max_dhead:.3f} ft",
              help="SAME schedule imposed in EPANET and replayed.")
    c4.metric("Min junction pressure", f"{case.min_pressure:.1f} ft",
              help="> 0 means hydraulically feasible in EPANET.")
    c5.metric("Solve time", f"{case.elapsed:.0f} s", help=f"{case.n_iter} iters · {case.solver}")
    if case.note:
        st.caption(f"note: {case.note}")

    has_prv = rec.get("prv_fig") is not None
    tab_names = ["Network map", "Flow animation", "Schedule", "Flows", "Heads",
                 "Convergence", "Error", "Solver log", "Download"]
    if has_prv:
        # appended LAST so the fixed tab indices below (0..8) stay valid
        tab_names.append("🔻 PRV")
    tabs = st.tabs(tab_names)
    if has_prv:
        with tabs[9]:
            st.plotly_chart(rec["prv_fig"], use_container_width=True,
                            key=f"prv_{rec['id']}")
            st.caption("Three-state PRV (closed / open / active): when **active** the "
                       "valve pins its downstream junction to h_set by absorbing "
                       "R_PRV of head. ✕ markers = EPANET **replay** of the optimized "
                       "schedule (model accuracy); dotted line = EPANET's own "
                       "**rule-based** operation (a *different* schedule at the .inp "
                       "setting). Compare the two regulations: if your h_set is higher "
                       "than the rules' setting, holding that extra downstream pressure "
                       "needs more pumped head — which is why the optimized cost can "
                       "*exceed* the rule-based baseline. Same h_set → the optimizer "
                       "matches or beats the rules.")
    with tabs[0]:
        md = rec["map_data"]
        if "flows" in md:
            hour = st.slider("Hour", 0, md["time"] - 1, 0, key=f"hr_{rec['id']}")
            st.plotly_chart(build_map_figure(md, hour), use_container_width=True, key=f"map_{rec['id']}")
        else:
            st.plotly_chart(build_map_figure(md), use_container_width=True, key=f"map_{rec['id']}")
    with tabs[1]:
        md = rec["map_data"]
        if "flows" in md:
            st.caption("▶ Play to animate. Arrow direction = flow direction, size = |flow|, node color = pressure.")
            st.plotly_chart(build_animated_map_figure(md), use_container_width=True, key=f"anim_{rec['id']}")
        else:
            st.info("No solution to animate.")
    plot_order = ["schedule", "flows", "heads", "convergence", "error"]
    by_kind = {Path(p).stem.split("_")[-1]: p for p in rec["plots"]}
    for tab, kind in zip(tabs[2:7], plot_order):
        with tab:
            p = by_kind.get(kind)
            st.image(str(p), use_container_width=True) if p and Path(p).exists() else st.info("plot not available")
    with tabs[7]:
        st.code(rec["log"] or "(no output)", language="text")
    with tabs[8]:
        cols = st.columns(len(rec["exports"]) or 1)
        for col, (name, blob) in zip(cols, rec["exports"].items()):
            col.download_button(name, data=blob, file_name=f"run{rec['id']}_{name}",
                                mime="text/csv", key=f"dl_{rec['id']}_{name}")


def _controls(st):
    section_header(st, WATER, "Optimal Water Flow — decoupled pump scheduling")
    r = st.columns([1.3, 1, 1, 1.2])
    net = r[0].selectbox("Network", list(NETWORKS),
                         format_func=lambda n: f"{NETWORKS[n].name} ({NET_BLURB[n]})", key="w_net")
    rec_mode, alt_mode = MODE_SUGGESTION[net]
    mode = r[1].selectbox("Mode", list(MODES), index=list(MODES).index(rec_mode),
                          help="\n\n".join(f"**{m}** — {d}" for m, d in MODES.items()), key="w_mode")
    price = r[2].selectbox("Price", [1, 0], format_func=lambda p: "Time-of-use" if p else "Flat", key="w_price")
    avail = available_solvers()
    solver = r[3].selectbox("MILP solver", SOLVER_CHOICES,
                            index=SOLVER_CHOICES.index(DEFAULT_SOLVER),
                            format_func=lambda s: s + ("" if s in avail else " (not installed)"), key="w_solver")
    if mode == rec_mode:
        st.caption(f"✅ Recommended mode for {NETWORKS[net].name}. {MODES[mode]}")
    elif mode == "direct" and net in (11, 36, 97):
        st.warning(f"Direct mode does not converge on looped networks — use '{rec_mode}'.")
    else:
        st.caption(MODES[mode])
    if net == 97 and mode == "optimize":
        st.warning("Net3 optimize takes ~3 min and honestly reports ~0% savings.")
    if net == 126:
        st.info("BWSN is a large network (126 junctions, 2 pumps, 2 tanks, 1 reservoir). "
                "Use **optimize** — its binary fixed-speed pumps must switch OFF when a "
                "tank is full (EPANET throttles instead), so 'epanet' mode overfills a "
                "tank. Optimize takes a few minutes.")
    # --- Pressure-reducing valves (PRV) ----------------------------------------
    # "Add PRVs" swaps in the PRV variant of the network (default valve locations
    # are fixed by the variant's .inp); the pressure setting h_set is the user's.
    prv = None
    PRV_VARIANT = {8: 108}          # base net -> its PRV variant
    with st.expander("🔻 Pressure-reducing valves (PRV)"):
        if net in PRV_VARIANT:
            n_prv = st.radio("Number of PRVs", [0, 1], horizontal=True, key="w_nprv",
                             help="PRV locations are fixed (8-node: junction 6 → 9); "
                                  "choose how many are installed.")
            if n_prv:
                net = PRV_VARIANT[net]
        prv_ids = _prv_ids(net)
        if prv_ids:
            st.caption(f"{len(prv_ids)} PRV(s) installed (fixed locations). The PRV "
                       "regulates its downstream junction to h_set = elevation + P_set. "
                       "PRV runs use the **warmstart** solve (reliable with valve binaries).")
            pset = st.slider("PRV pressure setting P_set (psi)", 5.0, 60.0, 20.0, 1.0,
                             key="w_prv_pset",
                             help="Downstream pressure setpoint; h_set = E_down + P_set·2.307 ft.")
            prv = {vid: float(pset) for vid in prv_ids}
        elif net not in PRV_VARIANT.values():
            st.caption("This network has no PRV variant yet (PRVs available on the 8-node).")

    # --- Variable-speed pumps (VSP) --------------------------------------------
    vsp = None
    with st.expander("⚙️ Variable-speed pumps (VSP)"):
        st.caption("Mark pumps to run at **variable speed**: the solver co-optimizes "
                   "each pump's relative speed ω ∈ [ω_min, 1]. Running at reduced speed "
                   "cuts energy (power ∝ ω³) when the head allows it. Unlisted pumps stay "
                   "fixed-speed. VSP uses a soft-bound damped **direct** solve.")
        ids = _pump_ids(net)
        vsp_sel = st.multiselect("Pumps to run as variable-speed", ids, key="w_vsp_sel")
        omin = st.slider("Minimum relative speed ω_min", 0.50, 1.0, 0.80, 0.05,
                         key="w_vsp_omin",
                         help="Lower bound on relative pump speed when running.")
        if vsp_sel:
            vsp = {p: (float(omin), 1.0) for p in vsp_sel}
            st.info(f"VSP active on {len(vsp)} pump(s); mode is overridden to the "
                    f"direct VSP solve.")
    with st.expander("EPANET operating rules (cost baseline)"):
        cr = read_controls_rules(NETWORKS[net].inp_path)
        if cr["CONTROLS"]:
            st.markdown("**[CONTROLS]**"); st.code("\n".join(cr["CONTROLS"]), language="text")
        if cr["RULES"]:
            st.markdown("**[RULES]**"); st.code("\n".join(cr["RULES"]), language="text")
        if not cr["CONTROLS"] and not cr["RULES"]:
            st.caption("This network defines no explicit controls or rules.")
    b1, b2 = st.columns([1, 1])
    run = b1.button("💧  Run water case", type="primary", use_container_width=True,
                    disabled=solver not in avail, key="w_run")
    if b2.button("Clear water history", use_container_width=True, key="w_clear"):
        st.session_state.water_records = []
        st.rerun()
    return net, mode, price, solver, run, vsp, prv


def render_water(st) -> None:
    if "water_records" not in st.session_state:
        st.session_state.water_records = []
    net, mode, price, solver, run, vsp, prv = _controls(st)
    # PRV networks need the warmstart solve: a free-binary direct solve is
    # unreliable with valve binaries (the recommended mode for net 108).
    if prv and mode == "direct":
        mode = "warmstart"

    if run:
        eta = {"direct": "a few seconds", "warmstart": "up to a minute",
               "epanet": "~30-60 s", "optimize": "1-3 minutes"}[mode]
        if vsp:
            eta = "up to a minute (VSP)"
        label = f"{mode} mode" if not vsp else "VSP direct solve"
        with st.spinner(f"Solving {NETWORKS[net].name} in {label} ({eta})..."):
            log_buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(log_buf):
                    case, wdn, result = run_case(net, mode, price, None, plot=True,
                                                 outdir="outputs", verbose=False,
                                                 solver=solver, vsp=vsp, prv=prv)
            except Exception as exc:
                st.error(f"Case failed: {exc}")
                case = None
        if case is not None:
            run_id = len(st.session_state.water_records) + 1
            plots = []
            for p in case.plots:
                p = Path(p)
                if p.exists():
                    q = p.with_name(f"run{run_id}_{p.name}")
                    shutil.copyfile(p, q)
                    plots.append(q)
            # paper-style PRV panel (model vs EPANET) when the net has a valve
            prv_fig = None
            if (result is not None and result.flows is not None
                    and getattr(result, "prv", None) and wdn.n_valves):
                try:
                    from owf.validation import validate_schedule
                    from owf.epanet_io import run_epanet
                    from .prv_plots import prv_panel
                    rep = validate_schedule(wdn, result)
                    # EPANET's own rule-based operation (a DIFFERENT schedule):
                    # shows the pressure regulation the rules achieve, so a cost
                    # increase vs the baseline is explained by the pressure target.
                    fl_r, hd_r, _, _ = run_epanet(wdn.raw)
                    prv_fig = prv_panel(wdn, result.heads, result.flows, result.prv,
                                        heads_ep=rep.heads_epanet,
                                        flows_ep=rep.flows_epanet,
                                        heads_rule=hd_r[: wdn.time].T,
                                        flows_rule=fl_r[: wdn.time].T)
                except Exception:
                    prv_fig = None
            st.session_state.water_records.append({
                "id": run_id, "case": case, "log": log_buf.getvalue(), "plots": plots,
                "map_data": extract_map_data(wdn, result),
                "prv_fig": prv_fig,
                "exports": (_build_exports(case, wdn, result)
                            if result is not None and result.flows is not None else {}),
            })

    records = st.session_state.water_records
    if not records:
        st.info("Configure a case above and press **Run water case**.")
        return
    latest = records[-1]
    st.subheader(f"Result — run {latest['id']}: {latest['case'].label}")
    _render_case(st, latest)

    st.subheader("Session comparison")
    df = _session_df(records)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("Download session table (CSV)", data=_csv_bytes(df.set_index("run")),
                       file_name="owf_session_comparison.csv", mime="text/csv", key="w_dlsess")
    st.caption("**Cost** columns compare *different* schedules (EPANET rules vs C-OWPF). "
               "**Replay** columns use the *same* schedule imposed back in EPANET — "
               "linearization fidelity, not the cost operating point.")
