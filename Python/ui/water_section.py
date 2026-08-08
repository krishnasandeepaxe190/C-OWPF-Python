"""Water tab: the decoupled FSP Optimal Water Flow (direct / warmstart / optimize /
EPANET modes), schedules, EPANET validation, plus per-session comparison and diff.

Self-contained: controls live in-tab so each app section is independent.
"""
from __future__ import annotations

import contextlib
import io
import os
import shutil
import tempfile
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


def _run_capturing_all(fn):
    """Run ``fn`` while capturing BOTH Python-level stdout and the C-level fd 1/2.

    HiGHS/MOSEK/EPANET are C libraries that write their logs (presolve, primal/
    dual simplex, branch-and-bound) straight to file descriptor 1 -- invisible to
    ``contextlib.redirect_stdout``. Temporarily point fd 1 and 2 into a temp file
    so the full solver log is captured. Returns (fn_result, captured_text).
    """
    fd_out, fd_err = os.dup(1), os.dup(2)
    tmp = tempfile.TemporaryFile(mode="w+b")
    try:
        os.dup2(tmp.fileno(), 1)
        os.dup2(tmp.fileno(), 2)
        try:
            result = fn()
        finally:
            try:
                import sys
                sys.stdout.flush(); sys.stderr.flush()
            except Exception:
                pass
            os.dup2(fd_out, 1)
            os.dup2(fd_err, 2)
        tmp.flush(); tmp.seek(0)
        text = tmp.read().decode("utf-8", "ignore")
    finally:
        os.close(fd_out); os.close(fd_err); tmp.close()
    return result, text


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
            "C-OWF cost": round(c.owf_cost, 5) if np.isfinite(c.owf_cost) else None,
            "saving %": round(c.savings_pct, 1) if np.isfinite(c.savings_pct) else None,
            "replay max |dHead| ft": round(c.max_dhead, 3) if np.isfinite(c.max_dhead) else None,
            "replay min press. ft": round(c.min_pressure, 1) if np.isfinite(c.min_pressure) else None,
            "feasible": "yes" if c.converged else "no", "time s": round(c.elapsed),
        })
    df = pd.DataFrame(rows)
    # computation-time difference vs the first run of the session, in %
    if len(rows) > 1 and rows[0]["time s"]:
        t0 = float(rows[0]["time s"])
        df["time Δ% vs run 1"] = [
            "—" if i == 0 else f"{100.0 * (r['time s'] - t0) / t0:+.0f}%"
            for i, r in enumerate(rows)]
    return df


def _render_inspector(st, rec) -> None:
    """Pick any links/nodes and plot their time series -- flow(t), head(t),
    pressure(t) -- straight from the solved fields (student exploration)."""
    import plotly.graph_objects as go
    md = rec["map_data"]
    if "flows" not in md:
        st.info("No solved time series to inspect.")
        return
    T = md["time"]
    hrs = list(range(T))
    link_lbl = [f"{md['link_kind'][i]} {md['link_id'][i]}" for i in range(len(md["link_id"]))]
    node_lbl = [f"{md['node_kind'][i]} {md['node_id'][i]}" for i in range(len(md["node_id"]))]
    c1, c2 = st.columns(2)
    with c1:
        sel_l = st.multiselect("Links → flow (GPM)", link_lbl,
                               default=[l for l in link_lbl if l.startswith("pump")][:1],
                               key=f"insp_l_{rec['id']}")
        if sel_l:
            fig = go.Figure()
            for lbl in sel_l:
                i = link_lbl.index(lbl)
                fig.add_trace(go.Scatter(x=hrs, y=md["flows"][i, :T],
                                         mode="lines+markers", name=lbl))
            fig.update_layout(height=330, xaxis_title="hour", yaxis_title="flow (GPM)",
                              margin=dict(l=10, r=10, t=25, b=10),
                              legend=dict(orientation="h", y=-0.3, x=0),
                              plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True, key=f"insp_lf_{rec['id']}")
    with c2:
        what = st.radio("Node quantity", ["head h(t) [ft]", "pressure p(t) [ft]"],
                        horizontal=True, key=f"insp_q_{rec['id']}")
        sel_n = st.multiselect("Nodes", node_lbl,
                               default=[n for n in node_lbl if n.startswith("tank")][:1],
                               key=f"insp_n_{rec['id']}")
        if sel_n:
            src = md["heads"] if what.startswith("head") else md["pressure"]
            fig = go.Figure()
            for lbl in sel_n:
                i = node_lbl.index(lbl)
                fig.add_trace(go.Scatter(x=hrs, y=src[i, :T],
                                         mode="lines+markers", name=lbl))
            fig.update_layout(height=330, xaxis_title="hour",
                              yaxis_title=what.split(" ")[1] + " (ft)",
                              margin=dict(l=10, r=10, t=25, b=10),
                              legend=dict(orientation="h", y=-0.3, x=0),
                              plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True, key=f"insp_nh_{rec['id']}")
    st.caption("Add any links or nodes to the plots — same values the Network-map "
               "hover shows, as full time series. Pressure = head − elevation.")


def _render_pump_curves(st, rec) -> None:
    """EPANET-style pump curve H(q) = h0·ω² − r·q^v with the run's operating
    points overlaid (colored by hour)."""
    import plotly.graph_objects as go
    pc = rec.get("pump_curve")
    if not pc:
        st.info("No pump-curve data for this run.")
        return
    sel = st.selectbox("Pump", pc["ids"], key=f"pc_sel_{rec['id']}")
    p = pc["ids"].index(sel)
    h0, r, v, qmax = pc["h0"][p], pc["r"][p], pc["v"][p], pc["maxf"][p]
    q = np.linspace(0.0, 1.05 * qmax, 200)
    fig = go.Figure()
    speeds = [1.0]
    if bool(pc["is_vsp"][p]) and pc["speed"] is not None:
        on = pc["onoff"][p] > 0.5
        used = sorted({round(float(s), 2) for s in pc["speed"][p][on]})
        speeds = sorted(set([1.0] + used))
    for om in speeds:
        H = h0 * om ** 2 - r * q ** v
        fig.add_trace(go.Scatter(x=q, y=np.maximum(H, 0), mode="lines",
                                 name=f"curve ω={om:.2f}",
                                 line=dict(dash="solid" if om == 1.0 else "dash")))
    on = pc["onoff"][p] > 0.5
    if on.any():
        hrs = np.where(on)[0]
        fig.add_trace(go.Scatter(
            x=pc["q_op"][p][on], y=pc["H_op"][p][on], mode="markers+text",
            text=[f"h{t}" for t in hrs], textposition="top center",
            textfont=dict(size=9),
            marker=dict(size=10, color=hrs, colorscale="Viridis",
                        colorbar=dict(title="hour"), line=dict(color="white", width=1)),
            name="operating points"))
    fig.update_layout(height=430, xaxis_title="flow q (GPM)",
                      yaxis_title="head gain H (ft)",
                      title=f"Pump {sel}:  H = {h0:.1f}·ω² − {r:.3g}·q^{v:.2f}",
                      margin=dict(l=10, r=10, t=45, b=10),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True, key=f"pc_fig_{rec['id']}")
    st.caption("The EPANET-style characteristic curve. Each marker is one hour's "
               "operating point (q, H) from the optimized solution — it must sit ON "
               "the curve for the speed the pump ran at. VSPs show one dashed curve "
               "per speed used: the affinity laws shift the curve down as ω falls.")


def _render_constraints(st, rec) -> None:
    """Teaching mode: per-iteration constraint residuals/margins with the math."""
    from .constraint_view import iteration_report, evolution_figure
    wdn = rec["teach_wdn"]
    iterates = rec["iterates"]
    st.markdown(f"**{len(iterates)} successive-linearization iterations recorded.** "
                "Equalities report the worst |residual|; inequalities report the "
                "worst violation (0 = satisfied). Watch the linearized physics "
                "tighten as the algorithm relinearizes.")
    st.plotly_chart(evolution_figure(wdn, iterates), use_container_width=True,
                    key=f"cons_ev_{rec['id']}")
    # st.slider requires min < max: a run that converges in ONE iteration has
    # nothing to slide over
    if len(iterates) > 1:
        k = st.slider("Iteration", 0, len(iterates) - 1, len(iterates) - 1,
                      key=f"cons_k_{rec['id']}")
    else:
        k = 0
        st.caption("Converged in a single iteration — showing it.")
    snap = iterates[k]
    st.caption(f"Iteration {k}: objective = {snap['obj']:.5f}, "
               f"iterate change ‖Δ[H;Q;z]‖ = {snap['err']:.4f}"
               + (f", max head-bound slack = {snap['slack']:.4g}" if snap.get("slack") else ""))
    for row in iteration_report(wdn, snap):
        c1, c2, c3 = st.columns([2.6, 1, 0.6])
        with c1:
            st.latex(row["latex"])
        c2.metric(row["family"], f"{row['worst']:.3g}",
                  help="worst |residual| (=) or worst violation (≤) across all "
                       "elements and hours at this iteration")
        c3.markdown("### " + ("✅" if row["ok"] else "⚠️"))
    st.caption("Equalities are enforced **on the linearized physics**, so their "
               "residuals here are measured against the *nonlinear* truth where "
               "possible (pump power) or the iteration's own linearization (pipes) — "
               "the gap that shrinks is exactly the successive-approximation error. "
               "Inequalities (bounds, big-M gating) must hold at every iteration.")


def _render_case(st, rec) -> None:
    case = rec["case"]
    if not case.converged and not np.isfinite(case.owf_cost):
        st.error(f"No feasible solution ({case.note}). Try the recommended mode.")
        return
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("EPANET cost (rules)", f"{case.epanet_cost:.4f}",
              help="EPANET's own tank-level rules — a DIFFERENT schedule; cost baseline only.")
    c2.metric("C-OWF cost", f"{case.owf_cost:.4f}", delta=f"{-case.savings_pct:.1f}%",
              delta_color="inverse", help="Optimized schedule, true nonlinear energy cost.")
    c3.metric("Max head error", f"{case.max_dhead:.3f} ft",
              help="SAME schedule imposed in EPANET and replayed.")
    c4.metric("Min junction pressure", f"{case.min_pressure:.1f} ft",
              help="> 0 means hydraulically feasible in EPANET.")
    c5.metric("Solve time", f"{case.elapsed:.0f} s", help=f"{case.n_iter} iters · {case.solver}")
    if case.note:
        st.caption(f"note: {case.note}")

    has_prv = rec.get("prv_fig") is not None
    has_teach = rec.get("iterates") and rec.get("teach_wdn") is not None
    tab_names = ["Network map", "Flow animation", "Schedule", "Flows", "Heads",
                 "Convergence", "Error", "Solver log", "Download"]
    # conditional tabs appended AFTER the fixed indices 0..8, order tracked below
    idx = 9
    if has_prv:
        tab_names.append("🔻 PRV"); prv_idx = idx; idx += 1
    tab_names.append("🔎 Inspector"); insp_idx = idx; idx += 1
    tab_names.append("📈 Pump curves"); curve_idx = idx; idx += 1
    if has_teach:
        tab_names.append("📐 Constraints"); cons_idx = idx; idx += 1
    tabs = st.tabs(tab_names)
    with tabs[insp_idx]:
        _render_inspector(st, rec)
    with tabs[curve_idx]:
        _render_pump_curves(st, rec)
    if has_teach:
        with tabs[cons_idx]:
            _render_constraints(st, rec)
    if has_prv:
        with tabs[prv_idx]:
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
    teach = st.checkbox("📐 Teaching mode — record every iteration",
                        value=False, key="w_teach",
                        help="Stores each successive-linearization iterate so the "
                             "result gains a **Constraints** tab: watch how every "
                             "equality residual and inequality margin evolves per "
                             "iteration. Best on the small networks.")
    b1, b2 = st.columns([1, 1])
    run = b1.button("💧  Run water case", type="primary", use_container_width=True,
                    disabled=solver not in avail, key="w_run")
    if b2.button("Clear water history", use_container_width=True, key="w_clear"):
        st.session_state.water_records = []
        st.rerun()
    return net, mode, price, solver, run, vsp, prv, teach


def render_water(st) -> None:
    if "water_records" not in st.session_state:
        st.session_state.water_records = []
    net, mode, price, solver, run, vsp, prv, teach = _controls(st)
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
                # verbose=True turns on the solver's own log (HiGHS/MOSEK presolve,
                # primal/dual simplex, branch-and-bound); those are C-level writes,
                # so capture fd 1/2 as well as Python stdout.
                def _job():
                    with contextlib.redirect_stdout(log_buf):
                        return run_case(net, mode, price, None, plot=True,
                                        outdir="outputs", verbose=True,
                                        solver=solver, vsp=vsp, prv=prv,
                                        teach=teach)
                (case, wdn, result), solver_log = _run_capturing_all(_job)
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
            # pump-curve data (H = h0 omega^2 - r q^v + the run's operating points)
            pump_curve = None
            if result is not None and result.flows is not None:
                p = wdn.pump
                pump_curve = dict(
                    ids=[str(wdn.raw.link_name_id[i]) for i in wdn.raw.link_pump_index],
                    h0=p.h0.copy(), r=p.r_m.copy(), v=p.v_m.copy(),
                    maxf=p.max_flow.copy(), is_vsp=p.is_vsp.copy(),
                    q_op=(wdn.M.Lambda @ result.flows[:, :wdn.time]),
                    H_op=-((wdn.M.Lambda @ wdn.M.Pi.T) @ result.heads[:, :wdn.time]),
                    onoff=np.round(result.onoff[:, :wdn.time]),
                    speed=(None if result.speed is None
                           else result.speed[:, :wdn.time]))
            st.session_state.water_records.append({
                "id": run_id, "case": case,
                # python-level prints + the C-level solver log (HiGHS/MOSEK
                # primal-dual iterations); keep the tail if it's enormous
                "log": (log_buf.getvalue() + "\n" + "=" * 60
                        + " SOLVER LOG (HiGHS/MOSEK/SCIP) " + "=" * 60 + "\n"
                        + solver_log)[-400_000:],
                "plots": plots,
                "map_data": extract_map_data(wdn, result),
                "prv_fig": prv_fig,
                "pump_curve": pump_curve,
                # teaching mode: the iterate snapshots + the WDN they refer to
                "iterates": (result.iterates if result is not None
                             and getattr(result, "iterates", None) else None),
                "teach_wdn": (wdn if teach and result is not None
                              and getattr(result, "iterates", None) else None),
                # for the DSO hand-off: this run's optimized pump electrical power
                "net": net, "horizon": wdn.time,
                "pump_link_ids": [str(wdn.raw.link_name_id[i])
                                  for i in wdn.raw.link_pump_index],
                "ppump_true": (np.abs(np.asarray(result.ppump_true)[:, :wdn.time])
                               if result is not None and result.flows is not None
                               else None),
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

    # --- transmit schedules to the DSO (Power tab) -----------------------------
    if latest.get("ppump_true") is not None:
        with st.expander("📡 Transmit pump schedules to the DSO (Power tab)"):
            st.caption("The water operator publishes its pump electrical load to the "
                       "distribution system operator. Choose which schedule(s) to "
                       "transmit — the Power tab acknowledges them and solves its "
                       "OPF with the transmitted load (both are compared side by "
                       "side if you send both).")
            opts = ["EPANET rule-based (baseline)",
                    f"C-OWF optimized (run {latest['id']})"]
            chosen = st.multiselect("Schedules to transmit", opts, default=opts,
                                    key="w_tx_sel")
            if st.button("📡 Transmit to Power tab", key="w_tx_btn",
                         disabled=not chosen):
                from .power_section import _epanet_pump_kw
                schedules = {}
                if any(c.startswith("EPANET") for c in chosen):
                    pk, _, _ = _epanet_pump_kw(latest["net"])
                    schedules["EPANET rules"] = pk[:, :latest["horizon"]]
                if any(c.startswith("C-OWF") for c in chosen):
                    schedules["C-OWF optimized"] = latest["ppump_true"]
                st.session_state["dso_handoff"] = dict(
                    net=latest["net"], label=NETWORKS[latest["net"]].name,
                    horizon=latest["horizon"], pump_ids=latest["pump_link_ids"],
                    schedules=schedules, case=latest["case"].label)
                st.success(f"Transmitted **{len(schedules)} schedule(s)** to the DSO "
                           f"— open **⚡ Power**, tick *Impose water pump load*, and "
                           f"the transmitted source is pre-selected.")

    st.subheader("Session comparison")
    df = _session_df(records)
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("Download session table (CSV)", data=_csv_bytes(df.set_index("run")),
                       file_name="owf_session_comparison.csv", mime="text/csv", key="w_dlsess")
    st.caption("**Cost** columns compare *different* schedules (EPANET rules vs C-OWF). "
               "**Replay** columns use the *same* schedule imposed back in EPANET — "
               "linearization fidelity, not the cost operating point.")
