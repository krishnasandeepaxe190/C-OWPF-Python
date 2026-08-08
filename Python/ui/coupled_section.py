"""Coupled tab: joint water-power C-OWPF vs the decoupled hand-off, cost compared.

Decoupled : optimize the water schedule alone, then impose its pump load on the
            feeder and dispatch PV reactive (PDN OPF).  Water-optimal, grid-blind.
Coupled   : optimize the schedule with the feeder voltage constraints in the loop
            (voltage-aware schedule search + trust-region MILP).

Both are scored on the paper's objective -- pump-energy cost -- plus the feeder
voltage feasibility, so the trade-off the coupling buys is explicit.
"""
from __future__ import annotations

import time as _time

import numpy as np
import pandas as pd
import streamlit as _st

from owf.config import (DEFAULT_SOLVER, SOLVER_CHOICES, available_solvers)
from pdn import FEEDERS, PDN
from coupled import (setup as setup_coupled, CoupledConfig, solve_coupled_schedule,
                     solve_coupled_epanet, optimize_coupled_schedule, validate_coupled,
                     coupled_loss_kwh)
from coupled.config import LOAD_PROFILE_24
from owf.warmstart import optimize_schedule as water_optimize, epanet_default_onoff
from .theme import COUPLED, section_header
from . import pdn_plots as P

NET_LABELS = {8: "8-node", 108: "8-node + PRV", 3: "3-node", 11: "Net1",
              36: "Net2", 97: "Net3", 126: "BWSN (large)"}


@_st.cache_data(show_spinner=False)
def _probe_pump_ids(net: int):
    """Pump ids for a water net (cached so switching widgets doesn't re-run EPANET)."""
    from owf.config import SolverConfig
    from owf.network import setup as setup_wdn
    wdn = setup_wdn(SolverConfig(net_num=net, time=24))
    return [str(wdn.raw.link_name_id[i]) for i in wdn.raw.link_pump_index]


def render_coupled(st) -> None:
    section_header(st, COUPLED, "Coupled Optimal Water-Power Flow (C-OWPF)",
                   "Co-optimize the pump schedule with the feeder: objective is the "
                   "paper's pump-energy cost; PV reactive holds voltages.")

    r1 = st.columns([1, 1, 1, 1])
    net = r1[0].selectbox("Water network", [8, 108, 3, 11, 36, 97, 126], format_func=NET_LABELS.get,
                          key="cpl_net",
                          help="8-node / 3-node solve in seconds; Net1/Net2 take longer; "
                               "Net3 is heavy (large network + many schedule evaluations).")
    if net in (11, 36):
        r1[0].caption("⏳ Net1/Net2: ~1–3 min.")
    elif net == 97:
        r1[0].caption("⏳ Net3: several minutes (large multi-pump network).")
    feeder = r1[1].selectbox("Feeder", list(FEEDERS),
                             format_func=lambda k: FEEDERS[k]["label"], key="cpl_feeder")
    price = r1[2].selectbox("Electricity price", [1, 0],
                            format_func=lambda p: "Time-of-use" if p else "Flat", key="cpl_price")
    n_pv_sites = int(np.sum(FEEDERS[feeder]["pv"]))
    pv_count = r1[3].slider("Active PV sites", 0, n_pv_sites, n_pv_sites, 1,
                            key="cpl_npv", help="Use the k largest PV sites.")

    r2 = st.columns([1, 1, 1, 1.6])
    pv_sizing = r2[0].slider("PV sizing Spv=k·Ppv", 1.0, 1.6, 1.2, 0.05, key="cpl_size")
    vmin = r2[1].number_input("Vmin (pu)", 0.85, 1.0, 0.95, 0.01, key="cpl_vmin")
    vmax = r2[2].number_input("Vmax (pu)", 1.0, 1.15, 1.05, 0.01, key="cpl_vmax")
    heavy = (net in (97, 126)) or (feeder == "sb128") or (net in (11, 36))
    effort = r2[3].selectbox("Search effort", ["Fast", "Thorough"],
                             index=0 if heavy else 1, key="cpl_effort",
                             help="Fast: EPANET baseline + light polish (recommended for "
                                  "Net3 / SB-128). Thorough: full water-optimal baseline + "
                                  "deep polish.")

    fmeta = FEEDERS[feeder]
    # pump -> bus connection
    tmp = PDN.build(feeder)
    vnom = tmp.model.voltage(tmp.model.p_load, tmp.model.q_load)
    weakest = int(np.argmin(vnom))
    # Feeder headroom check: if the feeder's own base load already sits below Vmin,
    # no pump schedule or PV reactive dispatch can hold Vmin -- the coupled solve will
    # (honestly) report a voltage violation. Surface this before the user runs it.
    base_vmin = float(vnom.min())
    if base_vmin < vmin:
        st.warning(
            f"⚠️ **Feeder voltage headroom.** {fmeta['label']}'s minimum bus voltage at "
            f"*base load alone* is {base_vmin:.3f} pu — already below Vmin = {vmin:.2f} pu, "
            f"before any pump load is added. This case will report a voltage violation "
            f"regardless of the schedule. To get a feasible coupled case, lower **Vmin** "
            f"(e.g. 0.90), connect the pumps to stronger buses (nearer the substation), "
            f"or add PV (raise Spv / active PV sites).")
    pump_ids = _probe_pump_ids(net)
    with r2[2]:
        st.caption("Pump → feeder bus (Xi coupling):")
        pcols = st.columns(min(len(pump_ids), 4) or 1)
        pump_bus = [int(pcols[i % len(pcols)].selectbox(
            f"pump {pid}", list(range(fmeta["N"])), index=weakest,
            format_func=lambda b: f"bus {fmeta['orig_id'][b]}", key=f"cpl_pb_{i}"))
            for i, pid in enumerate(pump_ids)]

    prv = None
    if net == 108:
        with st.expander("🔻 Pressure-reducing valve (PRV)", expanded=True):
            st.caption("The 8-node+PRV net has one PRV (junction 6 → 9, fixed location). "
                       "Optimal PRV scheduling stops head being wasted across the valve, "
                       "so the pump works less — a lighter feeder load and better "
                       "voltages. Tune the pressure setting h_set:")
            pset = st.slider("PRV pressure setting P_set (psi)", 5.0, 60.0, 20.0, 1.0,
                             key="cpl_prv_pset")
            prv = {"10": float(pset)}

    vsp = None
    with st.expander("⚙️ Variable-speed pumps (VSP)"):
        st.caption("Run selected pumps at variable speed ω ∈ [ω_min, 1]. Reduced speed "
                   "cuts pump energy (∝ ω³) **and** the load the pump imposes on the "
                   "feeder, so it usually lifts voltages too.")
        vsp_sel = st.multiselect("Variable-speed pumps", pump_ids, key="cpl_vsp_sel")
        omin = st.slider("Minimum relative speed ω_min", 0.50, 1.0, 0.80, 0.05,
                         key="cpl_vsp_omin")
        if vsp_sel:
            vsp = {p: (float(omin), 1.0) for p in vsp_sel}
            st.info(f"VSP active on {len(vsp)} pump(s); both the decoupled and coupled "
                    f"solves co-optimize pump speed.")

    rs = st.columns([1, 3])
    avail = available_solvers()
    solver = rs[0].selectbox("MILP solver", SOLVER_CHOICES,
                             index=SOLVER_CHOICES.index(DEFAULT_SOLVER),
                             format_func=lambda s: s + ("" if s in avail else " (not installed)"),
                             key="cpl_solver",
                             help="Solver for every coupled/decoupled MILP-LP step. "
                                  "HiGHS is the bundled default; MOSEK/Gurobi need a license.")

    run = st.button("🔗  Run coupled vs decoupled", type="primary", key="cpl_run",
                    disabled=solver not in avail)
    if not run:
        st.info("Set the water net, feeder and DER, then **Run coupled vs decoupled**.")
        if st.session_state.get("cpl_result"):
            _render_coupled_result(st)
        return

    fast = (effort == "Fast")
    cc = CoupledConfig(feeder=feeder, pump_bus=pump_bus, pv_sizing=pv_sizing,
                       vmin=vmin, vmax=vmax, load_profile=LOAD_PROFILE_24,
                       vsp_pumps=vsp, prv_settings=prv)
    eta = ("~1 min" if (net == 97 and feeder in ("sb128", "sce56")) else
           "up to a minute" if heavy else "a few seconds")
    with st.spinner(f"Solving decoupled and coupled ({effort} search, {eta}, {solver})..."):
        wdn, pdn = setup_coupled(net, cc, time=24, price_choice=price, solver=solver)
        pdn.limit_pv(pv_count)

        # --- decoupled: water schedule, then imposed on the feeder --------------
        # Thorough uses the full water-optimal schedule; Fast uses EPANET's own
        # (skips the slow multi-candidate water search on big networks).
        t0 = _time.time()
        if fast:
            dec_sched = epanet_default_onoff(wdn)
        else:
            try:
                _, w_info = water_optimize(wdn, verbose=False)
                dec_sched = w_info["schedule"]
            except Exception:
                dec_sched = epanet_default_onoff(wdn)
        dec = solve_coupled_schedule(wdn, pdn, cc, dec_sched)
        t_dec = _time.time() - t0

        # --- coupled: voltage-aware joint schedule search ----------------------
        opt_kw = dict(verbose=False, inner_iter=8 if fast else 15,
                      polish=not fast, max_flips=6 if fast else 20, use_milp=not fast)
        if fast and heavy:
            # minimal candidate set on big networks: the tank-safe load-shift plus
            # two price quantiles (load_shift is the feasible saver on high-duty
            # systems like BWSN; the quantiles under-pump and drain tanks there).
            from owf.warmstart import price_threshold_schedules
            allc = price_threshold_schedules(wdn)
            opt_kw["candidates"] = {k: allc[k] for k in
                                    ("load_shift", "cheapest_40pct", "cheapest_60pct")
                                    if k in allc}
        t0 = _time.time()
        cpl, cpl_info = optimize_coupled_schedule(wdn, pdn, cc, **opt_kw)
        t_cpl = _time.time() - t0

        val = validate_coupled(wdn, pdn, cpl, vmin=vmin, vmax=vmax)
        dec_loss = coupled_loss_kwh(pdn, dec)      # true Z-bus loss (kW·h)
        cpl_loss = coupled_loss_kwh(pdn, cpl)
        prv_fig = None
        if getattr(cpl, "prv", None) and wdn.n_valves:
            try:
                from owf.epanet_io import run_epanet
                from .prv_plots import prv_panel
                fl_r, hd_r, _, _ = run_epanet(wdn.raw)   # EPANET rule-based operation
                prv_fig = prv_panel(wdn, cpl.heads, cpl.flows, cpl.prv,
                                    heads_ep=val.water.heads_epanet,
                                    flows_ep=val.water.flows_epanet,
                                    heads_rule=hd_r[: wdn.time].T,
                                    flows_rule=fl_r[: wdn.time].T)
            except Exception:
                prv_fig = None

    st.session_state["cpl_result"] = dict(
        dec=dec, cpl=cpl, cpl_info=cpl_info, val=val, fmeta=fmeta, feeder=feeder,
        pump_bus=np.asarray(pump_bus), vmin=vmin, vmax=vmax,
        pv_buses=pdn.pv_buses, dec_loss=dec_loss, cpl_loss=cpl_loss,
        prv_fig=prv_fig, t_dec=t_dec, t_cpl=t_cpl,
        link_ids=[str(s) for s in wdn.raw.link_name_id],
        node_ids=[str(s) for s in wdn.raw.node_name_id],
        pump_ids=[str(wdn.raw.link_name_id[i]) for i in wdn.raw.link_pump_index],
        water_md=__import__("owf.netmap", fromlist=["extract_map_data"]).extract_map_data(wdn),
        valve_link_ids=[str(wdn.raw.link_name_id[i]) for i in wdn.M.valve_index],
        attrs=dict(
            link=dict(id=[str(s) for s in wdn.raw.link_name_id],
                      kind=[str(k) for k in wdn.raw.link_type],
                      diameter_in=wdn.raw.link_diameter.tolist(),
                      length_ft=wdn.raw.link_length.tolist(),
                      roughness=wdn.raw.link_roughness.tolist()),
            node=dict(id=[str(s) for s in wdn.raw.node_name_id],
                      elevation_ft=wdn.raw.node_elevations.tolist())))
    _render_coupled_result(st)


def _render_solution_tables(st, res, R, prefix: str, title: str) -> None:
    """Full decision variables of one solution: schedule (with speeds), pump power,
    flows and heads -- as labeled tables with CSV downloads."""
    T = res.onoff.shape[1] if res.onoff is not None and res.onoff.size else 0
    if not T:
        st.info("No solution stored.")
        return
    hours = [f"h{t}" for t in range(T)]
    pump_ids = R.get("pump_ids") or [f"pump {i}" for i in range(res.onoff.shape[0])]
    link_ids = R.get("link_ids") or [f"link {i}" for i in range(res.flows.shape[0])]
    node_ids = R.get("node_ids") or [f"node {i}" for i in range(res.heads.shape[0])]
    st.markdown(f"**{title}**")
    sched = np.round(res.onoff)
    if getattr(res, "speed", None) is not None:
        sch = pd.DataFrame(np.round(sched * res.speed[:, :T], 2),
                           index=pump_ids, columns=hours)
        st.caption("Pump schedule — cell = relative speed ω while running (0 = off).")
    else:
        sch = pd.DataFrame(sched.astype(int), index=pump_ids, columns=hours)
        st.caption("Pump on/off schedule (1 = on).")
    st.dataframe(sch, use_container_width=True)
    with st.expander("Pump electrical power (kW, true nonlinear)"):
        st.dataframe(pd.DataFrame(np.round(res.ppump_true[:, :T], 2),
                                  index=pump_ids, columns=hours),
                     use_container_width=True)
    with st.expander(f"Flows (GPM) — {len(link_ids)} links × {T} h"):
        st.dataframe(pd.DataFrame(np.round(res.flows[:, :T], 1),
                                  index=link_ids, columns=hours),
                     use_container_width=True)
    with st.expander(f"Heads (ft) — {len(node_ids)} nodes × {T} h"):
        st.dataframe(pd.DataFrame(np.round(res.heads[:, :T], 2),
                                  index=node_ids, columns=hours),
                     use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.download_button("⬇ schedule.csv", data=sch.to_csv().encode(),
                       file_name=f"{prefix}_schedule.csv", key=f"{prefix}_dl_s")
    c2.download_button("⬇ flows.csv",
                       data=pd.DataFrame(res.flows[:, :T], index=link_ids,
                                         columns=hours).to_csv().encode(),
                       file_name=f"{prefix}_flows.csv", key=f"{prefix}_dl_f")
    c3.download_button("⬇ heads.csv",
                       data=pd.DataFrame(res.heads[:, :T], index=node_ids,
                                         columns=hours).to_csv().encode(),
                       file_name=f"{prefix}_heads.csv", key=f"{prefix}_dl_h")


def _render_coupled_inspector(st, cpl, val, R, fmeta) -> None:
    """Pick feeder buses / lines / water elements and plot their time series:
    V(t) linear vs Z-bus, line P(t) & Q(t), water flow(t) and head(t)."""
    import plotly.graph_objects as go
    from pdn import PDN
    from pdn.lindistflow import branch_flow_matrix
    T = cpl.voltage.shape[1]
    hrs = list(range(T))
    c1, c2 = st.columns(2)
    with c1:
        bus_lbl = [f"bus {fmeta['orig_id'][b]}" for b in range(fmeta["N"])]
        sel_b = st.multiselect("Feeder buses → V(t)", bus_lbl, default=bus_lbl[:1],
                               key="cpl_insp_b")
        if sel_b:
            fig = go.Figure()
            for lbl in sel_b:
                b = bus_lbl.index(lbl)
                fig.add_trace(go.Scatter(x=hrs, y=val.v_nl[b], mode="lines+markers",
                                         name=f"{lbl} (Z-bus)"))
                fig.add_trace(go.Scatter(x=hrs, y=cpl.voltage[b], mode="lines",
                                         line=dict(dash="dot"), name=f"{lbl} (linear)"))
            fig.add_hline(y=R["vmin"], line_dash="dash", line_color="#d62728")
            fig.update_layout(height=330, xaxis_title="hour", yaxis_title="|V| (pu)",
                              legend=dict(orientation="h", y=-0.3, x=0),
                              margin=dict(l=10, r=10, t=25, b=10),
                              plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True, key="cpl_insp_vfig")
        # line P/Q from the subtree matrix: line n carries everything below bus n
        line_lbl = [f"line → bus {fmeta['orig_id'][b]}" for b in range(fmeta["N"])]
        sel_ln = st.multiselect("Feeder lines → P(t), Q(t)", line_lbl,
                                default=line_lbl[:1], key="cpl_insp_ln")
        if sel_ln and cpl.p_net.size:
            m = PDN.build(R["feeder"]).model
            Tmat = branch_flow_matrix(m)
            from pdn.feeders import FEEDERS as _F
            skw = _F[R["feeder"]]["SBase"] / 1000.0
            Pb = Tmat @ cpl.p_net * skw
            Qb = Tmat @ cpl.q_net * skw
            fig = go.Figure()
            for lbl in sel_ln:
                b = line_lbl.index(lbl)
                fig.add_trace(go.Scatter(x=hrs, y=Pb[b], mode="lines+markers",
                                         name=f"{lbl} P (kW)"))
                fig.add_trace(go.Scatter(x=hrs, y=Qb[b], mode="lines",
                                         line=dict(dash="dash"),
                                         name=f"{lbl} Q (kVAr)"))
            fig.update_layout(height=330, xaxis_title="hour",
                              yaxis_title="line flow (kW / kVAr)",
                              legend=dict(orientation="h", y=-0.3, x=0),
                              margin=dict(l=10, r=10, t=25, b=10),
                              plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True, key="cpl_insp_pqfig")
    with c2:
        link_ids = R.get("link_ids") or []
        node_ids = R.get("node_ids") or []
        sel_l = st.multiselect("Water links → flow (GPM)", link_ids,
                               default=(R.get("pump_ids") or [])[:1], key="cpl_insp_wl")
        if sel_l and cpl.flows is not None and cpl.flows.size:
            fig = go.Figure()
            for lbl in sel_l:
                i = link_ids.index(lbl)
                fig.add_trace(go.Scatter(x=hrs, y=cpl.flows[i, :T],
                                         mode="lines+markers", name=f"link {lbl}"))
            fig.update_layout(height=330, xaxis_title="hour", yaxis_title="flow (GPM)",
                              legend=dict(orientation="h", y=-0.3, x=0),
                              margin=dict(l=10, r=10, t=25, b=10),
                              plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True, key="cpl_insp_wffig")
        sel_n = st.multiselect("Water nodes → head (ft)", node_ids, default=[],
                               key="cpl_insp_wn")
        if sel_n and cpl.heads is not None and cpl.heads.size:
            fig = go.Figure()
            for lbl in sel_n:
                i = node_ids.index(lbl)
                fig.add_trace(go.Scatter(x=hrs, y=cpl.heads[i, :T],
                                         mode="lines+markers", name=f"node {lbl}"))
            fig.update_layout(height=330, xaxis_title="hour", yaxis_title="head (ft)",
                              legend=dict(orientation="h", y=-0.3, x=0),
                              margin=dict(l=10, r=10, t=25, b=10),
                              plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, use_container_width=True, key="cpl_insp_whfig")
    st.caption("Both systems, one inspector: feeder V(t) (linear vs Z-bus truth), "
               "line active/reactive flows, water flows and heads — pick any elements "
               "and compare across the coupling.")


def _render_coupled_result(st) -> None:
    R = st.session_state.get("cpl_result")
    if not R:
        return
    dec, cpl, info, val = R["dec"], R["cpl"], R["cpl_info"], R["val"]
    fmeta, vmin, vmax = R["fmeta"], R["vmin"], R["vmax"]

    dec_loss, cpl_loss = R.get("dec_loss", float("nan")), R.get("cpl_loss", float("nan"))
    dec_total, cpl_total = dec.total_cost, cpl.total_cost
    save = dec_total - cpl_total                       # dollar savings vs decoupled
    m = st.columns(5)
    m[0].metric("Coupled total cost ($)", f"{cpl_total:.4f}",
                delta=f"-${save:.4f} vs decoupled" if np.isfinite(save) else None,
                delta_color="inverse", help="Pump energy + priced network loss (paper 33d).")
    m[1].metric("Pump / loss cost ($)", f"{cpl.energy_cost:.4f} / {cpl.loss_cost:.4f}",
                help="Pump-energy cost and priced network-loss cost.")
    m[2].metric("Coupled Vmin", f"{cpl.voltage.min():.4f} pu",
                delta="feasible" if cpl.v_violation < 1e-3 else f"viol {cpl.v_violation:.3f}",
                delta_color="normal" if cpl.v_violation < 1e-3 else "inverse")
    m[3].metric("Water valid. (EPANET)", f"{val.water.max_abs_head:.3f} ft",
                help="Max head error, optimized schedule re-run in EPANET.")
    m[4].metric("Power valid. (Z-bus)", f"{val.v_err_max:.4f} pu",
                help="Max linear-vs-nonlinear voltage error.")

    # ---- cost / voltage / loss comparison table ----
    st.markdown("##### Decoupled vs coupled")
    def _r(name, d, c, fmt="{:.4f}"):
        if np.isfinite(d) and np.isfinite(c) and abs(d) > 1e-12:
            pct = 100.0 * (c - d) / abs(d)
            pct_s = f"{'▼' if pct < 0 else '▲'} {abs(pct):.1f}%"
        else:
            pct_s = "—"
        return {"metric": name, "decoupled (water-only)": fmt.format(d),
                "coupled (C-OWPF)": fmt.format(c),
                "Δ": fmt.format(c - d), "Δ %": pct_s}
    cmp = pd.DataFrame([
        _r("pump-energy cost ($)", dec.energy_cost, cpl.energy_cost),
        _r("network-loss cost ($)", dec.loss_cost, cpl.loss_cost),
        _r("TOTAL cost ($)", dec_total, cpl_total),
        _r("min voltage (pu)", dec.voltage.min(), cpl.voltage.min()),
        _r("voltage violation (pu)", dec.v_violation, cpl.v_violation),
        _r("true loss, Z-bus (kW·h)", dec_loss, cpl_loss, "{:,.0f}"),
        _r("water head slack (ft)", dec.water_max_slack, cpl.water_max_slack, "{:.3f}"),
        _r("solve time (s)", R.get("t_dec", float("nan")),
           R.get("t_cpl", float("nan")), "{:.1f}"),
    ])
    st.dataframe(cmp, use_container_width=True, hide_index=True)
    st.caption("Objective (paper 33d): **pump energy + priced network loss**, both at the "
               "WDN electricity price. Decoupled optimizes the water schedule blind to the "
               "grid then imposes its pump load; coupled co-optimizes pump timing and PV "
               "reactive to cut loss and hold voltages — the Δ is the dollar saving. "
               "**Solve time**: decoupled = its schedule + one fixed-schedule coupled LP; "
               "coupled = the full voltage-aware schedule search (baseline, candidates, "
               "trust-region MILP, polish) — the Δ% is the computational price of coupling.")

    T = cpl.voltage.shape[1]
    h0 = min(12, T - 1)
    tab_names = ["🔗 Coupling map", "Feeder map", "Voltage profile", "Pump schedule",
                 "PV reactive", "Search trace", "Decoupled solution", "🔎 Inspector"]
    if R.get("prv_fig") is not None:
        tab_names.append("PRV")
    tabs = st.tabs(tab_names)
    with tabs[0]:
        if R.get("water_md") is not None:
            o1, o2, o3 = st.columns(3)
            show_n = o1.checkbox("Node labels", value=False, key="cpl_map_nlab")
            show_b = o2.checkbox("Bus labels", value=False, key="cpl_map_blab")
            show_c = o3.checkbox("Coupling labels", value=False, key="cpl_map_clab")
            st.plotly_chart(
                P.coupled_network_map(R["water_md"], fmeta, R["pump_bus"],
                                      R.get("pump_ids"), R.get("pv_buses"),
                                      R.get("valve_link_ids"),
                                      show_node_labels=show_n,
                                      show_bus_labels=show_b,
                                      show_coupling_labels=show_c),
                use_container_width=True, key="cpl_couplemap")
            st.caption("The interdependent energy systems: pumps (red links) in the "
                       "water network draw their electrical power from the feeder "
                       "buses they connect to (dashed red = the Ξ coupling). PRVs are "
                       "purple; PV inverters are stars. Hover any element for its id; "
                       "toggle the label checkboxes to annotate the figure.")
            attrs = R.get("attrs")
            if attrs:
                with st.expander("🗂 Element attributes (nodes & sections)"):
                    a1, a2 = st.columns(2)
                    with a1:
                        ln = st.selectbox("Section (link)", attrs["link"]["id"],
                                          key="cpl_attr_l")
                        i = attrs["link"]["id"].index(ln)
                        st.table(pd.DataFrame({
                            "attribute": ["kind", "diameter (in)", "length (ft)",
                                          "roughness C"],
                            "value": [attrs["link"]["kind"][i],
                                      attrs["link"]["diameter_in"][i],
                                      attrs["link"]["length_ft"][i],
                                      attrs["link"]["roughness"][i]]}))
                    with a2:
                        nd = st.selectbox("Node", attrs["node"]["id"], key="cpl_attr_n")
                        j = attrs["node"]["id"].index(nd)
                        st.table(pd.DataFrame({
                            "attribute": ["elevation (ft)"],
                            "value": [attrs["node"]["elevation_ft"][j]]}))
        else:
            st.info("Re-run the case to build the coupling map.")
    tabs = list(tabs)[1:]     # drop the map tab so the code below keeps indices 0..5
                              # (list() first: st.tabs' return is not list-concatable)
    with tabs[5]:
        _render_solution_tables(st, dec, R, prefix="dec",
                                title="Decoupled (water-only) solution")
    with tabs[6]:
        _render_coupled_inspector(st, cpl, val, R, fmeta)
    if R.get("prv_fig") is not None:
        with tabs[-1]:
            st.plotly_chart(R["prv_fig"], use_container_width=True, key="cpl_prv_fig")
            st.caption("Optimal PRV scheduling in the coupled problem: when the valve "
                       "is **active** it holds the downstream zone exactly at h_set "
                       "instead of over-pressurizing it, so less pump head (and less "
                       "electrical power) is needed — a lighter feeder load, which "
                       "shows up as lower loss and better voltages in the table above. "
                       "Dotted line = EPANET's own **rule-based** regulation (at the "
                       ".inp setting): if your h_set is higher, the extra downstream "
                       "pressure costs more pump energy than the rules — the cost "
                       "delta buys the pressure target you chose.")
    with tabs[0]:
        hr = st.slider("Hour", 0, T - 1, h0, key="cpl_map_hr")
        st.plotly_chart(P.feeder_map(fmeta, val.v_nl, hr, R["pv_buses"], R["pump_bus"],
                                     vmin, vmax), use_container_width=True, key="cpl_map")
    with tabs[1]:
        hr = st.slider("Hour", 0, T - 1, h0, key="cpl_prof_hr")
        st.plotly_chart(P.voltage_profile(cpl.voltage, val.v_nl, hr, vmin, vmax,
                                          fmeta["orig_id"]),
                        use_container_width=True, key="cpl_prof")
    with tabs[2]:
        sched = np.round(cpl.onoff)
        pump_labels = [f"pump {i}" for i in range(sched.shape[0])]
        if getattr(cpl, "speed", None) is not None:
            # VSP: cell value = relative speed omega while running (0 = off)
            vals = np.round(sched * cpl.speed[:, :sched.shape[1]], 2)
            df = pd.DataFrame(vals, index=pump_labels,
                              columns=[f"h{t}" for t in range(sched.shape[1])])
            st.dataframe(df, use_container_width=True)
            st.caption("Coupled pump schedule — cell value = **relative speed ω** "
                       "while the pump runs (1.0 = full speed, 0 = off). "
                       "Fixed-speed pumps show 1.0 when on.")
        else:
            df = pd.DataFrame(sched.astype(int), index=pump_labels,
                              columns=[f"h{t}" for t in range(sched.shape[1])])
            st.dataframe(df, use_container_width=True)
            st.caption("Coupled pump on/off schedule (1 = on).")
    with tabs[3]:
        fig = P.pv_reactive_chart(cpl.pv_q, cpl.pv_p, cpl.pv_buses, fmeta["orig_id"])
        if fig:
            st.plotly_chart(fig, use_container_width=True, key="cpl_pv")
        else:
            st.info("No PV on this feeder.")
    with tabs[4]:
        tr = pd.DataFrame(info["trace"], columns=["candidate", "pump cost", "V violation"])
        st.dataframe(tr.round(4), use_container_width=True, hide_index=True)
        st.caption(f"Trust-region MILP: {info.get('trust_milp')}  ·  1-opt flips: "
                   f"{info.get('n_flips')}")
