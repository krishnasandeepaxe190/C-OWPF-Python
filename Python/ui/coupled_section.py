"""Coupled tab: joint water-power C-OWPF vs the decoupled hand-off, cost compared.

Decoupled : optimize the water schedule alone, then impose its pump load on the
            feeder and dispatch PV reactive (PDN OPF).  Water-optimal, grid-blind.
Coupled   : optimize the schedule with the feeder voltage constraints in the loop
            (voltage-aware schedule search + trust-region MILP).

Both are scored on the paper's objective -- pump-energy cost -- plus the feeder
voltage feasibility, so the trade-off the coupling buys is explicit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as _st

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

    run = st.button("🔗  Run coupled vs decoupled", type="primary", key="cpl_run")
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
    with st.spinner(f"Solving decoupled and coupled ({effort} search, {eta})..."):
        wdn, pdn = setup_coupled(net, cc, time=24, price_choice=price)
        pdn.limit_pv(pv_count)

        # --- decoupled: water schedule, then imposed on the feeder --------------
        # Thorough uses the full water-optimal schedule; Fast uses EPANET's own
        # (skips the slow multi-candidate water search on big networks).
        if fast:
            dec_sched = epanet_default_onoff(wdn)
        else:
            try:
                _, w_info = water_optimize(wdn, verbose=False)
                dec_sched = w_info["schedule"]
            except Exception:
                dec_sched = epanet_default_onoff(wdn)
        dec = solve_coupled_schedule(wdn, pdn, cc, dec_sched)

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
        cpl, cpl_info = optimize_coupled_schedule(wdn, pdn, cc, **opt_kw)

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
        prv_fig=prv_fig)
    _render_coupled_result(st)


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
        return {"metric": name, "decoupled (water-only)": fmt.format(d),
                "coupled (C-OWPF)": fmt.format(c),
                "Δ": fmt.format(c - d)}
    cmp = pd.DataFrame([
        _r("pump-energy cost ($)", dec.energy_cost, cpl.energy_cost),
        _r("network-loss cost ($)", dec.loss_cost, cpl.loss_cost),
        _r("TOTAL cost ($)", dec_total, cpl_total),
        _r("min voltage (pu)", dec.voltage.min(), cpl.voltage.min()),
        _r("voltage violation (pu)", dec.v_violation, cpl.v_violation),
        _r("true loss, Z-bus (kW·h)", dec_loss, cpl_loss, "{:,.0f}"),
        _r("water head slack (ft)", dec.water_max_slack, cpl.water_max_slack, "{:.3f}"),
    ])
    st.dataframe(cmp, use_container_width=True, hide_index=True)
    st.caption("Objective (paper 33d): **pump energy + priced network loss**, both at the "
               "WDN electricity price. Decoupled optimizes the water schedule blind to the "
               "grid then imposes its pump load; coupled co-optimizes pump timing and PV "
               "reactive to cut loss and hold voltages — the Δ is the dollar saving.")

    T = cpl.voltage.shape[1]
    h0 = min(12, T - 1)
    tab_names = ["Feeder map", "Voltage profile", "Pump schedule",
                 "PV reactive", "Search trace"]
    if R.get("prv_fig") is not None:
        tab_names.append("PRV")
    tabs = st.tabs(tab_names)
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
