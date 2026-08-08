"""Power tab: standalone distribution-network reactive OPF with DER controls.

Flow: pick a feeder + DER setup (PV count/sizing, pump connection buses), optionally
impose the water schedule's pump load, solve the reactive OPF, then verify with the
nonlinear Z-bus -> true voltages and true loss (with vs without VAr support).
"""
from __future__ import annotations

import time as _time

import numpy as np
import pandas as pd
import streamlit as _st

from pdn import FEEDERS, PDN, solve_pdn_opf, pump_load_to_bus
from coupled.config import LOAD_PROFILE_24
from .theme import POWER, section_header
from . import pdn_plots as P


def _feeder_choice_labels():
    return {k: f"{FEEDERS[k]['label']} — {FEEDERS[k]['N']} buses, "
               f"{int(np.sum(FEEDERS[k]['pv']))} PV sites" for k in FEEDERS}


@_st.cache_data(show_spinner=False)
def _epanet_pump_kw(net: int):
    """EPANET (rule-based) pump electrical power for a water net.

    Returns ``(ppump, ids, Tw)`` at the water net's OWN horizon ``Tw`` -- the PDN
    OPF then runs over exactly that horizon so pump load and feeder load line up.
    """
    from owf.config import SolverConfig
    from owf.network import setup as setup_wdn
    from owf.epanet_io import run_epanet
    from owf.solver import _true_pump_power
    wdn = setup_wdn(SolverConfig(net_num=net))
    flows_ep, _, _, _ = run_epanet(wdn.raw)
    ppump = np.abs(_true_pump_power(wdn, flows_ep[:wdn.time].T))   # (Pu x Tw) kW
    ids = [str(wdn.raw.link_name_id[i]) for i in wdn.raw.link_pump_index]
    return ppump, ids, wdn.time


def render_power(st) -> None:
    section_header(st, POWER, "Distribution-network reactive-power OPF",
                   "Dispatch PV reactive setpoints on a real feeder, then verify with "
                   "the nonlinear Z-bus for true voltages and true loss.")

    cfg = st.columns([1.1, 1, 1])
    labels = _feeder_choice_labels()
    feeder = cfg[0].selectbox("Feeder", list(FEEDERS), format_func=labels.get,
                              key="pwr_feeder")
    fmeta = FEEDERS[feeder]
    n_pv_sites = int(np.sum(fmeta["pv"]))

    pv_sizing = cfg[1].slider("PV sizing  Spv = k · Ppv,max", 1.0, 1.6, 1.2, 0.05,
                              key="pwr_size",
                              help="Inverter apparent-power rating as a multiple of "
                                   "the bus's PV active. Larger k -> more reactive headroom.")
    pv_count = cfg[2].slider("Active PV sites", 0, n_pv_sites, n_pv_sites, 1,
                             key="pwr_npv", help="Use the k largest PV sites.")

    lim = st.columns([1, 1, 1, 1.4])
    vmin = lim[0].number_input("Vmin (pu)", 0.85, 1.0, 0.95, 0.01, key="pwr_vmin")
    vmax = lim[1].number_input("Vmax (pu)", 1.0, 1.15, 1.05, 0.01, key="pwr_vmax")
    daily = lim[2].checkbox("Daily load shape", True, key="pwr_daily",
                            help="Vary the feeder base load over 24 h (else static nominal).")
    couple = lim[3].checkbox("Impose water pump load (decoupled hand-off)", False,
                             key="pwr_couple",
                             help="Add a water network's EPANET pump power as bus load.")

    pump_load = None
    pump_buses = np.array([], int)
    horizon = 24                       # standalone feeder day; matched to pumps when coupling
    if couple:
        cc = st.columns([1, 2])
        net = cc[0].selectbox("Water network", [8, 3, 11, 36, 97],
                              format_func=lambda n: {8: "8-node", 3: "3-node", 11: "Net1",
                                                     36: "Net2", 97: "Net3"}[n], key="pwr_net")
        try:
            ppump_kw, pump_ids, horizon = _epanet_pump_kw(net)
            cc[0].caption(f"OPF horizon matched to pump schedule: **{horizon} h**")
        except Exception as exc:
            st.warning(f"Could not get pump load for that network: {exc}")
            ppump_kw, pump_ids = None, []
        if ppump_kw is not None:
            bus_opts = list(range(fmeta["N"]))
            bus_fmt = lambda b: f"bus {fmeta['orig_id'][b]}"
            weakest = int(np.argmin(PDN.build(feeder).model.voltage(
                PDN.build(feeder).model.p_load, PDN.build(feeder).model.q_load)))
            sel = []
            with cc[1]:
                st.caption("Connect each pump to a feeder bus (Xi coupling):")
                pcols = st.columns(min(len(pump_ids), 4) or 1)
                for i, pid in enumerate(pump_ids):
                    b = pcols[i % len(pcols)].selectbox(
                        f"pump {pid}", bus_opts, index=weakest,
                        format_func=bus_fmt, key=f"pwr_pb_{i}")
                    sel.append(int(b))
            pump_buses = np.array(sel, int)

    if not st.button("⚡  Solve PDN OPF", type="primary", key="pwr_run"):
        st.info("Configure the feeder and DER above, then **Solve PDN OPF**.")
        return

    pdn = PDN.build(feeder, pv_sizing=pv_sizing, vmin=vmin, vmax=vmax).limit_pv(pv_count)
    load_shape = LOAD_PROFILE_24 if daily else None
    if couple and pump_buses.size:
        pump_load = pump_load_to_bus(pdn, ppump_kw, pump_buses)

    with st.spinner(f"Solving reactive OPF on {fmeta['label']} over {horizon} h "
                    f"+ Z-bus verification..."):
        t0 = _time.time()
        res = solve_pdn_opf(pdn, T=horizon, pump_load_pu=pump_load, load_shape=load_shape,
                            vmin=vmin, vmax=vmax)
        elapsed = _time.time() - t0

    prev = st.session_state.get("pwr_result") or {}
    st.session_state["pwr_result"] = dict(res=res, feeder=feeder, fmeta=fmeta,
                                          pump_buses=pump_buses, vmin=vmin, vmax=vmax,
                                          elapsed=elapsed,
                                          prev_elapsed=prev.get("elapsed"))
    _render_power_result(st)


def _render_power_result(st) -> None:
    R = st.session_state.get("pwr_result")
    if not R:
        return
    res, fmeta = R["res"], R["fmeta"]
    vmin, vmax = R["vmin"], R["vmax"]

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("True Vmin (Z-bus)", f"{np.nanmin(res.v_nl):.4f} pu",
              help="Worst nonlinear bus voltage across the day.")
    m2.metric("Voltage violation", f"{res.v_violation:.4f} pu",
              delta="within limits" if res.v_violation < 1e-3 else "over limit",
              delta_color="normal" if res.v_violation < 1e-3 else "inverse")
    m3.metric("Daily loss (optimized)", f"{res.loss_kw.sum():.0f} kW·h",
              delta=f"-{res.loss_reduction_pct:.1f}% vs no VAr", delta_color="inverse")
    m4.metric("LinDistFlow vs Z-bus", f"{np.nanmax(np.abs(res.v_lin - res.v_nl)):.4f} pu",
              help="Max linear-model error against the nonlinear replay.")
    el, pel = R.get("elapsed"), R.get("prev_elapsed")
    dt = (f"{100.0 * (el - pel) / pel:+.0f}% vs last run"
          if el is not None and pel else None)
    m5.metric("Solve time", f"{el:.1f} s" if el is not None else "—", delta=dt,
              delta_color="inverse",
              help="OPF + Z-bus verification wall-clock; Δ% vs the previous Power run.")

    T = res.v_nl.shape[1]
    h0 = min(12, T - 1)
    tabs = st.tabs(["Feeder map", "Voltage profile", "Voltage heatmap",
                    "PV reactive", "Loss", "Setpoints (CSV)"])
    with tabs[0]:
        hr = st.slider("Hour", 0, T - 1, h0, key="pwr_map_hr")
        st.plotly_chart(P.feeder_map(fmeta, res.v_nl, hr, res.pv_buses,
                                     R["pump_buses"], vmin, vmax),
                        use_container_width=True, key="pwr_map")
    with tabs[1]:
        hr = st.slider("Hour", 0, T - 1, h0, key="pwr_prof_hr")
        st.plotly_chart(P.voltage_profile(res.v_lin, res.v_nl, hr, vmin, vmax,
                                          fmeta["orig_id"]),
                        use_container_width=True, key="pwr_prof")
    with tabs[2]:
        st.plotly_chart(P.voltage_heatmap(res.v_nl, vmin, vmax),
                        use_container_width=True, key="pwr_heat")
    with tabs[3]:
        fig = P.pv_reactive_chart(res.q_pv, res.p_pv, res.pv_buses, fmeta["orig_id"])
        if fig:
            st.plotly_chart(fig, use_container_width=True, key="pwr_pv")
        else:
            st.info("No PV on this feeder configuration.")
    with tabs[4]:
        st.plotly_chart(P.loss_chart(res.loss_base_kw, res.loss_kw),
                        use_container_width=True, key="pwr_loss")
        st.caption(f"Reactive support cuts true daily loss from "
                   f"**{res.loss_base_kw.sum():.0f}** to **{res.loss_kw.sum():.0f} kW·h** "
                   f"({res.loss_reduction_pct:.1f}%).")
    with tabs[5]:
        if res.q_pv.size:
            df = pd.DataFrame(res.q_pv, index=[f"bus {fmeta['orig_id'][b]}" for b in res.pv_buses],
                              columns=[f"h{t}" for t in range(res.q_pv.shape[1])]).round(4)
            st.dataframe(df, use_container_width=True)
            st.download_button("Download reactive setpoints (CSV)",
                               df.to_csv().encode(), file_name=f"{R['feeder']}_qpv.csv",
                               mime="text/csv", key="pwr_dl")
        else:
            st.info("No PV setpoints.")
