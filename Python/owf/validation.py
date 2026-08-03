"""Validate the OWF solution against EPANET hydraulics (ports the tail of
Main_OWF_IEEE_ACCESS.m): error norms on heads, pipe flows and pump flows, plus
a true-power comparison.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from epyt import epanet

from .epanet_io import run_epanet
from .network import WDN
from .solver import OWFResult, _true_pump_power


@dataclass
class ValidationReport:
    error_heads: float
    error_pipe_flows: float
    error_pump_flows: float
    pump_power_epanet: np.ndarray
    pump_power_opt_true: np.ndarray
    heads_epanet: np.ndarray
    flows_epanet: np.ndarray

    def summary(self) -> str:
        return (
            "EPANET validation (Euclidean error norms):\n"
            f"  heads      : {self.error_heads:.6g}\n"
            f"  pipe flows : {self.error_pipe_flows:.6g}\n"
            f"  pump flows : {self.error_pump_flows:.6g}\n"
            f"  total pump power (EPANET) : {self.pump_power_epanet.sum():.4f} kW\n"
            f"  total pump power (opt,true): {self.pump_power_opt_true.sum():.4f} kW"
        )


def validate(wdn: WDN, result: OWFResult) -> ValidationReport:
    flows_ep, heads_ep, _, _ = run_epanet(wdn.raw)
    T = wdn.time
    steps = min(T, flows_ep.shape[0])

    flows_ep_t = flows_ep[:steps, :].T   # (L x steps)
    heads_ep_t = heads_ep[:steps, :].T   # (N x steps)

    flows_opt = result.flows[:, :steps]
    heads_opt = result.heads[:, :steps]

    pipe_ep = wdn.M.Pi_prime @ flows_ep_t
    pump_ep = wdn.M.Lambda @ flows_ep_t
    pipe_opt = wdn.M.Pi_prime @ flows_opt
    pump_opt = wdn.M.Lambda @ flows_opt

    return ValidationReport(
        error_heads=float(np.linalg.norm(heads_ep_t - heads_opt)),
        error_pipe_flows=float(np.linalg.norm(pipe_ep - pipe_opt)),
        error_pump_flows=float(np.linalg.norm(pump_ep - pump_opt)),
        pump_power_epanet=_true_pump_power(wdn, flows_ep_t),
        pump_power_opt_true=_true_pump_power(wdn, flows_opt),
        heads_epanet=heads_ep_t,
        flows_epanet=flows_ep_t,
    )


@dataclass
class ScheduleValidationReport:
    """Result of imposing the optimized pump schedule back into EPANET
    (reproduces the paper's Fig. 4 hydraulic-feasibility check)."""

    max_abs_head: float
    max_abs_pipe_flow: float
    max_abs_pump_flow: float
    norm_head: float
    norm_pipe_flow: float
    norm_pump_flow: float
    heads_epanet: np.ndarray   # (N x T)  EPANET heads under the imposed schedule
    flows_epanet: np.ndarray   # (L x T)  EPANET flows under the imposed schedule

    def summary(self) -> str:
        return (
            "EPANET schedule-imposed validation (optimized pump schedule re-run in EPANET):\n"
            f"  max |dHead|      : {self.max_abs_head:.6g} ft\n"
            f"  max |dPipeFlow|  : {self.max_abs_pipe_flow:.6g} GPM\n"
            f"  max |dPumpFlow|  : {self.max_abs_pump_flow:.6g} GPM\n"
            f"  ||dHead||        : {self.norm_head:.6g}\n"
            f"  ||dPipeFlow||    : {self.norm_pipe_flow:.6g}\n"
            f"  ||dPumpFlow||    : {self.norm_pump_flow:.6g}"
        )


def validate_schedule(wdn: WDN, result: OWFResult) -> ScheduleValidationReport:
    """Fix the optimized pump on/off schedule into EPANET, re-simulate the true
    nonlinear hydraulics, and compare against the optimizer's heads/flows.

    Unlike :func:`validate` (which compares against EPANET's own rule-based
    operation), this drives EPANET with the *optimized* schedule, so a good
    linearization should reproduce EPANET closely.
    """
    T = wdn.time
    pump_links = (wdn.raw.link_pump_index + 1).tolist()  # EPANET 1-based

    d = epanet(str(wdn.spec.inp_path))
    # Remove existing operating logic so our schedule governs the pumps.
    try:
        d.deleteControls()
    except Exception:
        pass
    try:
        if d.getRuleCount() > 0:
            d.deleteRules()
    except Exception:
        pass

    d.openHydraulicAnalysis()
    d.initializeHydraulicAnalysis()
    rows: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    tstep = 1
    t_cur = 0
    while tstep > 0:
        idx = min(int(round(t_cur / 3600.0)), T - 1)
        for p, lk in enumerate(pump_links):
            d.setLinkStatus(lk, 1 if result.onoff[p, idx] > 0.5 else 0)
        t = int(d.runHydraulicAnalysis())
        rows[t] = (np.asarray(d.getNodeHydraulicHead()),
                   np.asarray(d.getLinkFlows()))
        tstep = d.nextHydraulicAnalysisStep()
        t_cur += tstep
    d.closeHydraulicAnalysis()
    d.unload()

    # Sample the state at each hour boundary (tanks can insert sub-steps).
    ordered = sorted(rows)
    heads_ep = np.zeros((wdn.n_nodes, T))
    flows_ep = np.zeros((wdn.n_links, T))
    for h in range(T):
        tt = h * 3600
        head, flow = rows.get(tt, rows[ordered[min(h, len(ordered) - 1)]])
        heads_ep[:, h] = head
        flows_ep[:, h] = flow

    dh = heads_ep - result.heads[:, :T]
    dpipe = wdn.M.Pi_prime @ (flows_ep - result.flows[:, :T])
    dpump = wdn.M.Lambda @ (flows_ep - result.flows[:, :T])

    return ScheduleValidationReport(
        max_abs_head=float(np.max(np.abs(dh))),
        max_abs_pipe_flow=float(np.max(np.abs(dpipe))),
        max_abs_pump_flow=float(np.max(np.abs(dpump))),
        norm_head=float(np.linalg.norm(dh)),
        norm_pipe_flow=float(np.linalg.norm(dpipe)),
        norm_pump_flow=float(np.linalg.norm(dpump)),
        heads_epanet=heads_ep,
        flows_epanet=flows_ep,
    )
