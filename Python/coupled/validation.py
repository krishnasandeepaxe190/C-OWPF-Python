"""Validate a coupled solution against BOTH ground-truth simulators.

Water side  -- impose the optimized pump schedule back into EPANET and compare
               the nonlinear hydraulics (heads / pipe & pump flows), reusing the
               water OWF's ``validate_schedule``.
Power side  -- replay the coupled bus injections through the nonlinear single-
               phase Z-bus power flow and compare voltages against the linear
               LinDistFlow model the optimizer used.

Both are the same idea: the optimizer works on a convex *approximation*; a good
approximation reproduces the exact simulator on the same decisions.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from owf.network import WDN
from owf.validation import validate_schedule, ScheduleValidationReport
from pdn.feeders import FEEDERS
from pdn.network import PDN
from pdn.powerflow import zbus_powerflow, zbus_solve

from .coupled_lp import CoupledResult


def coupled_loss_kwh(pdn: PDN, res: CoupledResult) -> float:
    """Total true (nonlinear Z-bus) real network loss over the horizon, in kW·h."""
    if res.p_net is None or res.p_net.size == 0:
        return float("nan")
    m = pdn.model
    SBase = float(FEEDERS[pdn.key]["SBase"])
    loss_pu = 0.0
    for t in range(res.p_net.shape[1]):
        # zbus takes q_net WITHOUT caps (caps live in the Y-bus); res.q_net excludes
        # them under the voltage-dependent-cap convention.
        _, _, loss = zbus_solve(m, res.p_net[:, t], res.q_net[:, t])
        loss_pu += loss
    return loss_pu * (SBase / 1000.0)   # pu-hours -> kW·h


@dataclass
class CoupledValidationReport:
    # water (EPANET replay of the optimized schedule)
    water: ScheduleValidationReport
    # power (nonlinear Z-bus replay of the coupled injections)
    v_lin: np.ndarray            # (N_p x T) linear voltage the optimizer used
    v_nl: np.ndarray             # (N_p x T) nonlinear Z-bus voltage
    v_err_max: float             # max |v_lin - v_nl| (pu)
    v_err_mean: float            # mean |v_lin - v_nl| (pu)
    vmin_nl: float               # worst nonlinear bus voltage (pu)
    v_violation_nl: float        # nonlinear voltage-limit violation (pu)

    def summary(self) -> str:
        w = self.water
        return (
            "Coupled validation\n"
            "  WATER (optimized schedule re-run in EPANET):\n"
            f"    max |dHead|     : {w.max_abs_head:.4g} ft\n"
            f"    max |dPumpFlow| : {w.max_abs_pump_flow:.4g} GPM\n"
            f"    max |dPipeFlow| : {w.max_abs_pipe_flow:.4g} GPM\n"
            "  POWER (coupled injections re-run in nonlinear Z-bus):\n"
            f"    max |dV|        : {self.v_err_max:.4g} pu\n"
            f"    mean |dV|       : {self.v_err_mean:.4g} pu\n"
            f"    nonlinear Vmin  : {self.vmin_nl:.4f} pu\n"
            f"    nonlinear Vviol : {self.v_violation_nl:.4g} pu"
        )


def validate_coupled(wdn: WDN, pdn: PDN, res: CoupledResult,
                     vmin: float = 0.95, vmax: float = 1.05) -> CoupledValidationReport:
    """Cross-check a coupled result against EPANET (water) and Z-bus (power)."""
    water = validate_schedule(wdn, res)

    m = pdn.model
    T = res.voltage.shape[1]
    v_nl = np.empty_like(res.voltage)
    for t in range(T):
        # zbus expects q_net WITHOUT caps (caps go in the Y-bus); res.q_net already
        # excludes them under the voltage-dependent-cap convention.
        v_nl[:, t] = zbus_powerflow(m, res.p_net[:, t], res.q_net[:, t])

    diff = np.abs(res.voltage - v_nl)
    vmin_nl = float(np.nanmin(v_nl))
    vmax_nl = float(np.nanmax(v_nl))
    return CoupledValidationReport(
        water=water, v_lin=res.voltage, v_nl=v_nl,
        v_err_max=float(np.nanmax(diff)), v_err_mean=float(np.nanmean(diff)),
        vmin_nl=vmin_nl,
        v_violation_nl=float(max(0.0, vmin - vmin_nl, vmax_nl - vmax)),
    )
