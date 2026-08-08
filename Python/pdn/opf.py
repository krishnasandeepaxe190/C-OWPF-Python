"""Standalone distribution-network reactive-power OPF.

Given the feeder loads (optionally over a day) plus any pump electrical load
imposed by the water schedule, dispatch the PV inverters' **reactive** setpoints
q_pv to hold bus voltages inside limits, then verify with the nonlinear Z-bus to
report the TRUE voltages and TRUE real loss.

Optimization (per hour, LinDistFlow linear voltages):
    min  voltage_penalty * sum(voltage slacks)      # keep V in [vmin, vmax]
       + sum |q_net|                                 # local reactive compensation
    s.t. V^2 = R(-p_net) + X(-q_net) + V_k
         |q_pv| <= sqrt(S^2 - p_pv^2)                # exact inverter capability
Minimizing the net reactive drawn from the substation pushes each PV to supply
its local reactive load, cutting reactive line flow -- and thus loss.  The loss
itself is not linearized into the objective; it is reported exactly from the
Z-bus replay (with and without the reactive setpoints, so the reduction is clear).
"""
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from .network import PDN
from .powerflow import zbus_solve


@dataclass
class PDNOPFResult:
    feeder: str
    q_pv: np.ndarray          # (npv x T) optimized reactive setpoints (pu)
    p_pv: np.ndarray          # (npv x T) PV active output (fixed availability, pu)
    pv_buses: np.ndarray      # (npv,)
    v_lin: np.ndarray         # (N x T) linear (LinDistFlow) voltages the OPF used
    v_nl: np.ndarray          # (N x T) nonlinear Z-bus voltages (with q setpoints)
    v_nl_base: np.ndarray     # (N x T) nonlinear voltages with q_pv = 0
    loss_nl: np.ndarray       # (T,) true loss per hour with q setpoints (pu)
    loss_base: np.ndarray     # (T,) true loss per hour, q_pv = 0 (pu)
    p_net: np.ndarray         # (N x T) net active load (pu)
    q_net: np.ndarray         # (N x T) net reactive load (pu, caps folded in)
    v_violation: float        # nonlinear voltage-limit violation (pu)
    SBase: float

    @property
    def loss_kw(self) -> np.ndarray:
        return self.loss_nl * (self.SBase / 1000.0)

    @property
    def loss_base_kw(self) -> np.ndarray:
        return self.loss_base * (self.SBase / 1000.0)

    @property
    def loss_reduction_pct(self) -> float:
        b, o = self.loss_base.sum(), self.loss_nl.sum()
        return 100.0 * (b - o) / b if b else 0.0


def pump_load_to_bus(pdn: PDN, ppump_kw: np.ndarray, pump_bus, scale: float = 1.0):
    """Map pump electrical power (Pu x T, kW) to a per-bus active load (N x T, pu)."""
    from .feeders import FEEDERS
    SBase = float(FEEDERS[pdn.key]["SBase"])
    n_pumps = ppump_kw.shape[0]
    B = np.zeros((pdn.N, n_pumps))
    for p, b in enumerate(np.asarray(pump_bus, int)):
        B[int(b), p] += 1.0
    return (B @ ppump_kw) * (1000.0 / SBase * scale)


def solve_pdn_opf(pdn: PDN, T: int, pump_load_pu: np.ndarray | None = None,
                  load_shape: np.ndarray | None = None, vmin: float = 0.95,
                  vmax: float = 1.05, soft_voltage: bool = True,
                  voltage_penalty: float = 1.0e4,
                  solver: str = "HIGHS", verbose: bool = False) -> PDNOPFResult:
    """Dispatch PV reactive to hold voltage, then verify with the nonlinear Z-bus."""
    from .feeders import FEEDERS
    SBase = float(FEEDERS[pdn.key]["SBase"])
    m = pdn.model
    N = m.N

    load = np.ones(T) if load_shape is None else np.asarray(load_shape, float)[:T]
    p_base = np.outer(m.p_load, load)                       # (N x T)
    q_base = np.outer(m.q_load, load)
    pump = np.zeros((N, T)) if pump_load_pu is None else np.asarray(pump_load_pu)

    pv = pdn.pv_buses
    npv = pv.size
    if npv:
        rating = pdn.pv_rating[pv]
        solar = pdn.solar(T)
        p_pv = np.outer(rating, solar)                      # fixed active (npv x T)
        qcap = np.sqrt(np.maximum(rating[:, None] ** 2 - p_pv ** 2, 0.0))
        qv = cp.Variable((npv, T), name="q_pv")
        Spv = np.zeros((N, npv))
        for k, b in enumerate(pv):
            Spv[int(b), k] = 1.0
        g_full = Spv @ p_pv
        q_full = Spv @ qv
        cons = [qv <= qcap, qv >= -qcap]
    else:
        p_pv = np.zeros((0, T)); qv = None
        g_full = np.zeros((N, T)); q_full = np.zeros((N, T))
        cons = []

    p_net = p_base + pump - g_full                          # (N x T)
    q_net = q_base - q_full                                 # caps are in R/X/V_k now
    V2 = (-m.R @ p_net) - (m.X @ q_net) + m.V_k[:, None]

    obj = 0
    if soft_voltage:
        s_lo = cp.Variable((N, T), nonneg=True)
        s_hi = cp.Variable((N, T), nonneg=True)
        cons += [V2 >= vmin ** 2 - s_lo, V2 <= vmax ** 2 + s_hi]
        obj = obj + voltage_penalty * (cp.sum(s_lo) + cp.sum(s_hi))
    else:
        cons += [V2 >= vmin ** 2, V2 <= vmax ** 2]
    if npv:
        obj = obj + cp.sum(cp.abs(q_net))                   # local reactive compensation

    prob = cp.Problem(cp.Minimize(obj), cons)
    if verbose:
        print(f"[pdn-opf] {pdn.key}: T={T}h, solver={solver}")
    for s in (solver, "SCIP", "CLARABEL", "SCS"):
        try:
            prob.solve(solver=s, verbose=verbose)
            if prob.status in ("optimal", "optimal_inaccurate"):
                break
        except Exception:
            continue

    q_val = np.asarray(qv.value) if qv is not None else np.zeros((0, T))
    p_net_v = np.asarray(p_net.value if hasattr(p_net, "value") else p_net)
    q_net_v = np.asarray(q_net.value if hasattr(q_net, "value") else q_net)
    v_lin = np.sqrt(np.maximum(np.asarray(V2.value), 0.0))

    # nonlinear Z-bus replay: with the setpoints, and a q_pv=0 baseline, for loss.
    # zbus takes q_net WITHOUT caps (caps live in the Y-bus); q_net already excludes
    # them under the voltage-dependent-cap convention.
    q_net_base = q_base                                     # no PV reactive
    p_net_base = p_base + pump - g_full                     # keep PV active in base
    v_nl = np.empty((N, T)); v_nl_base = np.empty((N, T))
    loss_nl = np.empty(T); loss_base = np.empty(T)
    for t in range(T):
        vmag, _, loss = zbus_solve(m, p_net_v[:, t], q_net_v[:, t])
        v_nl[:, t] = vmag; loss_nl[t] = loss
        vmag0, _, loss0 = zbus_solve(m, p_net_base[:, t], q_net_base[:, t])
        v_nl_base[:, t] = vmag0; loss_base[t] = loss0

    vmn, vmx = float(np.nanmin(v_nl)), float(np.nanmax(v_nl))
    return PDNOPFResult(
        feeder=pdn.key, q_pv=q_val, p_pv=p_pv, pv_buses=pv, v_lin=v_lin, v_nl=v_nl,
        v_nl_base=v_nl_base, loss_nl=loss_nl, loss_base=loss_base, p_net=p_net_v,
        q_net=q_net_v, v_violation=float(max(0.0, vmin - vmn, vmx - vmax)), SBase=SBase,
    )
