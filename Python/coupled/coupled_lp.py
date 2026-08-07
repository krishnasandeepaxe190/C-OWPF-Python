"""Coupled Optimal Water-Power Flow (C-OWPF) solver.

Augments the water OWF problem (``owf``) with a linear distribution-feeder layer
(``pdn`` LinDistFlow) and couples them through the pumps: each pump's electrical
power ``Ppump`` (kW, from the water model) becomes an active load, in per-unit, at
its assigned PDN bus.  PV inverters at the feeder dispatch active and reactive
power (linear inverter-capability octagon) to hold bus voltages inside limits.

The combined problem is solved with the SAME successive-linearization loop as the
water side -- only the water head-loss / pump-power terms are relinearized each
iteration; the entire power layer is already linear.  With the pump schedule fixed
it is a pure LP (converges in ~1-2 warm-started iterations); with the schedule
free it is a MILP whose only integers are the pump on/off variables -- the feeder
adds none.

Objective (all in the water side's cost units):

    min  sum_t price(t) * [ sum_p Ppump/1000  -  sum_pv g_kW/1000 ]      (energy)
       + voltage_penalty * sum(voltage slacks)                          (V limits)
       + reactive_reg    * sum|q_pv|                                    (regularize)
       + water head-bound-slack penalty                                 (if soft)

PV active energy is credited at the time-of-use price, so the optimizer is pushed
to run pumps when solar is available and voltages are healthy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cvxpy as cp
import numpy as np

from owf import constraints as C
from owf.linearization import LinPoint, error_norm, linearize, stack_eps
from owf.network import WDN
from owf.solver import _build_model, _set_params, _max_slack, _true_pump_power

from pdn.network import PDN
from .config import CoupledConfig


@dataclass
class CoupledResult:
    status: str
    n_iter: int
    converged: bool
    # water
    objective: float                 # full coupled objective (with penalties)
    energy_cost: float               # pump-energy cost ($, priced at the WDN price)
    heads: np.ndarray                # (N_w x T)
    flows: np.ndarray                # (L x T)
    onoff: np.ndarray                # (Pu x T)
    ppump_true: np.ndarray           # (Pu x T) true nonlinear pump power (kW)
    water_max_slack: float
    # power
    voltage: np.ndarray              # (N_p x T) linear |V| (pu)
    v2: np.ndarray                   # (N_p x T) squared voltage (pu^2)
    pv_p: np.ndarray                 # (N_pv x T) PV active dispatch (pu)
    pv_q: np.ndarray                 # (N_pv x T) PV reactive dispatch (pu)
    pv_buses: np.ndarray             # (N_pv,) bus indices
    p_net: np.ndarray                # (N_p x T) net active bus load (pu)
    q_net: np.ndarray                # (N_p x T) net reactive bus load (pu, caps NOT in q)
    grid_kw: np.ndarray              # (T,) substation active import (kW, linear)
    pump_bus: np.ndarray             # (Pu,) PDN bus each pump feeds
    v_min: float
    v_violation: float               # worst voltage-limit violation (pu)
    loss_kw: np.ndarray = None       # (T,) LinDistFlow network loss (kW)
    loss_cost: float = float("nan")  # priced loss cost ($, at the WDN price)
    total_cost: float = float("nan") # energy_cost + loss_cost ($)
    speed: np.ndarray = None         # (Pu x T) VSP relative speed (None if all FSP)
    errors: list = field(default_factory=list)
    objectives: list = field(default_factory=list)


def default_pump_buses(pdn: PDN, n_pumps: int) -> np.ndarray:
    """Place pumps at the feeder's electrically weakest (lowest nominal-voltage)
    buses, where a pump load most stresses the network -- the interesting case."""
    m = pdn.model
    v_nom = m.voltage(m.p_load, m.q_load)                # nominal (caps in R/X/V_k)
    order = np.argsort(v_nom)                            # weakest first
    return np.array([order[i % len(order)] for i in range(n_pumps)], dtype=int)


def _pv_capability(pdn: PDN, cc: CoupledConfig, T: int):
    """PV active output (fixed at availability) and the exact reactive capability.

    PV *active* p_pv[k,t] = rating[k] * solar(t) is fixed (the paper controls PV
    reactive, not active).  With active known, the smart-inverter capability circle
    |q| <= sqrt(S^2 - p^2) is a CONSTANT bound per (bus, hour) -- exact and LP-safe.
    """
    pv = pdn.pv_buses
    rating = pdn.pv_rating[pv]                           # (npv,) = inverter S rating
    solar = pdn.solar(T)                                 # (T,)
    p_pv = np.outer(rating, solar)                       # (npv, T) fixed active
    qcap = np.sqrt(np.maximum(rating[:, None] ** 2 - p_pv ** 2, 0.0))   # (npv, T)
    return pv, rating, p_pv, qcap


def build_coupled_problem(wdn: WDN, pdn: PDN, cc: CoupledConfig,
                          trust_region: Optional[tuple] = None):
    """Construct the augmented CVXPY problem; returns (problem, model, ctx).

    ``trust_region=(z0, K)`` -- only meaningful when the schedule is free (MILP) --
    limits the Hamming distance of the pump on/off variable to at most ``K`` flips
    from an incumbent schedule ``z0``, keeping the warm-started linearization valid.
    """
    T = wdn.time
    m = pdn.model
    SBase = _feeder_sbase(pdn)   # VA base -> converts pump kW to per-unit

    # --- water sub-model (fixed schedule -> LP; free -> MILP), same as solve_owf --
    wmodel = _build_model(wdn)
    soft = wdn.config.soft_bounds
    wcons = C.build_constraints(wmodel, wdn, include_mass_balance=True, soft=soft)

    # --- trust region on the (free) pump schedule -------------------------------
    if trust_region is not None and hasattr(wmodel.OnOff, "value"):
        z0, K = trust_region
        z0 = np.round(np.asarray(z0)).astype(float)
        # Hamming distance: sum of z where z0==0, plus (1-z) where z0==1  <= K
        ham = (cp.sum(wmodel.OnOff[z0 == 0]) + cp.sum(1.0 - wmodel.OnOff[z0 == 1]))
        wcons = wcons + [ham <= K]

    # --- coupling: pump kW -> per-unit bus load ---------------------------------
    pump_bus = (np.asarray(cc.pump_bus, int) if cc.pump_bus is not None
                else default_pump_buses(pdn, wdn.n_pumps))
    Bpump = np.zeros((m.N, wdn.n_pumps))
    for p, b in enumerate(pump_bus):
        Bpump[int(b), p] += 1.0
    kw_to_pu = 1000.0 / SBase * cc.pump_load_scale       # kW -> W -> pu (* demo scale)
    pump_load_pu = (Bpump @ wmodel.Ppump) * kw_to_pu     # (N x T) active, linear in Ppump
    # pump reactive draw: q_pump = p_pump * sqrt(1/PF^2 - 1) = p_pump * tan(acos PF)
    tan_pf = float(np.sqrt(max(1.0 / cc.pump_pf ** 2 - 1.0, 0.0)))
    pump_qload_pu = pump_load_pu * tan_pf                # (N x T) reactive at pump bus

    # --- feeder base load over the day -----------------------------------------
    loadmult = cc.load_shape(T)                          # (T,)
    p_base = np.outer(m.p_load, loadmult)                # (N x T)
    q_base = np.outer(m.q_load, loadmult)

    # --- PV: active fixed at availability, reactive q_pv is the control ----------
    # (Gamma maps PV buses to the network, as in the MATLAB ConnectMatrices.)
    pcons = []
    if cc.enable_pv and pdn.pv_buses.size:
        pv, rating, p_pv, qcap = _pv_capability(pdn, cc, T)
        npv = pv.size
        qv = cp.Variable((npv, T), name="pv_q")           # reactive control lever
        Spv = np.zeros((m.N, npv))                        # = Gamma
        for k, b in enumerate(pv):
            Spv[int(b), k] = 1.0
        pcons += [qv <= qcap, qv >= -qcap]                # exact inverter capability
        g_full = Spv @ p_pv                               # constant PV active (N x T)
        q_full = Spv @ qv                                 # PV reactive (N x T)
    else:
        pv = np.array([], int); npv = 0
        qv = None; p_pv = np.zeros((0, T))
        g_full = np.zeros((m.N, T))
        q_full = np.zeros((m.N, T))

    # --- net injections and linear (Kekatos) voltages ---------------------------
    #   p_net = Psi p_load + Xi Ppump_pu - Gamma p_pv              (pumps add load)
    #   q_net = Psi q_load + Xi Qpump_pu - Gamma q_pv              (pump reactive too;
    #           caps are voltage-dependent and folded into R/X/V_k, so NOT here)
    #   V^2   = R(-p_net) + X(-q_net) + V_k
    p_net = p_base - g_full + pump_load_pu                       # (N x T)
    q_net = q_base - q_full + pump_qload_pu
    V2 = (-m.R @ p_net) - (m.X @ q_net) + m.V_k[:, None]         # (N x T) squared V

    # --- branch flows & (linearized) network loss -------------------------------
    # Lossless LinDistFlow branch flows: P_branch = T p_net, Q_branch = T q_net,
    # with the voltage-dependent cap injection qˢ·v subtracted from the reactive
    # side (paper eq. 1b). Loss = sum_i r_i (P_i^2 + Q_i^2) is convex-quadratic; we
    # linearize it around the current branch flows each iteration (same successive-
    # approximation idea as the water side), so every step stays an LP/MILP.
    from pdn.lindistflow import branch_flow_matrix
    Tmat = branch_flow_matrix(m)                                 # (N x N)
    q_net_cap = q_net - cp.multiply(m.b_shunt[:, None], V2)      # cap injection qˢ·v
    P_branch = Tmat @ p_net                                      # (N x T)
    Q_branch = Tmat @ q_net_cap
    Pbk = cp.Parameter((m.N, T), name="Pbk", value=np.zeros((m.N, T)))
    Qbk = cp.Parameter((m.N, T), name="Qbk", value=np.zeros((m.N, T)))
    rP = np.tile(m.r[:, None], (1, T))                          # (N x T) branch r
    # Tangent of sum_i r_i(P^2+Q^2) at (Pbk,Qbk): the linear-in-decision part is
    # 2r·Pbk·P + 2r·Qbk·Q; the constant −r(Pbk²+Qbk²) is dropped (it doesn't change
    # the minimizer, and keeping a parameter² would break DPP / cached solves). The
    # true loss is reported from the solved branch flows, not this tangent.
    loss_terms = (cp.multiply(2.0 * rP, cp.multiply(Pbk, P_branch))
                  + cp.multiply(2.0 * rP, cp.multiply(Qbk, Q_branch)))          # (N x T)
    loss_lin = cp.sum(loss_terms, axis=0)                       # (T,)
    loss_kw = loss_lin * (SBase / 1000.0)                       # (T,) pu -> kW

    # --- voltage limits: hard, or the water side's soft-slack feasibility device -
    if cc.soft_voltage:
        s_vlo = cp.Variable((m.N, T), nonneg=True, name="s_vlo")
        s_vhi = cp.Variable((m.N, T), nonneg=True, name="s_vhi")
        pcons += [V2 >= cc.vmin ** 2 - s_vlo, V2 <= cc.vmax ** 2 + s_vhi]
    else:
        s_vlo = s_vhi = None
        pcons += [V2 >= cc.vmin ** 2, V2 <= cc.vmax ** 2]

    # --- objective: pump energy + network-loss cost, both at the WDN price -------
    #   min  sum_t price(t) * [ (sum_pumps Ppump)/1000  +  loss ]        (paper 33d)
    # PV reactive dispatch and pump timing both cut loss -> real dollar savings.
    # The slack terms are the SAME penalty-CCP feasibility device the water side
    # uses -- they drive the slacks to zero, they are not economics.
    water_energy = wdn.price_final @ (cp.sum(wmodel.Ppump, axis=0) / 1000.0)
    obj = water_energy
    loss_cost_expr = wdn.price_final @ (loss_kw / 1000.0)        # priced loss
    if cc.include_loss:
        obj = obj + loss_cost_expr
    if cc.soft_voltage:
        obj = obj + cc.voltage_penalty * (cp.sum(s_vlo) + cp.sum(s_vhi))
    if soft:
        wpen = (cp.sum(wmodel.s_jlo) + cp.sum(wmodel.s_jhi) + cp.sum(wmodel.s_tlo)
                + cp.sum(wmodel.s_thi) + cp.sum(wmodel.s_term))
        obj = obj + wmodel.penalty * wpen

    problem = cp.Problem(cp.Minimize(obj), wcons + pcons)
    ctx = dict(pump_bus=pump_bus, pv=pv, p_pv=p_pv, qv=qv, V2=V2,
               p_net=p_net, q_net=q_net, s_vlo=s_vlo, s_vhi=s_vhi, SBase=SBase,
               m=m, water_energy=water_energy, P_branch=P_branch, Q_branch=Q_branch,
               Pbk=Pbk, Qbk=Qbk, price=np.asarray(wdn.price_final))
    return problem, wmodel, ctx


def _feeder_sbase(pdn: PDN) -> float:
    from pdn.feeders import FEEDERS
    return float(FEEDERS[pdn.key]["SBase"])


def solve_coupled(wdn: WDN, pdn: PDN, cc: CoupledConfig,
                  lin_override: Optional[LinPoint] = None,
                  eps_override: Optional[np.ndarray] = None,
                  trust_region: Optional[tuple] = None) -> CoupledResult:
    """Run the coupled successive-linearization loop and return a CoupledResult."""
    cfg = wdn.config
    soft = cfg.soft_bounds
    problem, wmodel, ctx = build_coupled_problem(wdn, pdn, cc, trust_region=trust_region)
    _set_params(wmodel, lin_override if lin_override is not None else wdn.lin0)

    prev_eps = eps_override if eps_override is not None else wdn.int_eps
    f_lin_prev = None
    errors, objectives = [], []
    status = "not_solved"
    converged = False
    n_iter = 0
    best = None
    best_key = (float("inf"), float("inf"))
    m = pdn.model

    for it in range(cfg.max_iter):
        if soft:
            wmodel.penalty.value = min(cfg.penalty_max,
                                       cfg.penalty_weight * cfg.penalty_growth ** it)
        status = None
        for solver_name in (cfg.solver, *cfg.fallback_solvers):
            try:
                problem.solve(solver=solver_name, verbose=cfg.verbose)
                status = problem.status
            except Exception as exc:
                if cfg.verbose:
                    print(f"[coupled iter {it}] solver {solver_name} error ({exc})")
                status = None
                continue
            if status in ("optimal", "optimal_inaccurate"):
                break
        if status not in ("optimal", "optimal_inaccurate"):
            print(f"[coupled iter {it}] problem {status} -- stopping")
            break

        heads = np.asarray(wmodel.Heads.value)
        flows = np.asarray(wmodel.Flows.value)
        onoff = np.asarray(wmodel.OnOff.value if hasattr(wmodel.OnOff, "value")
                           else wmodel.OnOff)
        cur_slack = _max_slack(wmodel) if soft else 0.0

        eps = stack_eps(heads, flows, onoff)
        err = error_norm(eps, prev_eps)
        prev_eps = eps
        errors.append(err)
        objectives.append(float(problem.value))
        n_iter = it + 1

        snap = _snapshot(wmodel, wdn, pdn, cc, ctx)
        key = (round(cur_slack, 6), err)
        if key < best_key:
            best_key = key
            best = snap

        if cfg.verbose:
            print(f"[coupled iter {it}] obj={problem.value:.6f} err={err:.6f} "
                  f"Vmin={snap['voltage'].min():.4f} status={status}")

        if f_lin_prev is None or cfg.damping >= 1.0:
            f_lin = flows
        else:
            f_lin = cfg.damping * flows + (1.0 - cfg.damping) * f_lin_prev
        f_lin_prev = f_lin
        speed = (np.asarray(wmodel.Speed.value) if wmodel.Speed is not None
                 and wmodel.Speed.value is not None else None)
        _set_params(wmodel, linearize(f_lin, wdn.M, wdn.pump, speed=speed))
        # relinearize the (convex) loss around the new branch flows
        ctx["Pbk"].value = np.asarray(ctx["P_branch"].value)
        ctx["Qbk"].value = np.asarray(ctx["Q_branch"].value)

        # VSP: the McCormick relaxation contracts the flow iterate slowly, so also
        # accept a stable objective over the last 3 iterates (bound-feasible). FSP
        # converges on the iterate norm first, so it is unaffected.
        obj_stable = (
            wdn.pump.any_vsp and len(objectives) >= 3
            and (max(objectives[-3:]) - min(objectives[-3:]))
            <= cfg.obj_rtol * max(1e-9, abs(objectives[-1]))
            and (not soft or cur_slack <= cfg.feas_tol)
        )
        if err < cfg.tol or obj_stable:
            converged = True
            break

    if best is None:                       # never got an optimal iterate
        return _failed_result(status, n_iter, ctx, errors, objectives)
    snap = best if (not converged) else _snapshot(wmodel, wdn, pdn, cc, ctx)
    if soft and converged and snap["water_max_slack"] > cfg.feas_tol:
        converged = False

    v = snap["voltage"]
    v_violation = float(max(0.0, cc.vmin - v.min(), v.max() - cc.vmax))
    return CoupledResult(
        status=status, n_iter=n_iter, converged=converged,
        objective=objectives[-1] if objectives else float("nan"),
        energy_cost=snap["energy_cost"],
        heads=snap["heads"], flows=snap["flows"], onoff=snap["onoff"],
        ppump_true=snap["ppump_true"], water_max_slack=snap["water_max_slack"],
        voltage=v, v2=snap["v2"], pv_p=snap["pv_p"], pv_q=snap["pv_q"],
        pv_buses=snap["pv_buses"], p_net=snap["p_net"], q_net=snap["q_net"],
        grid_kw=snap["grid_kw"], pump_bus=np.asarray(ctx["pump_bus"]),
        v_min=float(v.min()), v_violation=v_violation,
        loss_kw=snap["loss_kw"], loss_cost=snap["loss_cost"],
        total_cost=snap["total_cost"], speed=snap.get("speed"),
        errors=errors, objectives=objectives,
    )


def _failed_result(status, n_iter, ctx, errors, objectives) -> CoupledResult:
    """Return a clearly-failed CoupledResult (e.g. infeasible hard voltage)."""
    empty = np.zeros((0, 0))
    return CoupledResult(
        status=status or "failed", n_iter=n_iter, converged=False,
        objective=float("nan"), energy_cost=float("nan"),
        heads=empty, flows=empty, onoff=empty, ppump_true=empty,
        water_max_slack=float("nan"), voltage=empty, v2=empty,
        pv_p=empty, pv_q=empty, pv_buses=np.asarray(ctx["pv"]),
        p_net=empty, q_net=empty, grid_kw=np.zeros(0),
        pump_bus=np.asarray(ctx["pump_bus"]), v_min=float("nan"),
        v_violation=float("nan"), errors=errors, objectives=objectives,
    )


def _snapshot(wmodel, wdn, pdn, cc, ctx) -> dict:
    """Read current variable values into plain numpy (safe to store)."""
    flows = np.asarray(wmodel.Flows.value)
    heads = np.asarray(wmodel.Heads.value)
    onoff = np.asarray(wmodel.OnOff.value if hasattr(wmodel.OnOff, "value")
                       else wmodel.OnOff)
    speed = (np.asarray(wmodel.Speed.value) if wmodel.Speed is not None
             and wmodel.Speed.value is not None else None)
    _val = lambda e: np.asarray(e.value if hasattr(e, "value") else e)
    p_net = _val(ctx["p_net"])
    q_net = _val(ctx["q_net"])
    v2 = _val(ctx["V2"])
    voltage = np.sqrt(np.maximum(v2, 0.0))
    SBase = ctx["SBase"]
    grid_kw = p_net.sum(axis=0) * (SBase / 1000.0)
    if ctx["qv"] is not None:
        pv_p = np.asarray(ctx["p_pv"])            # fixed PV active (npv x T)
        pv_q = np.asarray(ctx["qv"].value)        # optimized PV reactive
    else:
        pv_p = np.zeros((0, wdn.time)); pv_q = np.zeros((0, wdn.time))
    energy_cost = float(ctx["water_energy"].value)   # pump energy cost ($)
    # LinDistFlow network loss from the solved branch flows: sum_i r_i(P^2+Q^2)
    m = ctx["m"]
    Pb = _val(ctx["P_branch"]); Qb = _val(ctx["Q_branch"])
    loss_pu = (m.r[:, None] * (Pb ** 2 + Qb ** 2)).sum(axis=0)     # (T,) pu
    loss_kw = loss_pu * (SBase / 1000.0)
    loss_cost = float(ctx["price"] @ (loss_kw / 1000.0))
    return dict(
        flows=flows, heads=heads, onoff=onoff, speed=speed,
        ppump_true=_true_pump_power(wdn, flows, speed),
        water_max_slack=_max_slack(wmodel) if wdn.config.soft_bounds else 0.0,
        voltage=voltage, v2=v2, pv_p=pv_p, pv_q=pv_q, pv_buses=np.asarray(ctx["pv"]),
        p_net=p_net, q_net=q_net, grid_kw=grid_kw, energy_cost=energy_cost,
        loss_kw=loss_kw, loss_cost=loss_cost, total_cost=energy_cost + loss_cost,
    )
