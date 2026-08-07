"""Successive linear-approximation loop (ports WDN_OWF_IEEEACCESS_cvx.m).

Each iteration solves a MILP (linear objective, linear constraints, binary pump
on/off) with HiGHS, then relinearizes the pipe head-loss and pump-power terms
around the new solution until the iterate stops moving.

For hard (e.g. looped) networks two warm-start aids are available via
``SolverConfig``: ``damping`` (trust-region blending of the linearization point)
and ``soft_bounds`` (penalty CCP -- head bounds are relaxed with penalized slacks
so every MILP is feasible and the slacks vanish as the linearization improves).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cvxpy as cp
import numpy as np

from . import constraints as C
from .linearization import LinPoint, error_norm, linearize, stack_eps
from .network import WDN


@dataclass
class Model:
    """CVXPY decision variables and linearization parameters."""
    Flows: cp.Variable
    Heads: cp.Variable
    Ppump: cp.Variable
    Hdummy: cp.Variable
    OnOff: cp.Variable
    TankFlow_aux: cp.Variable
    # linearization coefficients (updated each iteration)
    Cp: cp.Parameter
    C1M: cp.Parameter
    C2M: cp.Parameter
    APrime: cp.Parameter
    BPrime: cp.Parameter
    # optional soft-bound slacks + penalty weight (penalty CCP warm-start)
    s_jlo: Optional[cp.Variable] = None
    s_jhi: Optional[cp.Variable] = None
    s_tlo: Optional[cp.Variable] = None
    s_thi: Optional[cp.Variable] = None
    s_term: Optional[cp.Variable] = None
    penalty: Optional[cp.Parameter] = None
    Cp_bypass: Optional[cp.Parameter] = None
    # Variable-speed pumps: relative speed, the McCormick bilinear aux WW=omega*f,
    # and the VSP power coefficient on WW. Present only when the network has a VSP;
    # when None the pump constraints take the fixed-speed (FSP) path unchanged.
    Speed: Optional[cp.Variable] = None
    WW: Optional[cp.Variable] = None
    DPrime: Optional[cp.Parameter] = None
    # Pressure-reducing valves: active/open binaries + valve head-loss variable.
    # Present only when the network has a PRV.
    x_act: Optional[cp.Variable] = None
    x_open: Optional[cp.Variable] = None
    R_prv: Optional[cp.Variable] = None


@dataclass
class OWFResult:
    status: str
    n_iter: int
    converged: bool
    objective: float
    heads: np.ndarray          # (N x T)
    flows: np.ndarray          # (L x T)
    onoff: np.ndarray          # (Pu x T)
    ppump_linear: np.ndarray   # (Pu x T)  model (linearized) pump power
    ppump_true: np.ndarray     # (Pu x T)  true nonlinear pump power at solution
    speed: np.ndarray = None   # (Pu x T)  VSP relative speed (None if all FSP)
    errors: list = field(default_factory=list)
    objectives: list = field(default_factory=list)
    max_slack: float = float("nan")   # max head-bound violation (soft_bounds only)


def _build_model(wdn: WDN) -> Model:
    T = wdn.time
    # Pump on/off: a binary variable, or a fixed Parameter when a schedule is
    # pinned (turns the per-iteration problem into a continuous LP -- used by the
    # multi-start warm-start's convergence phase).
    if wdn.config.fixed_schedule is not None:
        # A plain array (not a Parameter): keeps products like BPrime*OnOff
        # DPP-compliant so CVXPY caches the compilation across iterations.
        onoff = np.asarray(wdn.config.fixed_schedule, dtype=float)
    else:
        onoff = cp.Variable((wdn.n_pumps, T), boolean=True, name="OnOff")
    model = Model(
        Flows=cp.Variable((wdn.n_links, T), name="Flows"),
        Heads=cp.Variable((wdn.n_nodes, T), name="Heads"),
        Ppump=cp.Variable((wdn.n_pumps, T), name="Ppump"),
        Hdummy=cp.Variable((wdn.n_tanks, T), name="Hdummy"),
        OnOff=onoff,
        TankFlow_aux=cp.Variable((wdn.n_tanks, T), name="TankFlow_aux"),
        Cp=cp.Parameter((wdn.n_pipes, T), name="Cp"),
        C1M=cp.Parameter((wdn.n_pumps, T), name="C1M"),
        C2M=cp.Parameter((wdn.n_pumps, T), name="C2M"),
        APrime=cp.Parameter((wdn.n_pumps, T), name="APrime"),
        BPrime=cp.Parameter((wdn.n_pumps, T), name="BPrime"),
    )
    if wdn.M.bypass_index.size:
        model.Cp_bypass = cp.Parameter((wdn.M.bypass_index.size, T), name="Cp_bypass")
    if wdn.pump.any_vsp:
        model.Speed = cp.Variable((wdn.n_pumps, T), name="Speed")
        model.WW = cp.Variable((wdn.n_pumps, T), name="WW")
        model.DPrime = cp.Parameter((wdn.n_pumps, T), name="DPrime")
    if wdn.n_valves:
        model.x_act = cp.Variable((wdn.n_valves, T), boolean=True, name="x_act")
        model.x_open = cp.Variable((wdn.n_valves, T), boolean=True, name="x_open")
        model.R_prv = cp.Variable((wdn.n_valves, T), nonneg=True, name="R_prv")
    if wdn.config.soft_bounds:
        model.s_jlo = cp.Variable((wdn.n_junctions, T), nonneg=True, name="s_jlo")
        model.s_jhi = cp.Variable((wdn.n_junctions, T), nonneg=True, name="s_jhi")
        model.s_tlo = cp.Variable((T, wdn.n_tanks), nonneg=True, name="s_tlo")
        model.s_thi = cp.Variable((T, wdn.n_tanks), nonneg=True, name="s_thi")
        model.s_term = cp.Variable((wdn.n_tanks,), nonneg=True, name="s_term")
        model.penalty = cp.Parameter(nonneg=True, name="penalty",
                                     value=wdn.config.penalty_weight)
    return model


def _set_params(model: Model, lin: LinPoint) -> None:
    model.Cp.value = lin.Cp
    model.C1M.value = lin.C1M
    model.C2M.value = lin.C2M
    model.APrime.value = lin.APrimePump
    model.BPrime.value = lin.BPrimePump
    if model.Cp_bypass is not None:
        model.Cp_bypass.value = lin.Cp_bypass
    if model.DPrime is not None:
        model.DPrime.value = lin.DPrime


def _max_slack(model: Model) -> float:
    vals = []
    for s in (model.s_jlo, model.s_jhi, model.s_tlo, model.s_thi, model.s_term):
        if s is not None and s.value is not None:
            vals.append(np.max(np.abs(s.value)))
    return float(max(vals)) if vals else 0.0


def _true_pump_power(wdn: WDN, flows: np.ndarray, speed: np.ndarray = None) -> np.ndarray:
    """Nonlinear pump power at a flow solution.

    FSP: c_m (h0 - r f^v) f.  VSP (with relative speed omega): c_m (h0 omega^2 - r f^v) f.
    """
    q = wdn.M.Lambda @ flows
    p = wdn.pump
    h0, r, v = p.h0[:, None], p.r_m[:, None], p.v_m[:, None]
    power = p.c_m * (h0 - r * np.abs(q) ** v) * q
    if speed is not None and p.any_vsp:
        vs = np.asarray(p.is_vsp, dtype=bool)
        vsp_power = p.c_m * (h0 * speed ** 2 - r * np.abs(q) ** v) * q
        power = power.copy()
        power[vs] = vsp_power[vs]
    return power


def solve_owf(
    wdn: WDN,
    lin_override: Optional[LinPoint] = None,
    eps_override: Optional[np.ndarray] = None,
) -> OWFResult:
    """Run the successive-linearization loop.

    ``lin_override`` / ``eps_override`` let a caller seed the initial linearization
    and iterate from an arbitrary warm-start point (e.g. EPANET flows under a
    chosen pump schedule) instead of ``wdn.lin0`` / ``wdn.int_eps``.
    """
    cfg = wdn.config
    soft = cfg.soft_bounds
    model = _build_model(wdn)
    _set_params(model, lin_override if lin_override is not None else wdn.lin0)

    warmup = cfg.mass_balance_warmup
    include_mb = not warmup
    obj = C.objective_soft(model, wdn) if soft else C.objective(model, wdn)
    problem = cp.Problem(
        obj, C.build_constraints(model, wdn, include_mass_balance=include_mb, soft=soft)
    )

    prev_eps = eps_override if eps_override is not None else wdn.int_eps
    f_lin_prev = None
    errors, objectives = [], []
    status = "not_solved"
    heads = flows = onoff = ppump = speed = None
    converged = False
    n_iter = 0
    max_slack = float("nan")
    # Best-so-far snapshot, so a late ill-conditioned solve can't lose a good
    # earlier iterate. Ranked by (bound violation, then iterate change).
    best = None
    best_key = (float("inf"), float("inf"))

    for it in range(cfg.max_iter):
        if warmup and it == cfg.mass_balance_warmup_iters:
            include_mb = True
            problem = cp.Problem(
                obj, C.build_constraints(model, wdn, include_mass_balance=True, soft=soft)
            )
        if soft:
            model.penalty.value = min(
                cfg.penalty_max, cfg.penalty_weight * cfg.penalty_growth ** it
            )

        # A single HiGHS failure (e.g. an ill-conditioned late iterate) must not
        # abort the whole run -- stop and fall back to the best iterate so far.
        status = None
        for solver_name in (cfg.solver, *cfg.fallback_solvers):
            try:
                problem.solve(solver=solver_name, verbose=cfg.verbose)
                status = problem.status
            except Exception as exc:  # numerical failure -- try the next solver
                if cfg.verbose:
                    print(f"[iter {it}] solver {solver_name} error ({exc})")
                status = None
                continue
            if status in ("optimal", "optimal_inaccurate"):
                break
        if status is None:
            print(f"[iter {it}] all solvers failed -- stopping")
            break
        if status not in ("optimal", "optimal_inaccurate"):
            print(f"[iter {it}] problem {status} -- stopping")
            break

        heads = np.asarray(model.Heads.value)
        flows = np.asarray(model.Flows.value)
        onoff = np.asarray(model.OnOff.value if hasattr(model.OnOff, "value")
                           else model.OnOff)
        ppump = np.asarray(model.Ppump.value)
        speed = (np.asarray(model.Speed.value) if model.Speed is not None
                 and model.Speed.value is not None else None)
        cur_slack = _max_slack(model) if soft else 0.0
        max_slack = cur_slack if soft else float("nan")

        eps = stack_eps(heads, flows, onoff)
        err = error_norm(eps, prev_eps)
        prev_eps = eps
        errors.append(err)
        objectives.append(float(C._energy_cost(model, wdn).value))
        n_iter = it + 1

        key = (round(cur_slack, 6), err)
        if key < best_key:
            best_key = key
            best = (heads, flows, onoff, ppump, speed, objectives[-1], max_slack)

        if cfg.verbose:
            extra = f"  max_slack={cur_slack:.4g}" if soft else ""
            print(f"[iter {it}] obj={objectives[-1]:.6f}  error={err:.6f}"
                  f"  status={status}{extra}")

        # relinearize around a (damped) blend of the new and previous flow fields
        if f_lin_prev is None or cfg.damping >= 1.0:
            f_lin = flows
        else:
            f_lin = cfg.damping * flows + (1.0 - cfg.damping) * f_lin_prev
        f_lin_prev = f_lin
        _set_params(model, linearize(f_lin, wdn.M, wdn.pump, speed=speed))

        # VSP: the McCormick relaxation contracts the flow iterate only slowly, so
        # also accept a stable objective (over the last 3 iterates) as converged,
        # provided the bounds are satisfied. FSP still converges on the norm first.
        obj_stable = (
            wdn.pump.any_vsp and len(objectives) >= 3
            and (max(objectives[-3:]) - min(objectives[-3:]))
            <= cfg.obj_rtol * max(1e-9, abs(objectives[-1]))
            and (not soft or cur_slack <= cfg.feas_tol)
        )
        if err < cfg.tol or obj_stable:
            converged = True
            break

    # Fall back to the best iterate if the loop stopped without converging.
    obj_final = objectives[-1] if objectives else float("nan")
    if not converged and best is not None:
        heads, flows, onoff, ppump, speed, obj_final, max_slack = best

    # With soft bounds, "converged" also requires the slacks to be ~0.
    if soft and converged and max_slack > cfg.feas_tol:
        converged = False

    return OWFResult(
        status=status,
        n_iter=n_iter,
        converged=converged,
        objective=obj_final,
        heads=heads,
        flows=flows,
        onoff=onoff,
        ppump_linear=ppump,
        ppump_true=_true_pump_power(wdn, flows, speed) if flows is not None else None,
        speed=speed,
        errors=errors,
        objectives=objectives,
        max_slack=max_slack,
    )
