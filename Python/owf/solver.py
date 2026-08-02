"""Successive linear-approximation loop (ports WDN_OWF_IEEEACCESS_cvx.m).

Each iteration solves a MILP (linear objective, linear constraints, binary pump
on/off) with HiGHS, then relinearizes the pipe head-loss and pump-power terms
around the new solution until the iterate stops moving.
"""
from __future__ import annotations

from dataclasses import dataclass, field

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
    ppump_true: np.ndarray     # (Pu x T)  true nonlinear FSP power at solution
    errors: list = field(default_factory=list)
    objectives: list = field(default_factory=list)


def _build_model(wdn: WDN) -> Model:
    T = wdn.time
    return Model(
        Flows=cp.Variable((wdn.n_links, T), name="Flows"),
        Heads=cp.Variable((wdn.n_nodes, T), name="Heads"),
        Ppump=cp.Variable((wdn.n_pumps, T), name="Ppump"),
        Hdummy=cp.Variable((wdn.n_tanks, T), name="Hdummy"),
        OnOff=cp.Variable((wdn.n_pumps, T), boolean=True, name="OnOff"),
        TankFlow_aux=cp.Variable((wdn.n_tanks, T), name="TankFlow_aux"),
        Cp=cp.Parameter((wdn.n_pipes, T), name="Cp"),
        C1M=cp.Parameter((wdn.n_pumps, T), name="C1M"),
        C2M=cp.Parameter((wdn.n_pumps, T), name="C2M"),
        APrime=cp.Parameter((wdn.n_pumps, T), name="APrime"),
        BPrime=cp.Parameter((wdn.n_pumps, T), name="BPrime"),
    )


def _set_params(model: Model, lin: LinPoint) -> None:
    model.Cp.value = lin.Cp
    model.C1M.value = lin.C1M
    model.C2M.value = lin.C2M
    model.APrime.value = lin.APrimePump
    model.BPrime.value = lin.BPrimePump


def _true_pump_power(wdn: WDN, flows: np.ndarray) -> np.ndarray:
    """Nonlinear FSP power c_m (h0 - r q^2) q at a flow solution."""
    q = wdn.M.Lambda @ flows
    p = wdn.pump
    return p.c_m * (p.h0[:, None] - p.r_m[:, None] * q ** 2) * q


def solve_owf(wdn: WDN) -> OWFResult:
    cfg = wdn.config
    model = _build_model(wdn)
    _set_params(model, wdn.lin0)

    warmup = cfg.mass_balance_warmup
    include_mb = not warmup
    problem = cp.Problem(C.objective(model, wdn),
                         C.build_constraints(model, wdn, include_mass_balance=include_mb))

    prev_eps = wdn.int_eps
    errors, objectives = [], []
    status = "not_solved"
    heads = flows = onoff = ppump = None
    converged = False
    n_iter = 0

    for it in range(cfg.max_iter):
        # MATLAB quirk: switch mass balance on after the warm-up iterations.
        if warmup and it == cfg.mass_balance_warmup_iters:
            include_mb = True
            problem = cp.Problem(
                C.objective(model, wdn),
                C.build_constraints(model, wdn, include_mass_balance=True),
            )

        problem.solve(solver=cfg.solver, verbose=cfg.verbose)
        status = problem.status
        if status not in ("optimal", "optimal_inaccurate"):
            print(f"[iter {it}] problem {status} -- stopping")
            break

        heads = np.asarray(model.Heads.value)
        flows = np.asarray(model.Flows.value)
        onoff = np.asarray(model.OnOff.value)
        ppump = np.asarray(model.Ppump.value)

        eps = stack_eps(heads, flows, onoff)
        err = error_norm(eps, prev_eps)
        prev_eps = eps
        errors.append(err)
        objectives.append(float(problem.value))
        n_iter = it + 1

        if cfg.verbose:
            print(f"[iter {it}] obj={problem.value:.6f}  error={err:.6f}  status={status}")

        # relinearize around the new solution for the next MILP
        _set_params(model, linearize(flows, wdn.M, wdn.pump))

        if err < cfg.tol:
            converged = True
            break

    return OWFResult(
        status=status,
        n_iter=n_iter,
        converged=converged,
        objective=objectives[-1] if objectives else float("nan"),
        heads=heads,
        flows=flows,
        onoff=onoff,
        ppump_linear=ppump,
        ppump_true=_true_pump_power(wdn, flows) if flows is not None else None,
        errors=errors,
        objectives=objectives,
    )
