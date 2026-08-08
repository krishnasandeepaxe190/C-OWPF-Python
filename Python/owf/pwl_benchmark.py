"""One-shot piecewise-linear MILP OWF benchmark (Oikonomou & Parvania, 2019).

Implements the PWL-MILP linearization strategy of

  K. Oikonomou and M. Parvania, "Optimal Coordination of Water Distribution
  Energy Flexibility With Power Systems Operation," IEEE Trans. Smart Grid,
  vol. 10, no. 1, pp. 1101-1110, Jan. 2019,

on THIS study's OWF problem -- same networks, same data, same objective (pump
energy at the TOU price) -- so the two linearization strategies compare
apples-to-apples:

  * paper method (here): ONE MILP. Every nonlinearity is replaced up front by a
    K-breakpoint piecewise-linear function with SOS2-style "consecutive
    lambdas" adjacency enforced by segment binaries (paper eqs. 17-24):
    pipes get S*(K-1)*T binaries, pumps another U*(K-1)*T on top of the U*T
    on/off. Accuracy is FIXED by K.
  * this study: successive linear approximation. Binaries are the U*T pump
    on/off ONLY (pipes/pumps carry none); a sequence of small MILPs converges
    the linearization to the operating point, and the schedule is validated by
    EPANET replay.

Both solutions are scored the same way: the TRUE nonlinear pump-energy cost of
the returned flows, and the max |dHead| when the returned schedule is replayed
in EPANET.

Scope: FSP networks without switched bypasses or PRVs (the paper's bivariate
(q, omega) triangulation for variable speed is a follow-on comparison).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Optional

import cvxpy as cp
import numpy as np
import scipy.sparse as sp

from . import constraints as C
from .epanet_io import run_epanet
from .network import WDN
from .solver import _true_pump_power


@dataclass
class PWLResult:
    status: str
    K: int
    objective: float            # the MILP's own (PWL surrogate) cost
    true_cost: float            # true nonlinear pump-energy cost of the flows
    heads: Optional[np.ndarray]
    flows: Optional[np.ndarray]
    onoff: Optional[np.ndarray]
    ppump_pwl: Optional[np.ndarray]
    ppump_true: Optional[np.ndarray]
    build_s: float              # model construction + CVXPY compile time
    solve_s: float              # solver wall time
    n_binary: int
    n_continuous: int
    n_constraint_rows: int
    speed: None = None          # duck-typing for validate_schedule (FSP only)
    notes: str = ""


def pipe_qmax_from_epanet(wdn: WDN, factor: float = 3.0,
                          floor: float = 50.0) -> np.ndarray:
    """Per-pipe |q| breakpoint range from the EPANET rules run (generous)."""
    flows_ep, _, _, _ = run_epanet(wdn.raw)
    pq = wdn.M.Pi_prime @ flows_ep[: wdn.time].T          # (S x T)
    return np.maximum(factor * np.max(np.abs(pq), axis=1), floor)


def _sos2_blocks(n_items: int, K: int, T: int, prefix: str):
    """Stacked lambda/alpha variables + the adjacency selector for n_items
    independent K-breakpoint PWL functions (paper eqs. 19-24, vectorized).

    Returns (lam (n*K x T), alp (n*(K-1) x T, boolean), constraints_fn) where
    constraints_fn(sum_target) enforces sum(lam)=sum(alp)=sum_target per item
    and the consecutive-lambdas adjacency lam_k <= alp_{k-1} + alp_k.
    """
    lam = cp.Variable((n_items * K, T), nonneg=True, name=f"{prefix}_lam")
    alp = cp.Variable((n_items * (K - 1), T), boolean=True, name=f"{prefix}_alp")

    # per-item summation selectors
    S_lam = sp.kron(sp.eye(n_items), np.ones((1, K)), format="csr")
    S_alp = sp.kron(sp.eye(n_items), np.ones((1, K - 1)), format="csr")
    # adjacency: lam_k <= alp_{k-1} + alp_k  (alp_0 / alp_K absent at the ends)
    A = sp.lil_matrix((K, K - 1))
    for k in range(K):
        if k - 1 >= 0:
            A[k, k - 1] = 1.0
        if k <= K - 2:
            A[k, k] = 1.0
    P_adj = sp.kron(sp.eye(n_items), A.tocsr(), format="csr")

    def cons(sum_target):
        return [
            S_lam @ lam == sum_target,
            S_alp @ alp == sum_target,
            lam <= P_adj @ alp,
        ]

    return lam, alp, cons


def build_pwl_milp(wdn: WDN, K: int = 9,
                   pipe_qmax: Optional[np.ndarray] = None):
    """Construct the one-shot PWL MILP. Returns (problem, handles)."""
    if wdn.M.bypass_index.size or wdn.n_valves or wdn.pump.any_vsp:
        raise NotImplementedError(
            "PWL benchmark covers FSP networks without bypasses/PRVs")
    if K < 3 or K % 2 == 0:
        raise ValueError("K must be odd and >= 3 (0 must be a pipe breakpoint)")

    T = wdn.time
    S = wdn.n_pipes
    U = wdn.n_pumps
    if pipe_qmax is None:
        pipe_qmax = pipe_qmax_from_epanet(wdn)
    R_pipe = np.asarray(wdn.M.Omega.diagonal()).ravel()

    model = SimpleNamespace(
        Heads=cp.Variable((wdn.n_nodes, T), name="Heads"),
        Flows=cp.Variable((wdn.n_links, T), name="Flows"),
        Ppump=cp.Variable((U, T), name="Ppump"),
        OnOff=cp.Variable((U, T), boolean=True, name="OnOff"),
        Hdummy=cp.Variable((wdn.n_tanks, T), name="Hdummy"),
        TankFlow_aux=cp.Variable((wdn.n_tanks, T), name="TankFlow_aux"),
        Speed=None,
    )

    cons = []
    # shared physics, reused verbatim from the study's constraint builder
    cons += C.reservoir_head(model, wdn)
    cons += C.tank_flow_aux(model, wdn)
    cons += C.tank_state_space(model, wdn)
    cons += C.tank_hdummy_head(model, wdn)
    cons += C.closed_pipes_zero(model, wdn)
    cons += C.mass_balance(model, wdn)
    cons += C.pump_flow(model, wdn)
    cons += C.pump_availability(model, wdn)
    cons += C.tank_head_bounds(model, wdn)
    cons += C.junction_head_bounds(model, wdn)
    cons += C.tank_terminal(model, wdn)

    # ---- pipes: K-breakpoint PWL of R sign(q)|q|^1.852 (paper eqs. 17-24) ----
    # breakpoints per pipe: symmetric linspace so 0 (the curve's kink) is exact
    q_bp = np.stack([np.linspace(-pipe_qmax[s], pipe_qmax[s], K)
                     for s in range(S)])                       # (S x K)
    g_bp = R_pipe[:, None] * np.sign(q_bp) * np.abs(q_bp) ** 1.852
    lam_p, alp_p, sos_p = _sos2_blocks(S, K, T, "pipe")
    cons += sos_p(np.ones((S, T)))
    # row-selector matrices carrying the breakpoint values (S x S*K)
    Bq = sp.block_diag([q_bp[s][None, :] for s in range(S)], format="csr")
    Bg = sp.block_diag([g_bp[s][None, :] for s in range(S)], format="csr")
    cons += [wdn.M.Pi_prime @ model.Flows == Bq @ lam_p,
             wdn.M.Pi_telda @ model.Heads == Bg @ lam_p]

    # ---- pumps: K-breakpoint PWL of head gain and power over q in [0, qmax] --
    # (paper Sec. III-B restricted to fixed speed: univariate in q). Sum of the
    # lambdas equals the on/off binary, so an OFF pump has q = P = 0 and its
    # head rows decouple via big-M exactly as in the study's model.
    p = wdn.pump
    qp_bp = np.stack([np.linspace(0.0, p.max_flow[u], K) for u in range(U)])
    hgain = -(p.h0[:, None] - p.r_m[:, None] * qp_bp ** p.v_m[:, None])
    power = p.c_m * (p.h0[:, None] - p.r_m[:, None]
                     * qp_bp ** p.v_m[:, None]) * qp_bp
    lam_u, alp_u, sos_u = _sos2_blocks(U, K, T, "pump")
    cons += sos_u(model.OnOff)
    Bqu = sp.block_diag([qp_bp[u][None, :] for u in range(U)], format="csr")
    Bh = sp.block_diag([hgain[u][None, :] for u in range(U)], format="csr")
    Bp = sp.block_diag([power[u][None, :] for u in range(U)], format="csr")
    cons += [wdn.M.Lambda @ model.Flows == Bqu @ lam_u,
             model.Ppump == Bp @ lam_u]
    Mbig = wdn.config.big_m
    LambdaPiT = wdn.M.Lambda @ wdn.M.Pi.T
    e = (LambdaPiT @ model.Heads) - Bh @ lam_u
    cons += [e >= Mbig * (model.OnOff - 1), e <= Mbig * (1 - model.OnOff)]

    objective = cp.Minimize(
        wdn.price_final @ (cp.sum(model.Ppump, axis=0) / 1000.0))
    problem = cp.Problem(objective, cons)

    n_binary = S * (K - 1) * T + U * (K - 1) * T + U * T
    n_continuous = (S * K * T + U * K * T + model.Heads.size + model.Flows.size
                    + model.Ppump.size + model.Hdummy.size
                    + model.TankFlow_aux.size)
    n_rows = sum(int(np.prod(c.shape)) for c in cons)
    handles = dict(model=model, n_binary=n_binary, n_continuous=n_continuous,
                   n_rows=n_rows, pipe_qmax=pipe_qmax)
    return problem, handles


def solve_pwl_owf(wdn: WDN, K: int = 9, time_limit: float = 600.0,
                  solver: str = "HIGHS", verbose: bool = False,
                  pipe_qmax: Optional[np.ndarray] = None) -> PWLResult:
    """Build and solve the one-shot PWL MILP; score with the TRUE power law."""
    t0 = time.time()
    problem, H = build_pwl_milp(wdn, K=K, pipe_qmax=pipe_qmax)
    build_s = time.time() - t0
    model = H["model"]

    t0 = time.time()
    kwargs = dict(solver=solver, verbose=verbose)
    try:
        problem.solve(time_limit=float(time_limit), **kwargs)
    except TypeError:                      # solver interface without time_limit
        problem.solve(**kwargs)
    solve_s = time.time() - t0

    ok = problem.status in ("optimal", "optimal_inaccurate")
    if not ok or model.Flows.value is None:
        return PWLResult(status=problem.status or "failed", K=K,
                         objective=float("nan"), true_cost=float("nan"),
                         heads=None, flows=None, onoff=None, ppump_pwl=None,
                         ppump_true=None, build_s=build_s, solve_s=solve_s,
                         n_binary=H["n_binary"], n_continuous=H["n_continuous"],
                         n_constraint_rows=H["n_rows"],
                         notes="MILP did not reach optimality")

    flows = np.asarray(model.Flows.value)
    ppump_true = _true_pump_power(wdn, flows)
    true_cost = float(wdn.price_final
                      @ (np.abs(ppump_true).sum(axis=0) / 1000.0))
    return PWLResult(
        status=problem.status, K=K, objective=float(problem.value),
        true_cost=true_cost,
        heads=np.asarray(model.Heads.value), flows=flows,
        onoff=np.round(np.asarray(model.OnOff.value)),
        ppump_pwl=np.asarray(model.Ppump.value), ppump_true=ppump_true,
        build_s=build_s, solve_s=solve_s,
        n_binary=H["n_binary"], n_continuous=H["n_continuous"],
        n_constraint_rows=H["n_rows"])
