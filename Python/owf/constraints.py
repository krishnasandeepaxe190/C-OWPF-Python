"""OWF constraints in CVXPY (ports the define*_CVX.m modules, one per function).

Each function returns a list of CVXPY constraints. The linearization
coefficients (Cp, C1M, C2M, A', B') are passed as ``cp.Parameter`` objects so
the problem is compiled once and only the numbers change between iterations.
"""
from __future__ import annotations

import cvxpy as cp
import numpy as np


def reservoir_head(model, wdn):
    """Theta H == Theta H_min  (fix reservoir heads to their elevation)."""
    Th = wdn.M.Theta
    return [Th @ model.Heads == Th @ wdn.bounds.min_nodal_heads]


def tank_flow_aux(model, wdn):
    """TankFlow_aux == Tau' (-Pi) Flows  (net inflow to each tank)."""
    C = wdn.M.Tau.T @ (-wdn.M.Pi)          # (Tk x L)
    return [model.TankFlow_aux == C @ model.Flows]


def tank_state_space(model, wdn):
    """Hdummy_i = A_tk H0_i + (del/A_i) Btk TankFlow_aux_i  (integrator)."""
    tank = wdn.tank
    cons = []
    for i in range(wdn.n_tanks):
        cons.append(
            model.Hdummy[i, :]
            == tank.Atk * tank.init_head[i]
            + tank.del_tk_tanks[i] * (tank.Btk @ model.TankFlow_aux[i, :])
        )
    return cons


def tank_hdummy_head(model, wdn):
    """Tau' H == [H0 , Hdummy(:,1:T-1)]  (tank head = previous dummy level)."""
    tank = wdn.tank
    rhs = cp.hstack([tank.init_head[:, None], model.Hdummy[:, : wdn.time - 1]])
    return [wdn.M.Tau.T @ model.Heads == rhs]


def tank_head_bounds(model, wdn):
    """H_min <= tank heads <= H_max (expressed over Tau)."""
    Tau = wdn.M.Tau
    lo = wdn.bounds.min_nodal_heads.T @ Tau
    hi = wdn.bounds.max_nodal_heads.T @ Tau
    return [model.Heads.T @ Tau >= lo, model.Heads.T @ Tau <= hi]


def pipe_head_loss(model, wdn):
    """Pi_telda H == Cp + Pi_prime Q  (linearized Hazen-Williams)."""
    return [wdn.M.Pi_telda @ model.Heads == model.Cp + wdn.M.Pi_prime @ model.Flows]


def mass_balance(model, wdn):
    """Pi_reduced Q == -demand  (conservation of mass at junctions)."""
    return [wdn.M.Pi_reduced @ model.Flows == -wdn.junction_demand_profile]


def pump_flow(model, wdn):
    """q_min OnOff <= Lambda Q <= q_max OnOff  (flow only when pump is on)."""
    T = wdn.time
    qmax = np.tile(wdn.pump.max_flow[:, None], (1, T))
    pump_flows = wdn.M.Lambda @ model.Flows
    return [
        pump_flows >= 0,                                   # q_min = 0
        pump_flows <= cp.multiply(qmax, model.OnOff),
    ]


def pump_negative_headloss(model, wdn):
    """C1M + C2M (Lambda Q) <= 0."""
    return [model.C1M + cp.multiply(model.C2M, wdn.M.Lambda @ model.Flows) <= 0]


def pump_power(model, wdn):
    """Ppump == A'(Lambda Q) + B' OnOff  (linearized FSP power)."""
    pump_flows = wdn.M.Lambda @ model.Flows
    return [
        model.Ppump
        == cp.multiply(model.APrime, pump_flows) + cp.multiply(model.BPrime, model.OnOff)
    ]


def pump_bigm(model, wdn):
    """Big-M pump head-gain curve, enforced only when OnOff == 1."""
    M_big = wdn.config.big_m
    LambdaPiT = wdn.M.Lambda @ wdn.M.Pi.T          # (Pu x N)
    expr = (LambdaPiT @ model.Heads) - (
        model.C1M + cp.multiply(model.C2M, wdn.M.Lambda @ model.Flows)
    )
    return [
        expr >= M_big * (model.OnOff - 1),
        expr <= M_big * (1 - model.OnOff),
    ]


def junction_head_bounds(model, wdn):
    """H_min <= junction heads <= H_max."""
    K = wdn.M.Kappa
    return [
        K @ model.Heads >= K @ wdn.bounds.min_nodal_heads,
        K @ model.Heads <= K @ wdn.bounds.max_nodal_heads,
    ]


def tank_terminal(model, wdn):
    """Hdummy(:,end) >= tank mean head  (terminal level target)."""
    return [model.Hdummy[:, -1] >= wdn.tank.mean_head]


def build_constraints(model, wdn, include_mass_balance: bool = True):
    """Assemble the full FSP OWF constraint set."""
    cons = []
    cons += reservoir_head(model, wdn)
    cons += tank_flow_aux(model, wdn)
    cons += tank_state_space(model, wdn)
    cons += tank_hdummy_head(model, wdn)
    cons += tank_head_bounds(model, wdn)
    cons += pipe_head_loss(model, wdn)
    if include_mass_balance:
        cons += mass_balance(model, wdn)
    cons += pump_flow(model, wdn)
    cons += pump_negative_headloss(model, wdn)
    cons += pump_power(model, wdn)
    cons += pump_bigm(model, wdn)
    cons += junction_head_bounds(model, wdn)
    cons += tank_terminal(model, wdn)
    return cons


def objective(model, wdn):
    """min  sum_t price(t) * sum_pumps(Ppump)/1000   (energy cost)."""
    total_power = cp.sum(model.Ppump, axis=0)          # (T,)
    return cp.Minimize(wdn.price_final @ (total_power / 1000.0))
