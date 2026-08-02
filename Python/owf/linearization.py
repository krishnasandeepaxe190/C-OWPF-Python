"""Successive-linearization coefficients (ports CalculateNewIterationValues.m).

Given a flow field, compute the first-order coefficients the MILP uses to
approximate the two nonlinearities in the FSP OWF problem:

  * Hazen-Williams pipe head loss  ->  Cp
  * FSP pump power  P(q) = c_m (h0 - r q^2) q  ->  A'q + B'  (Taylor about q)

and the pump head-gain curve linearization coefficients C1M, C2M.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .connection_matrices import Matrices


@dataclass
class PumpParams:
    h0: np.ndarray       # (Pu,)  pump shutoff head
    r_m: np.ndarray      # (Pu,)  pump curve resistance
    v_m: np.ndarray      # (Pu,)  pump curve exponent (2 for FSP power model)
    c_m: float           # scalar power coefficient
    max_flow: np.ndarray # (Pu,)  sqrt(h0/r)


@dataclass
class LinPoint:
    """Frozen linearization coefficients for one MILP solve."""
    Cp: np.ndarray          # (P_pipes x T)
    C1M: np.ndarray         # (Pu x T)
    C2M: np.ndarray         # (Pu x T)
    APrimePump: np.ndarray  # (Pu x T)
    BPrimePump: np.ndarray  # (Pu x T)


def linearize(flows: np.ndarray, M: Matrices, pump: PumpParams) -> LinPoint:
    """Build linearization coefficients around a flow field ``flows`` (L x T)."""
    pipe_flows = M.Pi_prime @ flows          # (P x T)
    pump_flows = M.Lambda @ flows            # (Pu x T)
    T = flows.shape[1]
    n_pumps = pump.h0.shape[0]

    # Hazen-Williams pipe head loss linearization
    Cp = pipe_flows * (M.Omega @ np.abs(pipe_flows) ** 0.852 - np.ones_like(pipe_flows))

    # Pump head-gain curve:  H_gain(q) ~ -(C1M + C2M q) = h0 - r q^(v-1) q
    C1M = -pump.h0[:, None] * np.ones((n_pumps, T))
    C2M = np.zeros((n_pumps, T))
    for i in range(n_pumps):
        C2M[i, :] = pump.r_m[i] * pump_flows[i, :] ** (pump.v_m[i] - 1)

    # FSP power  P(q) = c_m (h0 - r q^2) q,  first-order Taylor about pump_flows
    pump_power = pump.c_m * (pump.h0[:, None] - pump.r_m[:, None] * pump_flows ** 2) * pump_flows
    pump_prime = pump.c_m * (pump.h0[:, None] - 3 * pump.r_m[:, None] * pump_flows ** 2)
    APrimePump = pump_prime
    BPrimePump = pump_power - pump_prime * pump_flows

    return LinPoint(Cp=Cp, C1M=C1M, C2M=C2M, APrimePump=APrimePump, BPrimePump=BPrimePump)


def stack_eps(heads: np.ndarray, flows: np.ndarray, onoff: np.ndarray) -> np.ndarray:
    """Concatenate [Heads; Flows; OnOff] (the successive-linearization iterate)."""
    return np.vstack([heads, flows, onoff])


def error_norm(eps: np.ndarray, prev_eps: np.ndarray) -> float:
    """Euclidean norm of the change in the stacked iterate."""
    return float(np.linalg.norm(eps - prev_eps))
