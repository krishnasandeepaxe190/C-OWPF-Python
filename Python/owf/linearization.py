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
    Cp_bypass: np.ndarray = None  # (Nb x T) head-loss lin. for switched bypasses


def linearize(flows: np.ndarray, M: Matrices, pump: PumpParams) -> LinPoint:
    """Build linearization coefficients around a flow field ``flows`` (L x T)."""
    pipe_flows = M.Pi_prime @ flows          # (P x T)
    pump_flows = M.Lambda @ flows            # (Pu x T)
    T = flows.shape[1]
    n_pumps = pump.h0.shape[0]

    h0 = pump.h0[:, None]
    r = pump.r_m[:, None]
    v = pump.v_m[:, None]
    absf = np.abs(pump_flows)                 # pump flows are >= 0; abs guards f^v

    # Hazen-Williams pipe head loss linearization
    Cp = pipe_flows * (M.Omega @ np.abs(pipe_flows) ** 0.852 - np.ones_like(pipe_flows))

    # same linearization for switched-bypass pipes (if any)
    if M.bypass_index.size:
        bp_flows = M.Pi_prime_bypass @ flows
        Cp_bypass = bp_flows * (M.Omega_bypass @ np.abs(bp_flows) ** 0.852 - np.ones_like(bp_flows))
    else:
        Cp_bypass = np.zeros((0, T))

    # Pump head-gain curve  H_gain(f) = h0 - r f^v  (general exponent v).
    # Monomial linearization freezes the coefficient: r f^v ~ (r f_prev^(v-1)) f,
    # so the enforced gain is -(C1M + C2M f) with C1M = -h0, C2M = r f_prev^(v-1).
    C1M = -h0 * np.ones((n_pumps, T))
    C2M = r * absf ** (v - 1)

    # FSP power  P(f) = c_m (h0 - r f^v) f,  first-order Taylor about pump_flows.
    # dP/df = c_m (h0 - (v+1) r f^v);  reduces to c_m(h0 - 3 r f^2) when v = 2.
    pump_power = pump.c_m * (h0 - r * absf ** v) * pump_flows
    pump_prime = pump.c_m * (h0 - (v + 1) * r * absf ** v)
    APrimePump = pump_prime
    BPrimePump = pump_power - pump_prime * pump_flows

    return LinPoint(Cp=Cp, C1M=C1M, C2M=C2M, APrimePump=APrimePump,
                    BPrimePump=BPrimePump, Cp_bypass=Cp_bypass)


def stack_eps(heads: np.ndarray, flows: np.ndarray, onoff: np.ndarray) -> np.ndarray:
    """Concatenate [Heads; Flows; OnOff] (the successive-linearization iterate)."""
    return np.vstack([heads, flows, onoff])


def error_norm(eps: np.ndarray, prev_eps: np.ndarray) -> float:
    """Euclidean norm of the change in the stacked iterate."""
    return float(np.linalg.norm(eps - prev_eps))
