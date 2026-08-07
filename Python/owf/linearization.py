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
    # Variable-speed pumps (VSP). ``is_vsp`` marks which pumps run at variable
    # speed; the relative speed omega is bounded in [omega_min, omega_max]. All
    # default to fixed-speed (omega == 1), which reproduces the FSP model exactly.
    is_vsp: np.ndarray = None     # (Pu,) bool
    omega_min: np.ndarray = None  # (Pu,)
    omega_max: np.ndarray = None  # (Pu,)

    def __post_init__(self):
        n = self.h0.shape[0]
        if self.is_vsp is None:
            self.is_vsp = np.zeros(n, dtype=bool)
        if self.omega_min is None:
            self.omega_min = np.ones(n)
        if self.omega_max is None:
            self.omega_max = np.ones(n)

    @property
    def any_vsp(self) -> bool:
        return bool(np.any(self.is_vsp))


@dataclass
class LinPoint:
    """Frozen linearization coefficients for one MILP solve."""
    Cp: np.ndarray          # (P_pipes x T)
    C1M: np.ndarray         # (Pu x T)   head-gain: -h0 * <omega>  (=-h0 for FSP)
    C2M: np.ndarray         # (Pu x T)   head-gain flow slope r <f>^(v-1)
    APrimePump: np.ndarray  # (Pu x T)   pump-power flow coefficient
    BPrimePump: np.ndarray  # (Pu x T)   pump-power on/off coefficient
    Cp_bypass: np.ndarray = None  # (Nb x T) head-loss lin. for switched bypasses
    DPrime: np.ndarray = None     # (Pu x T) VSP power coeff on WW=omega*f (0 for FSP)


def linearize(flows: np.ndarray, M: Matrices, pump: PumpParams,
              speed: np.ndarray = None) -> LinPoint:
    """Build linearization coefficients around a flow field ``flows`` (L x T).

    ``speed`` (Pu x T relative pump speed) is used only for variable-speed pumps;
    when ``None`` it defaults to 1 (fixed speed), which recovers the FSP model.
    """
    pipe_flows = M.Pi_prime @ flows          # (P x T)
    pump_flows = M.Lambda @ flows            # (Pu x T)
    T = flows.shape[1]
    n_pumps = pump.h0.shape[0]

    h0 = pump.h0[:, None]
    r = pump.r_m[:, None]
    v = pump.v_m[:, None]
    absf = np.abs(pump_flows)                 # pump flows are >= 0; abs guards f^v
    if speed is None:
        speed = np.ones((n_pumps, T))

    # Hazen-Williams pipe head loss linearization (minor losses are ignored, to
    # match the study; the EPANET data files carry no minor-loss coefficients).
    Cp = pipe_flows * (M.Omega @ np.abs(pipe_flows) ** 0.852 - np.ones_like(pipe_flows))

    # same linearization for switched-bypass pipes (if any)
    if M.bypass_index.size:
        bp_flows = M.Pi_prime_bypass @ flows
        Cp_bypass = bp_flows * (M.Omega_bypass @ np.abs(bp_flows) ** 0.852 - np.ones_like(bp_flows))
    else:
        Cp_bypass = np.zeros((0, T))

    # Pump head-gain curve  H_gain = h0 omega^2 - r f^v  (general exponent v).
    # Linearized as -(C1M omega + C2M f) with C1M = -h0 <omega>, C2M = r <f>^(v-1)
    # (for FSP, <omega>=1 so C1M = -h0, matching the fixed-speed model).
    C1M = -h0 * speed
    C2M = r * absf ** (v - 1)

    # FSP power  P(f) = c_m (h0 - r f^v) f,  first-order Taylor about pump_flows.
    # dP/df = c_m (h0 - (v+1) r f^v);  reduces to c_m(h0 - 3 r f^2) when v = 2.
    pump_power = pump.c_m * (h0 - r * absf ** v) * pump_flows
    pump_prime = pump.c_m * (h0 - (v + 1) * r * absf ** v)
    APrimePump = np.array(np.broadcast_to(pump_prime, (n_pumps, T)), dtype=float)
    BPrimePump = np.array(np.broadcast_to(pump_power - pump_prime * pump_flows,
                                          (n_pumps, T)), dtype=float)
    DPrime = np.zeros((n_pumps, T))

    # VSP power (paper eq. linearVSPpower): c_m(-2 C2M <f> f + C2M <f>^2 x + h0<omega> WW),
    # with WW = omega*f (McCormick) folding the bilinear omega*f term. c_m is folded
    # into the coefficients here so the unified power constraint form applies to all
    # pumps. FSP rows keep the concave-cubic Taylor coefficients above (DPrime = 0).
    if pump.any_vsp:
        vs = np.asarray(pump.is_vsp, dtype=bool)
        APrimePump[vs] = pump.c_m * (-2.0 * C2M[vs] * pump_flows[vs])
        BPrimePump[vs] = pump.c_m * (C2M[vs] * pump_flows[vs] ** 2)
        DPrime[vs] = pump.c_m * (h0[vs] * speed[vs])       # = -c_m * C1M[vs]

    return LinPoint(Cp=Cp, C1M=C1M, C2M=C2M, APrimePump=APrimePump,
                    BPrimePump=BPrimePump, Cp_bypass=Cp_bypass, DPrime=DPrime)


def stack_eps(heads: np.ndarray, flows: np.ndarray, onoff: np.ndarray) -> np.ndarray:
    """Concatenate [Heads; Flows; OnOff] (the successive-linearization iterate)."""
    return np.vstack([heads, flows, onoff])


def error_norm(eps: np.ndarray, prev_eps: np.ndarray) -> float:
    """Euclidean norm of the change in the stacked iterate."""
    return float(np.linalg.norm(eps - prev_eps))
