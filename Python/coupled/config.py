"""Configuration for the coupled Optimal Water-Power Flow (C-OWPF).

A ``CoupledConfig`` bolts a power feeder (``pdn.PDN``) onto a water network
(``owf`` ``SolverConfig``) and says how the two are wired: which PDN bus carries
each water pump's electrical load, how the feeder's own load and solar vary over
the day, and how heavily voltage-limit violations are penalized.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np

# Normalized daily feeder-load shape (fraction of nominal), hour 0..23. A typical
# residential/mixed curve: low overnight, morning and (dominant) evening peaks.
LOAD_PROFILE_24 = np.array([
    0.55, 0.50, 0.47, 0.46, 0.48, 0.55, 0.68, 0.80,
    0.82, 0.80, 0.78, 0.77, 0.78, 0.80, 0.82, 0.85,
    0.90, 0.97, 1.00, 0.98, 0.92, 0.82, 0.70, 0.60,
])

# Power-factor floor for PV inverters -> reactive headroom |q| <= sin(acos(pf))*S.
PV_POWER_FACTOR = 0.9


@dataclass
class CoupledConfig:
    """How a water network and a distribution feeder are co-optimized."""

    feeder: str = "ieee13"                 # pdn.FEEDER_CHOICES key
    # Which non-slack PDN bus (0-based) carries each water pump's electrical load
    # (the paper's Xi coupling matrix; pumps attach to load buses). None -> auto-
    # place pumps at the electrically weakest buses, where coupling is most visible.
    pump_bus: Optional[Sequence[int]] = None
    # Variable-speed pumps: {pump_id: (omega_min, omega_max)}. Listed pumps run at
    # variable speed; the reduced-speed power (~omega^3) lowers the feeder load. None
    # -> all fixed-speed. Passed straight to the water SolverConfig.
    vsp_pumps: Optional[dict] = None
    # PRV pressure-setting overrides {valve_id: P_set_psi}; None -> .inp settings.
    # PRVs draw no electrical power; their effect on the feeder is indirect (less
    # wasted head -> different pump operation -> different bus load).
    prv_settings: Optional[dict] = None
    pump_load_scale: float = 1.0           # amplify pump electrical load (demo aid)
    load_scale: float = 1.0                # scale the feeder base load (Psi p)
    # Base feeder load over the day. None -> STATIC nominal load every hour (as in
    # the MATLAB ConnectMatrices, which uses PDN.p directly); provide a (T,) shape
    # only to profile it.
    load_profile: Optional[np.ndarray] = None
    # PV / DER.  PV *active* is fixed at the available solar (rating * solar
    # profile); PV *reactive* q_pv is the control lever (smart-inverter Volt/VAR).
    enable_pv: bool = True
    pv_sizing: float = 1.2                 # inverter S rating = pv_sizing * bus load
    # Pump motor power factor: the pump draws reactive q_pump = p_pump*sqrt(1/PF^2-1)
    # at its feeder bus (paper eq. 33), coupling pump load to the reactive flow.
    pump_pf: float = 0.9
    # Include the cost of network losses in the objective (paper 33d):
    #   min sum_t price_t * ( pump energy + loss ),  both at the WDN price.
    # This is what makes PV reactive dispatch and loss reduction worth money.
    include_loss: bool = True
    # Voltage limits (pu).  Enforced as HARD constraints (Vmin^2 <= V^2 <= Vmax^2).
    vmin: float = 0.95
    vmax: float = 1.05
    # Soft voltage bounds: the SAME penalty-slack feasibility device the water side
    # uses for head bounds -- NOT an economic objective term.  On stressed feeders
    # a hard limit can be infeasible; enabling this always returns a solution and
    # reports the residual violation, while the optimized cost stays the paper's
    # pure pump-energy cost.  Off -> hard limits (may return `infeasible`).
    soft_voltage: bool = True
    voltage_penalty: float = 1.0e4         # feasibility-slack weight (not economic)

    def load_shape(self, T: int) -> np.ndarray:
        if self.load_profile is None:
            return np.full(T, self.load_scale)     # static nominal load
        prof = np.asarray(self.load_profile, float)
        if T == len(prof):
            out = prof.copy()
        else:
            reps = int(np.ceil(T / len(prof)))
            out = np.tile(prof, reps)[:T]
        return out * self.load_scale
