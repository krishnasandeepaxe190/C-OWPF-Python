"""Configuration for the FSP Optimal Water Flow (OWF) solver.

Mirrors the per-network constants that ``Prepare_net_WDN.m`` hard-codes in the
MATLAB implementation (pump curve coefficients, which ``.inp`` file to load),
plus the successive-linearization / solver settings.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

# Repository-relative data directory (Python/data/...).
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Pump power coefficient  c_m = 0.7457 / (3960 * eff),  eff = 0.81  [kW per (gpm*ft)]
# 0.7457 converts hp -> kW; 3960 converts gpm*ft -> hp.
C_M = 0.7457 / (3960 * 0.81)

# Tank integration constant: converts gpm -> ft^3/hr over a 1-hour step
# (matches ``del_tk = 1 * 8.0208`` in defineCommonOptimizationParametersWDN.m).
DEL_TK = 8.0208


@dataclass(frozen=True)
class NetworkSpec:
    """Everything network-specific needed to build the OWF problem.

    ``pump_coefficients`` is an optional override: one ``[h0, r, v]`` row per pump
    giving the pump head-gain curve  H_gain(q) = h0 - r * q**v.  When ``None``
    (the default) the coefficients are derived from the EPANET pump head curve,
    which keeps the optimizer consistent with EPANET and avoids the value drift
    seen in the hard-coded ``Prepare_net_WDN.m`` tables.
    """

    net_num: int
    name: str
    inp_relpath: str  # relative to DATA_DIR
    pump_coefficients: Optional[List[List[float]]] = None

    @property
    def inp_path(self) -> Path:
        return DATA_DIR / self.inp_relpath


# Registry of supported FSP networks. Only the 8-node case ships with data for
# now; 3-node / Net1 specs are kept here so they drop in once their .inp files
# are copied into Python/data/.
# Pump coefficients are derived from each network's EPANET head curve by default
# (pump_coefficients=None). Set an explicit override only to force specific values.
NETWORKS: dict[int, NetworkSpec] = {
    8: NetworkSpec(
        net_num=8,
        name="eightnode",
        inp_relpath="eightnode/tutorial8node_modified.inp",
    ),
    3: NetworkSpec(
        net_num=3,
        name="threenode",
        inp_relpath="threenode/Threenodes-gp_largepump.inp",
    ),
    11: NetworkSpec(
        net_num=11,
        name="net1",
        inp_relpath="net1/Net1_Shen_extendedtime.inp",
    ),
}

# Time-of-use price pattern (24h) from WDN_setup_IEEE_ACCESS.m.
PRICE_PATTERN = [
    24, 23, 22, 20, 20.5, 23, 27, 27.5, 29, 32, 36, 36.5,
    37, 40, 42, 45, 46, 46, 40, 37.5, 40, 35, 27.5, 25,
]
PRICE_BASE = 0.005


@dataclass
class SolverConfig:
    """Successive-linearization loop and MILP solver settings."""

    net_num: int = 8
    time: Optional[int] = None       # horizon; None -> use EPANET pattern length
    price_choice: int = 1            # 1 = time-of-use price, 0 = flat price of 1
    choice: int = 1                  # 1 = initialize from EPANET, 0 = user-defined
    tol: float = 0.5                 # convergence tolerance on ||[H;Q;OnOff]|| change
    max_iter: int = 50               # MATLAB used inf; a finite cap is safer
    big_m: float = 10e06             # Big-M for pump on/off head-gain constraints
    # MATLAB quirk: mass balance is disabled for the first 10 iterations. Off by
    # default here (physically-correct: always enforce mass balance).
    mass_balance_warmup: bool = False
    mass_balance_warmup_iters: int = 10
    # Warm-start controls for hard (e.g. looped) networks:
    #  * damping < 1 relinearizes around a blend of the new and previous flow
    #    fields, keeping the successive linearization inside a trust region.
    #  * soft_bounds penalizes head-bound / terminal violations with slacks
    #    (penalty CCP), so every MILP is feasible and the slacks are driven to
    #    zero as the linearization improves. penalty_growth ramps the weight.
    damping: float = 1.0
    fixed_schedule: Optional["np.ndarray"] = None  # if set, pump OnOff is fixed to it
    soft_bounds: bool = False
    penalty_weight: float = 1.0e3
    penalty_growth: float = 2.0
    penalty_max: float = 1.0e7
    feas_tol: float = 1.0e-3   # max slack accepted as "feasible" when soft_bounds
    solver: str = "HIGHS"
    verbose: bool = False
    data_dir: Path = field(default=DATA_DIR)

    @property
    def spec(self) -> NetworkSpec:
        if self.net_num not in NETWORKS:
            raise ValueError(
                f"Unknown net_num={self.net_num}. Available: {sorted(NETWORKS)}"
            )
        return NETWORKS[self.net_num]
