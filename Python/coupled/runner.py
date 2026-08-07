"""High-level drivers for the coupled water-power optimization.

``setup`` builds the water + power objects; ``solve_coupled_schedule`` fixes a
pump schedule, warm-starts the coupled problem from EPANET hydraulics under that
schedule (exactly as the water side does), and converges the coupled LP.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Optional

import numpy as np

from owf.config import SolverConfig
from owf.network import WDN, setup as setup_wdn
from owf.warmstart import warmstart_point, epanet_default_onoff
from owf.linearization import linearize, stack_eps

from pdn.network import PDN
from .config import CoupledConfig
from .coupled_lp import solve_coupled, CoupledResult


def setup(net_num: int, cc: CoupledConfig, time: Optional[int] = None,
          price_choice: int = 1, solver: str = "HIGHS") -> tuple[WDN, PDN]:
    """Build the water network and the distribution feeder for a coupled run."""
    wcfg = SolverConfig(net_num=net_num, time=time, price_choice=price_choice,
                        solver=solver, vsp_pumps=cc.vsp_pumps)
    wdn = setup_wdn(wcfg)
    pdn = PDN.build(cc.feeder, pv_sizing=cc.pv_sizing, vmin=cc.vmin, vmax=cc.vmax)
    return wdn, pdn


def solve_coupled_schedule(wdn: WDN, pdn: PDN, cc: CoupledConfig,
                           onoff: np.ndarray, soft_bounds: bool = True,
                           max_iter: int = 20) -> CoupledResult:
    """Fix ``onoff``, warm-start from EPANET, and converge the coupled problem."""
    onoff = np.round(np.asarray(onoff)).astype(float)
    # VSP needs the damped homotopy (the McCormick relaxation) and a few more
    # iterations to settle; FSP keeps the single-shot warm-started convergence.
    vsp = wdn.pump.any_vsp
    cfg = replace(wdn.config, fixed_schedule=onoff, soft_bounds=soft_bounds,
                  damping=0.5 if vsp else 1.0, penalty_weight=1.0e3,
                  penalty_growth=1.2, penalty_max=1.0e5,
                  max_iter=max(max_iter, 80) if vsp else max_iter, feas_tol=2.0)
    wdn_fixed = replace(wdn, config=cfg)
    lin, eps = warmstart_point(wdn_fixed, onoff)
    return solve_coupled(wdn_fixed, pdn, cc, lin_override=lin, eps_override=eps)


def solve_coupled_epanet(wdn: WDN, pdn: PDN, cc: CoupledConfig) -> CoupledResult:
    """Coupled solve reproducing EPANET's own (rule-based) pump schedule."""
    onoff = epanet_default_onoff(wdn)
    return solve_coupled_schedule(wdn, pdn, cc, onoff)
