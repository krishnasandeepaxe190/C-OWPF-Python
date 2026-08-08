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
    # Fallbacks mirror main_owf: try the bundled solvers that are NOT the primary.
    from owf.config import DEFAULT_SOLVER, DEFAULT_FALLBACK, available_solvers
    fallbacks = tuple(s for s in (DEFAULT_SOLVER, DEFAULT_FALLBACK)
                      if s != solver and s in available_solvers())
    wcfg = SolverConfig(net_num=net_num, time=time, price_choice=price_choice,
                        solver=solver, fallback_solvers=fallbacks,
                        vsp_pumps=cc.vsp_pumps, prv_settings=cc.prv_settings)
    wdn = setup_wdn(wcfg)
    pdn = PDN.build(cc.feeder, pv_sizing=cc.pv_sizing, vmin=cc.vmin, vmax=cc.vmax)
    return wdn, pdn


def solve_coupled_schedule(wdn: WDN, pdn: PDN, cc: CoupledConfig,
                           onoff: np.ndarray, soft_bounds: bool = True,
                           max_iter: int = 20,
                           replay_polish: bool = False) -> CoupledResult:
    """Fix ``onoff``, warm-start from EPANET, and converge the coupled problem.

    ``replay_polish=True`` (VSP networks only) finishes with the EPANET
    speed-pinned polish -- the same fix the decoupled path uses -- so the
    returned heads/flows replay in EPANET at the solved speeds.
    """
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
    res = solve_coupled(wdn_fixed, pdn, cc, lin_override=lin, eps_override=eps)
    # VSP: speeds must be pinned at the replay point (McCormick gap). PRV: the
    # binaries are exact but the pipe linearization can sit off the replay flows
    # near the valve -- the same schedule-pinned reconverge closes that too.
    if replay_polish and (vsp or wdn.n_valves > 0):
        res = _replay_polish(wdn_fixed, pdn, cc, res)
    return res


def _replay_polish(wdn: WDN, pdn: PDN, cc: CoupledConfig,
                   res: CoupledResult, rounds: int = 2) -> CoupledResult:
    """EPANET speed-pinned polish for a coupled VSP result.

    The damped homotopy can stop at a point where the McCormick relaxation gap
    leaves the model's pump flow far from what EPANET's affinity-scaled curve
    delivers at the solved speeds (tens of ft / hundreds of GPM of replay
    error). Mirror of the decoupled fix in ``main_owf``: replay (schedule,
    speeds) in EPANET, re-converge the COUPLED problem with the schedule AND
    speeds pinned (with omega fixed, WW = omega*f is exact) linearized at the
    replay point, and keep whichever candidate -- including the original --
    replays best. Monotone by construction.
    """
    if res is None or res.flows is None or not getattr(res.flows, "size", 0):
        return res
    try:
        from owf.epanet_io import simulate_with_schedule

        T = wdn.time
        pl = (wdn.raw.link_pump_index + 1).tolist()
        bl = [(int(lk) + 1, int(np.argmax(wdn.M.S_bypass_pump[i])))
              for i, lk in enumerate(wdn.M.bypass_index)]

        def _replay_err(r):
            s = np.round(r.onoff)
            h_ep, f_ep = simulate_with_schedule(
                wdn.spec.inp_path, pl, s, T, wdn.n_nodes, wdn.n_links,
                bypass_links=bl, pump_speeds=r.speed)
            return float(np.max(np.abs(h_ep - r.heads[:, :T]))), h_ep, f_ep

        best, cur = res, res
        best_err, h_ep, f_ep = _replay_err(res)
        for _ in range(rounds):
            sched = np.round(cur.onoff)
            cfg2 = replace(wdn.config, fixed_schedule=sched,
                           fixed_speed=cur.speed, damping=0.7, max_iter=40)
            wdn2 = replace(wdn, config=cfg2)
            lin = linearize(f_ep, wdn.M, wdn.pump, speed=cur.speed)
            r2 = solve_coupled(wdn2, pdn, cc, lin_override=lin,
                               eps_override=stack_eps(h_ep, f_ep, sched))
            if r2.flows is None or not r2.flows.size:
                break
            err2, h2, f2 = _replay_err(r2)
            if err2 < best_err - 1e-6:
                best, best_err = r2, err2
            cur, h_ep, f_ep = r2, h2, f2
        return best
    except Exception:
        return res


def solve_coupled_epanet(wdn: WDN, pdn: PDN, cc: CoupledConfig,
                         replay_polish: bool = False) -> CoupledResult:
    """Coupled solve reproducing EPANET's own (rule-based) pump schedule."""
    onoff = epanet_default_onoff(wdn)
    return solve_coupled_schedule(wdn, pdn, cc, onoff, replay_polish=replay_polish)
