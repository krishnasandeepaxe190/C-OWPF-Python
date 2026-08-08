"""Coupled pump-schedule optimization (voltage-aware).

Extends the water side's honest schedule search (``owf.warmstart.optimize_schedule``)
to the interdependent problem: every candidate schedule is scored by *fixing* it,
converging the coupled LP, and reading (a) the TRUE nonlinear pump-energy cost --
the paper's objective -- and (b) the feeder voltage feasibility.  A schedule that
pumps into a peak-load hour may be cheap on the water side yet cause an
undervoltage; scoring on the coupled solve captures that trade-off.

Stages:
  1. Baseline    -- EPANET's own (rule-based) schedule.
  2. Candidates  -- price-quantile schedules (pump only during the cheapest hours).
  3. Trust-region MILP -- free pump binaries, warm-started at the incumbent's
     linearization, with a Hamming-distance cap of K flips from the incumbent.
     This is the MATLAB-style free-binary MILP made reliable: the warm start keeps
     the linearization valid and the trust region blocks the far-away schedule
     flips that make a single cold MILP return junk.
  4. Polish      -- 1-opt single pump-hour flips.

Ranking: coupled-feasible first (water head slack <= feas_tol AND voltage
violation <= v_tol), then lowest true pump-energy cost.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from owf.linearization import linearize, stack_eps
from owf.warmstart import (epanet_default_onoff, price_threshold_schedules,
                           true_energy_cost, _apply_availability)

from .coupled_lp import solve_coupled, CoupledResult
from .runner import solve_coupled_schedule


def _score(wdn, res: CoupledResult, feas_tol: float, v_tol: float):
    """Rank key: (infeasible?, total violation, total coupled cost pump+loss)."""
    if res is None or res.flows is None or res.flows.size == 0:
        return (2, 1e9, 1e9), 1e9, 1e9
    wsl = res.water_max_slack if np.isfinite(res.water_max_slack) else 0.0
    vviol = res.v_violation if np.isfinite(res.v_violation) else 1e9
    infeas = (wsl > feas_tol) or (vviol > v_tol)
    viol = max(0.0, wsl - feas_tol) + max(0.0, vviol - v_tol)
    # true pump cost (honest, nonlinear) + priced network loss from the coupled solve
    pump_cost = true_energy_cost(wdn, res.flows, getattr(res, "speed", None))
    loss_cost = res.loss_cost if np.isfinite(res.loss_cost) else 0.0
    cost = pump_cost + loss_cost
    return (1 if infeas else 0, round(viol, 4), cost), cost, vviol


def optimize_coupled_schedule(wdn, pdn, cc, v_tol: float = 0.02,
                              feas_tol: float = 2.0, inner_iter: int = 15,
                              trust_k: int | None = None, polish: bool = True,
                              max_flips: int = 20, verbose: bool = True,
                              use_milp: bool = True, candidates: dict | None = None):
    """Search for the pump schedule minimizing coupled cost + voltage feasibility.

    ``candidates`` overrides the default price-quantile candidate set (pass a small
    dict to cut runtime on big networks); ``use_milp=False`` skips the trust-region
    MILP (the most expensive stage on a large feeder + many-pump network).

    Returns (best CoupledResult, info dict).
    """
    def evaluate(sched):
        s = _apply_availability(wdn, np.round(sched))
        r = solve_coupled_schedule(wdn, pdn, cc, s, soft_bounds=True, max_iter=inner_iter)
        key, cost, vviol = _score(wdn, r, feas_tol, v_tol)
        return r, s, key, cost, vviol

    trace = []

    # --- stage 1: baseline (EPANET) --------------------------------------------
    base_sched = epanet_default_onoff(wdn)
    best, best_sched, best_key, best_cost, best_v = evaluate(base_sched)
    baseline_cost, baseline_v = best_cost, best_v
    seen = {best_sched.tobytes()}
    trace.append(("epanet", best_cost, best_v))
    if verbose:
        print(f"[coupled-opt] baseline (EPANET): cost={best_cost:.5f} Vviol={best_v:.4f}")

    # --- stage 2: price-quantile candidates ------------------------------------
    cand_set = candidates if candidates is not None else price_threshold_schedules(wdn)
    for name, sched in cand_set.items():
        s = _apply_availability(wdn, np.round(sched))
        if s.tobytes() in seen:
            continue
        seen.add(s.tobytes())
        r, s, key, cost, vviol = evaluate(s)
        trace.append((name, cost, vviol))
        if verbose:
            print(f"[coupled-opt] {name:16s} cost={cost:.5f} Vviol={vviol:.4f} "
                  f"{'ACCEPT' if key < best_key else ''}")
        if key < best_key:
            best, best_sched, best_key, best_cost, best_v = r, s, key, cost, vviol

    # --- stage 3: warm-started trust-region MILP -------------------------------
    milp_info = None
    try:
        if not use_milp:
            raise StopIteration  # skip the MILP stage on heavy cases
        K = trust_k if trust_k is not None else max(2, (wdn.n_pumps * wdn.time) // 4)
        lin = linearize(best.flows, wdn.M, wdn.pump)
        eps = stack_eps(best.heads, best.flows, best_sched)
        cfg = replace(wdn.config, fixed_schedule=None, soft_bounds=True, damping=1.0,
                      penalty_weight=1.0e3, penalty_growth=1.0, penalty_max=1.0e4,
                      max_iter=2, feas_tol=feas_tol)
        milp = solve_coupled(replace(wdn, config=cfg), pdn, cc, lin_override=lin,
                             eps_override=eps, trust_region=(best_sched, K))
        if milp.onoff is not None and np.asarray(milp.onoff).size:
            s = _apply_availability(wdn, np.round(milp.onoff))
            milp_info = {"K": K, "flips": int(np.abs(s - best_sched).sum())}
            if s.tobytes() not in seen:
                seen.add(s.tobytes())
                r, s, key, cost, vviol = evaluate(s)
                trace.append(("trust_milp", cost, vviol))
                if verbose:
                    print(f"[coupled-opt] {'trust_milp':16s} cost={cost:.5f} "
                          f"Vviol={vviol:.4f} (K={K}) "
                          f"{'ACCEPT' if key < best_key else ''}")
                if key < best_key:
                    best, best_sched, best_key, best_cost, best_v = r, s, key, cost, vviol
    except Exception as exc:
        if verbose:
            print(f"[coupled-opt] trust-region MILP skipped: {exc}")

    # --- stage 4: 1-opt polish -------------------------------------------------
    n_flips = 0
    if polish:
        improved = True
        while improved and n_flips < max_flips:
            improved = False
            order = sorted(range(wdn.time), key=lambda t: -wdn.price_final[t])
            for p in range(wdn.n_pumps):
                for t in order:
                    if n_flips >= max_flips:
                        break
                    avail = wdn.pump_avail or {}
                    if p in avail and not (avail[p][0] <= t < avail[p][1]):
                        continue
                    s = best_sched.copy()
                    s[p, t] = 1.0 - s[p, t]
                    if s.tobytes() in seen:
                        continue
                    seen.add(s.tobytes())
                    n_flips += 1
                    r, s, key, cost, vviol = evaluate(s)
                    if key < best_key:
                        if verbose:
                            print(f"[coupled-opt] polish flip pump{p} hour{t}: "
                                  f"cost={cost:.5f} Vviol={vviol:.4f} ACCEPT")
                        best, best_sched, best_key, best_cost, best_v = r, s, key, cost, vviol
                        trace.append((f"flip_p{p}_t{t}", cost, vviol))
                        improved = True
                        break
                if improved:
                    break

    # --- final clean convergence -------------------------------------------------
    # The candidate/polish solves cap iterations for fast ranking, so a candidate
    # can win on an under-converged score yet be worse once fully solved. Re-solve
    # BOTH the winner and the EPANET baseline cleanly and keep the genuinely best,
    # which guarantees the coupled result is never worse than the decoupled one.
    # replay_polish: on VSP networks, finish each final with the EPANET
    # speed-pinned polish so the returned result replays at the solved speeds.
    finals = []
    for sched in (best_sched, base_sched):
        f = solve_coupled_schedule(wdn, pdn, cc, sched, soft_bounds=True,
                                   max_iter=max(inner_iter, 25),
                                   replay_polish=True)
        if f.flows is not None and f.flows.size:
            fkey, fcost, fv = _score(wdn, f, feas_tol, v_tol)
            finals.append((fkey, fcost, fv, f, _apply_availability(wdn, np.round(sched))))
    if finals:
        fkey, fcost, fv, f, fsched = min(finals, key=lambda t: t[0])
        best, best_cost, best_v, best_sched = f, fcost, fv, fsched

    info = {
        "baseline_cost": baseline_cost,
        "baseline_vviol": baseline_v,
        "best_cost": best_cost,
        "best_vviol": best_v,
        "savings_pct": (100.0 * (baseline_cost - best_cost) / baseline_cost
                        if baseline_cost else 0.0),
        "schedule": best_sched,
        "trace": trace,
        "n_flips": n_flips,
        "trust_milp": milp_info,
    }
    return best, info
