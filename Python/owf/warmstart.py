"""EPANET-based multi-start warm-starting for the successive approximation.

Idea (suggested for hard/looped networks): instead of linearizing around a single
default point, generate several candidate pump on/off schedules, run EPANET with
each schedule *imposed* to get a physically-consistent set of flows, seed the
successive-linearization loop from each, and keep the best result. Starting from a
feasible EPANET operating point puts the linearization in a good basin.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Optional

import numpy as np

from .epanet_io import run_epanet, simulate_with_schedule
from .linearization import linearize, stack_eps
from .network import WDN
from .solver import OWFResult, solve_owf


def epanet_default_onoff(wdn: WDN, thresh: float = 1e-3) -> np.ndarray:
    """Pump on/off implied by EPANET's own (rule-based) operation."""
    flows_ep, _, _, _ = run_epanet(wdn.raw)
    T = wdn.time
    pump_flows = wdn.M.Lambda @ flows_ep[:T, :].T   # (Pu x T)
    return (np.abs(pump_flows) > thresh).astype(float)


def candidate_schedules(wdn: WDN) -> dict[str, np.ndarray]:
    """A small library of pump on/off schedules to warm-start from."""
    T, P = wdn.time, wdn.n_pumps
    scheds: dict[str, np.ndarray] = {}
    scheds["epanet"] = epanet_default_onoff(wdn)
    scheds["all_on"] = np.ones((P, T))
    scheds["all_off"] = np.zeros((P, T))
    # pump during the cheapest ~half of hours (off-peak filling)
    price = np.asarray(wdn.price_final, dtype=float)
    scheds["cheap_hours"] = np.tile((price <= np.median(price)).astype(float), (P, 1))
    return scheds


def warmstart_point(wdn: WDN, onoff: np.ndarray):
    """Linearization point + stacked iterate from EPANET flows under ``onoff``."""
    pump_links = (wdn.raw.link_pump_index + 1).tolist()
    bypass_links = [
        (int(lk) + 1, int(np.argmax(wdn.M.S_bypass_pump[i])))
        for i, lk in enumerate(wdn.M.bypass_index)
    ]
    heads, flows = simulate_with_schedule(
        wdn.spec.inp_path, pump_links, onoff, wdn.time, wdn.n_nodes, wdn.n_links,
        bypass_links=bypass_links,
    )
    lin = linearize(flows, wdn.M, wdn.pump)
    eps = stack_eps(heads, flows, onoff)
    return lin, eps


def _score(r: OWFResult) -> tuple:
    """Rank results: converged first, then smaller bound violation, then cheaper."""
    slack = 0.0 if np.isnan(r.max_slack) else r.max_slack
    return (0 if r.converged else 1, round(slack, 3), r.objective)


def solve_multistart(
    wdn: WDN,
    schedules: Optional[dict[str, np.ndarray]] = None,
    verbose: bool = True,
) -> tuple[OWFResult, str, dict[str, OWFResult]]:
    """Run the successive approximation from each candidate EPANET warm-start.

    Returns (best_result, best_name, all_results). Best = converged, then least
    head-bound violation, then lowest energy cost.
    """
    if schedules is None:
        schedules = candidate_schedules(wdn)

    results: dict[str, OWFResult] = {}
    for name, onoff in schedules.items():
        try:
            lin, eps = warmstart_point(wdn, onoff)
        except Exception as exc:  # EPANET may fail for infeasible schedules (e.g. all_off)
            if verbose:
                print(f"[warmstart {name}] EPANET failed: {exc}")
            continue
        if not np.isfinite(eps).all():   # EPANET produced NaNs (disconnected/negative)
            if verbose:
                print(f"[warmstart {name}] non-finite EPANET hydraulics -- skipping")
            continue
        r = solve_owf(wdn, lin_override=lin, eps_override=eps)
        results[name] = r
        if verbose:
            slack = "" if np.isnan(r.max_slack) else f" max_slack={r.max_slack:.3g}"
            print(f"[warmstart {name}] status={r.status} iters={r.n_iter} "
                  f"conv={r.converged} obj={r.objective:.4f}{slack}")

    if not results:
        raise RuntimeError("all warm-start schedules failed in EPANET")

    best_name = min(results, key=lambda k: _score(results[k]))
    return results[best_name], best_name, results


def true_energy_cost(wdn: WDN, flows: np.ndarray, speed: np.ndarray = None) -> float:
    """Energy cost using the TRUE nonlinear pump power (not the linearized model).

    This is the honest basis for comparing two schedules -- the per-iteration
    objective uses linearized power, which differs slightly between linearization
    points. ``speed`` (Pu x T) applies the variable-speed power form on VSP pumps.
    """
    if flows is None:
        return float("inf")
    from .solver import _true_pump_power
    T = wdn.time
    sp = speed[:, :T] if speed is not None else None
    power = _true_pump_power(wdn, flows[:, :T], sp)
    return float(wdn.price_final @ (power.sum(axis=0) / 1000.0))


def _converge_fixed(wdn: WDN, onoff: np.ndarray, seed_flows: np.ndarray,
                    seed_heads: np.ndarray, max_iter: int = 20) -> OWFResult:
    """Fix a schedule and converge the continuous problem from a given seed."""
    cfg = replace(wdn.config, fixed_schedule=np.round(onoff), soft_bounds=True,
                  damping=1.0, penalty_weight=1.0e3, penalty_growth=1.2,
                  penalty_max=1.0e5, max_iter=max_iter, feas_tol=2.0)
    wdn_fixed = replace(wdn, config=cfg)
    lin = linearize(seed_flows, wdn_fixed.M, wdn_fixed.pump)
    eps = stack_eps(seed_heads, seed_flows, np.round(onoff))
    return solve_owf(wdn_fixed, lin_override=lin, eps_override=eps)


def _apply_availability(wdn: WDN, sched: np.ndarray) -> np.ndarray:
    """Zero out hours where a pump's source is unavailable."""
    sched = np.array(sched, dtype=float)
    for pos, (start, end) in (wdn.pump_avail or {}).items():
        for t in range(wdn.time):
            if not (start <= t < end):
                sched[pos, t] = 0.0
    return sched


def duty_preserving_schedule(wdn: WDN) -> np.ndarray:
    """Load-shift EPANET's operation: keep each pump's total on-hours but move
    them to the cheapest hours.

    The price-quantile candidates run pumps *fewer* hours (only the cheapest),
    which under-pumps and drains the tanks below their minimum. Preserving each
    pump's EPANET duty and merely re-timing it to cheap hours keeps the tanks in
    bounds while still cutting the time-of-use bill -- the feasible way to save on
    high-duty-cycle systems (e.g. BWSN's 11 h / 21 h pumps).
    """
    T = wdn.time
    price = np.asarray(wdn.price_final, dtype=float)
    order = np.argsort(price)                      # cheapest hours first
    ep = np.round(epanet_default_onoff(wdn))       # EPANET duty per pump
    s = np.zeros((wdn.n_pumps, T))
    for p in range(wdn.n_pumps):
        k = int(ep[p].sum())
        s[p, order[:k]] = 1.0                      # same duty, cheapest hours
    return s


def price_threshold_schedules(wdn: WDN) -> dict[str, np.ndarray]:
    """Candidate schedules that run pumps only during the cheapest hours.

    A family of price quantiles (plus all-on and a duty-preserving load-shift) --
    physically motivated for time-of-use pricing and cheap to enumerate.
    """
    T, P = wdn.time, wdn.n_pumps
    price = np.asarray(wdn.price_final, dtype=float)
    out: dict[str, np.ndarray] = {"all_on": np.ones((P, T))}
    for q in (0.25, 0.4, 0.5, 0.6, 0.75, 0.9):
        thresh = np.quantile(price, q)
        out[f"cheapest_{int(q * 100)}pct"] = np.tile(
            (price <= thresh).astype(float), (P, 1)
        )
    try:
        out["load_shift"] = duty_preserving_schedule(wdn)   # feasible TOU shift
    except Exception:
        pass
    return out


def optimize_schedule(
    wdn: WDN,
    inner_iter: Optional[int] = None,
    feas_tol: float = 2.0,
    polish: bool = True,
    max_flips: int = 30,
    verbose: bool = True,
) -> tuple[OWFResult, dict]:
    """Optimize the pump on/off schedule and report savings vs EPANET.

    Each candidate schedule is evaluated **honestly**: fix it, converge the
    continuous hydraulics, and score the TRUE (nonlinear) energy cost plus the
    head-bound violation. We do not let the frozen linearization *predict* a
    distant schedule's quality -- that is what makes a single free-binary MILP
    return the incumbent (its coefficients simply don't apply far away).

    Stages:
      1. **Baseline** -- EPANET's own operation.
      2. **Candidates** -- price-quantile schedules (run only during the cheapest
         hours), all-on, and a MILP proposal at the incumbent's linearization.
      3. **Polish** (optional) -- 1-opt: flip single pump-hours of the winner and
         keep any flip that lowers cost, giving a schedule that is locally optimal
         w.r.t. single-hour changes.

    Ranking: feasible (slack <= feas_tol) first, then lowest true cost.

    Returns (best result, info dict with baseline/best cost, savings and trace).
    """
    # Large networks converge slowly (tank-driven fixed point); a candidate scored
    # with too few inner iterations is mis-ranked. Auto-scale when not specified.
    # Each 1-opt flip is a full converge, so cap the polish depth on big networks.
    large = wdn.n_nodes > 100
    if inner_iter is None:
        inner_iter = 100 if large else 15
    if large:
        max_flips = min(max_flips, 6)

    def evaluate(name, sched, seed_flows=None, seed_heads=None):
        sched = _apply_availability(wdn, np.round(sched))
        if seed_flows is None:
            seed_flows, seed_heads = base.flows, base.heads
        r = _converge_fixed(wdn, sched, seed_flows, seed_heads, inner_iter)
        if r.flows is None:
            return None, float("inf"), float("inf")
        return r, true_energy_cost(wdn, r.flows), float(r.max_slack)

    def rank(cost, slack):
        return (0 if slack <= feas_tol else 1, round(slack, 3) if slack > feas_tol else 0.0, cost)

    base = solve_from_epanet(wdn)
    if base.flows is None:
        raise RuntimeError("EPANET baseline solve failed")
    baseline_cost = true_energy_cost(wdn, base.flows)
    baseline_slack = float(base.max_slack)

    best, best_cost, best_slack = base, baseline_cost, baseline_slack
    best_sched = _apply_availability(wdn, np.round(base.onoff))
    best_key = rank(best_cost, best_slack)
    trace = [("epanet", baseline_cost, baseline_slack)]
    if verbose:
        print(f"[opt] baseline (EPANET): cost={baseline_cost:.5f} slack={baseline_slack:.3f}")

    # --- stage 2: candidate schedules -------------------------------------
    candidates = price_threshold_schedules(wdn)
    seen = {best_sched.tobytes()}
    for name, sched in candidates.items():
        s = _apply_availability(wdn, np.round(sched))
        if s.tobytes() in seen:
            continue
        seen.add(s.tobytes())
        r, cost, slack = evaluate(name, s)
        trace.append((name, cost, slack))
        key = rank(cost, slack)
        better = key < best_key
        if verbose:
            print(f"[opt] {name:16s} cost={cost:.5f} slack={slack:.3f} "
                  f"{'ACCEPT' if better else ''}")
        if better and r is not None:
            best, best_cost, best_slack, best_sched, best_key = r, cost, slack, s, key

    # MILP proposal at the incumbent's linearization (cheap; sometimes helps).
    # Skip on large networks: the free-binary MILP over many nodes is memory-heavy
    # (SCIP can OOM), and the price/load-shift candidates + polish already cover the
    # savings there.
    try:
        if wdn.n_nodes > 100:
            raise StopIteration(f"large network ({wdn.n_nodes} nodes)")
        lin = linearize(best.flows, wdn.M, wdn.pump)
        cfg = replace(wdn.config, fixed_schedule=None, soft_bounds=True, damping=1.0,
                      penalty_weight=1.0e3, penalty_growth=1.0, penalty_max=1.0e4,
                      max_iter=1, feas_tol=feas_tol)
        milp = solve_owf(replace(wdn, config=cfg), lin_override=lin,
                         eps_override=stack_eps(best.heads, best.flows, best_sched))
        if milp.onoff is not None:
            s = _apply_availability(wdn, np.round(milp.onoff))
            if s.tobytes() not in seen:
                seen.add(s.tobytes())
                r, cost, slack = evaluate("milp", s, milp.flows, milp.heads)
                trace.append(("milp", cost, slack))
                key = rank(cost, slack)
                if verbose:
                    print(f"[opt] {'milp':16s} cost={cost:.5f} slack={slack:.3f} "
                          f"{'ACCEPT' if key < best_key else ''}")
                if key < best_key and r is not None:
                    best, best_cost, best_slack, best_sched, best_key = r, cost, slack, s, key
    except Exception as exc:
        if verbose:
            print(f"[opt] milp proposal skipped: {exc}")

    # --- stage 3: 1-opt polish --------------------------------------------
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
                    if p in avail:
                        start, end = avail[p]
                        if not (start <= t < end):
                            continue
                    s = best_sched.copy()
                    s[p, t] = 1.0 - s[p, t]
                    if s.tobytes() in seen:
                        continue
                    seen.add(s.tobytes())
                    n_flips += 1
                    r, cost, slack = evaluate(f"flip_p{p}_t{t}", s)
                    key = rank(cost, slack)
                    if key < best_key and r is not None:
                        if verbose:
                            print(f"[opt] polish flip pump{p} hour{t}: "
                                  f"cost={cost:.5f} slack={slack:.3f} ACCEPT")
                        best, best_cost, best_slack, best_sched, best_key = r, cost, slack, s, key
                        trace.append((f"flip_p{p}_t{t}", cost, slack))
                        improved = True
                        break
                if improved:
                    break

    # --- final clean convergence of the winning schedule ------------------
    # The candidate/polish solves use soft bounds + capped iterations tuned for
    # fast *ranking*; the winner's heads/flows from that pass can be a poor
    # iterate (large EPANET-replay error) even though its cost/feasibility are
    # fine. Re-solve the chosen schedule cleanly so the returned result is
    # accurate for reporting and plotting.
    final = solve_fixed_schedule(wdn, best_sched)
    final_ok = (final.flows is not None
                and final.status in ("optimal", "optimal_inaccurate"))
    if final_ok:
        # hard solve -> max_slack is nan (no slack vars); a returned optimal
        # solution already satisfies the hard bounds, so treat that as slack 0.
        final_slack = final.max_slack if np.isfinite(final.max_slack) else 0.0
        if final_slack <= feas_tol:
            # The clean solve is authoritative for the chosen schedule: its cost
            # is the true cost (candidate solves are under-converged for speed and
            # may understate it), and its heads/flows reproduce EPANET.
            best, best_cost, best_slack = final, true_energy_cost(wdn, final.flows), final_slack

    # --- honest EPANET validation of the winner (large networks) --------------
    # The soft-bounds ranking can rate a tank-draining schedule as "feasible", but
    # EPANET is the ground truth: imposing such a schedule back in EPANET gives a
    # large head error (a tank drifts). So validate the candidate schedules by
    # re-running them in EPANET and keep the CHEAPEST one that actually reproduces
    # (replay error small) -- the reported schedule is then hydraulically
    # consistent, not merely soft-feasible.
    if wdn.n_nodes > 100:
        try:
            from .validation import validate_schedule
            REPLAY_OK = 5.0  # ft; a faithful reproduction is well under this
            ep_sched = _apply_availability(wdn, np.round(base.onoff))
            ls = _apply_availability(wdn, np.round(duty_preserving_schedule(wdn)))
            r_ls, c_ls, _ = evaluate("load_shift", ls)
            options = [("winner", best, best_sched, best_cost),
                       ("epanet", base, ep_sched, baseline_cost),
                       ("load_shift", r_ls, ls, c_ls)]
            scored = []
            for name, r, s, c in options:
                if r is None or r.flows is None:
                    continue
                try:
                    dh = float(validate_schedule(wdn, r).max_abs_head)
                except Exception:
                    dh = float("inf")
                # reproduces-EPANET first, then cheapest true cost
                scored.append(((0 if dh <= REPLAY_OK else 1), c, dh, name, r, s))
            if scored:
                scored.sort(key=lambda t: (t[0], t[1]))
                _, c, dh, name, r, s = scored[0]
                if name != "winner":
                    if verbose:
                        print(f"[opt] winner failed EPANET check -> using {name} "
                              f"(cost={c:.5f} replay dHead={dh:.2f} ft)")
                    best, best_cost, best_sched = r, c, s
                    best_slack = r.max_slack if np.isfinite(r.max_slack) else 0.0
                    trace.append((f"epanet_validated:{name}", c, dh))
        except Exception as exc:
            if verbose:
                print(f"[opt] validation fallback skipped: {exc}")

    info = {
        "baseline_cost": baseline_cost,
        "baseline_slack": baseline_slack,
        "best_cost": best_cost,
        "best_slack": float(best_slack),
        "savings_pct": (100.0 * (baseline_cost - best_cost) / baseline_cost
                        if baseline_cost else 0.0),
        "trace": trace,
        "schedule": best_sched,
        "n_flips": n_flips,
    }
    return best, info


def solve_from_epanet(wdn: WDN) -> OWFResult:
    """Reproduce EPANET's own operation: seed the linearization from EPANET's
    computed hydraulics, fix that pump schedule, and converge the continuous
    problem. Robust for multi-pump / switched-bypass networks (e.g. Net3) where
    EPANET's controls already give a valid, feasible operating point.
    """
    flows_ep, heads_ep, _, _ = run_epanet(wdn.raw)
    T = wdn.time
    Fseed = flows_ep[:T].T
    Hseed = heads_ep[:T].T
    onoff = np.round((wdn.M.Lambda @ Fseed > 1.0).astype(float))
    # soft bounds (EPANET may carry small negative pressures the model forbids),
    # modest penalty growth to stay well-conditioned.
    cfg = replace(wdn.config, fixed_schedule=onoff, soft_bounds=True, damping=1.0,
                  penalty_weight=1.0e3, penalty_growth=1.2, penalty_max=1.0e5,
                  max_iter=max(wdn.config.max_iter, 20), feas_tol=2.0)
    wdn_fixed = replace(wdn, config=cfg)
    lin = linearize(Fseed, wdn_fixed.M, wdn_fixed.pump)
    eps = stack_eps(Hseed, Fseed, onoff)
    return solve_owf(wdn_fixed, lin_override=lin, eps_override=eps)


def solve_fixed_schedule(wdn: WDN, onoff: np.ndarray) -> OWFResult:
    """Pin the pump on/off schedule and converge the continuous problem.

    With the binaries fixed the per-iteration problem is a pure LP, so the
    successive linearization converges reliably (as on the tree networks).
    """
    onoff = np.round(np.asarray(onoff)).astype(float)
    wdn_fixed = replace(wdn, config=replace(wdn.config, fixed_schedule=onoff))
    lin, eps = warmstart_point(wdn_fixed, onoff)
    return solve_owf(wdn_fixed, lin_override=lin, eps_override=eps)


def solve_warmstart(
    wdn: WDN,
    schedules: Optional[dict[str, np.ndarray]] = None,
    verbose: bool = True,
) -> tuple[OWFResult, str]:
    """Two-phase warm-start solve for hard networks.

    Phase 1: multi-start over candidate EPANET schedules to pick a good binary
    pump schedule. Phase 2: fix that schedule and converge the continuous problem.
    Returns (result, best_schedule_name).
    """
    best, name, _ = solve_multistart(wdn, schedules, verbose)
    if verbose:
        print(f"[warmstart] phase 1 best schedule = {name}; fixing and converging")
    refined = solve_fixed_schedule(wdn, best.onoff)
    # keep whichever is bound-feasible and cheaper
    if best.converged and not refined.converged:
        return best, name
    return refined, name
