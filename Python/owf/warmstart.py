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
    heads, flows = simulate_with_schedule(
        wdn.spec.inp_path, pump_links, onoff, wdn.time, wdn.n_nodes, wdn.n_links
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
