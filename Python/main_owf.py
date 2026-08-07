"""Entry point for the FSP Optimal Water Flow (OWF) solver.

Run with no arguments for the **interactive** driver: it asks for the network,
suggests the right solve mode, constructs the case, runs it, writes plots and
prints an EPANET-vs-C-OWPF comparison table (plus an aggregate table when you
run several cases in one session).

    python main_owf.py

Or run non-interactively with flags:

    python main_owf.py --net 8                       # auto mode, no plots
    python main_owf.py --net 36 --mode optimize --plot
    python main_owf.py --net 97 --mode epanet --plot
"""
from __future__ import annotations

import argparse
import sys
import time as _time
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from owf import (
    NETWORKS,
    SolverConfig,
    setup,
    solve_owf,
    validate_schedule,
)
from owf.config import (
    DEFAULT_FALLBACK,
    DEFAULT_SOLVER,
    SOLVER_CHOICES,
    available_solvers,
)
from owf.warmstart import (
    optimize_schedule,
    solve_from_epanet,
    solve_warmstart,
    true_energy_cost,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Modes and per-network recommendations
# ---------------------------------------------------------------------------
MODES = {
    "direct": "plain successive linearization, free pump binaries "
              "(fast; tree networks only)",
    "warmstart": "EPANET multi-start warm-start, then fix the best schedule "
                 "(reliable on looped networks)",
    "epanet": "reproduce EPANET's own rule-based operation "
              "(validation / baseline; the only reliable mode for Net3)",
    "optimize": "search for the cheapest feasible pump schedule and report "
                "savings vs EPANET (slowest, most informative)",
}

# (recommended, alternative worth trying)
MODE_SUGGESTION = {
    8: ("direct", "optimize"),
    3: ("direct", "optimize"),
    108: ("warmstart", "optimize"),
    11: ("warmstart", "optimize"),
    36: ("warmstart", "optimize"),
    97: ("epanet", "optimize"),
    126: ("optimize", "epanet"),
}

NET_BLURB = {
    8: "8-node tutorial -- tree, 1 pump, 1 tank",
    3: "3-node -- tree, 1 pump, 1 tank",
    108: "8-node + PRV -- pressure-reducing valve (junction 6->9, 20 psi)",
    11: "Net1 -- looped, 1 pump, 1 tank",
    36: "Net2 -- looped, 1 pump, 1 tank",
    97: "Net3 -- looped, 2 pumps, 3 tanks, switched bypass",
    126: "BWSN -- large looped, 2 pumps, 2 tanks, 1 reservoir (126 junctions)",
}


def _fallbacks(solver: str) -> tuple:
    """Fall back to HiGHS then SCIP (whichever are installed), skipping the primary."""
    avail = available_solvers()
    return tuple(s for s in (DEFAULT_SOLVER, DEFAULT_FALLBACK)
                 if s != solver and s in avail)


def _warmstart_config(net: int, price: int, horizon, solver: str) -> SolverConfig:
    return SolverConfig(net_num=net, price_choice=price, time=horizon,
                        soft_bounds=True, damping=0.6, penalty_weight=1e3,
                        penalty_growth=1.5, max_iter=60, feas_tol=0.5,
                        solver=solver, fallback_solvers=_fallbacks(solver))


# ---------------------------------------------------------------------------
# Case runner
# ---------------------------------------------------------------------------
@dataclass
class CaseResult:
    label: str
    net_num: int
    mode: str
    price: str
    horizon: int
    elapsed: float
    converged: bool
    n_iter: int
    epanet_cost: float = float("nan")   # cost of EPANET's own operation
    owf_cost: float = float("nan")      # true nonlinear cost of the C-OWPF result
    savings_pct: float = float("nan")
    max_dhead: float = float("nan")     # schedule-imposed EPANET replay errors
    max_dpumpflow: float = float("nan")
    min_pressure: float = float("nan")  # min junction pressure in the EPANET replay
    solver: str = "HIGHS"
    note: str = ""
    plots: list = field(default_factory=list)


def run_case(net: int, mode: str, price: int, horizon, plot: bool,
             outdir: str, verbose: bool,
             solver: str = "HIGHS",
             vsp: dict = None,
             prv: dict = None) -> tuple[Optional[CaseResult], object, object]:
    """Construct and solve one case; returns (CaseResult, wdn, result).

    ``vsp`` maps a pump id to its (omega_min, omega_max) speed bounds; listed
    pumps run at variable speed. When any VSP is present the case uses a
    soft-bound, damped direct solve (the McCormick relaxation needs it).
    ``prv`` maps a valve id to its pressure setting P_set (psi), overriding the
    .inp value (h_set = downstream elevation + P_set * 2.3072).
    """
    label = f"{NETWORKS[net].name}/{mode}/{'TOU' if price == 1 else 'flat'}"
    if vsp:
        label += f"/VSP{len(vsp)}"
    print(f"\n=== case: {label}  (solver={solver}) ===")
    t0 = _time.time()

    if vsp:
        # Variable-speed pumps: the bilinear McCormick relaxation needs the soft-
        # bound / damped homotopy to converge; run a single direct solve.
        wdn = setup(SolverConfig(
            net_num=net, price_choice=price, time=horizon, vsp_pumps=vsp,
            prv_settings=prv,
            soft_bounds=True, damping=0.5, penalty_weight=1e3, penalty_growth=1.5,
            max_iter=80, feas_tol=0.5, verbose=verbose, solver=solver,
            fallback_solvers=_fallbacks(solver)))
    elif mode in ("warmstart",):
        cfgw = _warmstart_config(net, price, horizon, solver)
        cfgw.prv_settings = prv
        wdn = setup(cfgw)
    else:
        wdn = setup(SolverConfig(net_num=net, price_choice=price, time=horizon,
                                 prv_settings=prv,
                                 verbose=verbose, solver=solver,
                                 fallback_solvers=_fallbacks(solver)))
        # PRV networks need the soft-bound homotopy: a hard-bounded direct solve is
        # infeasible whenever the setpoint pins a head outside the reachable range.
        if wdn.n_valves and not wdn.config.soft_bounds:
            cfgv = _warmstart_config(net, price, horizon, solver)
            cfgv.prv_settings = prv
            wdn = setup(cfgv)
    print(f"  network: nodes={wdn.n_nodes} links={wdn.n_links} pumps={wdn.n_pumps} "
          f"tanks={wdn.n_tanks} horizon={wdn.time}h")

    # Cost baseline: EPANET running its OWN tank-level rules (a *different*
    # schedule from the optimizer). This is the only place EPANET's rule-based
    # operation is used, and it is used for cost only -- never for head/flow
    # error, since the schedules differ.
    from owf.epanet_io import run_epanet, epanet_pump_speeds
    flows_ep, _, _, _ = run_epanet(wdn.raw)
    # Use the pump speeds EPANET actually applied (from its own SETTING controls),
    # so a network that schedules variable speed has an honest baseline. For pumps
    # that run at full speed this is 1.0 and the cost is unchanged.
    base_speed = epanet_pump_speeds(wdn.raw, wdn.time)
    epanet_cost = true_energy_cost(wdn, flows_ep[: wdn.time].T, speed=base_speed)

    note = ""
    if vsp:
        result = solve_owf(wdn)
        note = f"VSP direct solve; speed in [{min(v[0] for v in vsp.values()):.2f},1.0]"
        # Polish: the free-binary VSP (+PRV) solve can hit the iteration cap while
        # the linearization is still moving. Iterate fix-schedule -> EPANET replay
        # -> reconverge from the replay-consistent point, and keep whichever
        # candidate (INCLUDING the original) replays best in EPANET -- monotone by
        # construction (never adopts a polish that widens the replay error).
        if result.flows is not None:
            try:
                from dataclasses import replace as _replace
                from owf.epanet_io import simulate_with_schedule
                from owf.linearization import linearize, stack_eps

                pl = (wdn.raw.link_pump_index + 1).tolist()
                bl = [(int(lk) + 1, int(np.argmax(wdn.M.S_bypass_pump[i])))
                      for i, lk in enumerate(wdn.M.bypass_index)]

                def _replay_err(res):
                    s = np.round(res.onoff)
                    h_ep, f_ep = simulate_with_schedule(
                        wdn.spec.inp_path, pl, s, wdn.time, wdn.n_nodes,
                        wdn.n_links, bypass_links=bl, pump_speeds=res.speed)
                    return float(np.max(np.abs(h_ep - res.heads[:, :wdn.time]))), h_ep, f_ep

                best_err, h_ep, f_ep = _replay_err(result)
                cur = result
                for _ in range(2):
                    sched = np.round(cur.onoff)
                    # pin schedule AND speeds: with omega fixed, WW = omega*f is
                    # exact, so the reconverge cannot drift off the replay point
                    cfg2 = _replace(wdn.config, fixed_schedule=sched,
                                    fixed_speed=cur.speed, damping=0.7, max_iter=40)
                    wdn2 = _replace(wdn, config=cfg2)
                    lin = linearize(f_ep, wdn.M, wdn.pump, speed=cur.speed)
                    r2 = solve_owf(wdn2, lin_override=lin,
                                   eps_override=stack_eps(h_ep, f_ep, sched))
                    if r2.flows is None:
                        break
                    err2, h2, f2 = _replay_err(r2)
                    if err2 < best_err - 1e-6:
                        result, best_err = r2, err2
                        note += f"; polished (replay {err2:.2f} ft)"
                    cur, h_ep, f_ep = r2, h2, f2
            except Exception:
                pass
    elif mode == "direct":
        result = solve_owf(wdn)
    elif mode == "warmstart":
        result, sched_name = solve_warmstart(wdn, verbose=verbose)
        note = f"schedule={sched_name}"
    elif mode == "epanet":
        result = solve_from_epanet(wdn)
        note = "EPANET's schedule reproduced"
        # "converged" here means the model reproduces EPANET within tolerance
        if result.flows is not None and np.isfinite(result.max_slack):
            result.converged = bool(result.max_slack <= 2.0)
    elif mode == "optimize":
        result, info = optimize_schedule(wdn, verbose=verbose)
        # NOTE: epanet_cost stays the raw EPANET rule-based cost (above), the same
        # baseline used by every other mode -- not optimize's internal LP-reproduced
        # baseline -- so savings are reported consistently across modes.
        note = f"searched {len(info['trace'])} schedules, {info['n_flips']} polish flips"
        # for optimize mode, "converged" means the winning schedule is feasible
        result.converged = bool(info["best_slack"] <= 2.0)
    else:
        raise ValueError(f"unknown mode {mode!r}")
    elapsed = _time.time() - t0

    if result.flows is None:
        print(f"  FAILED: {result.status} -- no feasible solution.")
        if mode == "direct" and net in (11, 36, 97):
            print("  hint: this looped network needs --mode warmstart (or epanet/optimize).")
        return (CaseResult(label, net, mode, "TOU" if price == 1 else "flat",
                           wdn.time, elapsed, False, result.n_iter,
                           epanet_cost=epanet_cost, note=f"failed: {result.status}"),
                wdn, result)

    owf_cost = true_energy_cost(wdn, result.flows, speed=result.speed)
    savings = 100.0 * (epanet_cost - owf_cost) / epanet_cost if epanet_cost else 0.0

    # honest check: re-simulate the resulting schedule in EPANET
    rep = validate_schedule(wdn, result)
    junction_head = wdn.M.Kappa @ rep.heads_epanet
    elev = (wdn.M.Kappa @ wdn.raw.node_elevations[:, None]).ravel()
    min_press = float((junction_head - elev[:, None]).min())

    case = CaseResult(
        label=label, net_num=net, mode=mode,
        price="TOU" if price == 1 else "flat", horizon=wdn.time,
        elapsed=elapsed, converged=result.converged, n_iter=result.n_iter,
        epanet_cost=epanet_cost, owf_cost=owf_cost, savings_pct=savings,
        max_dhead=rep.max_abs_head, max_dpumpflow=rep.max_abs_pump_flow,
        min_pressure=min_press, solver=solver, note=note,
    )

    if plot:
        from owf.plots import plot_all
        prefix = f"{NETWORKS[net].name}_{mode}"
        case.plots = plot_all(wdn, result, rep, outdir=outdir, prefix=prefix)

    print(comparison_table([case]))
    if case.plots:
        print("  plots:")
        for p in case.plots:
            print(f"    {p}")
    return case, wdn, result


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------
def comparison_table(cases: list) -> str:
    """EPANET(rules) vs C-OWPF comparison for one or more cases."""
    head = (f"\n{'case':28s} {'EPANET(rules)':>13s} {'C-OWPF':>9s} {'saving':>8s} "
            f"{'dHead':>8s} {'dPumpQ':>8s} {'minP':>7s} {'feas':>5s} {'time':>6s}")
    sep = "-" * len(head)
    lines = [head, sep]
    for c in cases:
        conv = "yes" if c.converged else "NO"
        sv = f"{c.savings_pct:7.1f}%" if np.isfinite(c.savings_pct) else "     --"
        dh = f"{c.max_dhead:8.3f}" if np.isfinite(c.max_dhead) else "      --"
        dq = f"{c.max_dpumpflow:8.3f}" if np.isfinite(c.max_dpumpflow) else "      --"
        mp = f"{c.min_pressure:7.1f}" if np.isfinite(c.min_pressure) else "     --"
        lines.append(
            f"{c.label:28s} {c.epanet_cost:13.5f} "
            f"{c.owf_cost:9.5f} {sv} {dh} {dq} {mp} {conv:>5s} {c.elapsed:5.0f}s"
        )
        if c.note:
            lines.append(f"    note: {c.note}")
    lines.append(sep)
    lines.append(
        "COST columns compare DIFFERENT schedules: EPANET(rules) = EPANET's own "
        "tank-level rule operation vs C-OWPF's optimized schedule (true nonlinear "
        "energy cost); saving = % reduction.")
    lines.append(
        "ERROR columns use the SAME schedule: the C-OWPF schedule is imposed in "
        "EPANET and replayed -- dHead [ft] / dPumpQ [GPM] are max |C-OWPF - EPANET|; "
        "minP = min junction pressure in that replay (>0 = hydraulically feasible).")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Interactive driver
# ---------------------------------------------------------------------------
def _ask(prompt: str, default: str, valid=None) -> str:
    while True:
        raw = input(f"{prompt} [{default}]: ").strip().lstrip("﻿\xff\xfe").lower()
        if not raw:
            return default
        if valid is None or raw in valid:
            return raw
        print(f"  please enter one of: {', '.join(valid)}")


def interactive() -> None:
    print("=" * 70)
    print("FSP Optimal Water Flow (OWF) -- interactive driver")
    print("=" * 70)
    session: list[CaseResult] = []
    while True:
        print("\nAvailable networks:")
        for n in NETWORKS:
            rec, alt = MODE_SUGGESTION[n]
            print(f"  {n:>3d}  {NET_BLURB[n]:52s} (recommended: {rec})")
        net = int(_ask("Network", "8", {str(n) for n in NETWORKS}))

        rec, alt = MODE_SUGGESTION[net]
        print(f"\nModes for {NETWORKS[net].name}:")
        for m, desc in MODES.items():
            tag = "  <-- recommended" if m == rec else (
                  "  <-- try for savings" if m == alt else "")
            print(f"  {m:9s} {desc}{tag}")
        if net == 97:
            print("  note: Net3's tank controls are already near-optimal; 'optimize'"
                  " honestly reports ~0% savings and takes ~3 min.")
        mode = _ask("Mode", rec, set(MODES))

        price = int(_ask("Price (1=time-of-use, 0=flat)", "1", {"0", "1"}))

        avail = available_solvers()
        print(f"\nMILP solver (available: {', '.join(avail)}; "
              f"default {DEFAULT_SOLVER}, fallback {DEFAULT_FALLBACK}):")
        for s in SOLVER_CHOICES:
            state = "available" if s in avail else "not installed"
            print(f"  {s:7s} {state}")
        solver = _ask("Solver", DEFAULT_SOLVER, {s.lower() for s in avail}).upper()

        plot = _ask("Write plots? (y/n)", "y", {"y", "n"}) == "y"

        try:
            case, _, _ = run_case(net, mode, price, None, plot, "outputs", False, solver=solver)
            if case:
                session.append(case)
        except KeyboardInterrupt:
            print("\n  case interrupted.")
        except Exception as exc:
            print(f"  case failed: {exc}")

        if _ask("\nRun another case? (y/n)", "n", {"y", "n"}) == "n":
            break

    if len(session) > 1:
        print("\n" + "=" * 70)
        print("SESSION SUMMARY -- all cases")
        print("=" * 70)
        print(comparison_table(session))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main() -> None:
    if len(sys.argv) == 1:
        interactive()
        return

    p = argparse.ArgumentParser(description="FSP Optimal Water Flow (OWF) solver")
    p.add_argument("--net", type=int, default=8, choices=list(NETWORKS))
    p.add_argument("--mode", default="auto",
                   choices=["auto", *MODES], help="auto = recommended per network")
    p.add_argument("--price", type=int, default=1, choices=[0, 1])
    p.add_argument("--time", type=int, default=None, help="horizon (hours)")
    p.add_argument("--solver", default=DEFAULT_SOLVER, choices=SOLVER_CHOICES,
                   help=f"MILP solver (default {DEFAULT_SOLVER}, fallback {DEFAULT_FALLBACK})")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--outdir", default="outputs")
    p.add_argument("--verbose", action="store_true")
    # kept for backward compatibility: --warmstart == --mode warmstart
    p.add_argument("--warmstart", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args()

    if a.solver not in available_solvers():
        print(f"error: solver {a.solver} is not installed. "
              f"Available: {', '.join(available_solvers())}")
        return

    mode = a.mode
    if a.warmstart and mode == "auto":
        mode = "warmstart"
    if mode == "auto":
        mode = MODE_SUGGESTION[a.net][0]
        print(f"mode=auto -> using recommended '{mode}' for {NETWORKS[a.net].name}")

    run_case(a.net, mode, a.price, a.time, a.plot, a.outdir, a.verbose, solver=a.solver)


if __name__ == "__main__":
    main()
