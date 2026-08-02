"""Entry point for the FSP OWF solver (ports Main_OWF_IEEE_ACCESS.m).

Usage:
    python main_owf.py                 # 8-node, time-of-use price, EPANET init
    python main_owf.py --net 8 --price 1 --choice 1 --time 12 --verbose
"""
from __future__ import annotations

import argparse
import time as _time

from owf import SolverConfig, setup, solve_owf, validate


def parse_args() -> SolverConfig:
    p = argparse.ArgumentParser(description="FSP Optimal Water Flow (OWF) solver")
    p.add_argument("--net", type=int, default=8, help="network number (8 = 8-node)")
    p.add_argument("--price", type=int, default=1, choices=[0, 1],
                   help="1 = time-of-use price, 0 = flat")
    p.add_argument("--choice", type=int, default=1, choices=[0, 1],
                   help="1 = init from EPANET, 0 = user-defined")
    p.add_argument("--time", type=int, default=None,
                   help="horizon (default: EPANET pattern length)")
    p.add_argument("--tol", type=float, default=0.5, help="convergence tolerance")
    p.add_argument("--max-iter", type=int, default=50)
    p.add_argument("--verbose", action="store_true")
    a = p.parse_args()
    return SolverConfig(
        net_num=a.net, price_choice=a.price, choice=a.choice, time=a.time,
        tol=a.tol, max_iter=a.max_iter, verbose=a.verbose,
    )


def main() -> None:
    config = parse_args()
    print(f"Building WDN (net={config.net_num}) ...")
    wdn = setup(config)
    print(f"  nodes={wdn.n_nodes}  links={wdn.n_links}  pumps={wdn.n_pumps}  "
          f"tanks={wdn.n_tanks}  reservoirs={wdn.n_reservoirs}  time={wdn.time}")

    t0 = _time.time()
    result = solve_owf(wdn)
    elapsed = _time.time() - t0

    print(f"\nSolve finished: status={result.status}  iterations={result.n_iter}  "
          f"converged={result.converged}  ({elapsed:.2f}s)")
    print(f"Objective (energy cost): {result.objective:.6f}")

    report = validate(wdn, result)
    print("\n" + report.summary())


if __name__ == "__main__":
    main()
