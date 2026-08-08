# OWF benchmark: successive linearization vs. piecewise-linear MILP

Comparison of this study's **successive linear approximation (SLA)** against the
piecewise-linearization strategy of

> K. Oikonomou and M. Parvania, "Optimal Coordination of Water Distribution
> Energy Flexibility With Power Systems Operation," *IEEE Trans. Smart Grid*,
> vol. 10, no. 1, pp. 1101–1110, Jan. 2019.

Their WDS operation model linearizes every nonlinearity **up front** into one
mixed-integer linear program: each pipe's Hazen–Williams curve becomes a
K-breakpoint piecewise-linear function whose "consecutive lambdas" adjacency is
enforced by segment **binaries** (their eqs. 17–24), and the pump curves are
piecewise-linearized the same way. Accuracy is fixed by the breakpoint count K.

**Apples-to-apples protocol.** Their PWL machinery is re-implemented in
`owf/pwl_benchmark.py` on *this study's* networks, data, objective (pump energy
at the TOU price), tank/mass-balance/bound constraints, and solver (HiGHS), so
the ONLY difference is the linearization strategy. Both methods are scored
identically and honestly:
- the **replayed cost** — the true nonlinear pump energy of the schedule as
  EPANET actually runs it (a model's own flows can be fantasy);
- the **EPANET replay error** max |dHead| (this project's ≤ 5 ft standard);
- **binaries**, **wall time** (PWL budget: 300 s + build).
Pipe breakpoint ranges were set *generously in the PWL method's favor* (3× the
EPANET-rules flow range per pipe). FSP networks without bypasses/PRVs.

## Results (2026-08-08, HiGHS, TOU price)

### This study — SLA (recommended mode after this benchmark)

| Network | Mode | Binaries per MILP | Wall (s) | Replayed cost | Replay dHead (ft) |
|---|---|---|---|---|---|
| 3-node (T=13) | optimize | **13** | 2.8 | **0.23528** | 0.0045 |
| 8-node (T=12) | direct | **12** | 0.7 | **0.27474** | 0.153 |
| Net1 (T=24) | optimize | **24** | 20.5 | **0.16492** | 0.0052 |
| Net2 (T=24) | warmstart | **24** | 46 | **0.19736** | 0.0127 |
| BWSN (T=25) | optimize | **48** | 424* | **2.48756** | 0.0131 |

\* full schedule search including EPANET replay validation of candidates.

### PWL-MILP (Oikonomou–Parvania linearization; 300 s solver budget)

| Network | K | Binaries | Solve (s) | MILP obj | Replayed cost | Replay dHead (ft) | Status |
|---|---|---|---|---|---|---|---|
| 3-node | 5 | 117 | 0.4 | 0.22618 | 0.23528 | 0.56 | optimal |
| 3-node | 9 | 221 | 1.1 | 0.23121 | 0.23528 | 0.20 | optimal |
| 3-node | 17 | 429 | 0.8 | 0.23395 | 0.23528 | 0.06 | optimal |
| 8-node | 5 | 444 | 14.0 | 0.58655 | 0.74077 | **21.95** | optimal, not deliverable |
| 8-node | 9 | 876 | 46.8 | 0.35095 | 0.36374 | **58.47** | optimal, not deliverable |
| 8-node | 17 | 1,740 | >300 | — | — | — | no incumbent |
| Net1 | 5 | 1,272 | 76.9 | 0.15641 | 0.16594 | **15.29** | optimal, not deliverable |
| Net1 | 9 | 2,520 | >300 | — | — | — | no incumbent |
| Net1 | 17 | 5,016 | >300 | — | — | — | no incumbent |
| Net2 | 5 | 3,864 | 16.7 | — | — | — | **infeasible** (grid too coarse) |
| Net2 | 9 | 7,704 | >300 | — | — | — | no incumbent |
| Net2 | 17 | 15,384 | >300 | — | — | — | no incumbent |
| BWSN | 9 | **35,050** | >300 | — | — | — | no incumbent |

"Not deliverable": the MILP is optimal for its PWL surrogate, but replaying its
schedule in EPANET deviates far beyond the 5 ft standard — the operating point
the model committed to does not physically occur (pump-flow errors of 112–707
GPM), so its objective is not what the utility would pay.

## Head-to-head

| Network | Binaries saved by SLA | Time | Replayed cost |
|---|---|---|---|
| 3-node | 13 vs 117–429 (**89–97% fewer**) | comparable (toy) | tie — both reach 0.23528; SLA replay is exact (0.0045 ft vs 0.06–0.56 ft; pump-flow error 0.004 vs 13–100 GPM) |
| 8-node | 12 vs 876 (**98.6% fewer**) | **67× faster** (0.7 vs 46.8 s) | **24% cheaper** (0.275 vs 0.364), and the PWL schedule replays 58 ft off |
| Net1 | 24 vs 1,272 (**98.1% fewer**) | **3.8× faster** (20.5 vs 76.9 s) | **0.6% cheaper** (0.16492 vs 0.16594) with exact replay (0.005 vs 15.3 ft) |
| Net2 | 24 vs 3,864+ | ≥6.5× (46 s vs no answer in 300 s) | PWL returns **no valid solution at any K** |
| BWSN | 48 vs 35,050 (**99.86% fewer**) | search completes in 424 s vs **no incumbent** in 300 s | PWL returns nothing |

## Why the structural gap

- **Binary count.** PWL: (S + U)(K−1)T + UT — grows with *pipe count ×
  accuracy × horizon*. SLA: UT only — independent of network size and of
  accuracy, because accuracy comes from re-linearizing at the operating point
  (1–4 iterations) instead of from a finer grid.
- **Accuracy vs. tractability is a forced trade in PWL.** K=5 grids are coarse
  enough to make Net2 *infeasible* and to mis-place the 8-node operating point
  by 700 GPM; K=17 grids are accurate but already intractable on a 9-link
  tutorial network. SLA sidesteps the trade: its MILPs stay small while the
  linearization sharpens exactly where the solution sits.
- **Honesty must be checked either way.** Both methods optimize a surrogate.
  This study validates every result by EPANET replay (≤ 5 ft or it is not
  accepted); applying the same standard to the PWL results disqualifies its
  8-node and Net1 solutions.

## Two self-findings from the benchmark (applied)

Running the comparison honestly also improved *our* defaults: the 3-node
"direct" mode was converging to the all-ON local fixed point (0% savings) —
`optimize` finds the 4-ON-hour schedule (83.5% savings) in 2.8 s; Net1's
`warmstart` candidate was 31% more expensive than `optimize` at the same
runtime. Both networks' recommended modes now default to `optimize`
(`main_owf.MODE_SUGGESTION`).

## Reproduce

```
python - <<'PY'
from owf.config import SolverConfig
from owf.network import setup
from owf.pwl_benchmark import solve_pwl_owf
r = solve_pwl_owf(setup(SolverConfig(net_num=8)), K=9, time_limit=300)
print(r.status, r.n_binary, r.solve_s, r.objective, r.true_cost)
PY
```

Next step: the same comparison inside C-OWPF (their Sec. IV couples the WDS to
a unit-commitment model; ours co-optimizes with the distribution feeder — the
WDS-layer linearization comparison carries over unchanged).
