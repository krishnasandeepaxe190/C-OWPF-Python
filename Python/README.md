# FSP Optimal Water Flow (OWF) — Python

Python translation of the water-side (WDN) **Optimal Water Flow** problem from the
MATLAB C-OWPF code, restricted to **fixed-speed pumps (FSPs)**. VSP and PRV models
are intentionally out of scope.

Each iteration of a **successive linear-approximation** loop is a **MILP** — linear
objective and constraints with binary pump on/off variables — solved with
**HiGHS** through **CVXPY**. EPANET `.inp` parsing and hydraulic ground-truth use
**epyt** (the EPANET-Python-Toolkit, which mirrors the EPANET-MATLAB-Toolkit).

## Install & run

```bash
pip install -r requirements.txt

streamlit run app.py          # WEB UI: case setup, run, plots, session comparison

python main_owf.py            # INTERACTIVE: pick network, get a recommended mode,
                              # run, plot, and print the comparison table

# or non-interactive:
python main_owf.py --net 8                        # auto mode (recommended per net)
python main_owf.py --net 36 --mode optimize --plot  # find savings vs EPANET + plots
python main_owf.py --net 97 --mode epanet --plot    # reproduce EPANET (Net3)
python main_owf.py --net 11 --mode warmstart        # looped-network warm-start
pytest tests/                                     # regression suite
```

**Modes** (`--mode`, or suggested interactively): `direct` (tree networks),
`warmstart` (looped networks), `epanet` (reproduce EPANET's rule-based operation),
`optimize` (search for the cheapest feasible schedule; reports savings).
Every run ends with an **EPANET-vs-C-OWPF comparison table** — true energy cost,
savings, EPANET-replay errors, and min junction pressure — and the interactive
driver prints an aggregate session table across the cases you ran.

## Plots

`--plot` (or `owf.plots.plot_all`) writes five PNGs to `--outdir` (default
`outputs/`):

| file | contents |
|---|---|
| `*_convergence.png` | successive-linearization error (log) and objective per iteration |
| `*_pump_schedule.png` | optimized pump on/off vs the electricity price |
| `*_flows.png` | pump + largest pipe flows, OWF vs EPANET |
| `*_heads.png` | tank (with bounds) and junction heads, OWF vs EPANET |
| `*_error.png` | max abs head/flow error vs EPANET per time step |

The EPANET series in the flow/head plots come from the **schedule-imposed**
re-simulation, so overlapping curves mean the linearized solution reproduces the
true nonlinear hydraulics.

## Network status

| `--net` | Network | Size | Status |
|---|---|---|---|
| 8  | 8-node tutorial | 8 nodes / 9 links | ✅ converges; validation ≈ 0.15 ft / 1 GPM |
| 3  | 3-node large-pump | 3 / 2 | ✅ converges; validation ≈ 0.005 ft / 0.01 GPM |
| 11 | Net1 (looped) | 11 / 13 | ✅ converges with `--warmstart`; ≈ 0.005 ft / 0.002 GPM |
| 36 | Net2 (looped) | 36 / 40 | ✅ converges with `--warmstart`; ≈ 0.013 ft / 0.001 GPM |

| 97 | Net3 (looped, 2 pumps, switched bypass) | 97 / 119 | ✅ reproduces EPANET ≈ 1.1 ft via `solve_from_epanet` |

Looped networks (Net1, Net2) do **not** converge from the default initialization —
use `--warmstart`. KY3 is archived under `data/archive/ky3/`: its 5 pumps carry no
EPANET head curve, so it needs explicit `[h0, r, v]` coefficients before it can be
registered in `config.NETWORKS`.

## Schedule optimization

`owf.optimize_schedule(wdn)` searches for a cheaper pump schedule than EPANET's own
operation. Every candidate is **evaluated honestly** — fixed, converged, and scored
on the *true nonlinear* energy cost plus head-bound violation — because a single
free-binary MILP cannot judge a distant schedule (its frozen linearization simply
doesn't apply there, so it returns the incumbent).

Stages: EPANET baseline → price-quantile candidates (+ all-on, + a MILP proposal)
→ optional 1-opt polish (flip single pump-hours). Ranking: feasible first, then
lowest true cost.

Savings vs EPANET's operation (all verified feasible by re-simulating the optimized
schedule in EPANET — positive junction pressures and in-bounds tanks):

| network | EPANET cost | optimized | saving | min pressure (EPANET) |
|---|---|---|---|---|
| 8-node | 0.2747 | 0.1926 | **29.9 %** | +7.4 ft |
| 3-node | 1.4251 | 0.4169 | **70.7 %** | +204.8 ft |
| Net1 | 0.2988 | 0.1631 | **45.4 %** | +242.4 ft |
| Net2 | 0.3930 | 0.0891 | **77.3 %** | +59.5 ft |
| Net3 | 0.3854 | 0.3854 | 0 % | — |

**Net3 is an honest negative result**: its tank-level control logic is already
well tuned, and every price-quantile alternative either costs more or violates the
tank bounds by 12–21 ft (three tanks with tight bounds, one availability-limited
source). Beating it would need candidates that respect its per-tank refill
structure, not a blanket "run during the cheapest hours" pattern.

Pump curve coefficients are **derived from each network's EPANET head curve**, and
the junction demand profile is taken from EPANET's computed time series, so the
optimizer stays consistent with EPANET regardless of pattern lengths or units.

## Warm-start for hard (looped) networks

Looped networks like Net1 don't converge from the default single-point
initialization — the tank head is over-determined by both the hydraulic network
and the tank integrator, and one linearization point isn't in a good basin. The
`warmstart` module solves this in two phases (`solve_warmstart`):

1. **Multi-start** — for several candidate pump on/off schedules (EPANET's own
   rule-based schedule, all-on, all-off, off-peak/`cheap_hours`), impose each in
   EPANET, take the resulting physically-consistent flows as the linearization
   seed, and run the successive approximation with soft (penalized) head bounds so
   every MILP stays feasible. Keep the schedule with the smallest bound violation.
2. **Fix & converge** — pin that binary schedule and converge the remaining
   continuous problem (a pure LP per iteration), which settles reliably.

For Net1 this picks the off-peak schedule and converges to ≈ 0.005 ft vs EPANET.

## Layout

```
Python/
├── main_owf.py                 # entry point            ≈ Main_OWF_IEEE_ACCESS.m
├── owf/
│   ├── config.py               # network specs, pump curves, prices, solver settings
│   ├── epanet_io.py            # read_inp.m + init_epanet.m   (via epyt)
│   ├── connection_matrices.py  # ConnectionMatrices_WDN.m  (Pi, Λ, Θ, Τ, Κ, Ω, Δ …)
│   ├── network.py              # WDN_setup_IEEE_ACCESS.m + definebounds_WDN.m
│   ├── initial_values.py       # Initial_Values_WDN.m
│   ├── linearization.py        # CalculateNewIterationValues.m
│   ├── constraints.py          # every define*_CVX.m  → one function
│   ├── solver.py               # WDN_OWF_IEEEACCESS_cvx.m  (the MILP loop)
│   ├── warmstart.py            # EPANET multi-start warm-start for looped nets
│   ├── validation.py           # EPANET error-norm + schedule-imposed check
│   └── plots.py                # convergence / flow / head / schedule plots
├── data/eightnode/…            # EPANET .inp (self-contained)
└── tests/test_eightnode.py
```

## MATLAB → Python map

| MATLAB module | Python | What it does |
|---|---|---|
| `Main_OWF_IEEE_ACCESS.m` | `main_owf.py` | CLI entry, solve + validate |
| `WDN_setup_IEEE_ACCESS.m` / `definebounds_WDN.m` | `network.py` | assemble `WDN`, bounds, prices |
| `Prepare_net_WDN.m` / `read_inp.m` / `init_epanet.m` | `epanet_io.py` | parse `.inp`, run EPANET |
| `ConnectionMatrices_WDN.m` | `connection_matrices.py` | incidence / selection matrices |
| `Initial_Values_WDN.m` | `initial_values.py` | initial linearization point |
| `CalculateNewIterationValues.m` | `linearization.py` | relinearize Cp, C1M, C2M, A′, B′ |
| `define*_CVX.m` | `constraints.py` | constraints + objective |
| `WDN_OWF_IEEEACCESS_cvx.m` | `solver.py` | successive-linearization MILP loop |

## Formulation (per iteration, all linear in the decision variables)

Decision variables over `Time` steps: pipe+pump `Flows`, nodal `Heads`, pump power
`Ppump`, pump `OnOff` (binary), tank auxiliaries `Hdummy`, `TankFlow_aux`.

- **Objective** — min Σₜ price(t)·Σ Ppump/1000 (energy cost)
- **Pipe head loss** — `Π̃·H = Cp + Π′·q` (linearized Hazen–Williams)
- **Mass balance** — `Π_reduced·q = −demand`
- **Pump flow** — `0 ≤ q_pump ≤ q_max·OnOff`
- **Pump head-gain** — Big-M curve enforced only when `OnOff = 1`
- **Pump power** — `Ppump = A′·q_pump + B′·OnOff` (Taylor of FSP power)
- **Tank dynamics** — integrator state-space + head bounds + terminal level
- **Reservoir / junction head bounds**

## Validation

Two levels, both in `validation.py`:

- `validate()` — Euclidean error norms vs EPANET's own (rule-based) operation.
- `validate_schedule()` — **fixes the optimized pump schedule back into EPANET**,
  deletes the existing controls/rules, re-simulates the true nonlinear hydraulics,
  and compares heads/flows (reproduces the paper's Fig. 4 feasibility check). This
  is the apples-to-apples check: 8-node ≈ 0.15 ft, 3-node ≈ 0.005 ft.

## Notes & deviations from the MATLAB

- **Pump coefficients** are derived from each network's EPANET head curve
  (`epanet_io._derive_pump_coefficients`) rather than hard-coded per `Net_num` as
  in `Prepare_net_WDN.m` — this fixes value drift (e.g. the 3-node "large pump")
  and keeps the optimizer consistent with EPANET. Override via
  `NetworkSpec.pump_coefficients`.
- **Demand** is taken from EPANET's computed time series, so pattern wrapping,
  default patterns and mixed pattern lengths match EPANET exactly.
- **Horizon (`Time`)** is driven by EPANET's simulation length, not hard-coded to
  24 as in `Main_OWF_IEEE_ACCESS.m`. Override with `--time`.
- **Mass-balance warm-up**: the MATLAB loop disables mass balance for the first 10
  iterations. Python enforces it from iteration 0 by default (physically correct);
  set `SolverConfig.mass_balance_warmup=True` to reproduce the original behavior.
- **Solver**: HiGHS via CVXPY replaces CVX + Gurobi/SDPT3. Each iteration is a
  genuine MILP.
- **Pump exponent `v`**: the head-gain / power model is `H(f) = h0 - r f^v` for
  general `v`, not hard-coded to 2. EPANET *single-design-point* curves are exactly
  quadratic (`v = 2`) — which is why all three in-scope networks use `v = 2` — but
  multi-point curves get their fitted exponent, and the power linearization
  `P(f) = c_m(h0 - r f^v) f` with `dP/df = c_m(h0 - (v+1) r f^v)` handles it.
- **Net1 warm-start**: see the "Warm-start" section. `SolverConfig` exposes the
  knobs: `damping` (trust-region blend of the linearization point), `soft_bounds`
  + `penalty_*` (penalty CCP), and `fixed_schedule` (pin the pump binaries).
- **Out of scope**: VSPs, PRVs, and the power-network (PDN) coupling.
```
