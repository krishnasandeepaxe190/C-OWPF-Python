# C-OWPF — Coupled Optimal Water-Power Flow (Python)

Python implementation of the **Coupled Optimal Water-Power Flow** problem from the
MATLAB C-OWPF code (IEEE Access-2024-18604), covering three problems:

- **💧 Water OWF** — schedule fixed-speed pumps (FSPs) to minimize energy cost.
- **⚡ Power OPF** — dispatch PV **reactive** setpoints on a distribution feeder,
  verified with a nonlinear **Z-bus** power flow for true voltages and true loss.
- **🔗 Coupled C-OWPF** — co-optimize the two grids through the pumps' electrical
  load, keeping the **paper's objective (pump-energy cost)**; the feeder enters as
  voltage constraints and PV reactive is the control.

Each iteration of a **successive linear-approximation** loop is a **MILP** (an LP
once the pump schedule is fixed) — solved with **HiGHS** through **CVXPY**. The
water physics is validated against **EPANET** (via **epyt**) and the power physics
against a nonlinear **Z-bus** solve.

**Variable-speed pumps (VSP)** and **pressure-reducing valves (PRV)** are modeled:

- **VSP** — the relative speed ω ∈ [ω_min, ω_max] is a decision variable (gated by
  on/off). Head gain `h0·ω² − σ·f^ν` is linearized in (ω, f); the bilinear ω·f in
  the power is handled by a McCormick auxiliary `WW = ω·f`. Power ∝ ω³, so reduced
  speed cuts energy — on the 8-node, VSP saves ~49% vs fixed speed (EPANET-validated),
  and in the coupled problem it also lightens the feeder load (better voltages).
  Select VSP pumps and ω_min in the Water/Coupled tabs (`SolverConfig.vsp_pumps`).
  The EPANET baseline honestly uses the speeds the network's own `SETTING`
  controls apply (read back from EPANET's applied settings).
- **PRV** — a three-state valve (closed / open / active) via two binaries with
  `x_act + x_open ≤ 1`, exact in the binaries (no relinearization). When *active*
  it pins its downstream junction to `h_set = E_down + P_set·2.3072` by absorbing
  `R_PRV` of head. The **8-node + PRV** network (net 108) reproduces EPANET's PRV
  logic to ~0.15 ft; `h_set` is user-tunable (`SolverConfig.prv_settings`), and the
  app plots the paper-style PRV panel (state timeline, downstream head vs h_set,
  R_PRV, valve flow — model vs EPANET).

## Install & run

```bash
pip install -r requirements.txt

# Windows: double-click Python\run_ui.bat  (installs deps on first run, opens the UI)
streamlit run app.py          # WEB UI — five tabs: Home · Water · Power · Coupled · Guide
                              # appearance toggle: System / Light / Dark

python main_owf.py            # WATER CLI: pick network, get a recommended mode,
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

## Deploy the web UI (Railway / any container host)

The app is a standard Streamlit app; all solvers (HiGHS, SCIP), CVXPY and epyt's
bundled EPANET have Linux wheels, so it runs on a Linux container unchanged.
Deploy files live in `Python/`: `Procfile`, `runtime.txt`, `.streamlit/config.toml`.

**Railway (from GitHub):**
1. Push the repo to GitHub (`git push`). Only `Python/` is tracked; `Matlab/` is
   git-ignored, so nothing proprietary ships.
2. On railway.app: **New Project → Deploy from GitHub repo** → pick this repo.
3. In the service **Settings**, set **Root Directory** = `Python` (so Railway sees
   `requirements.txt` and the `Procfile`).
4. Railway builds with Nixpacks (reads `runtime.txt` → Python 3.11) and starts with
   the `Procfile`, which binds Streamlit to Railway's `$PORT` on `0.0.0.0`.
5. Open the generated public URL.

Notes:
- **No auth** — anyone with the URL can run solves (which cost CPU). Add Railway
  access control or a password if that matters.
- The filesystem is **ephemeral** — generated plots/CSVs live only for the session.
- `optimize`/Net3 runs are minutes-long and memory-hungry; a small instance is fine
  for the tree/looped demos but size up if you rely on Net3 `optimize`.
- MOSEK/Gurobi stay unavailable in the cloud unless you add their packages **and**
  a license; HiGHS/SCIP need neither.

## Commercial solvers locally (MOSEK / Gurobi academic licenses)

Running locally — e.g. by double-clicking **`run_ui.bat`** — the app can use MOSEK
and Gurobi in addition to the bundled HiGHS/SCIP. Both offer **free academic
licenses**. Nothing in the app needs configuring: the solver dropdowns in the
**Water** and **Coupled** tabs detect installed solvers automatically and show
missing ones greyed out as *"(not installed)"*.

**MOSEK** (personal academic license — user-locked, works on all your machines):
1. `pip install Mosek` (once; deliberately not in `requirements.txt` so the
   cloud deploy doesn't advertise a solver it can't license).
2. Request the license at mosek.com → *Academic Licenses* with your university
   email; they email you a `mosek.lic`.
3. Put the file in **either** place — both work:
   - `mosek.lic` at the **repo root** (next to this `Python/` folder). It is
     git-ignored (`*.lic`) and Docker-ignored, so it can never be committed or
     baked into an image; `owf/config.py` points `MOSEKLM_LICENSE_FILE` at it
     automatically, or
   - MOSEK's default `C:\Users\<you>\mosek\mosek.lic`.
   > Gotcha: an **old expired copy in `~\mosek\`** silently shadows a newer one
   > and fails with `err_license_version` — refresh both copies on renewal. The
   > license major version must match the pip package (`Mosek` 11.x ↔ v11 lic).

**Gurobi** (named-user academic license — machine-locked, one per machine):
1. `pip install gurobipy` (once).
2. portal.gurobi.com → *Licenses → Request → Named-User Academic* (free); needs
   a campus network or university VPN at activation time.
3. `grbgetkey <your-key>` on the machine that will run the app. It writes
   `gurobi.lic` to your home directory (Gurobi's default search path). A copy at
   the **repo root** also works (`GRB_LICENSE_FILE` is pointed at it, same
   git/Docker-ignore protection as MOSEK).

**Then just launch as usual** — `run_ui.bat` (or `streamlit run app.py`): the
bat file installs `requirements.txt` on first run, starts the app on
`http://localhost:8501`, and the dropdowns un-grey MOSEK/GUROBI. Pick one and
every MILP/LP step of that run uses it, with automatic fallback to the bundled
solvers if a solve errors. Cross-check: the 8-node case gives the **same cost on
HiGHS and MOSEK (0.27474)** — a good first sanity test after installing.

## Web UI (`streamlit run app.py`)

Five tabs, plus a **System / Light / Dark** appearance toggle:

- **🏠 Home** — paper intro, a flowchart of how the coupled problem is solved, an
  **animation** of successive linearization converging (with a "why it's better"
  explainer), the networks, and the key formulations.
- **💧 Water** — the decoupled OWF: pick a network + mode (`direct` / `warmstart` /
  `optimize` / `epanet`) + price; interactive network map, flow animation, schedule,
  flows/heads/convergence/error plots, EPANET rules, per-case CSVs and a session
  comparison table. A **⚡ pump-power check** compares the optimizer's Σ
  *linearized* power against the Σ *true* nonlinear power at the solution
  (total + per-pump Δ% — the linearization-honesty evidence). A **📡
  Transmit-to-DSO** panel publishes the pump schedules (EPANET rule-based,
  C-OWF optimized, or both, with their linearized counterparts) to the Power
  tab — the water-utility → DSO hand-off, live.
- **⚡ Power** — the standalone reactive OPF on any feeder. DER controls: **PV
  sizing** (Spv = k·Ppv,max), **# active PV sites**, voltage limits, solver choice,
  and the water **pump-load hand-off**: transmitted schedules from the Water tab
  (acknowledged with a banner; **both schedules are solved separately and their
  OPF results compared** when two arrive), or pick EPANET rules / C-OWF optimized
  directly. A **pump-load table** reports the Σ true kWh each schedule imposes
  on the feeder, plus the water LP's linearized Σ and Δ% where the schedule
  came from an optimization. Outputs: feeder voltage map, voltage profile
  (linear vs Z-bus),
  voltage heatmap, PV reactive dispatch + **capacity utilization**, **true
  loss** (with vs without VAr support), a setpoints CSV, the full **solver log**
  (fd-level capture: HiGHS/MOSEK presolve + primal-dual iterations), and a
  **🧬 correlation explorer** (playable signal overlay, Pearson matrix,
  user-drawn pair scatter).
- **🔗 Coupled** — joint C-OWPF against **both decoupled practices**: EPANET rules
  + OPF (always) and C-OWF + OPF (Thorough effort), with Δ% savings vs each.
  Controls: water net, feeder, pump→bus, PV sizing/count, voltage limits, VSP/PRV,
  solver choice, and a **Fast / Thorough** search-effort switch (Fast is
  recommended for Net3 / SB-128). Coupling map, decoupled-solution tables, the
  full pipeline **solver log** (every search candidate + primal-dual output), a
  **🧬 correlation explorer** (price / demand / pump kW / PV / voltage / loss /
  tank heads co-evolving, with Pearson matrix and pair scatter), and validation
  against EPANET (water) and Z-bus (power). The **water-side plot suite is the
  same as the Water tab's** (💧 water map + hour slider, flow animation,
  schedule-vs-price with VSP speeds, flows/heads vs the EPANET replay,
  convergence, error summary), plus the same ⚡ pump-power check — VSP coupled
  results finish with the **speed-pinned EPANET replay polish**, so the
  reported max |dHead| is the replayed one (~0.005 ft on the tutorials, not a
  relaxation artifact).
- **📖 Guide** — the full methodology, including the power and coupled sections.

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

## Power distribution (PDN)

The `pdn` package models a radial single-phase feeder with the **LinDistFlow /
Kekatos** linear voltage model — squared bus voltages are *affine* in the net
injections, `v² = R(−p) + X(−q) + V_k` with `R = 2F·diag(r)·F′`, `X = 2F·diag(x)·F′`,
`F = −A⁻¹` — so it stays convex (LP-friendly). A nonlinear **Z-bus** fixed-point
power flow (`pdn.zbus_solve`, a port of `func_zbussan.m`) is the ground truth: it
returns the true voltages and the true real loss `Σ|Iℓ|²rℓ`.

`pdn.solve_pdn_opf` dispatches PV **reactive** setpoints (active fixed at available
solar; capability `|q| ≤ √(S²−p²)`) to hold `Vmin ≤ |V| ≤ Vmax`, then verifies with
the Z-bus and reports **loss with vs without** VAr support.

Feeders (checked in under `pdn/feeders.py` + `pdn/feeders_paper.py`, self-contained):

| key | feeder | buses | PV sites | source |
|---|---|---|---|---|
| `ieee13` | IEEE-13 | 12 | 5 | IEEE 13-node single-phase |
| `ieee33` | IEEE-33 | 32 | 15 | Baran & Wu 33-bus |
| `sb128`  | SB-128  | 128 | 27 | SB-128 test feeder (stressed) |
| `sce47`  | SCE-47  | 47 | 5 | paper Table I / Fig. 5 |
| `sce56`  | SCE-56  | 57 | 1 | paper Table II / Fig. 6 |

The `ieee13/33/sb128` data is extracted from `Matlab/Codes/PDN Networks/*.mat`; the
two SCE feeders are transcribed from the paper tables (MVA loads @pf 0.9, explicit
PV/cap nameplates, tie lines dropped for the base radial config). All validate
against the nonlinear Z-bus to < 0.01 pu on the healthy feeders.

## Coupled C-OWPF

The `coupled` package ties water to power: each pump's electrical power `P_pump`
(kW) becomes a per-unit active load at its feeder bus (the paper's `Ξ` coupling),
and the pump also draws reactive `q_pump = P_pump·√(1/PF²−1)`. The **objective is
the paper's (eq. 33d) — pump energy plus the cost of network losses**, both priced
at the WDN electricity price:

```
min  Σ_t π_t · [ (Σ_p Ppump)/1000  +  Ĉloss_t ] ,   Ĉloss_t = Σ_ℓ r_ℓ (P_ℓ² + Q_ℓ²)
```

The loss is a convex quadratic in the LinDistFlow branch flows `P_ℓ = Σ_{k∈subtree} p_net`;
it is **linearized around the current flows each iteration** (same successive-
approximation idea as the water side), so every step stays an LP/MILP. This is what
makes PV reactive dispatch and pump timing worth **real dollars** — the tool reports
pump / loss / total cost and the coupled-vs-decoupled saving.

**Shunt caps are voltage-dependent** (paper eq. 1b/2b): the cap injects `qˢ·vₙ`, so
`(I − X·diag(qˢ)) v = R(−p)+X(−q)+V_k` — the `(I−Qsx)⁻¹` correction is folded into
`R,X,V_k` (matching `func_branchbus.m`'s `V_nsh`), and `q_net` then excludes caps.

With the schedule fixed the coupled problem is a pure **LP**; with the schedule free
the only integers are the pump binaries (the feeder adds none).

- `solve_coupled` — the coupled successive-linearization loop (LP/MILP).
- `optimize_coupled_schedule` — **voltage-aware** schedule search: each candidate is
  scored on the true pump cost **and** the coupled voltage feasibility, then a
  **warm-started trust-region MILP** (free binaries, Hamming-distance cap from the
  incumbent) and a 1-opt polish. A `Fast` path (fewer candidates, no MILP) keeps
  Net3 / SB-128 tractable; the final result is guaranteed **never worse than the
  decoupled** baseline.
- `validate_coupled` — replays the schedule in EPANET (water ΔH) and the injections
  in the Z-bus (voltage error, true loss).

Typical: water ΔHead ≈ 0.02 ft, power ΔV ≈ 0.005–0.013 pu; coupled beats the
decoupled hand-off on voltage feasibility at equal or lower pump cost.

## Layout

```
Python/
├── app.py                      # Streamlit app: Home · Water · Power · Coupled · Guide
├── main_owf.py                 # water CLI            ≈ Main_OWF_IEEE_ACCESS.m
├── owf/                        # WATER side (successive-linearization MILP)
│   ├── config.py               # network specs, pump curves, prices, solver settings
│   ├── epanet_io.py            # read_inp.m + init_epanet.m   (via epyt)
│   ├── connection_matrices.py  # ConnectionMatrices_WDN.m  (Pi, Λ, Θ, Τ, Κ, Ω, Δ …)
│   ├── network.py · initial_values.py · linearization.py
│   ├── constraints.py          # every define*_CVX.m  → one function
│   ├── solver.py               # WDN_OWF_IEEEACCESS_cvx.m  (the MILP loop)
│   ├── warmstart.py · validation.py · plots.py · netmap.py
├── pdn/                        # POWER side
│   ├── feeders.py · feeders_paper.py   # feeder data (self-contained)
│   ├── lindistflow.py          # Kekatos linear voltage model
│   ├── powerflow.py            # nonlinear Z-bus + true loss  (func_zbussan.m)
│   ├── network.py              # PDN object, PV sizing, solar profile
│   └── opf.py                  # standalone reactive-power OPF
├── coupled/                    # COUPLED C-OWPF
│   ├── config.py               # feeder + coupling + DER settings
│   ├── coupled_lp.py           # coupled successive-linearization loop
│   ├── schedule.py             # voltage-aware search + trust-region MILP
│   ├── validation.py           # EPANET + Z-bus cross-check, true loss
│   └── runner.py
├── ui/                         # Streamlit sections (theme, landing, water, power, coupled, plots)
├── guide.py                    # methodology (water + power + coupled)
├── data/…                      # EPANET .inp files (self-contained)
└── tests/                      # test_eightnode.py … test_coupled.py
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
- **Power coupling**: the PDN voltage model is the Kekatos LinDistFlow relaxation
  with **voltage-dependent shunt caps** (`qˢ·v`, eq. 1b/2b) folded into `R,X,V_k`;
  the loss term uses the lossless branch-flow approximation and is linearized each
  iteration. The nonlinear **Z-bus** replay (caps in the Y-bus) is the ground truth
  the linear model is checked against. PV active is fixed at available solar; PV
  **reactive** is the control. Voltage limits are hard, with an optional soft-slack
  feasibility device (the same one used for water head bounds — not an economic term).
- **VSP / PRV**: implemented (see the feature summary at the top). The MATLAB
  reference toggles FSP-vs-VSP per script; the Python port has a genuine per-pump
  partition (`PumpParams.is_vsp`), and the PRV uses the two-binary three-state
  model (the MATLAB's third binary `ValveStatusOnOff` is redundant: it equals
  `x_act + x_open`).
```
