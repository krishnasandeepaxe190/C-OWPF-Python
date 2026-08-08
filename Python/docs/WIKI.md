# C-OWPF Project Wiki — Deployment Errors & Lessons Learned

Living log of every deployment failure, environment gotcha, and modeling lesson
hit while building this app, with the fix that worked. Check here FIRST when
something breaks.

---

## 1. Deployment (Railway / Docker / Streamlit)

### Railpack builder: `failed to resolve ghcr.io/railwayapp/railpack-frontend ... denied`
Railway's own builder failed to pull its frontend image (upstream registry auth
outage — not our code). **Fix (permanent):** `Python/Dockerfile` — Railway
auto-detects it and skips Railpack entirely. Requirements: Railway **Root
Directory must be `Python/`**; base `python:3.11.9-slim` + `libgomp1` (EPANET
needs the OpenMP runtime); shell-form CMD so `${PORT}` expands.

### Streamlit 500s on every request after a fresh cloud install
Streamlit 1.60 runs on Starlette/uvicorn; a newer Starlette breaks Streamlit's
gzip middleware (`GZipResponder.__init__` missing `thread_minimum_size`).
**Fix:** pin the trio in lockstep in `requirements.txt`:
`streamlit==1.60.0`, `starlette==1.0.0`, `uvicorn==0.44.0`.

### Crash rendering a result tab: `TypeError` on `st.tabs()` concatenation
`st.tabs()` returns a sequence that is **not list-concatable**; code like
`[tabs[0], tabs[1]] + tabs[3:]` crashes the page — and only when a result is
actually rendered, so an app-loads smoke test misses it. **Rules:** (1) append
conditional tabs **last** so fixed indices stay valid; (2) if you must slice,
`list(tabs)` first; (3) verify UI changes by driving a REAL run through
`AppTest` (set widgets, click run, render), not just app-loads.

### `use_container_width` deprecation warnings
We pin Streamlit 1.60 where `use_container_width=True` is correct. Do NOT
migrate to `width=` while the pin is in place.

### Solver licenses and the cloud
`*.lic` is ignored in `.gitignore` AND `Python/.dockerignore` — a license can
never be committed or baked into an image. The deployed app therefore runs
HiGHS/SCIP only. For cloud Gurobi use a WLS license via env vars
(`WLSACCESSID`/`WLSSECRET`/`LICENSEID`); MOSEK needs a floating/server license.

---

## 2. Solver & environment gotchas

### MOSEK: `rescode.err_license_version(1002): Feature PTS version X not supported`
The pip package major version must match the license. **Gotcha that cost us
time:** a stale expired v10 `mosek.lic` in `C:\Users\<user>\mosek\` silently
shadowed the fresh v11 license at the repo root. Fix: refresh the home copy
(old one backed up as `mosek.lic.v10.bak`). The app points
`MOSEKLM_LICENSE_FILE` at the repo-root `mosek.lic` (git-ignored) via a
bootstrap in `owf/config.py`. Verified: 8-node MILP on MOSEK == HiGHS
(0.27474, bit-for-bit).

### SCIP out-of-memory on large free-binary MILPs
BWSN (129 nodes) with free pump binaries killed SCIP. Mitigations:
`optimize_schedule` skips the MILP proposal for >100-node networks, caps
`max_flips`, and validates candidates by EPANET replay. With MOSEK licensed,
the full trust-region MILP could be re-enabled on large nets (untested).

### EPANET/epyt quirks
- `setLinkStatus(OPEN)` **resets a pump's speed setting to 1.0** — always set
  status FIRST, then `setLinkSettings(speed)`.
- Pump speed via `[CONTROLS]` uses the bare value: `LINK 9 0.771 AT CLOCKTIME
  11 AM` (the word `SETTING` in a control line is a parse error, Error 202).
  `getComputedTimeSeries().Setting` reports the applied relative speed;
  `getLinkInitialSetting()` gives a PRV's pressure setpoint (psi).
- epyt leaves `*_temp.txt` / `.rpt` / `.bin` files it can't always delete
  (WinError 32) — harmless; they're gitignored.
- epyt occasionally fails a run outright with `[Errno 2] No such file or
  directory: '@#XXXXXXXX.bin'` (its randomly-named scratch file) — transient;
  rerun the case. If persistent, clear stray `@#*` files from the CWD.
- Repeated multi-minute solves in one bash session can hit cygwin fork errors —
  run long jobs via background tasks.
- **Parallel solve sweeps can exhaust Windows commit memory and freeze the PC.**
  8 concurrent Python solver processes (cvxpy + scipy + EPANET each) blew the
  paging file: `ImportError: DLL load failed ... The paging file is too small`,
  `uv_spawn` failures, and a plotly `MemoryError` in an unrelated process.
  Throttle batch validation to ~2 concurrent solves on this machine, run
  BWSN/Net3-class cases alone, and never run an AppTest probe concurrently
  with a sweep. Also: two concurrent solves must never share the same .inp
  (epyt writes `*_temp` scratch files next to it).

### AppTest (Streamlit test harness) gotchas
- `at.session_state.get("key")` does NOT exist on the session-state proxy — it
  raises `AttributeError: get not found`. Use `"key" in at.session_state` then
  index. (A crashed *probe* is easy to misread as a crashed *app*.)
- Printing widget values that contain emoji (📡 etc.) from a verification
  script crashes on Windows' cp1252 console — run probes with
  `PYTHONIOENCODING=utf-8`.
- Drive REAL runs (set widgets, click, re-run, inspect session state); an
  app-loads-clean smoke test misses every render-with-results bug.
- **AppTest cannot catch console-state bugs.** A served app's `sys.stdout` can
  be closed/detached (run_ui.bat windows, cp1252 consoles) — a bare `print`
  in a tab then CRASHES the page, while AppTest's healthy console passes.
  `capture_fds` therefore rebinds `sys.stdout/stderr` to fresh UTF-8 writers
  over the captured fds for the duration of the block (restored after), so any
  print — ours or a library's — is safe and lands in the log. Never `print`
  in UI code outside a capture block; use `st.*` elements instead.
- **MOSEK logs via Python `logging`, not stdout.** CVXPY's MOSEK interface
  routes the solver log through `logging` handlers that hold a reference to the
  ORIGINAL console stream — under run_ui.bat that handle can be invalid, so
  every MOSEK line dumped an `OSError: [WinError 6]` "--- Logging error ---"
  traceback wall (the solve itself still succeeded). `capture_fds` now also
  retargets plain StreamHandlers bound to the original stdio into the captured
  stream (restored after) and sets `logging.raiseExceptions = False` for the
  block — MOSEK's interior-point log lands in the Solver-log tab like HiGHS's.

### Windows / OneDrive / LaTeX
- OneDrive **online-only placeholder files** read as *Permission denied* /
  "cloud provider exited unexpectedly". Fix: right-click → "Always keep on this
  device", or copy the file elsewhere. (Burned us on the MATLAB reference AND
  the PRV .inp.)
- MiKTeX: `\usepackage[table]{xcolor}` pulls a broken `colortbl`/`array` →
  "Undefined control sequence \insert@pcolumn" on `p{}` columns. Drop the
  `table` option. Sanitize non-ASCII (·, Δ, ‖, ≤, →) before pdflatex.
- matplotlib font cache can throw a random access violation under pytest when
  the whole suite runs — `test_plots.py` passes in isolation; rerun before
  blaming your change.

---

## 3. Modeling lessons (why results looked "wrong" and what was true)

### BWSN head error was 300+ ft
Two causes, both legitimate: (1) a K=800 minor loss on LINK-166 while the model
ignores minor losses (study convention) — zeroed in the .inp; (2) EPANET's
continuous pump throttling at a full tank vs our binary FSP — added tank-level
CONTROLS so EPANET switches like the model. Result: 0.8 ft reproduction.

### Optimizer "wins" that EPANET rejects
Soft-bound LP scores can rank a tank-draining schedule as cheapest. For >100
node nets the winner AND the EPANET baseline AND load_shift are re-validated by
EPANET replay; cheapest schedule with replay error <= 5 ft wins.

### VSP: three traps
1. **McCormick envelope is invalid when the pump is OFF** (omega=0 is outside
   [omega_min, omega_max]) — the corner rows carrying `-omega*fmax` constants
   made ANY schedule with an off-hour infeasible. Fix: gate row 4 by on/off.
2. **Slow convergence tail:** the flow-iterate norm shrinks asymptotically —
   added VSP-only objective-stability convergence (`obj_rtol`, 3 stable iters).
3. **High replay error (16.5 ft) on VSP+PRV:** free-binary solve hits the
   iteration cap mid-descent; re-solving with only the schedule fixed lets the
   speeds drift again. Fix: **speed-pinned polish** — replay (schedule, speeds)
   in EPANET, pin BOTH (`fixed_schedule` + `fixed_speed`; with omega pinned,
   `WW = omega*f` is exact), reconverge from the replay point, keep the best-
   replaying candidate. 16.5 → 0.01 ft.

### Coupled VSP had the SAME replay disease — the polish must live in BOTH paths
3-node + IEEE-13, VSP, coupled: max |dHead| = **27.8 ft**, pump-flow error
**808 GPM** (user: "why is the head diff very high?"). Same root cause as the
decoupled 16.5-ft case: the damped homotopy stops at a point where the
McCormick relaxation gap leaves the model's pump flow far from what EPANET's
affinity-scaled curve delivers at the solved speeds — and the coupled path
(`solve_coupled_schedule` / `optimize_coupled_schedule`) never replayed or
re-pinned. Fix: `_replay_polish` in `coupled/runner.py` — replay (schedule,
speeds) in EPANET, re-converge the **coupled** problem with `fixed_schedule`
AND `fixed_speed` (omega pinned makes `WW = omega*f` exact) linearized at the
replay point, keep the best-replaying candidate (monotone). Wired into the
final clean convergence of `optimize_coupled_schedule` and the UI's
`dec_rules`/`dec_owf` solves (`replay_polish=True`). Result: 27.8 → **0.005 ft**,
808 → **0.005 GPM**, same schedule and speeds — only replay-consistent now.
**Lesson: every solve path that produces a VSP result must END with the
speed-pinned EPANET replay polish. A fix applied to one path (decoupled) does
not protect its siblings (coupled).**

### PRV: what worked immediately
The two-binary three-state MILP (x_act/x_open, big-M, h_set = downstream
elevation + psi*2.30724939) validated at 0.15 ft on the first correct build.
PRV binaries are exact — do NOT relinearize them; only pumps/pipes relinearize.
**Known marginal-state gap (documented, not a bug):** on the coupled FSP+PRV
8-node case the optimized schedule replays at ~1.5 ft — localized to the PRV's
downstream zone in the hours where upstream pressure sits exactly at the
setpoint margin: the model picks "open" (R_prv = 0) while EPANET still
throttles ~1.5 ft. Flows agree to 0.004 GPM, the diff is bounded by the
pressure margin in those hours, and it is within feas_tol (2 ft). The replay
polish cannot close it (nothing to relinearize — the flows already match);
closing it would need EPANET-state-pinned valve binaries.
The PRV panel overlays EPANET's **rule-based** regulation so a cost increase at
a user-raised h_set is visibly "paying for pressure", not solver failure.

### Feeder physics
- IEEE-13's own base load sits at Vmin ≈ 0.945 pu — a 0.95 floor is infeasible
  before any pump is added (the UI warns). Use ~0.90 or stronger buses.
- Caps are voltage-dependent: folded into R/X/V_k for the linear model, in the
  Y-bus for the nonlinear Z-bus. `q_net` must NOT subtract them again.
- PV sized strictly by bus load — a pu floor over-sizes on IEEE-33's 100 MVA
  base and causes reverse-flow overvoltage (1.29 pu).

---

## 3b. Feature map for the institutional workflow (added 2026-08-07 evening)

- **Water → DSO transmit:** Water tab publishes pump schedules (EPANET rules /
  C-OWF optimized / both) into `st.session_state["dso_handoff"]`; the Power tab
  acknowledges, pre-selects the transmitted source, and solves ONE OPF PER
  schedule, comparing them (Vmin, violation, loss, time, Δ%).
- **Coupled three-way comparison:** `dec_rules` (always) and `dec_owf`
  (Thorough) are both solved and tabled against C-OWPF with Δ% per practice —
  the coupling's value vs *both* real decoupled workflows.
- **Solver logs everywhere:** `ui/capture.py` (`capture_fds` context manager /
  `run_capturing_all`) redirects fd 1/2 so the C-level HiGHS/MOSEK/EPANET
  output is captured. Water, Power AND Coupled each have a 🖥 Solver-log tab;
  Power prints one banner block per handed-off schedule; Coupled captures the
  whole pipeline (search candidates + primal-dual logs), 400 kB tail.
- **🧬 Correlation explorer** (Power + Coupled): signal library built at run
  time (`R["signals"]` — price, demand, pump kW, pumps ON, VSP speed, PV P/Q,
  min Z-bus voltage, grid import, loss, tank heads, PRV head loss). Three
  views in `ui/correlations.py`: playable normalized overlay, Pearson matrix,
  user-drawn pair scatter with trend + r.
- **Pump-power fidelity check (all three tabs):** `ui/pump_power.py`
  (`pump_power_check`) compares Σ_t of the LP's *linearized* pump power P-hat
  against Σ_t of the *true* nonlinear power at the solved point, total and
  per-pump with Δ%. `CoupledResult` gained `ppump_linear` (snapshot of
  `wmodel.Ppump.value`); the Water record stores it per run; the DSO hand-off
  carries `schedules_linear` so the Power tab can show the gap on transmitted
  optimized schedules. A near-zero Δ% is the linearization-honesty evidence
  (3-node coupled VSP: 2339.130 vs 2339.131 kWh).
- **Plot consistency (user feedback, 2026-08-08):** *"the same plots [as the
  Water tab] are not there in the coupled tab; maintain plots consistent."*
  The Coupled tab now renders the SAME water-side figure suite via
  `owf.plots.plot_all` on the `CoupledResult` (works because the plot functions
  only touch `errors/objectives/onoff/speed/flows/heads` + the EPANET-replay
  report): 💧 Water map (+hour slider), Flow animation, Flows, Heads,
  Convergence, Error tabs, and the schedule-vs-price figure above the schedule
  table. **Standing rule: when a visualization is added to one tab, mirror it
  in the sibling tab if the same data exists there.** New conditional tabs
  still go LAST (`tabs[-1]` must remain the PRV tab when present).
- **Doc rule (standing):** every commit batch ends by updating README, the
  codebase PDF (`ast_extract.py` + `codebase_summary.tex` recompile), and this
  wiki.

## 4. Verification checklist (what "done" means here)

1. `python -m pytest Python/tests` — all green (run `test_plots.py` separately
   if the font-cache crash appears).
2. `AppTest.from_file('app.py')` — not just loads: **drive a real run** (set
   widgets, click, render results) for any UI change.
3. Water: EPANET replay of the SAME schedule — small max |dHead|, positive min
   pressure. Power: Z-bus voltages within ~0.02 pu of LinDistFlow.
4. Cost claims: compare TRUE nonlinear costs, never linearized objectives; the
   EPANET-rules baseline uses the speeds EPANET actually applied
   (`epanet_pump_speeds`).
5. Nothing proprietary in git: `Matlab/`, `Titlelabel1/`, `*.lic` are ignored;
   check `git status` before every commit.
