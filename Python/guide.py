"""Methodology / user guide rendered inside the Streamlit app (Guide tab)."""
from __future__ import annotations


def render_guide(st) -> None:
    st.header("📖 Methodology & Guide")
    st.markdown(
        "This tool schedules **fixed-speed pumps (FSPs)** on a water distribution "
        "network to minimise pumping-energy cost, by a **successive linear "
        "approximation** whose every iteration is a **MILP** solved with HiGHS "
        "(via CVXPY). EPANET (through *epyt*) provides the hydraulic ground truth."
    )

    # ---- overview ----
    with st.expander("1 · What it computes", expanded=True):
        st.markdown(
            "- **Decision variables** over a horizon of *T* hours: pipe & pump "
            "flows *q*, nodal heads *h*, pump power *P*, and binary pump on/off *z*.\n"
            "- **Objective**: minimise the electricity cost of pumping under a "
            "time-of-use (or flat) price.\n"
            "- **Constraints**: mass balance, (linearised) Hazen–Williams pipe head "
            "loss, pump head-gain & power, tank dynamics, and head/pressure bounds.\n"
            "- Every result is checked by **re-simulating the optimised schedule in "
            "EPANET**."
        )
        st.latex(r"\min_{q,h,P,z}\;\; \sum_{t} \pi_t \sum_{m} \frac{P_{m,t}}{1000}")

    # ---- model ----
    with st.expander("2 · The model (per-iteration, all linear)"):
        st.markdown("**Mass balance** at junctions (demand from EPANET):")
        st.latex(r"\Pi_{J}\, q_t = -d_t")
        st.markdown("**Pipe head loss** — Hazen–Williams, linearised each iteration "
                    "around the previous flow $\\bar q$ (monomial linearisation):")
        st.latex(r"\tilde\Pi\, h_t = C_{p,t} + \Pi' q_t,\qquad "
                 r"C_{p,t} = \bar q\,(\omega\,|\bar q|^{0.852}-1)")
        st.markdown("**Pump head-gain curve** with a *general* exponent $\\nu$ "
                    "(not restricted to quadratic):")
        st.latex(r"\Delta h_m = h_{0,m} - \sigma_m\, f^{\nu}")
        st.markdown("**FSP power** and its first-order Taylor model about $\\bar f$ "
                    "(reduces to $c_m(h_0-3\\sigma f^2)$ when $\\nu=2$):")
        st.latex(r"P(f)=c_m\,(h_0-\sigma f^{\nu})\,f,\qquad "
                 r"P=A' f + B' z")
        st.markdown("**Pump on/off gating** (Big-M) and **flow bound**:")
        st.latex(r"M(z-1)\le \Lambda\Pi^{\!\top}h-(C1M+C2M\,f)\le M(1-z),\qquad "
                 r"0\le f\le f_{\max} z")
        st.markdown("**Tank dynamics** — level integrates net inflow $u$; bounds and "
                    "a terminal level are enforced:")
        st.latex(r"H_{i,t}=H_{i,0}+\frac{\delta}{A_i}\sum_{\tau\le t}u_{i,\tau}")

    # ---- special elements ----
    with st.expander("3 · Special link logic"):
        st.markdown(
            "- **Closed pipes** (EPANET initial status *Closed*): pinned to zero "
            "flow and excluded from the head-loss equation.\n"
            "- **Switched bypass** (e.g. Net3's pipe 330): a pipe that is **open "
            "exactly when its pump is off** — modelled with Big-M gated by the pump "
            "binary so the tank can feed the network by gravity when the pump rests.\n"
            "- **Availability windows**: a pump on a time-limited source (Net3's "
            "lake pump, hours 1–14) is forced off outside its window."
        )
        st.markdown("**Pump curves are derived from each network's EPANET head "
                    "curve** — single design point → exact quadratic ($\\nu=2$); "
                    "multi-point → fitted $\\nu$ (Net3: 1.77 & 1.09).")

    # ---- successive linearization ----
    with st.expander("4 · Successive linearisation"):
        st.markdown(
            "With the coefficients frozen, each MILP is solved; its solution "
            "becomes the next linearisation point; repeat until the stacked iterate "
            "stops moving:")
        st.latex(r"\varepsilon_k=[\,h;\,q;\,z\,],\qquad "
                 r"\lVert\varepsilon_k-\varepsilon_{k-1}\rVert_2 < 0.5")
        st.markdown(
            "Tree networks converge in 2–4 iterations. Looped networks do **not** "
            "from a default start — the tank head is over-determined (set by both "
            "the hydraulic path *and* the tank integrator), which is what warm "
            "starts fix.")

    # ---- warm starts ----
    with st.expander("5 · Warm starts (how we make looped networks converge)", expanded=True):
        st.markdown(
            "**Core idea:** don't guess a linearisation point — pick a pump "
            "schedule, run it through **EPANET** to get a *physically-consistent* "
            "operating point, and linearise there. EPANET's solution satisfies mass "
            "balance, head loss and the pump curves exactly for that schedule, so "
            "the linearised model's fixed point sits right at the seed (these solves "
            "often converge in a **single iteration**).")
        st.markdown("**The mechanism** (`warmstart_point`):")
        st.markdown(
            "1. Take a candidate on/off schedule.\n"
            "2. Impose it in EPANET (delete its controls/rules; drive each pump "
            "hour-by-hour, plus any switched bypass).\n"
            "3. Run the true nonlinear hydraulics → EPANET flows & heads.\n"
            "4. Build the linearisation coefficients at those flows; seed the loop "
            "with $[h;q;z]$.")
        st.markdown("**Candidate library** — EPANET's own schedule, all-on, all-off, "
                    "and cheap-hours (pump the cheapest ~half of hours). Candidates "
                    "whose EPANET run fails or returns negative/NaN hydraulics are "
                    "dropped. Best = feasible → least head-bound violation → cheapest.")
        st.markdown("**Two aids** for hard cases: **soft head bounds** (penalised "
                    "slacks so every MILP stays feasible) and **damping** (blend the "
                    "new linearisation point with the previous one, a trust region).")

    # ---- optimization ----
    with st.expander("6 · Honest schedule optimisation"):
        st.markdown(
            "A single MILP at a frozen linearisation *cannot* judge a schedule far "
            "from its linearisation point, so `optimize` mode never lets the model "
            "predict — **every candidate is evaluated honestly**: fix the schedule → "
            "converge the LP → score by the **true nonlinear cost**. It searches "
            "price-quantile schedules + an MILP proposal, then does a **1-opt "
            "polish** (flip single pump-hours, keep a flip only if it stays feasible "
            "and lowers true cost), and finally a **clean re-solve** of the winner.")

    # ---- modes ----
    with st.expander("7 · Solve modes"):
        st.markdown(
            "| mode | use it for | what it does |\n"
            "|---|---|---|\n"
            "| **direct** | tree networks | plain loop, free pump binaries |\n"
            "| **warmstart** | looped networks | multi-start over candidates, then "
            "fix the best schedule and converge |\n"
            "| **epanet** | validation / Net3 | reproduce EPANET's own rule-based "
            "operation |\n"
            "| **optimize** | savings | search for the cheapest feasible schedule, "
            "report savings vs EPANET |")

    # ---- validation ----
    with st.expander("8 · Validation & the EPANET comparison"):
        st.markdown(
            "**Two separate EPANET runs, for two purposes** — a distinction worth "
            "keeping straight:")
        st.markdown(
            "- **Cost** compares *different* schedules: **EPANET(rules)** — EPANET "
            "running its own tank-level rules — versus the C-OWPF optimised "
            "schedule. Cost only; heads/flows are not compared here.\n"
            "- **Head/flow error** uses the *same* schedule: the C-OWPF schedule is "
            "**imposed back into EPANET** and replayed, and the heads/flows are "
            "compared point-by-point. This measures linearisation fidelity.\n"
            "- **min pressure** in that replay > 0 confirms the schedule is "
            "hydraulically feasible.")
        st.info("Accuracy achieved (same-schedule replay): 8-node ≈ 0.15 ft, "
                "3-node/Net1/Net2 ≈ 0.005–0.02 ft.")

    # ---- power distribution ----
    with st.expander("10 · Power side — LinDistFlow / Kekatos voltage model"):
        st.markdown(
            "Each radial feeder (IEEE-13 / IEEE-33 / SB-128) is reduced to a "
            "single-phase model with the slack bus at index 0. The **squared** bus "
            "voltages are an *affine* function of the net bus injections:")
        st.latex(r"v^2 = R\,(-p_{\text{net}}) + X\,(-q_{\text{net}}) + V_k")
        st.latex(r"F=-A^{-1},\quad R = 2F\,\mathrm{diag}(r)\,F^\top,\quad "
                 r"X = 2F\,\mathrm{diag}(x)\,F^\top,\quad V_k = F a_0\, v_0^2")
        st.markdown(
            "with $A$ the reduced branch-bus incidence. This is the loss-less "
            "LinDistFlow relaxation; it is convex (LP-friendly) and matches a full "
            "nonlinear solve to a few $10^{-3}$ pu on moderate feeders.\n"
            "- **Validation**: after a schedule is fixed we replay the injections "
            "through the nonlinear **Z-bus** fixed-point power flow "
            "($I=\\mathrm{conj}(S)/\\mathrm{conj}(V)$, $V=Y^{-1}I+w$) and report the "
            "true voltages and the true loss $\\sum |I_\\ell|^2 r_\\ell$.\n"
            "- **Voltage-dependent caps** (eq. 1b/2b): a shunt cap injects $q^s v_n$, "
            "so $(I - X\\,\\mathrm{diag}(q^s))\\,v = R(-p)+X(-q)+V_k$; the "
            "$(I-Q_{sx})^{-1}$ correction is folded into $R,X,V_k$ (matching "
            "`func_branchbus.m`'s $V_{nsh}$), and $q_{net}$ then excludes caps. The "
            "nonlinear Z-bus keeps them in the Y-bus.")

    with st.expander("11 · Power OPF — PV reactive dispatch"):
        st.markdown(
            "PV **active** is fixed at the available solar; PV **reactive** $q^{pv}$ is "
            "the control (smart-inverter Volt/VAr). With active known, the inverter "
            "capability is an exact per-hour box:")
        st.latex(r"|q^{pv}| \le \sqrt{S^2 - (p^{pv})^2},\qquad S = k\,P^{pv}_{\max}")
        st.markdown(
            "The standalone OPF holds $V_{\\min}\\le|V|\\le V_{\\max}$ (soft slacks) "
            "while minimising the net reactive drawn from the substation "
            "$\\sum|q_{\\text{net}}|$ — local reactive compensation, which cuts "
            "reactive line flow and hence loss. The loss reduction is *reported* from "
            "the Z-bus (with vs. without the setpoints), not linearised into the "
            "objective.")

    with st.expander("12 · Coupled C-OWPF — the interdependency"):
        st.markdown(
            "The pumps' electrical power couples the two grids. Each pump's "
            "$P^{\\text{pump}}$ (kW) becomes a per-unit active load at its feeder bus "
            "($\\Xi$ coupling) and also draws reactive $q^{\\text{pump}} = "
            "P^{\\text{pump}}_{\\text{pu}}\\sqrt{1/PF^2-1}$, so the net injections are")
        st.latex(r"p_{\text{net}} = \Psi p + \Xi\,P^{\text{pump}}_{\text{pu}} - \Gamma p^{pv},"
                 r"\qquad q_{\text{net}} = \Psi q + \Xi\,q^{\text{pump}} - \Gamma q^{pv}")
        st.markdown("**The objective is the paper's eq. 33d — pump energy + the cost of "
                    "network losses**, both priced at the WDN electricity price:")
        st.latex(r"\min\ \sum_t \pi_t\Big[\tfrac{1}{1000}\textstyle\sum_p P^{\text{pump}}_{p,t}"
                 r"\;+\;\hat C^{\text{loss}}_t\Big],\qquad "
                 r"\hat C^{\text{loss}}_t=\sum_\ell r_\ell\big(P_{\ell,t}^2+Q_{\ell,t}^2\big)")
        st.markdown(
            "The loss is convex-quadratic in the LinDistFlow branch flows and is "
            "**linearized each iteration**, so every step stays an LP/MILP. This makes "
            "PV reactive dispatch and pump timing worth **real dollars**. With the "
            "schedule fixed it is a pure LP; with the schedule free the only integers "
            "are the pump binaries — the feeder adds none.\n"
            "- **Schedule search** is *cost- and voltage-aware*: each candidate is "
            "scored on the true total cost (pump + loss) **and** voltage feasibility.\n"
            "- **Warm-started trust-region MILP**: free binaries, warm-started at the "
            "incumbent's linearisation, with a Hamming-distance cap of $K$ flips — the "
            "MATLAB-style free-binary MILP made reliable (the trust region blocks the "
            "far-away schedule flips a cold MILP would take on a stale linearisation).\n"
            "- **Decoupled vs coupled**: the app runs both — optimise water blind then "
            "impose it on the grid, vs. co-optimise — and compares cost against voltage "
            "feasibility.")

    # ---- robustness ----
    with st.expander("13 · Numerical robustness"):
        st.markdown(
            "- **Adaptive Big-M** scaled per network from the head span and the "
            "*actual* EPANET flows (a fixed 1e7 ill-conditions HiGHS on Net3).\n"
            "- **Solver fallback**: an iterate one solver reports UNKNOWN is retried "
            "on the next (HiGHS → SCIP); a single failed solve stops gracefully and "
            "returns the best iterate so far.\n"
            "- **DPP discipline**: linearisation coefficients are CVXPY *parameters* "
            "(compile once); a fixed schedule enters as a plain constant so the "
            "problem isn't recompiled every iteration.")
        st.caption("Full derivations are in Python/docs/OWF_Methodology.docx.")
