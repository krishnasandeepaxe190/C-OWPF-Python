"""Water-tab benchmark panel: this study's SLA vs the PWL-MILP of
Oikonomou & Parvania (IEEE Trans. Smart Grid, 2019), run live in the app.

Both methods solve the SAME network/data/objective with HiGHS and are scored
identically: EPANET-replayed cost + replay max |dHead| (the honest metrics --
a surrogate's own objective is not what the utility pays), plus each method's
model size (binary/continuous variables, constraint rows). The PWL schedule is
kept in session state so the 📡 transmit panel can broadcast it to the DSO —
the Power tab then solves one OPF per schedule for a fair power-side
comparison. Full offline study: docs/benchmark_pwl.md.
"""
from __future__ import annotations

import time as _time

import numpy as np
import pandas as pd

# FSP networks without bypasses/PRVs (the PWL port's scope)
BENCH_NETS = {3: "3-node", 8: "8-node", 11: "Net1", 36: "Net2",
              126: "BWSN (large — several minutes)"}


def _sla_model_size(net: int, price: int):
    """Size of ONE of our per-iterate MILPs (canonical hard-bound model)."""
    import cvxpy as cp
    from owf import constraints as OC
    from owf.config import SolverConfig
    from owf.network import setup
    from owf.solver import _build_model
    wdn = setup(SolverConfig(net_num=net, price_choice=price))
    m = _build_model(wdn)
    cons = OC.build_constraints(m, wdn)
    prob = cp.Problem(cp.Minimize(0), cons)
    n_bin = sum(v.size for v in prob.variables()
                if v.attributes.get("boolean"))
    n_cont = sum(v.size for v in prob.variables()) - n_bin
    n_rows = sum(int(np.prod(c.shape)) for c in cons)
    return n_bin, n_cont, n_rows


def _run_both(net: int, K: int, budget: float, price: int):
    """Run SLA (recommended mode) and the PWL-MILP; returns (rows, pwl_store)."""
    from main_owf import run_case, MODE_SUGGESTION
    from owf.config import SolverConfig
    from owf.network import setup
    from owf.pwl_benchmark import solve_pwl_owf
    from owf.solver import _true_pump_power
    from owf.validation import validate_schedule

    rows = []
    # --- this study: successive linear approximation ------------------------
    mode = MODE_SUGGESTION[net][0]
    t0 = _time.time()
    case, wdn, result = run_case(net, mode, price, None, False, "outputs",
                                 False, solver="HIGHS")
    wall = _time.time() - t0
    sb, sc, sr = _sla_model_size(net, price)
    if case is None or result is None or result.flows is None:
        rows.append({"method": f"SLA ({mode})", "status": "failed",
                     "binary vars": sb, "continuous vars": sc,
                     "constraint rows": sr})
    else:
        rows.append({
            "method": f"SLA — this study ({mode})", "status": "optimal",
            "binary vars": sb, "continuous vars": sc, "constraint rows": sr,
            "wall (s)": round(wall, 1),
            "model objective ($)": round(case.owf_cost, 5),
            "replayed cost ($)": round(case.owf_cost, 5),
            "replay max|dHead| (ft)": round(case.max_dhead, 3),
            "deliverable (≤5 ft)": "✅" if case.max_dhead <= 5.0 else "❌",
        })

    # --- PWL-MILP (Oikonomou-Parvania linearization) ------------------------
    wdn2 = setup(SolverConfig(net_num=net, price_choice=price))
    r = solve_pwl_owf(wdn2, K=K, time_limit=float(budget))
    pwl_store = None
    if r.flows is None:
        rows.append({"method": f"PWL-MILP (K={K})", "status": r.status,
                     "binary vars": int(r.n_binary),
                     "continuous vars": int(r.n_continuous),
                     "constraint rows": int(r.n_constraint_rows),
                     "wall (s)": round(r.build_s + r.solve_s, 1),
                     "deliverable (≤5 ft)": "❌ no incumbent"})
    else:
        rep = validate_schedule(wdn2, r)
        p_ep = np.abs(_true_pump_power(wdn2, rep.flows_epanet))
        replay_cost = float(wdn2.price_final @ (p_ep.sum(axis=0) / 1000.0))
        rows.append({
            "method": f"PWL-MILP (K={K})", "status": r.status,
            "binary vars": int(r.n_binary),
            "continuous vars": int(r.n_continuous),
            "constraint rows": int(r.n_constraint_rows),
            "wall (s)": round(r.build_s + r.solve_s, 1),
            "model objective ($)": round(r.objective, 5),
            "replayed cost ($)": round(replay_cost, 5),
            "replay max|dHead| (ft)": round(rep.max_abs_head, 3),
            "deliverable (≤5 ft)": "✅" if rep.max_abs_head <= 5.0 else "❌",
        })
        # keep the benchmark schedule for the DSO transmit panel (fair OPF
        # comparison: their schedule must be broadcastable too)
        pwl_store = dict(
            K=int(K), onoff=np.round(r.onoff), ppump=p_ep,
            horizon=int(wdn2.time),
            pump_ids=[str(wdn2.raw.link_name_id[i])
                      for i in wdn2.raw.link_pump_index])
    return rows, pwl_store


def render_benchmark(st) -> None:
    """The ⚔ benchmark expander at the bottom of the Water tab."""
    with st.expander("⚔ Benchmark vs notable methods — PWL-MILP "
                     "(Oikonomou & Parvania, IEEE TSG 2019)"):
        st.caption(
            "The comparison method piecewise-linearizes every pipe/pump curve "
            "up front into ONE MILP: each pipe gets **K breakpoints** whose "
            "segment adjacency needs **binaries** — (pipes+pumps)·(K−1)·T of "
            "them, vs this study's pump-on/off-only. Both run here on the same "
            "network, data, objective and solver (HiGHS); both are judged by "
            "the **EPANET-replayed cost**, replay error, and **model size**. "
            "After a run, the PWL schedule can be 📡-transmitted to the DSO "
            "alongside ours for a fair OPF comparison. Offline study: "
            "`docs/benchmark_pwl.md`.")
        nets = list(BENCH_NETS)
        cur = st.session_state.get("w_net")
        c = st.columns([1.4, 1, 1.2, 1])
        net = c[0].selectbox("Network", nets,
                             index=nets.index(cur) if cur in nets else 1,
                             format_func=lambda n: BENCH_NETS[n], key="bm_net")
        K = c[1].selectbox("K (breakpoints)", [5, 9, 17], index=1, key="bm_k",
                           help="Breakpoints per pipe/pump curve in the PWL "
                                "method. More K = more accurate AND more "
                                "binaries — that trade-off is the comparison.")
        budget = c[2].slider("PWL solver budget (s)", 30, 300, 120, 30,
                             key="bm_budget")
        price = 1 if st.session_state.get("w_price", 1) == 1 else 0
        if net in (36, 126):
            st.warning("On this network the PWL MILP typically returns **no "
                       "incumbent** within the budget (that is the finding — "
                       "the SLA side still completes). Expect the full budget "
                       "to be consumed.")
        if c[3].button("Run benchmark", key="bm_run"):
            from .capture import capture_fds
            with st.spinner(f"Solving both methods on {BENCH_NETS[net]} "
                            f"(SLA + PWL K={K}, ≤{budget}s budget)..."):
                with capture_fds() as cap:
                    rows, pwl_store = _run_both(net, K, budget, price)
            for r in rows:
                r["net"] = BENCH_NETS[net]
            st.session_state.setdefault("bm_rows", []).append(
                {"rows": rows, "log": cap["text"][-100_000:]})
            if pwl_store is not None:
                st.session_state.setdefault("bm_pwl", {})[net] = pwl_store
                st.success(f"PWL schedule stored — run a water case on "
                           f"{BENCH_NETS[net]} and 📡-transmit it to the DSO "
                           f"to benchmark the OPF side too.")

        for i, entry in enumerate(st.session_state.get("bm_rows") or []):
            rows = entry["rows"]
            st.markdown(f"**Run {i + 1} — {rows[0].get('net', '')}**")
            st.dataframe(pd.DataFrame(rows).drop(columns=["net"],
                                                 errors="ignore"),
                         use_container_width=True, hide_index=True)
            if len(rows) == 2 and rows[1].get("binary vars"):
                b0, b1 = rows[0]["binary vars"], rows[1]["binary vars"]
                saved = 100.0 * (b1 - b0) / b1 if b1 else 0.0
                msg = f"SLA uses **{saved:.1f}% fewer binaries** ({b0} vs {b1})"
                t0, t1 = rows[0].get("wall (s)"), rows[1].get("wall (s)")
                if t0 and t1 and t1 > t0:
                    msg += f"; **{t1 / t0:.1f}× faster**"
                c0, c1 = (rows[0].get("replayed cost ($)"),
                          rows[1].get("replayed cost ($)"))
                if c0 and c1:
                    d = 100.0 * (c1 - c0) / c1
                    msg += (f"; replayed cost **{d:+.1f}%** vs PWL"
                            if abs(d) >= 0.05 else "; replayed cost tie")
                st.caption(msg + ".")
        if st.session_state.get("bm_rows"):
            with st.expander("🖥 Benchmark solver log (last run)"):
                st.code(st.session_state["bm_rows"][-1]["log"] or "(empty)",
                        language="text")
