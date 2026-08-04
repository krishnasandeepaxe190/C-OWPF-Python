"""C-OWPF Streamlit UI (Phase 0: FSP water engine).

Run with:  streamlit run app.py
"""
from __future__ import annotations

import contextlib
import io
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
warnings.filterwarnings("ignore")

from owf import NETWORKS  # noqa: E402
from owf.netmap import build_map_figure, extract_map_data  # noqa: E402
from main_owf import (  # noqa: E402
    MODES,
    MODE_SUGGESTION,
    NET_BLURB,
    run_case,
)

st.set_page_config(page_title="C-OWPF | Optimal Water Flow", page_icon="💧",
                   layout="wide")

st.title("💧 C-OWPF — Optimal Water Flow (FSP)")
st.caption(
    "Pump scheduling on water distribution networks by successive linear "
    "approximation — CVXPY + HiGHS MILP, validated against EPANET. "
    "Water-side FSP engine; VSPs, PRVs and the power network are upcoming phases."
)

if "records" not in st.session_state:
    st.session_state.records = []       # list of dicts, one per completed run


# ---------------------------------------------------------------- helpers
def _csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv().encode()


def _build_exports(case, wdn, result) -> dict[str, bytes]:
    """CSV exports for a solved case (bytes ready for download buttons)."""
    T = wdn.time
    hours = [f"h{t}" for t in range(T)]
    link_ids = list(wdn.raw.link_name_id)
    node_ids = list(wdn.raw.node_name_id)
    out = {}
    out["flows_gpm.csv"] = _csv_bytes(
        pd.DataFrame(result.flows[:, :T], index=link_ids, columns=hours))
    out["heads_ft.csv"] = _csv_bytes(
        pd.DataFrame(result.heads[:, :T], index=node_ids, columns=hours))
    pump_ids = [link_ids[i] for i in wdn.raw.link_pump_index]
    sched = pd.DataFrame(result.onoff[:, :T], index=pump_ids, columns=hours)
    power = pd.DataFrame(result.ppump_true[:, :T], index=pump_ids, columns=hours)
    out["pump_schedule.csv"] = _csv_bytes(sched.astype(int))
    out["pump_power_kw.csv"] = _csv_bytes(power.round(3))
    return out


def _session_df() -> pd.DataFrame:
    rows = []
    for rec in st.session_state.records:
        c = rec["case"]
        rows.append({
            "run": rec["id"],
            "case": c.label,
            "EPANET cost": round(c.epanet_cost, 5),
            "C-OWPF cost": round(c.owf_cost, 5) if np.isfinite(c.owf_cost) else None,
            "saving %": round(c.savings_pct, 1) if np.isfinite(c.savings_pct) else None,
            "max |dHead| ft": round(c.max_dhead, 3) if np.isfinite(c.max_dhead) else None,
            "max |dPumpQ| GPM": round(c.max_dpumpflow, 3) if np.isfinite(c.max_dpumpflow) else None,
            "min press. ft": round(c.min_pressure, 1) if np.isfinite(c.min_pressure) else None,
            "feasible": "yes" if c.converged else "no",
            "time s": round(c.elapsed),
        })
    return pd.DataFrame(rows)


def _render_case(rec: dict) -> None:
    case = rec["case"]
    ok = case.converged and np.isfinite(case.owf_cost)
    if not ok and not np.isfinite(case.owf_cost):
        st.error(f"No feasible solution ({case.note}). "
                 "Try the recommended mode for this network.")
        return

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("EPANET cost", f"{case.epanet_cost:.4f}")
    c2.metric("C-OWPF cost", f"{case.owf_cost:.4f}",
              delta=f"{-case.savings_pct:.1f}%", delta_color="inverse")
    c3.metric("Max head error", f"{case.max_dhead:.3f} ft",
              help="EPANET replay of the optimized schedule")
    c4.metric("Min junction pressure", f"{case.min_pressure:.1f} ft",
              help="> 0 means hydraulically feasible in EPANET")
    c5.metric("Solve time", f"{case.elapsed:.0f} s", help=f"{case.n_iter} iterations")
    if case.note:
        st.caption(f"note: {case.note}")

    tab_names = ["Network map", "Schedule", "Flows", "Heads", "Convergence",
                 "Error", "Solver log", "Download"]
    tabs = st.tabs(tab_names)

    with tabs[0]:
        md = rec["map_data"]
        if "flows" in md:
            hour = st.slider("Hour", 0, md["time"] - 1, 0, key=f"hr_{rec['id']}")
            st.plotly_chart(build_map_figure(md, hour), use_container_width=True,
                            key=f"map_{rec['id']}")
        else:
            st.plotly_chart(build_map_figure(md), use_container_width=True,
                            key=f"map_{rec['id']}")

    plot_order = ["schedule", "flows", "heads", "convergence", "error"]
    by_kind = {Path(p).stem.split("_")[-1]: p for p in rec["plots"]}
    for tab, kind in zip(tabs[1:6], plot_order):
        with tab:
            p = by_kind.get(kind)
            if p and Path(p).exists():
                st.image(str(p), use_container_width=True)
            else:
                st.info("plot not available")

    with tabs[6]:
        st.code(rec["log"] or "(no output)", language="text")

    with tabs[7]:
        st.write("Per-case results as CSV:")
        cols = st.columns(len(rec["exports"]) or 1)
        for col, (name, blob) in zip(cols, rec["exports"].items()):
            col.download_button(name, data=blob, file_name=f"run{rec['id']}_{name}",
                                mime="text/csv", key=f"dl_{rec['id']}_{name}")


# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Case setup")

    net = st.selectbox("Network", options=list(NETWORKS),
                       format_func=lambda n: f"{NETWORKS[n].name}  ({NET_BLURB[n]})")
    rec_mode, alt_mode = MODE_SUGGESTION[net]

    mode = st.selectbox("Solve mode", options=list(MODES),
                        index=list(MODES).index(rec_mode),
                        help="\n\n".join(f"**{m}** — {d}" for m, d in MODES.items()))
    st.caption(MODES[mode])
    if mode == rec_mode:
        st.success(f"Recommended mode for {NETWORKS[net].name}.")
    elif mode == alt_mode:
        st.info("Worth trying: reports savings vs EPANET (slower).")
    elif mode == "direct" and net in (11, 36, 97):
        st.warning(f"Direct mode does not converge on looped networks — use '{rec_mode}'.")
    if net == 97 and mode == "optimize":
        st.warning("Net3 optimize takes ~3 min and honestly reports ~0% savings "
                   "(its tank-level controls are already near-optimal).")
    if net == 97 and mode == "warmstart":
        st.warning("The multi-start search does not scale to Net3 (50 binaries); "
                   "'epanet' is the reliable mode here.")

    price = st.radio("Electricity price", [1, 0], horizontal=True,
                     format_func=lambda p: "Time-of-use" if p == 1 else "Flat")

    run_clicked = st.button("▶  Run case", type="primary", use_container_width=True)

    st.divider()
    if st.button("Clear session history", use_container_width=True):
        st.session_state.records = []
        st.rerun()

# ---------------------------------------------------------------- run
if run_clicked:
    eta = {"direct": "a few seconds", "warmstart": "up to a minute",
           "epanet": "~30-60 s", "optimize": "1-3 minutes"}[mode]
    with st.spinner(f"Solving {NETWORKS[net].name} in {mode} mode ({eta})..."):
        log_buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(log_buf):
                case, wdn, result = run_case(net, mode, price, None,
                                             plot=True, outdir="outputs",
                                             verbose=False)
        except Exception as exc:
            st.error(f"Case failed: {exc}")
            case = None

    if case is not None:
        run_id = len(st.session_state.records) + 1
        # copy plots to run-unique names so reruns don't overwrite older cases
        plots = []
        for p in case.plots:
            p = Path(p)
            if p.exists():
                q = p.with_name(f"run{run_id}_{p.name}")
                shutil.copyfile(p, q)
                plots.append(q)
        record = {
            "id": run_id,
            "case": case,
            "log": log_buf.getvalue(),
            "plots": plots,
            "map_data": extract_map_data(wdn, result),
            "exports": (_build_exports(case, wdn, result)
                        if result is not None and result.flows is not None else {}),
        }
        st.session_state.records.append(record)

# ---------------------------------------------------------------- results
records = st.session_state.records
if records:
    latest = records[-1]
    st.subheader(f"Result — run {latest['id']}: {latest['case'].label}")
    _render_case(latest)

    # ------------------------------------------------ session comparison
    st.subheader("Session comparison")
    df = _session_df()
    st.dataframe(df, use_container_width=True, hide_index=True)
    st.download_button("Download session table (CSV)", data=_csv_bytes(df.set_index("run")),
                       file_name="owf_session_comparison.csv", mime="text/csv")
    st.caption("cost = true nonlinear energy cost; errors from the EPANET replay "
               "of each case's schedule.")

    # ------------------------------------------------ side-by-side diff
    if len(records) >= 2:
        st.subheader("Case diff")
        labels = {r["id"]: f"run {r['id']} — {r['case'].label}" for r in records}
        colA, colB = st.columns(2)
        a_id = colA.selectbox("Case A", list(labels), index=len(labels) - 2,
                              format_func=labels.get)
        b_id = colB.selectbox("Case B", list(labels), index=len(labels) - 1,
                              format_func=labels.get)
        A = next(r for r in records if r["id"] == a_id)
        B = next(r for r in records if r["id"] == b_id)
        ca, cb = A["case"], B["case"]

        def _row(name, va, vb, fmt="{:.4f}", better="lower"):
            da = fmt.format(va) if np.isfinite(va) else "—"
            db = fmt.format(vb) if np.isfinite(vb) else "—"
            delta = (vb - va) if (np.isfinite(va) and np.isfinite(vb)) else None
            return {"metric": name, labels[a_id]: da, labels[b_id]: db,
                    "Δ (B − A)": (fmt.format(delta) if delta is not None else "—")}

        diff_df = pd.DataFrame([
            _row("EPANET cost", ca.epanet_cost, cb.epanet_cost),
            _row("C-OWPF cost", ca.owf_cost, cb.owf_cost),
            _row("saving %", ca.savings_pct, cb.savings_pct, "{:.1f}"),
            _row("max |dHead| (ft)", ca.max_dhead, cb.max_dhead, "{:.3f}"),
            _row("max |dPumpQ| (GPM)", ca.max_dpumpflow, cb.max_dpumpflow, "{:.3f}"),
            _row("min pressure (ft)", ca.min_pressure, cb.min_pressure, "{:.1f}"),
            _row("solve time (s)", ca.elapsed, cb.elapsed, "{:.0f}"),
        ])
        st.dataframe(diff_df, use_container_width=True, hide_index=True)

        # pump schedules side by side
        pa = {Path(p).stem.split("_")[-1]: p for p in A["plots"]}.get("schedule")
        pb = {Path(p).stem.split("_")[-1]: p for p in B["plots"]}.get("schedule")
        img_a, img_b = st.columns(2)
        if pa and Path(pa).exists():
            img_a.image(str(pa), caption=labels[a_id], use_container_width=True)
        if pb and Path(pb).exists():
            img_b.image(str(pb), caption=labels[b_id], use_container_width=True)
else:
    st.info("Set up a case in the sidebar and press **Run case**.")
