"""Shared "optimized vs true pump power" check panel.

The successive-linearization optimizer decides on a *linearized* pump power
P-hat (the LP surrogate); the honest results everywhere (costs, hand-offs,
validation) use the TRUE nonlinear pump power re-evaluated at the solved
flows/speeds. This panel compares the two, summed over the horizon -- a small
gap means the linearization was tight at the solution, exactly the fidelity
claim the paper makes. Used by the Water, Power and Coupled tabs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def pump_power_check(st, ppump_linear, ppump_true, pump_ids=None,
                     key: str = "pp", label: str = "Pump power check") -> None:
    """Expander comparing Σ_t optimizer (linear) vs Σ_t true pump power."""
    if ppump_linear is None or ppump_true is None:
        return
    pl = np.abs(np.asarray(ppump_linear, float))
    pt = np.abs(np.asarray(ppump_true, float))
    T = min(pl.shape[-1], pt.shape[-1])
    pl, pt = pl[..., :T], pt[..., :T]
    tot_l, tot_t = float(pl.sum()), float(pt.sum())
    dpct = 100.0 * (tot_l - tot_t) / tot_t if tot_t > 1e-12 else 0.0
    with st.expander(f"⚡ {label} — optimizer Σ {tot_l:,.1f} kWh vs true "
                     f"Σ {tot_t:,.1f} kWh (Δ {dpct:+.3f}%)"):
        c1, c2, c3 = st.columns(3)
        c1.metric("Σ optimized (linearized)", f"{tot_l:,.1f} kWh",
                  help="Sum over pumps and hours of the LP's pump-power "
                       "variable P-hat -- the linear surrogate the optimizer "
                       "actually decided on.")
        c2.metric("Σ true (nonlinear)", f"{tot_t:,.1f} kWh",
                  help="Same schedule/flows/speeds pushed through the true "
                       "nonlinear pump power law -- what the costs use.")
        c3.metric("linearization gap", f"{dpct:+.3f}%",
                  delta="tight" if abs(dpct) < 1.0 else "loose",
                  delta_color="normal" if abs(dpct) < 1.0 else "inverse",
                  help="(Σ linear − Σ true) / Σ true. Small = the successive "
                       "linearization converged onto the true power law.")
        if pl.ndim == 2 and pl.shape[0] >= 1:
            ids = (list(pump_ids) if pump_ids and len(pump_ids) == pl.shape[0]
                   else [f"pump {i + 1}" for i in range(pl.shape[0])])
            sl, stt = pl.sum(axis=1), pt.sum(axis=1)
            dd = np.where(stt > 1e-12, 100.0 * (sl - stt) / np.maximum(stt, 1e-12), 0.0)
            df = pd.DataFrame({
                "pump": ids,
                "Σ optimized (kWh)": np.round(sl, 2),
                "Σ true (kWh)": np.round(stt, 2),
                "Δ %": [f"{v:+.3f}%" for v in dd],
            })
            st.dataframe(df, use_container_width=True, hide_index=True)
        st.caption("The optimizer works on a per-iteration **linearization** of the "
                   "pump power law; every cost and hand-off in this app is computed "
                   "from the **true** nonlinear power at the solved operating point. "
                   "A near-zero Δ% is the model-fidelity evidence: the linear "
                   "surrogate and the physics agree at the solution.")
