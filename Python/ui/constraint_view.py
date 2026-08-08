"""Teaching mode: per-iteration constraint-satisfaction viewer.

Given the recorded successive-linearization iterates (``OWFResult.iterates``,
enabled by ``SolverConfig.record_iterates``), compute -- for every constraint
family of the OWF problem -- the equality residual or inequality margin at each
iteration, so students can *watch* the linearized physics tighten as the
algorithm converges. Everything is recomputed post-hoc from the snapshots; the
solver itself is untouched.

Families (paper notation):
  equalities  : mass balance, pipe head loss (linearized), tank dynamics,
                reservoir head, pump power
  inequalities: pump big-M head gain (active when on), pump flow gating,
                junction head bounds, tank head bounds, terminal tank level
"""
from __future__ import annotations

import numpy as np


# (name, type, latex) -- latex strings match the Guide / paper notation
FAMILIES = [
    ("mass balance", "=", r"\Pi_r\,Q_t = -d_t"),
    ("pipe head loss (lin.)", "=", r"\tilde{\Pi}\,H_t = \kappa_t + \Pi'\,Q_t"),
    ("tank dynamics", "=", r"H^{tk}_{t} = H^{tk}_{0} + \tfrac{\delta}{A}\textstyle\sum_{s\le t}q^{tk}_s"),
    ("reservoir head", "=", r"\Theta H_t = \Theta H^{min}"),
    ("pump power (lin.)", "=", r"P^{pump}_t = A'\,(\Lambda Q_t) + B' z_t\,(+\,D'\,\zeta_t)"),
    ("pump head big-M", "≤", r"|\Lambda\Pi^{T}H_t - (C^{1M}(\omega_t) + C^{2M}\Lambda Q_t)| \le M(1-z_t)"),
    ("pump flow gating", "≤", r"0 \le \Lambda Q_t \le q^{max} z_t"),
    ("junction head bounds", "≤", r"K H^{min} \le K H_t \le K H^{max}"),
    ("tank head bounds", "≤", r"H^{tk,min} \le \mathrm{T}^{T} H_t \le H^{tk,max}"),
    ("terminal tank level", "≤", r"H^{tk}_{T} \ge \bar{H}^{tk}"),
]


def iteration_report(wdn, snap: dict) -> list[dict]:
    """One row per constraint family: worst residual (=) or worst violation (<=)."""
    M = wdn.M
    T = wdn.time
    H, Q, z = snap["heads"][:, :T], snap["flows"][:, :T], snap["onoff"][:, :T]
    speed = snap.get("speed")
    rows = []

    def eq(name, latex, resid):
        r = float(np.max(np.abs(resid))) if np.size(resid) else 0.0
        rows.append(dict(family=name, type="=", latex=latex, worst=r,
                         ok=bool(r <= 1e-3 * max(1.0, np.abs(H).max()))))

    def ineq(name, latex, violation):
        v = float(np.max(violation)) if np.size(violation) else 0.0
        v = max(v, 0.0)
        rows.append(dict(family=name, type="≤", latex=latex, worst=v,
                         ok=bool(v <= 1e-2)))

    # --- equalities -----------------------------------------------------------
    eq("mass balance", FAMILIES[0][2],
       M.Pi_reduced @ Q + wdn.junction_demand_profile[:, :T])
    eq("pipe head loss (lin.)", FAMILIES[1][2],
       M.Pi_telda @ H - (snap["Cp"][:, :T] + M.Pi_prime @ Q))
    # tank dynamics: recompute the integrator from flows
    tank = wdn.tank
    tank_inflow = (M.Tau.T @ (-M.Pi)) @ Q                       # (Tk x T)
    Hdummy = (tank.init_head[:, None]
              + tank.del_tk_tanks[:, None] * np.cumsum(tank_inflow, axis=1))
    tank_heads = M.Tau.T @ H                                    # (Tk x T)
    expected = np.hstack([tank.init_head[:, None], Hdummy[:, :T - 1]])
    eq("tank dynamics", FAMILIES[2][2], tank_heads - expected)
    eq("reservoir head", FAMILIES[3][2],
       M.Theta @ H - M.Theta @ wdn.bounds.min_nodal_heads[:, :T])
    # pump power vs the linearization used in that iterate
    pf = M.Lambda @ Q
    ppred = None
    if "ppump" in snap:
        C1M, C2M = snap["C1M"][:, :T], snap["C2M"][:, :T]
        # reconstruct A'/B' is iteration-specific; compare model power to the
        # *nonlinear* power instead -- the honest gap students should see
        h0 = wdn.pump.h0[:, None]
        r_ = wdn.pump.r_m[:, None]
        v_ = wdn.pump.v_m[:, None]
        om = speed[:, :T] if speed is not None else 1.0
        p_true = wdn.pump.c_m * (h0 * np.asarray(om) ** 2
                                 - r_ * np.abs(pf) ** v_) * pf
        eq("pump power (lin.)", FAMILIES[4][2], snap["ppump"][:, :T] - p_true)

    # --- inequalities ---------------------------------------------------------
    C1M, C2M = snap["C1M"][:, :T], snap["C2M"][:, :T]
    om = speed[:, :T] if speed is not None else None
    gain = (C1M * om if om is not None else C1M) + C2M * pf
    g = (M.Lambda @ M.Pi.T) @ H - gain
    Mb = wdn.config.big_m
    ineq("pump head big-M", FAMILIES[5][2], np.abs(g) - Mb * (1.0 - z))
    qmax = wdn.pump.max_flow[:, None]
    ineq("pump flow gating", FAMILIES[6][2],
         np.maximum(pf - qmax * z, -pf))
    K = M.Kappa
    jl = K @ wdn.bounds.min_nodal_heads[:, :T] - K @ H
    jh = K @ H - K @ wdn.bounds.max_nodal_heads[:, :T]
    ineq("junction head bounds", FAMILIES[7][2], np.maximum(jl, jh))
    tl = wdn.tank.min_head[:, None] - tank_heads
    th = tank_heads - wdn.tank.max_head[:, None]
    ineq("tank head bounds", FAMILIES[8][2], np.maximum(tl, th))
    ineq("terminal tank level", FAMILIES[9][2],
         wdn.tank.mean_head - Hdummy[:, -1])
    return rows


def evolution(wdn, iterates: list[dict]) -> dict:
    """{family: [worst per iteration]} for the evolution plot."""
    out: dict[str, list] = {}
    for snap in iterates:
        for row in iteration_report(wdn, snap):
            out.setdefault(row["family"], []).append(max(row["worst"], 1e-12))
    return out


def evolution_figure(wdn, iterates: list[dict]):
    """Log-scale line plot: each family's worst residual/violation vs iteration."""
    import plotly.graph_objects as go
    ev = evolution(wdn, iterates)
    fig = go.Figure()
    for fam, series in ev.items():
        fig.add_trace(go.Scatter(x=list(range(len(series))), y=series,
                                 mode="lines+markers", name=fam))
    fig.update_layout(
        height=420, xaxis_title="successive-linearization iteration",
        yaxis_title="worst |residual| (=) or violation (≤)",
        yaxis_type="log", legend=dict(orientation="h", y=-0.25, x=0),
        margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig
