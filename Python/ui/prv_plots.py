"""Paper-style PRV figures (Plotly): valve state timeline, downstream head vs the
setpoint h_set, valve head loss R_prv, and valve flow -- model vs EPANET where
available. Mirrors the manuscript's PRV accuracy/status figures."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def prv_states(prv: dict) -> np.ndarray:
    """(Nv x T) state code: 0 = closed, 1 = open, 2 = active."""
    return (2.0 * prv["x_act"] + 1.0 * prv["x_open"]).astype(float)


def prv_panel(wdn, heads: np.ndarray, flows: np.ndarray, prv: dict,
              heads_ep: np.ndarray = None, flows_ep: np.ndarray = None,
              heads_rule: np.ndarray = None, flows_rule: np.ndarray = None,
              valve_pos: int = 0):
    """2x2 PRV panel for one valve (paper Fig. 4-style, plus the state timeline).

    ``heads_ep``/``flows_ep``: EPANET replay of the OPTIMIZED schedule (validation).
    ``heads_rule``/``flows_rule``: EPANET running its OWN rule-based controls -- a
    DIFFERENT schedule, overlaid so the pressure regulation the rules achieve can
    be compared with the optimized regulation (and cost differences explained:
    the rules hold the .inp setting, the optimizer holds the user's h_set).
    """
    T = wdn.time
    hrs = list(range(T))
    lk = int(wdn.M.valve_index[valve_pos])
    hset = float(wdn.valve_hset[valve_pos])
    down_row = wdn.M.valve_down_sel[valve_pos]
    hdn = down_row @ heads[:, :T]
    fval = (wdn.M.Pi_prime_valve @ flows[:, :T])[valve_pos]
    states = prv_states(prv)[valve_pos, :T]
    R = prv["R_prv"][valve_pos, :T]

    fig = make_subplots(
        rows=2, cols=2, vertical_spacing=0.16, horizontal_spacing=0.1,
        subplot_titles=("PRV state (0 closed · 1 open · 2 active)",
                        "Downstream head vs setpoint h_set",
                        "Valve head loss R_PRV (ft)", "Valve flow (GPM)"))
    fig.add_trace(go.Scatter(x=hrs, y=states, mode="lines+markers",
                             line=dict(shape="hv", width=2), name="state",
                             showlegend=False), row=1, col=1)
    fig.update_yaxes(tickvals=[0, 1, 2], ticktext=["closed", "open", "active"],
                     range=[-0.3, 2.3], row=1, col=1)

    fig.add_trace(go.Scatter(x=hrs, y=hdn, mode="lines+markers",
                             name="optimized (model)"), row=1, col=2)
    if heads_ep is not None:
        fig.add_trace(go.Scatter(x=hrs, y=down_row @ heads_ep[:, :T], mode="markers",
                                 marker_symbol="x", name="EPANET replay"), row=1, col=2)
    if heads_rule is not None:
        fig.add_trace(go.Scatter(x=hrs, y=down_row @ heads_rule[:, :T],
                                 mode="lines", line=dict(dash="dot", width=2),
                                 name="EPANET rules"), row=1, col=2)
    fig.add_hline(y=hset, line_dash="dash", line_color="#d62728",
                  annotation_text=f"h_set = {hset:.1f} ft", row=1, col=2)

    fig.add_trace(go.Scatter(x=hrs, y=R, mode="lines+markers", name="R_PRV",
                             showlegend=False), row=2, col=1)

    fig.add_trace(go.Scatter(x=hrs, y=fval, mode="lines+markers",
                             name="valve flow (optimized)", showlegend=False),
                  row=2, col=2)
    if flows_ep is not None:
        fig.add_trace(go.Scatter(
            x=hrs, y=(wdn.M.Pi_prime_valve @ flows_ep[:, :T])[valve_pos],
            mode="markers", marker_symbol="x", name="EPANET replay",
            showlegend=False), row=2, col=2)
    if flows_rule is not None:
        fig.add_trace(go.Scatter(
            x=hrs, y=(wdn.M.Pi_prime_valve @ flows_rule[:, :T])[valve_pos],
            mode="lines", line=dict(dash="dot", width=2), name="EPANET rules",
            showlegend=False), row=2, col=2)

    vid = str(wdn.raw.link_name_id[lk])
    up = str(wdn.raw.node_name_id[wdn.raw.from_node[lk]])
    dn = str(wdn.raw.node_name_id[wdn.raw.to_node[lk]])
    fig.update_layout(height=560, margin=dict(l=40, r=20, t=60, b=40),
                      title=f"PRV {vid}  ({up} → {dn})",
                      legend=dict(orientation="h", y=-0.12))
    fig.update_xaxes(title_text="hour", row=2, col=1)
    fig.update_xaxes(title_text="hour", row=2, col=2)
    return fig
