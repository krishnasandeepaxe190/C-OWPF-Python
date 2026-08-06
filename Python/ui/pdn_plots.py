"""Plotly figures for the power-distribution side (voltages, PV VAr, loss, map)."""
from __future__ import annotations

import numpy as np

from .theme import POWER, WATER, GOOD, BAD


# ---------------------------------------------------------------- tree layout
def feeder_layout(feeder: dict):
    """Layered left-to-right layout of a radial feeder from its parent array.

    Returns (x, y) over GLOBAL nodes 0..N (0 = slack) and the edge list.
    """
    N = feeder["N"]
    parent = [None] + list(feeder["parent"])       # global: parent[g] for g=1..N
    children = {g: [] for g in range(N + 1)}
    for g in range(1, N + 1):
        children[parent[g]].append(g)
    depth = {0: 0}
    order = [0]
    stack = [0]
    while stack:
        g = stack.pop()
        for c in children[g]:
            depth[c] = depth[g] + 1
            order.append(c)
            stack.append(c)
    # y by post-order leaf counter so subtrees don't overlap
    y = {}
    counter = [0]

    def assign(g):
        if not children[g]:
            y[g] = counter[0]
            counter[0] += 1
        else:
            for c in children[g]:
                assign(c)
            y[g] = float(np.mean([y[c] for c in children[g]]))

    assign(0)
    x = np.array([depth[g] for g in range(N + 1)], float)
    yy = np.array([y[g] for g in range(N + 1)], float)
    edges = [(parent[g], g) for g in range(1, N + 1)]
    return x, yy, edges


def feeder_map(feeder: dict, v_nl: np.ndarray, hour: int, pv_buses, pump_buses,
               vmin=0.95, vmax=1.05):
    """One-line feeder diagram, buses colored by nonlinear voltage at ``hour``."""
    import plotly.graph_objects as go

    x, y, edges = feeder_layout(feeder)
    N = feeder["N"]
    volt = np.concatenate(([1.0], v_nl[:, hour]))          # slack + non-slack
    orig = [feeder["slack_id"]] + list(feeder["orig_id"])

    ex, ey = [], []
    for a, b in edges:
        ex += [x[a], x[b], None]; ey += [y[a], y[b], None]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines",
                             line=dict(color="rgba(150,150,160,.5)", width=1.5),
                             hoverinfo="skip", showlegend=False))

    pv_set = set(int(b) + 1 for b in np.asarray(pv_buses, int))     # global
    pu_set = set(int(b) + 1 for b in np.asarray(pump_buses, int))
    sym = []
    for g in range(N + 1):
        if g == 0:
            sym.append("star")
        elif g in pu_set:
            sym.append("square")
        elif g in pv_set:
            sym.append("diamond")
        else:
            sym.append("circle")
    txt = []
    for g in range(N + 1):
        role = ("slack" if g == 0 else "pump-bus" if g in pu_set
                else "PV-bus" if g in pv_set else "bus")
        txt.append(f"{role} {orig[g]}<br>|V| = {volt[g]:.4f} pu")
    fig.add_trace(go.Scatter(
        x=x, y=y, mode="markers",
        marker=dict(size=[16 if g == 0 else 12 for g in range(N + 1)],
                    symbol=sym, color=volt, colorscale="RdYlGn",
                    cmin=vmin - 0.03, cmax=min(vmax + 0.03, volt.max() + 1e-6),
                    line=dict(width=1, color="#333"),
                    colorbar=dict(title=dict(text="|V| pu", side="right"),
                                  thickness=12, len=0.8)),
        text=txt, hoverinfo="text", showlegend=False))
    fig.update_layout(
        height=460, margin=dict(l=8, r=60, t=30, b=8),
        title=f"Feeder voltage map — hour {hour} (★ slack · ◇ PV · ▪ pump)",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


# ---------------------------------------------------------------- voltage plots
def voltage_profile(v_lin: np.ndarray, v_nl: np.ndarray, hour: int,
                    vmin=0.95, vmax=1.05, orig_id=None):
    """Per-bus voltage at one hour: linear (LinDistFlow) vs nonlinear (Z-bus)."""
    import plotly.graph_objects as go

    N = v_nl.shape[0]
    idx = np.arange(1, N + 1)
    labels = [str(o) for o in (orig_id or idx)]
    fig = go.Figure()
    fig.add_hrect(y0=vmin, y1=vmax, fillcolor="rgba(46,139,87,.08)",
                  line_width=0, annotation_text="ANSI band", annotation_position="top left")
    fig.add_hline(y=vmin, line=dict(color=BAD, dash="dash", width=1))
    fig.add_hline(y=vmax, line=dict(color=BAD, dash="dash", width=1))
    fig.add_trace(go.Scatter(x=idx, y=v_lin[:, hour], mode="lines+markers",
                             name="LinDistFlow (opt)", line=dict(color=WATER, width=2),
                             text=labels, hovertemplate="bus %{text}<br>%{y:.4f} pu"))
    fig.add_trace(go.Scatter(x=idx, y=v_nl[:, hour], mode="lines+markers",
                             name="Z-bus (true)", line=dict(color=POWER, width=2, dash="dot"),
                             text=labels, hovertemplate="bus %{text}<br>%{y:.4f} pu"))
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=36, b=30),
                      title=f"Voltage profile — hour {hour}",
                      xaxis_title="bus", yaxis_title="|V| (pu)",
                      legend=dict(orientation="h", y=1.12, x=0),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def voltage_heatmap(v_nl: np.ndarray, vmin=0.95, vmax=1.05, title="Nonlinear voltage |V| (pu)"):
    """Bus x hour heatmap of the true (Z-bus) voltage magnitude."""
    import plotly.graph_objects as go

    N, T = v_nl.shape
    fig = go.Figure(go.Heatmap(
        z=v_nl, x=[f"h{t}" for t in range(T)], y=[f"{i+1}" for i in range(N)],
        colorscale="RdYlGn", zmid=1.0, zmin=min(vmin - 0.05, v_nl.min()),
        zmax=max(vmax + 0.02, v_nl.max()),
        colorbar=dict(title="|V| pu", thickness=12)))
    fig.update_layout(height=max(320, 12 * N + 80), margin=dict(l=10, r=10, t=36, b=30),
                      title=title, xaxis_title="hour", yaxis_title="bus",
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def pv_reactive_chart(q_pv: np.ndarray, p_pv: np.ndarray, pv_buses, orig_id=None):
    """PV reactive setpoints (stacked) and total active, over the day."""
    import plotly.graph_objects as go

    if q_pv.size == 0:
        return None
    T = q_pv.shape[1]
    hours = list(range(T))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=hours, y=q_pv.sum(axis=0), name="Σ PV reactive (pu)",
                         marker_color=POWER, opacity=0.85))
    fig.add_trace(go.Scatter(x=hours, y=p_pv.sum(axis=0), name="Σ PV active (pu)",
                             mode="lines+markers", line=dict(color=GOOD, width=2)))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=36, b=30),
                      title="PV dispatch (feeder total)", barmode="relative",
                      xaxis_title="hour", yaxis_title="power (pu)",
                      legend=dict(orientation="h", y=1.14, x=0),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def loss_chart(loss_base_kw: np.ndarray, loss_opt_kw: np.ndarray):
    """True network loss per hour: no-reactive baseline vs optimized setpoints."""
    import plotly.graph_objects as go

    T = len(loss_base_kw)
    hours = list(range(T))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=hours, y=loss_base_kw, name="loss — no VAr support",
                         marker_color="rgba(150,150,160,.6)"))
    fig.add_trace(go.Bar(x=hours, y=loss_opt_kw, name="loss — optimized VAr",
                         marker_color=POWER))
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=36, b=30),
                      title="True network loss (Z-bus) — reactive support reduces it",
                      barmode="group", xaxis_title="hour", yaxis_title="loss (kW)",
                      legend=dict(orientation="h", y=1.14, x=0),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig
