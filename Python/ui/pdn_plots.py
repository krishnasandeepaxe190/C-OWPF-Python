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


def coupled_network_map(water_md: dict, feeder: dict, pump_bus, pump_ids,
                        pv_buses=None, valve_links=None,
                        show_node_labels: bool = True,
                        show_bus_labels: bool = True,
                        show_coupling_labels: bool = True):
    """Side-by-side depiction of the interconnected energy systems (paper Fig. 1
    style): the WDN on the left, the feeder on the right, and dashed pump->bus
    coupling lines (the Xi matrix) between them."""
    import plotly.graph_objects as go

    fig = go.Figure()

    # ---- water side: normalize coordinates into x in [0, 0.40] ----------------
    wx = np.asarray(water_md["x"], float).copy()
    wy = np.asarray(water_md["y"], float).copy()

    def _norm(v):
        rng = float(v.max() - v.min())
        return (v - v.min()) / (rng if rng > 1e-9 else 1.0)

    wxn = 0.02 + 0.38 * _norm(wx)
    wyn = 0.08 + 0.84 * _norm(wy)
    frm = np.asarray(water_md["from_node"], int)
    to = np.asarray(water_md["to_node"], int)
    lkind = water_md["link_kind"]
    lids = water_md["link_id"]
    valve_links = set(map(str, valve_links or []))

    for i in range(len(frm)):
        kind = ("valve" if str(lids[i]) in valve_links else str(lkind[i]))
        color = {"pipe": "#8a8f99", "pump": "#d62728", "bypass": "#1f77b4",
                 "closed": "#c0c0c0", "valve": "#9467bd"}.get(kind, "#8a8f99")
        width = 4 if kind in ("pump", "valve") else 1.6
        dash = "dot" if kind == "closed" else None
        fig.add_trace(go.Scatter(
            x=[wxn[frm[i]], wxn[to[i]]], y=[wyn[frm[i]], wyn[to[i]]],
            mode="lines", line=dict(color=color, width=width, dash=dash),
            hoverinfo="text", text=f"{kind} {lids[i]}", showlegend=False))

    nkind = water_md["node_kind"]
    sym = {"junction": "circle", "tank": "square", "reservoir": "diamond"}
    colr = {"junction": "#1f77b4", "tank": "#2ca02c", "reservoir": "#17becf"}
    for kind in ("junction", "tank", "reservoir"):
        idx = [i for i in range(len(wxn)) if str(nkind[i]) == kind]
        if not idx:
            continue
        fig.add_trace(go.Scatter(
            x=wxn[idx], y=wyn[idx],
            mode="markers+text" if show_node_labels else "markers",
            marker=dict(symbol=sym[kind], size=13 if kind != "junction" else 10,
                        color=colr[kind], line=dict(color="white", width=1)),
            text=[water_md["node_id"][i] for i in idx], textposition="top center",
            textfont=dict(size=9), name=f"WDN {kind}"))

    # ---- power side: layered feeder into x in [0.60, 1.0] ---------------------
    fx, fy, edges = feeder_layout(feeder)
    fxn = 0.60 + 0.38 * _norm(np.asarray(fx, float))
    fyn = 0.08 + 0.84 * _norm(np.asarray(fy, float))
    for a, b in edges:
        fig.add_trace(go.Scatter(
            x=[fxn[a], fxn[b]], y=[fyn[a], fyn[b]], mode="lines",
            line=dict(color="#b9781a", width=1.6), hoverinfo="skip",
            showlegend=False))
    orig = feeder.get("orig_id")
    lbl = [("slack" if g == 0 else str(orig[g - 1]) if orig is not None else str(g))
           for g in range(len(fxn))]
    fig.add_trace(go.Scatter(
        x=fxn, y=fyn,
        mode="markers+text" if show_bus_labels else "markers",
        marker=dict(symbol="square", size=9, color="#b9781a",
                    line=dict(color="white", width=1)),
        text=lbl, textposition="bottom center", textfont=dict(size=9),
        name="PDN bus"))
    fig.add_trace(go.Scatter(
        x=[fxn[0]], y=[fyn[0]], mode="markers",
        marker=dict(symbol="star-square", size=16, color="#ff7f0e",
                    line=dict(color="white", width=1)),
        name="substation (slack)"))
    if pv_buses is not None and len(pv_buses):
        g = [int(b) + 1 for b in pv_buses]
        fig.add_trace(go.Scatter(
            x=fxn[g], y=fyn[g], mode="markers",
            marker=dict(symbol="star", size=13, color="#e6c229",
                        line=dict(color="#8a6d00", width=1)),
            name="PV inverter"))

    # ---- coupling: dashed pump -> bus lines (the Xi matrix) -------------------
    pump_links = [i for i in range(len(frm)) if str(lkind[i]) == "pump"]
    for p, b in enumerate(np.asarray(pump_bus, int)):
        if p >= len(pump_links):
            break
        i = pump_links[p]
        mx, my = (wxn[frm[i]] + wxn[to[i]]) / 2, (wyn[frm[i]] + wyn[to[i]]) / 2
        g = int(b) + 1
        fig.add_trace(go.Scatter(
            x=[mx, fxn[g]], y=[my, fyn[g]], mode="lines",
            line=dict(color="#d62728", width=2.2, dash="dash"),
            name="pump load coupling (Xi)" if p == 0 else None,
            showlegend=(p == 0)))
        pid = pump_ids[p] if pump_ids and p < len(pump_ids) else str(p)
        bus_lbl = (str(orig[int(b)]) if orig is not None else str(g))
        if show_coupling_labels:
            fig.add_annotation(x=(mx + fxn[g]) / 2, y=(my + fyn[g]) / 2 + 0.03,
                           text=f"pump {pid} → bus {bus_lbl}", showarrow=False,
                           font=dict(size=10, color="#d62728"))

    fig.add_annotation(x=0.20, y=1.03, text="<b>Water distribution network</b>",
                       showarrow=False, xref="x", yref="paper", font=dict(size=13))
    fig.add_annotation(x=0.80, y=1.03, text="<b>Power distribution feeder</b>",
                       showarrow=False, xref="x", yref="paper", font=dict(size=13))
    fig.update_layout(
        height=520, margin=dict(l=10, r=10, t=48, b=10),
        xaxis=dict(visible=False, range=[-0.02, 1.05]),
        yaxis=dict(visible=False, range=[0, 1.1]),
        legend=dict(orientation="h", y=-0.04, x=0),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def pv_capacity_animation(p_pv: np.ndarray, q_pv: np.ndarray, rating: np.ndarray,
                          labels: list):
    """Animated per-hour inverter loading: ||(p,q)|| / S_max per PV site.

    One bar per site; ▶ Play steps through the hours. The 100% line is the
    inverter apparent-power rating S -- the capability circle the OPF respects.
    """
    import plotly.graph_objects as go

    T = p_pv.shape[1]
    S = np.maximum(np.asarray(rating, float), 1e-9)[:, None]        # (npv,1)
    util = 100.0 * np.sqrt(p_pv ** 2 + q_pv ** 2) / S               # (npv,T) %
    qshare = 100.0 * np.abs(q_pv) / S
    pshare = 100.0 * p_pv / S

    def bars(t):
        return [
            go.Bar(x=labels, y=pshare[:, t], name="active |p|/S",
                   marker_color="#e6c229"),
            go.Bar(x=labels, y=qshare[:, t], name="reactive |q|/S",
                   marker_color="#1f77b4"),
            go.Scatter(x=labels, y=util[:, t], mode="markers+text",
                       text=[f"{u:.0f}%" for u in util[:, t]],
                       textposition="top center", textfont=dict(size=10),
                       marker=dict(symbol="diamond", size=10, color="#d62728"),
                       name="total ‖(p,q)‖/S"),
        ]

    fig = go.Figure(data=bars(0),
                    frames=[go.Frame(data=bars(t), name=str(t)) for t in range(T)])
    fig.add_hline(y=100.0, line_dash="dash", line_color="#d62728",
                  annotation_text="S_max (100%)")
    fig.update_layout(
        height=440, barmode="group",
        yaxis=dict(title="% of inverter rating S", range=[0, 130]),
        legend=dict(orientation="h", y=1.12, x=0),
        margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        updatemenus=[dict(type="buttons", showactive=False, y=1.18, x=1.0,
                          xanchor="right",
                          buttons=[dict(label="▶ Play", method="animate",
                                        args=[None, dict(frame=dict(duration=550,
                                                                    redraw=True),
                                                         fromcurrent=True)]),
                                   dict(label="⏸", method="animate",
                                        args=[[None], dict(frame=dict(duration=0),
                                                           mode="immediate")])])],
        sliders=[dict(steps=[dict(args=[[str(t)], dict(mode="immediate",
                                                       frame=dict(duration=0,
                                                                  redraw=True))],
                                  label=f"h{t}", method="animate")
                             for t in range(T)],
                      currentvalue=dict(prefix="hour: "), y=-0.07)])
    return fig
