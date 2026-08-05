"""Interactive network-map view (Plotly) for the Streamlit UI.

``extract_map_data`` distills a solved case into plain numpy/dict data (no epyt
handle, safely storable in Streamlit session state); ``build_map_figure`` turns
that into a Plotly figure, optionally colored by the solution at a given hour.
"""
from __future__ import annotations

import numpy as np


def extract_map_data(wdn, result=None) -> dict:
    """Collect everything the map needs from a WDN (+ optional OWFResult)."""
    raw = wdn.raw
    x, y = raw.node_x.copy(), raw.node_y.copy()
    # Nodes without coordinates land at (0,0); nudge them to the layout edge so
    # they don't draw long fake edges through the middle of the map.
    missing = (x == 0) & (y == 0)
    if missing.any() and (~missing).any():
        x[missing] = x[~missing].min() - 0.05 * (np.ptp(x[~missing]) or 1.0)
        y[missing] = y[~missing].min()

    node_kind = np.array(["junction"] * raw.n_nodes, dtype=object)
    node_kind[raw.tank_index] = "tank"
    node_kind[raw.reservoir_index] = "reservoir"

    link_kind = np.array(["pipe"] * raw.n_links, dtype=object)
    link_kind[raw.link_pump_index] = "pump"
    if wdn.M.bypass_index.size:
        link_kind[wdn.M.bypass_index] = "bypass"
    if len(raw.closed_pipe_index):
        link_kind[raw.closed_pipe_index] = "closed"

    data = {
        "x": x, "y": y,
        "node_id": list(raw.node_name_id),
        "node_kind": node_kind,
        "elev": raw.node_elevations,
        "from_node": raw.from_node, "to_node": raw.to_node,
        "link_id": list(raw.link_name_id),
        "link_kind": link_kind,
        "time": wdn.time,
    }
    if result is not None and result.flows is not None:
        data["heads"] = np.asarray(result.heads)          # (N x T)
        data["flows"] = np.asarray(result.flows)          # (L x T)
        data["pressure"] = data["heads"] - raw.node_elevations[:, None]
    return data


_NODE_STYLE = {
    "junction": dict(symbol="circle", color="#4C78A8", size=9),
    "tank": dict(symbol="square", color="#2CA02C", size=13),
    "reservoir": dict(symbol="diamond", color="#9467BD", size=14),
}
_LINK_STYLE = {
    "pipe": dict(color="#9AA0A6", dash="solid", width=2),
    "pump": dict(color="#D62728", dash="solid", width=4),
    "bypass": dict(color="#FF7F0E", dash="dash", width=3),
    "closed": dict(color="#444444", dash="dot", width=2),
}


def build_map_figure(data: dict, hour: int | None = None):
    """Plotly figure of the network; colored by solution state at ``hour``."""
    import plotly.graph_objects as go

    x, y = data["x"], data["y"]
    fig = go.Figure()
    have_sol = hour is not None and "flows" in data
    if have_sol:
        hour = int(np.clip(hour, 0, data["time"] - 1))
        qmax = max(float(np.abs(data["flows"]).max()), 1.0)

    # --- links, grouped by kind so each gets one legend entry ---
    for kind, style in _LINK_STYLE.items():
        idx = np.where(data["link_kind"] == kind)[0]
        if idx.size == 0:
            continue
        xs, ys, hover_x, hover_y, hover_t = [], [], [], [], []
        for l in idx:
            a, b = data["from_node"][l], data["to_node"][l]
            xs += [x[a], x[b], None]
            ys += [y[a], y[b], None]
            hover_x.append((x[a] + x[b]) / 2)
            hover_y.append((y[a] + y[b]) / 2)
            label = f"{kind} {data['link_id'][l]}"
            if have_sol:
                label += f"<br>flow: {data['flows'][l, hour]:,.0f} GPM"
            hover_t.append(label)
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", name=kind,
            line=dict(color=style["color"], dash=style["dash"], width=style["width"]),
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(  # invisible midpoints carrying link hovers
            x=hover_x, y=hover_y, mode="markers",
            marker=dict(size=14, color="rgba(0,0,0,0)"),
            hovertext=hover_t, hoverinfo="text", showlegend=False,
        ))

    # --- nodes ---
    for kind, style in _NODE_STYLE.items():
        idx = np.where(data["node_kind"] == kind)[0]
        if idx.size == 0:
            continue
        text = []
        for n in idx:
            t = f"{kind} {data['node_id'][n]}<br>elev: {data['elev'][n]:.0f} ft"
            if have_sol:
                t += (f"<br>head: {data['heads'][n, hour]:.1f} ft"
                      f"<br>pressure: {data['pressure'][n, hour]:.1f} ft")
            text.append(t)
        marker = dict(symbol=style["symbol"], size=style["size"],
                      line=dict(width=1, color="#222"))
        if have_sol and kind == "junction":
            marker.update(color=data["pressure"][idx, hour],
                          colorscale="RdYlGn", cmin=0.0,
                          cmax=max(float(data["pressure"].max()), 1.0),
                          colorbar=dict(title="pressure (ft)", thickness=12))
        else:
            marker["color"] = style["color"]
        fig.add_trace(go.Scatter(
            x=x[idx], y=y[idx], mode="markers", name=kind,
            marker=marker, hovertext=text, hoverinfo="text",
        ))

    title = "Network map" if not have_sol else f"Network state at hour {hour}"
    fig.update_layout(
        title=title, showlegend=True, height=560,
        margin=dict(l=10, r=10, t=40, b=10),
        xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.02),
    )
    return fig


def build_animated_map_figure(data: dict):
    """Animated map: a Play button steps through the hours, drawing each link's
    flow as a direction arrow (angle = flow direction, size = |flow|) over the
    junction-pressure field. Falls back to the static map if no solution."""
    import plotly.graph_objects as go

    if "flows" not in data:
        return build_map_figure(data)

    x, y = data["x"], data["y"]
    T = data["time"]
    flows = data["flows"]                      # (L x T)
    pressure = data["pressure"]                # (N x T)
    from_n, to_n = data["from_node"], data["to_node"]
    qmax = max(float(np.abs(flows).max()), 1.0)
    pmax = max(float(pressure[data["node_kind"] == "junction"].max()), 1.0)

    # per-link geometry: midpoint + compass angle of the from->to direction
    ax_, ay_ = x[from_n], y[from_n]
    bx_, by_ = x[to_n], y[to_n]
    midx, midy = (ax_ + bx_) / 2, (ay_ + by_) / 2
    base_ang = np.degrees(np.arctan2(bx_ - ax_, by_ - ay_))   # 0 = north, CW

    # static link lines (drawn once, behind the arrows)
    lx, ly = [], []
    for l in range(len(from_n)):
        lx += [x[from_n[l]], x[to_n[l]], None]
        ly += [y[from_n[l]], y[to_n[l]], None]
    link_trace = go.Scatter(x=lx, y=ly, mode="lines",
                            line=dict(color="#9AA0A6", width=1.2),
                            hoverinfo="skip", showlegend=False)

    def arrows(t):
        f = flows[:, t]
        ang = np.where(f >= 0, base_ang, base_ang + 180.0)     # flip when reversed
        size = 9 + 16 * np.abs(f) / qmax
        text = [f"{data['link_kind'][l]} {data['link_id'][l]}<br>"
                f"{f[l]:,.0f} GPM ({'→' if f[l] >= 0 else '←'})"
                for l in range(len(f))]
        return go.Scatter(
            x=midx, y=midy, mode="markers", name="flow",
            marker=dict(symbol="arrow", angle=ang, size=size,
                        angleref="up", color="#1f77b4",
                        line=dict(width=0.5, color="#08306b")),
            hovertext=text, hoverinfo="text", showlegend=False)

    ji = np.where(data["node_kind"] == "junction")[0]

    def junctions(t):
        text = [f"junction {data['node_id'][n]}<br>head {data['heads'][n, t]:.1f} ft"
                f"<br>pressure {pressure[n, t]:.1f} ft" for n in ji]
        return go.Scatter(
            x=x[ji], y=y[ji], mode="markers", name="junction",
            marker=dict(size=9, color=pressure[ji, t], colorscale="RdYlGn",
                        cmin=0.0, cmax=pmax, line=dict(width=1, color="#222"),
                        colorbar=dict(title="pressure (ft)", thickness=12)),
            hovertext=text, hoverinfo="text", showlegend=False)

    # static tank / reservoir markers
    extra = []
    for kind in ("tank", "reservoir"):
        idx = np.where(data["node_kind"] == kind)[0]
        if idx.size:
            st = _NODE_STYLE[kind]
            extra.append(go.Scatter(
                x=x[idx], y=y[idx], mode="markers", name=kind,
                marker=dict(symbol=st["symbol"], size=st["size"], color=st["color"],
                            line=dict(width=1, color="#222")),
                hovertext=[f"{kind} {data['node_id'][n]}" for n in idx],
                hoverinfo="text"))

    frames = [go.Frame(name=str(t), data=[arrows(t), junctions(t)], traces=[1, 2])
              for t in range(T)]

    fig = go.Figure(
        data=[link_trace, arrows(0), junctions(0), *extra],
        frames=frames,
    )
    play = dict(label="▶ Play", method="animate",
                args=[None, dict(frame=dict(duration=650, redraw=True),
                                 fromcurrent=True, transition=dict(duration=0))])
    pause = dict(label="⏸ Pause", method="animate",
                 args=[[None], dict(frame=dict(duration=0, redraw=False),
                                    mode="immediate")])
    steps = [dict(method="animate", label=str(t),
                  args=[[str(t)], dict(mode="immediate",
                                       frame=dict(duration=0, redraw=True))])
             for t in range(T)]
    fig.update_layout(
        title="Flow animation — arrow direction = flow direction, size = |flow|",
        height=600, margin=dict(l=10, r=10, t=70, b=10),
        xaxis=dict(visible=False), yaxis=dict(visible=False, scaleanchor="x"),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        updatemenus=[dict(type="buttons", direction="left", showactive=False,
                          x=0.0, y=1.08, xanchor="left", yanchor="top",
                          buttons=[play, pause])],
        sliders=[dict(active=0, x=0.12, y=0, len=0.85,
                      currentvalue=dict(prefix="hour "), steps=steps)],
    )
    return fig
