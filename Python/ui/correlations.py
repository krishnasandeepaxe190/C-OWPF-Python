"""Correlation explorer: how the system's signals co-evolve over the day.

Teaching view for the Power and Coupled tabs. The user picks any set of signals
(price, pump power, PV active/reactive, water demand, min voltage, loss, tank
levels, ...), sees them overlaid on a normalized axis with a ▶ Play hour cursor,
gets the Pearson correlation matrix of the selection, and can draw any pair as
a scatter (colored by hour) with the fitted trend and r value.
"""
from __future__ import annotations

import numpy as np


def _norm(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    rng = float(np.nanmax(v) - np.nanmin(v))
    return (v - np.nanmin(v)) / (rng if rng > 1e-12 else 1.0)


def _overlay_fig(sigs: dict, sel: list):
    """Normalized overlay of the selected signals with a playable hour cursor."""
    import plotly.graph_objects as go

    T = len(np.asarray(sigs[sel[0]]).ravel())
    hrs = list(range(T))
    base = []
    for name in sel:
        v = np.asarray(sigs[name], float).ravel()[:T]
        base.append(go.Scatter(
            x=hrs, y=_norm(v), mode="lines", name=name,
            customdata=v.reshape(-1, 1),
            hovertemplate=name + ": %{customdata[0]:.4g}<extra></extra>"))
    dots = [go.Scatter(x=[0], y=[_norm(np.asarray(sigs[n], float).ravel()[:T])[0]],
                       mode="markers", marker=dict(size=11),
                       showlegend=False, hoverinfo="skip") for n in sel]
    fig = go.Figure(data=base + dots)
    frames = []
    for k in hrs:
        fdots = [go.Scatter(x=[k], y=[_norm(np.asarray(sigs[n], float).ravel()[:T])[k]])
                 for n in sel]
        frames.append(go.Frame(
            name=str(k), data=fdots, traces=list(range(len(base), len(base) + len(sel))),
            layout=go.Layout(shapes=[dict(type="line", x0=k, x1=k, y0=0, y1=1,
                                          yref="paper",
                                          line=dict(dash="dot", color="#8a8f99"))])))
    fig.frames = frames
    fig.update_layout(
        height=420, xaxis_title="hour", yaxis_title="normalized 0–1",
        legend=dict(orientation="h", y=-0.22, x=0),
        margin=dict(l=10, r=10, t=30, b=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        updatemenus=[dict(type="buttons", x=0.0, y=1.16, xanchor="left",
                          buttons=[
                              dict(label="▶ Play", method="animate",
                                   args=[None, dict(frame=dict(duration=350, redraw=True),
                                                    fromcurrent=True)]),
                              dict(label="⏸ Pause", method="animate",
                                   args=[[None], dict(mode="immediate")]),
                          ])],
        sliders=[dict(steps=[dict(label=str(k), method="animate",
                                  args=[[str(k)], dict(mode="immediate",
                                                       frame=dict(redraw=True))])
                             for k in hrs],
                      currentvalue=dict(prefix="hour "), y=-0.42)])
    return fig


def _corr_fig(sigs: dict, sel: list):
    """Pearson correlation heatmap of the selected signals."""
    import plotly.graph_objects as go

    T = len(np.asarray(sigs[sel[0]]).ravel())
    M = np.vstack([np.asarray(sigs[n], float).ravel()[:T] for n in sel])
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.corrcoef(M)
    C = np.nan_to_num(C, nan=0.0)      # constant signals correlate with nothing
    fig = go.Figure(go.Heatmap(
        z=C, x=sel, y=sel, zmin=-1.0, zmax=1.0, colorscale="RdBu", reversescale=True,
        text=np.round(C, 2), texttemplate="%{text}",
        colorbar=dict(title="r")))
    fig.update_layout(height=380 + 14 * len(sel),
                      margin=dict(l=10, r=10, t=30, b=10),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def _pair_fig(sigs: dict, xname: str, yname: str):
    """Scatter of one signal against another, colored by hour, with trend + r."""
    import plotly.graph_objects as go

    x = np.asarray(sigs[xname], float).ravel()
    y = np.asarray(sigs[yname], float).ravel()
    T = min(x.size, y.size)
    x, y = x[:T], y[:T]
    with np.errstate(invalid="ignore", divide="ignore"):
        r = float(np.corrcoef(x, y)[0, 1]) if np.std(x) > 0 and np.std(y) > 0 else 0.0
    fig = go.Figure(go.Scatter(
        x=x, y=y, mode="markers",
        marker=dict(size=10, color=list(range(T)), colorscale="Viridis",
                    colorbar=dict(title="hour"), line=dict(color="white", width=1)),
        customdata=np.arange(T).reshape(-1, 1),
        hovertemplate="hour %{customdata[0]}<br>" + xname + ": %{x:.4g}<br>"
                      + yname + ": %{y:.4g}<extra></extra>"))
    if np.std(x) > 1e-12:
        a, b = np.polyfit(x, y, 1)
        xs = np.linspace(float(x.min()), float(x.max()), 20)
        fig.add_trace(go.Scatter(x=xs, y=a * xs + b, mode="lines",
                                 line=dict(dash="dash", color="#d62728"),
                                 name="trend", showlegend=False))
    fig.update_layout(height=420, xaxis_title=xname, yaxis_title=yname,
                      title=f"Pearson r = {r:+.2f}",
                      margin=dict(l=10, r=10, t=45, b=10),
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def render_correlations(st, sigs: dict, key: str) -> None:
    """The full explorer: overlay + matrix + user-drawn pair scatter."""
    if not sigs:
        st.info("Re-run the case to build the signal library.")
        return
    names = list(sigs)
    default = names[: min(5, len(names))]
    sel = st.multiselect("Signals to overlay (normalized 0–1)", names, default=default,
                         key=f"{key}_corr_sel")
    if sel:
        st.plotly_chart(_overlay_fig(sigs, sel), use_container_width=True,
                        key=f"{key}_corr_overlay")
        st.caption("▶ Play sweeps the hour cursor — watch the causal chain: cheap "
                   "hours → pumps on → pump kW up → feeder voltage down → PV "
                   "reactive responds; tank levels integrate the pumping.")
    if len(sel) >= 2:
        st.plotly_chart(_corr_fig(sigs, sel), use_container_width=True,
                        key=f"{key}_corr_matrix")
        st.caption("Pearson correlation over the horizon. Blue = move together, "
                   "red = move oppositely. Remember: correlation here mixes the "
                   "*physics* (pump load ⇒ voltage drop) with the *optimization* "
                   "(price ⇒ schedule) — use the pair view to reason about which.")
    c1, c2 = st.columns(2)
    xname = c1.selectbox("X signal", names, index=0, key=f"{key}_corr_x")
    yname = c2.selectbox("Y signal", names, index=min(1, len(names) - 1),
                         key=f"{key}_corr_y")
    st.plotly_chart(_pair_fig(sigs, xname, yname), use_container_width=True,
                    key=f"{key}_corr_pair")
