"""Diagnostic plots: convergence, EPANET-vs-OWF flows/heads, and pump schedules.

All functions take the solved ``WDN``/``OWFResult`` (plus the schedule-imposed
``ScheduleValidationReport``, which holds the EPANET hydraulics under the
optimized schedule) and return a matplotlib Figure. Use ``plot_all`` to write the
full set to a directory.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

from .network import WDN
from .solver import OWFResult
from .validation import ScheduleValidationReport, validate_schedule


def _plt():
    import matplotlib
    matplotlib.use("Agg")           # file output; no GUI needed
    import matplotlib.pyplot as plt
    return plt


def plot_convergence(result: OWFResult):
    """Successive-linearization error and objective per iteration."""
    plt = _plt()
    from matplotlib.ticker import MaxNLocator

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    it = np.arange(1, len(result.errors) + 1)

    ax1.semilogy(it, result.errors, "o-", color="tab:blue")
    ax1.set_xlabel("iteration")
    ax1.set_ylabel(r"$\|\epsilon_k - \epsilon_{k-1}\|$")
    ax1.set_title("Successive-linearization convergence")
    ax1.grid(True, which="both", alpha=0.3)

    ax2.plot(it, result.objectives, "s-", color="tab:red")
    ax2.set_xlabel("iteration")
    ax2.set_ylabel("energy cost")
    ax2.set_title("Objective per iteration")
    ax2.grid(True, alpha=0.3)

    for ax in (ax1, ax2):
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.tight_layout()
    return fig


def plot_pump_schedule(wdn: WDN, result: OWFResult):
    """Optimized pump on/off schedule against the electricity price."""
    plt = _plt()
    T = wdn.time
    hours = np.arange(T)
    n_p = wdn.n_pumps
    fig, axes = plt.subplots(n_p + 1, 1, figsize=(10, 2.2 * (n_p + 1)), sharex=True)
    axes = np.atleast_1d(axes)

    for p in range(n_p):
        ax = axes[p]
        ax.step(hours, result.onoff[p, :], where="mid", color="tab:green", lw=2)
        ax.fill_between(hours, 0, result.onoff[p, :], step="mid", alpha=0.3,
                        color="tab:green")
        ax.set_ylim(-0.1, 1.15)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["off", "on"])
        ax.set_ylabel(f"pump {p + 1}")
        ax.grid(True, alpha=0.3)
        if p == 0:
            ax.set_title("Optimized pump schedule vs price")

    ax = axes[-1]
    ax.plot(hours, wdn.price_final, "o-", color="tab:orange")
    ax.set_ylabel("price")
    ax.set_xlabel("hour")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig


def plot_flows(wdn: WDN, result: OWFResult, report: ScheduleValidationReport,
               max_pipes: int = 6):
    """Pump flows and a sample of pipe flows: OWF vs EPANET (imposed schedule)."""
    plt = _plt()
    T = wdn.time
    hours = np.arange(T)

    pump_opt = wdn.M.Lambda @ result.flows[:, :T]
    pump_ep = wdn.M.Lambda @ report.flows_epanet
    pipe_opt = wdn.M.Pi_prime @ result.flows[:, :T]
    pipe_ep = wdn.M.Pi_prime @ report.flows_epanet

    # show the pipes with the largest average flow
    order = np.argsort(-np.abs(pipe_opt).mean(axis=1))[:max_pipes]
    n_rows = wdn.n_pumps + len(order)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, 1.9 * n_rows), sharex=True)
    axes = np.atleast_1d(axes)

    k = 0
    for p in range(wdn.n_pumps):
        ax = axes[k]; k += 1
        ax.plot(hours, pump_ep[p], "o--", color="k", label="EPANET", ms=4)
        ax.plot(hours, pump_opt[p], "-", color="tab:blue", label="OWF", lw=2)
        ax.set_ylabel(f"pump {p + 1}\n(GPM)")
        ax.grid(True, alpha=0.3)
        if k == 1:
            ax.set_title("Flows: OWF vs EPANET (optimized schedule imposed)")
            ax.legend(fontsize=8, loc="best")

    pipe_ids = [wdn.raw.link_name_id[i] for i in wdn.M.pipe_index]
    for idx in order:
        ax = axes[k]; k += 1
        ax.plot(hours, pipe_ep[idx], "o--", color="k", ms=4)
        ax.plot(hours, pipe_opt[idx], "-", color="tab:blue", lw=2)
        ax.set_ylabel(f"pipe {pipe_ids[idx]}\n(GPM)")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("hour")
    fig.tight_layout()
    return fig


def plot_heads(wdn: WDN, result: OWFResult, report: ScheduleValidationReport,
               max_junctions: int = 5):
    """Tank and junction heads: OWF vs EPANET (imposed schedule)."""
    plt = _plt()
    T = wdn.time
    hours = np.arange(T)
    H_opt = result.heads[:, :T]
    H_ep = report.heads_epanet

    tanks = list(wdn.raw.tank_index)
    juncs = list(wdn.raw.junction_index)[:max_junctions]
    rows = len(tanks) + len(juncs)
    fig, axes = plt.subplots(rows, 1, figsize=(10, 1.9 * rows), sharex=True)
    axes = np.atleast_1d(axes)

    k = 0
    for t_i, n in enumerate(tanks):
        ax = axes[k]; k += 1
        ax.plot(hours, H_ep[n], "o--", color="k", label="EPANET", ms=4)
        ax.plot(hours, H_opt[n], "-", color="tab:purple", label="OWF", lw=2)
        ax.axhline(wdn.tank.min_head[t_i], color="r", ls=":", lw=1, label="bounds")
        ax.axhline(wdn.tank.max_head[t_i], color="r", ls=":", lw=1)
        ax.set_ylabel(f"tank {t_i + 1}\n(ft)")
        ax.grid(True, alpha=0.3)
        if k == 1:
            ax.set_title("Heads: OWF vs EPANET (optimized schedule imposed)")
            ax.legend(fontsize=8, loc="best")

    for n in juncs:
        ax = axes[k]; k += 1
        ax.plot(hours, H_ep[n], "o--", color="k", ms=4)
        ax.plot(hours, H_opt[n], "-", color="tab:purple", lw=2)
        ax.axhline(wdn.raw.node_elevations[n], color="r", ls=":", lw=1)
        ax.set_ylabel(f"node {wdn.raw.node_index[n] + 1}\n(ft)")
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("hour")
    fig.tight_layout()
    return fig


def plot_error_summary(wdn: WDN, result: OWFResult, report: ScheduleValidationReport):
    """Per-time-step max |error| in heads and flows vs EPANET."""
    plt = _plt()
    T = wdn.time
    hours = np.arange(T)
    dh = np.abs(report.heads_epanet - result.heads[:, :T]).max(axis=0)
    df = np.abs(report.flows_epanet - result.flows[:, :T]).max(axis=0)

    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.semilogy(hours, np.maximum(dh, 1e-12), "o-", label="max |dHead| (ft)")
    ax.semilogy(hours, np.maximum(df, 1e-12), "s-", label="max |dFlow| (GPM)")
    ax.set_xlabel("hour")
    ax.set_ylabel("abs. error")
    ax.set_title("OWF vs EPANET error per time step")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    return fig


def plot_all(wdn: WDN, result: OWFResult,
             report: Optional[ScheduleValidationReport] = None,
             outdir: str | Path = "outputs", prefix: Optional[str] = None) -> list[Path]:
    """Write the full plot set to ``outdir``; returns the written paths."""
    if report is None:
        report = validate_schedule(wdn, result)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    prefix = prefix or wdn.spec.name

    figs = {
        "convergence": plot_convergence(result),
        "pump_schedule": plot_pump_schedule(wdn, result),
        "flows": plot_flows(wdn, result, report),
        "heads": plot_heads(wdn, result, report),
        "error": plot_error_summary(wdn, result, report),
    }
    paths = []
    for name, fig in figs.items():
        path = outdir / f"{prefix}_{name}.png"
        fig.savefig(path, dpi=130)
        fig.clf()
        paths.append(path)
    return paths
