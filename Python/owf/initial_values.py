"""Initial linearization point (ports Initial_Values_WDN.m).

Choice == 1: initialize from an EPANET hydraulic run (recommended).
Choice == 0: user-defined constant seed flows/heads.
"""
from __future__ import annotations

import numpy as np

from .connection_matrices import Matrices
from .epanet_io import RawNetwork, run_epanet
from .linearization import LinPoint, PumpParams, linearize, stack_eps


def initial_point(
    raw: RawNetwork,
    M: Matrices,
    pump: PumpParams,
    bounds,
    time: int,
    choice: int,
) -> tuple[LinPoint, np.ndarray, np.ndarray]:
    """Return (initial LinPoint, stacked iterate int_eps, initial OnOff)."""
    n_pumps = len(raw.link_pump_index)

    if choice == 1:
        flows_ep, heads_ep, _ = run_epanet(raw)   # (steps, L), (steps, N)
        steps = min(time, flows_ep.shape[0])
        int_flows = flows_ep[:steps, :].T          # (L x T)
        int_heads = heads_ep[:steps, :].T          # (N x T)
        if steps < time:  # pad by repeating the last available step
            int_flows = np.pad(int_flows, ((0, 0), (0, time - steps)), mode="edge")
            int_heads = np.pad(int_heads, ((0, 0), (0, time - steps)), mode="edge")
        int_onoff = np.zeros((n_pumps, time))
    else:
        # user-defined constant seed (Initial_Values_WDN.m, Choice == 0 branch)
        int_flows = 6000.0 * np.ones((raw.n_links, time))
        int_heads = bounds.min_nodal_heads.copy()
        int_onoff = np.ones((n_pumps, time))

    # Force tank rows of the head seed to their initial heads.
    for t, n in enumerate(raw.tank_index):
        int_heads[n, :] = raw.tank_init_level[t]

    lin = linearize(int_flows, M, pump)
    int_eps = stack_eps(int_heads, int_flows, int_onoff)
    return lin, int_eps, int_onoff
