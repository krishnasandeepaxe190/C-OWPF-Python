"""EPANET input parsing and hydraulic ground-truth via epyt.

Ports ``read_inp.m`` (static network data) and ``init_epanet.m`` (extended-period
hydraulic simulation) onto the EPANET-Python-Toolkit (epyt), whose API mirrors
the EPANET-MATLAB-Toolkit used by the original code.

All node/link indices returned here are **0-based** (EPANET's 1-based indices are
converted on the way out) so they can be used directly as NumPy array indices.
EPANET orders nodes as: junctions, then reservoirs, then tanks -- the ordering
the incidence matrices in ``connection_matrices.py`` rely on.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from epyt import epanet


@dataclass
class RawNetwork:
    """Static network data extracted from an EPANET .inp file (0-based indices)."""

    # Links
    link_name_id: list[str]
    link_diameter: np.ndarray        # inches
    link_length: np.ndarray          # ft
    link_roughness: np.ndarray       # Hazen-Williams C
    link_resistance: np.ndarray      # per-link HW resistance (pumps -> 0)
    link_minor_resistance: np.ndarray
    link_type: list[str]
    link_pump_index: np.ndarray      # 0-based
    link_pump_count: int
    link_valve_index: np.ndarray     # 0-based
    pump_coefficients: np.ndarray    # (Pu, 3) [h0, r, v] derived from EPANET curves

    # Nodes
    node_index: np.ndarray           # 0-based [0..N-1]
    junction_index: np.ndarray       # 0-based
    reservoir_index: np.ndarray      # 0-based
    tank_index: np.ndarray           # 0-based
    node_elevations: np.ndarray      # ft

    # Connectivity: for each link, (from_node, to_node) as 0-based indices
    from_node: np.ndarray
    to_node: np.ndarray

    # Tanks
    tank_diameter: np.ndarray
    tank_min_level: np.ndarray       # min head = min level + elevation
    tank_max_level: np.ndarray       # max head = max level + elevation
    tank_init_level: np.ndarray      # init head = init level + elevation
    tank_area: np.ndarray            # A_tk = pi*(D/2)^2

    # keep the epyt handle so we can run the hydraulic simulation later
    handle: object

    @property
    def n_nodes(self) -> int:
        return len(self.node_index)

    @property
    def n_links(self) -> int:
        return len(self.link_name_id)


def _derive_pump_coefficients(d, n_pumps: int) -> np.ndarray:
    """Derive pump curve coefficients [h0, r, v=2] from EPANET head curves.

    For a single design point (Qd, Hd) EPANET's convention gives
    ``h0 = 1.3333 Hd`` and ``r = h0 / (2 Qd)**2`` (so H_gain = h0 - r q^2 passes
    through (0, 1.3333 Hd), (Qd, Hd) and (2 Qd, 0)); multi-point curves are fit
    to H = h0 - r q^2 by least squares. Assumes single-pump FSP networks
    (all in-scope cases); override via ``NetworkSpec.pump_coefficients`` otherwise.
    """
    raw = list(np.asarray(d.getLinkPumpHeadCurveIndex()).ravel().astype(int))
    # epyt returns [curveIdx, pumpLinkIdx] per pump for single-pump networks.
    curve_idx = raw[:n_pumps]
    ci = d.getCurvesInfo()
    coeffs = []
    for c in curve_idx:
        X = np.asarray(ci.CurveXvalue[c - 1], dtype=float)
        Y = np.asarray(ci.CurveYvalue[c - 1], dtype=float)
        if X.size == 1:
            Qd, Hd = float(X[0]), float(Y[0])
            h0 = 1.3333 * Hd
            r = h0 / ((2.0 * Qd) ** 2)
        else:
            A = np.vstack([np.ones_like(X), -X ** 2]).T
            (h0, r), *_ = np.linalg.lstsq(A, Y, rcond=None)
        coeffs.append([float(h0), float(r), 2.0])
    return np.asarray(coeffs, dtype=float)


def read_inp(inp_path) -> RawNetwork:
    """Load an EPANET .inp file and extract static OWF data (ports read_inp.m)."""
    d = epanet(str(inp_path))

    # --- node classification (EPANET 1-based -> 0-based) ---
    node_index = np.asarray(d.getNodeIndex(), dtype=int) - 1
    junction_index = np.asarray(d.getNodeJunctionIndex(), dtype=int) - 1
    reservoir_index = np.asarray(d.getNodeReservoirIndex(), dtype=int) - 1
    tank_index = np.asarray(d.getNodeTankIndex(), dtype=int) - 1
    node_elevations = np.asarray(d.getNodeElevations(), dtype=float)

    # --- links ---
    link_name_id = list(d.getLinkNameID())
    link_type = list(d.getLinkType())
    link_diameter = np.asarray(d.getLinkDiameter(), dtype=float)
    link_length = np.asarray(d.getLinkLength(), dtype=float)
    link_roughness = np.asarray(d.getLinkRoughnessCoeff(), dtype=float)
    link_minor = np.asarray(d.getLinkMinorLossCoeff(), dtype=float)
    link_pump_index = np.asarray(d.getLinkPumpIndex(), dtype=int).ravel() - 1
    link_pump_count = int(d.getLinkPumpCount())
    valve_raw = np.asarray(d.getLinkValveIndex(), dtype=float).ravel()
    link_valve_index = (valve_raw.astype(int) - 1) if valve_raw.size else np.array([], dtype=int)
    pump_coefficients = _derive_pump_coefficients(d, link_pump_count)

    # Hazen-Williams resistance (read_inp.m). Only defined for pipes; pumps have
    # roughness/diameter 0 -> guard against divide-by-zero and set them to 0.
    n_links = len(link_name_id)
    link_resistance = np.zeros(n_links)
    link_minor_resistance = np.zeros(n_links)
    with np.errstate(divide="ignore", invalid="ignore"):
        for i in range(n_links):
            if link_roughness[i] > 0 and link_diameter[i] > 0:
                link_resistance[i] = (
                    10.4622
                    * link_roughness[i] ** (-1.852)
                    * link_diameter[i] ** (-4.871)
                    * link_length[i]
                )
                link_minor_resistance[i] = (
                    0.002596 * link_minor[i]
                ) / (link_diameter[i] ** 4)

    # --- connectivity (0-based from/to per link) ---
    ncl = np.asarray(d.getNodesConnectingLinksIndex(), dtype=int)  # (n_links, 2), 1-based
    from_node = ncl[:, 0] - 1
    to_node = ncl[:, 1] - 1

    # --- tanks: heads referenced to absolute datum (level + elevation) ---
    tank_elev = node_elevations[tank_index]
    tank_diameter = np.asarray(d.getNodeTankDiameter(), dtype=float)
    tank_min_level = np.asarray(d.getNodeTankMinimumWaterLevel(), dtype=float) + tank_elev
    tank_max_level = np.asarray(d.getNodeTankMaximumWaterLevel(), dtype=float) + tank_elev
    tank_init_level = np.asarray(d.getNodeTankInitialLevel(), dtype=float) + tank_elev
    tank_area = np.pi * (tank_diameter / 2.0) ** 2

    # Demands are taken from EPANET's computed time series (run_epanet) rather
    # than rebuilt from base*pattern here -- that way pattern wrapping, default
    # patterns and mixed pattern lengths match EPANET exactly.

    return RawNetwork(
        link_name_id=link_name_id,
        link_diameter=link_diameter,
        link_length=link_length,
        link_roughness=link_roughness,
        link_resistance=link_resistance,
        link_minor_resistance=link_minor_resistance,
        link_type=link_type,
        link_pump_index=link_pump_index,
        link_pump_count=link_pump_count,
        link_valve_index=link_valve_index,
        pump_coefficients=pump_coefficients,
        node_index=node_index,
        junction_index=junction_index,
        reservoir_index=reservoir_index,
        tank_index=tank_index,
        node_elevations=node_elevations,
        from_node=from_node,
        to_node=to_node,
        tank_diameter=tank_diameter,
        tank_min_level=tank_min_level,
        tank_max_level=tank_max_level,
        tank_init_level=tank_init_level,
        tank_area=tank_area,
        handle=d,
    )


def run_epanet(raw: RawNetwork) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Extended-period hydraulic simulation (ports init_epanet.m).

    Returns (flows, heads, headloss, demand) each shaped (n_steps, n_links|n_nodes),
    i.e. time along axis 0 -- matching the raw EPANET output ordering the MATLAB
    code transposes downstream.
    """
    d = raw.handle
    res = d.getComputedTimeSeries()
    flows = np.asarray(res.Flow, dtype=float)
    heads = np.asarray(res.Head, dtype=float)
    headloss = np.asarray(res.HeadLoss, dtype=float)
    demand = np.asarray(res.Demand, dtype=float)
    return flows, heads, headloss, demand
