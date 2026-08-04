"""WDN assembly (ports WDN_setup_IEEE_ACCESS.m + definebounds_WDN.m).

Ties together EPANET parsing, topology matrices, pump/tank parameters, bounds,
prices and the initial linearization into a single ``WDN`` object consumed by
the solver.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import config as cfg
from .config import SolverConfig
from .connection_matrices import Matrices, build_matrices
from .epanet_io import RawNetwork, read_inp, run_epanet
from .initial_values import initial_point
from .linearization import LinPoint, PumpParams


@dataclass
class TankParams:
    area: np.ndarray          # (Tk,)
    init_head: np.ndarray     # (Tk,)
    min_head: np.ndarray      # (Tk,)
    max_head: np.ndarray      # (Tk,)
    mean_head: np.ndarray     # (Tk,)  terminal-level target
    del_tk_tanks: np.ndarray  # (Tk,)  DEL_TK / area
    Atk: np.ndarray           # (T,)   ones
    Btk: np.ndarray           # (T, T) lower-triangular integrator


@dataclass
class Bounds:
    min_nodal_heads: np.ndarray  # (N x T)
    max_nodal_heads: np.ndarray  # (N x T)


@dataclass
class WDN:
    config: SolverConfig
    spec: cfg.NetworkSpec
    raw: RawNetwork
    M: Matrices
    pump: PumpParams
    tank: TankParams
    bounds: Bounds
    time: int
    price_final: np.ndarray            # (T,)
    junction_demand_profile: np.ndarray  # (J x T)
    lin0: LinPoint                     # initial linearization coefficients
    int_eps: np.ndarray                # initial stacked iterate [H;Q;OnOff]
    int_onoff: np.ndarray              # (Pu x T)
    pump_avail: dict = None            # pump_pos -> (start_hour, end_hour) availability

    # convenient sizes
    @property
    def n_nodes(self) -> int:
        return self.raw.n_nodes

    @property
    def n_links(self) -> int:
        return self.raw.n_links

    @property
    def n_pumps(self) -> int:
        return len(self.raw.link_pump_index)

    @property
    def n_pipes(self) -> int:
        return len(self.M.pipe_index)

    @property
    def n_tanks(self) -> int:
        return len(self.raw.tank_index)

    @property
    def n_reservoirs(self) -> int:
        return len(self.raw.reservoir_index)

    @property
    def n_junctions(self) -> int:
        return len(self.raw.junction_index)


def _pump_params(raw: RawNetwork, spec: cfg.NetworkSpec) -> PumpParams:
    # Prefer an explicit config override; otherwise use EPANET-derived coefficients.
    if spec.pump_coefficients is not None:
        coeff = np.asarray(spec.pump_coefficients, dtype=float)
    else:
        coeff = np.asarray(raw.pump_coefficients, dtype=float)
    if coeff.shape[0] != raw.link_pump_count:
        raise ValueError(
            f"{spec.name}: {coeff.shape[0]} pump coeff rows but EPANET reports "
            f"{raw.link_pump_count} pumps."
        )
    if not np.isfinite(coeff).all():
        bad = np.where(~np.isfinite(coeff).all(axis=1))[0].tolist()
        raise ValueError(
            f"{spec.name}: pumps {bad} have no usable EPANET head curve (e.g. "
            f"constant-power pumps). Supply NetworkSpec.pump_coefficients "
            f"([h0, r, v] per pump) for this network."
        )
    h0, r_m, v_m = coeff[:, 0], coeff[:, 1], coeff[:, 2]
    max_flow = (h0 / r_m) ** (1.0 / v_m)   # H_gain(max_flow) = 0; sqrt when v = 2
    return PumpParams(h0=h0, r_m=r_m, v_m=v_m, c_m=cfg.C_M, max_flow=max_flow)


def _tank_params(raw: RawNetwork, time: int) -> TankParams:
    area = raw.tank_area
    init_head = raw.tank_init_level
    min_head = raw.tank_min_level
    max_head = raw.tank_max_level
    mean_head = 0.5 * (init_head + min_head)          # definebounds_WDN.m
    del_tk_tanks = cfg.DEL_TK / area
    Atk = np.ones(time)
    Btk = np.tril(np.ones((time, time)))
    return TankParams(area=area, init_head=init_head, min_head=min_head,
                      max_head=max_head, mean_head=mean_head,
                      del_tk_tanks=del_tk_tanks, Atk=Atk, Btk=Btk)


def _bounds(raw: RawNetwork, tank: TankParams, time: int) -> Bounds:
    N = raw.n_nodes
    min_nodal = np.tile(raw.node_elevations[:, None], (1, time)).astype(float)
    max_nodal = np.full((N, time), 3000.0)
    for t, n in enumerate(raw.tank_index):
        min_nodal[n, :] = tank.min_head[t]
        max_nodal[n, :] = tank.max_head[t]
    return Bounds(min_nodal_heads=min_nodal, max_nodal_heads=max_nodal)


def _price(config: SolverConfig, time: int) -> np.ndarray:
    if config.price_choice == 1:
        price = np.asarray(cfg.PRICE_PATTERN, dtype=float) * cfg.PRICE_BASE
        if time <= price.size:
            return price[:time]
        reps = int(np.ceil(time / price.size))
        return np.tile(price, reps)[:time]
    return np.ones(time)


def _resolve_bypasses(raw: RawNetwork, spec: cfg.NetworkSpec):
    """Map spec.bypasses {bypass_id: pump_id} to link indices + pump positions."""
    if not spec.bypasses:
        return np.array([], dtype=int), np.array([], dtype=int)
    names = list(raw.link_name_id)
    bidx, ppos = [], []
    for bid, pid in spec.bypasses.items():
        b = names.index(str(bid))
        p_link = names.index(str(pid))
        pos = int(np.where(raw.link_pump_index == p_link)[0][0])
        bidx.append(b)
        ppos.append(pos)
    return np.array(bidx, dtype=int), np.array(ppos, dtype=int)


def _resolve_availability(raw: RawNetwork, spec: cfg.NetworkSpec) -> dict:
    """Map spec.pump_availability {pump_id: (start,end)} to {pump_pos: (start,end)}."""
    if not spec.pump_availability:
        return {}
    names = list(raw.link_name_id)
    out = {}
    for pid, window in spec.pump_availability.items():
        p_link = names.index(str(pid))
        pos = int(np.where(raw.link_pump_index == p_link)[0][0])
        out[pos] = tuple(window)
    return out


def setup(config: SolverConfig) -> WDN:
    """Build the full WDN problem data for the given configuration."""
    spec = config.spec
    raw = read_inp(spec.inp_path)

    # switched bypasses: resolve, and drop them from the permanently-closed set
    bypass_index, bypass_pump_pos = _resolve_bypasses(raw, spec)
    if bypass_index.size:
        raw.closed_pipe_index = np.setdiff1d(raw.closed_pipe_index, bypass_index)
    pump_avail = _resolve_availability(raw, spec)

    M = build_matrices(raw, bypass_index, bypass_pump_pos)

    # One EPANET run supplies the demand profile and the choice=1 initial point.
    flows_ep, heads_ep, _, demand_ep = run_epanet(raw)

    # Resolve horizon: default to EPANET's simulation length, clamped.
    sim_steps = flows_ep.shape[0]
    time = config.time if config.time is not None else sim_steps
    time = int(min(time, sim_steps))

    pump = _pump_params(raw, spec)
    tank = _tank_params(raw, time)
    bounds = _bounds(raw, tank, time)
    price_final = _price(config, time)

    # Auto-scale Big-M to the network if not overridden: comfortably larger than
    # the max head difference and the actual operating flows, but not so large it
    # ill-conditions HiGHS (Net3's ~300 ft heads can't share a 1e7 Big-M with
    # 1e-5 resistances). Uses EPANET operating flows, not the pump's theoretical
    # shutoff-based max flow, which can be an order of magnitude larger.
    if config.big_m is None:
        head_span = float(bounds.max_nodal_heads.max() - raw.node_elevations.min())
        flow_span = float(np.abs(flows_ep[:time]).max()) if time else 0.0
        config.big_m = max(1.0e4, 20.0 * head_span, 5.0 * flow_span)

    # Junction demand from EPANET's computed time series: (steps x N) -> (J x T).
    junction_demand_profile = demand_ep[:time, raw.junction_index].T

    lin0, int_eps, int_onoff = initial_point(
        raw, M, pump, bounds, time, config.choice, flows_ep, heads_ep
    )

    return WDN(
        config=config, spec=spec, raw=raw, M=M, pump=pump,
        tank=tank, bounds=bounds, time=time, price_final=price_final,
        junction_demand_profile=junction_demand_profile, lin0=lin0,
        int_eps=int_eps, int_onoff=int_onoff, pump_avail=pump_avail,
    )
