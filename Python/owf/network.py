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
from .epanet_io import RawNetwork, read_inp
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
    coeff = np.asarray(spec.pump_coefficients, dtype=float)  # (Pu, 3)
    if coeff.shape[0] != raw.link_pump_count:
        raise ValueError(
            f"{spec.name}: {coeff.shape[0]} pump coeff rows but EPANET reports "
            f"{raw.link_pump_count} pumps."
        )
    h0, r_m, v_m = coeff[:, 0], coeff[:, 1], coeff[:, 2]
    max_flow = np.sqrt(h0 / r_m)
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
        return price[:time]
    return np.ones(time)


def setup(config: SolverConfig) -> WDN:
    """Build the full WDN problem data for the given configuration."""
    spec = config.spec
    raw = read_inp(spec.inp_path)

    # Resolve horizon: default to the demand-pattern length, clamped.
    pattern_len = raw.junction_profile.shape[1]
    time = config.time if config.time is not None else pattern_len
    time = int(min(time, pattern_len))

    pump = _pump_params(raw, spec)
    tank = _tank_params(raw, time)
    bounds = _bounds(raw, tank, time)
    price_final = _price(config, time)
    junction_demand_profile = raw.junction_profile[:, :time]

    lin0, int_eps, int_onoff = initial_point(
        raw, build_matrices(raw), pump, bounds, time, config.choice
    )

    return WDN(
        config=config, spec=spec, raw=raw, M=build_matrices(raw), pump=pump,
        tank=tank, bounds=bounds, time=time, price_final=price_final,
        junction_demand_profile=junction_demand_profile, lin0=lin0,
        int_eps=int_eps, int_onoff=int_onoff,
    )
