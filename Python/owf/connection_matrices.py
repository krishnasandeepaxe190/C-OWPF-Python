"""Topology / incidence matrices for the WDN (ports ConnectionMatrices_WDN.m).

All matrices use 0-based node/link ordering. Symbol names follow the paper:

    Pi         (N x L)      node-arc incidence: +1 at the 'from' node, -1 at 'to'
    Pi_telda   (P x N)      incidence rows for pipes only (energy equation)
    Pi_prime   (P x L)      selects pipe flows from the full flow vector
    Pi_reduced (J x L)      incidence rows for junctions only (mass balance)
    Lambda     (Pu x L)     selects pump flows from the full flow vector
    Theta      (R x N)      selects reservoir nodes
    Tau        (N x Tk)     selects tank nodes (columns)
    Kappa      (J x N)      selects junction nodes
    Omega      (P x P)      diagonal pipe Hazen-Williams resistance
    Delta      (Tk x Tk)    diagonal tank cross-sectional areas
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .epanet_io import RawNetwork


@dataclass
class Matrices:
    Pi: np.ndarray
    Pi_telda: np.ndarray
    Pi_prime: np.ndarray
    Pi_reduced: np.ndarray
    Lambda: np.ndarray
    Theta: np.ndarray
    Tau: np.ndarray
    Kappa: np.ndarray
    Omega: np.ndarray
    Delta: np.ndarray
    pipe_index: np.ndarray  # 0-based link indices that are pipes (non-pump)
    # switched-bypass pipes (open iff their pump is off); empty for most networks
    bypass_index: np.ndarray          # (Nb,) 0-based link indices
    Pi_telda_bypass: np.ndarray       # (Nb x N) incidence rows for bypasses
    Pi_prime_bypass: np.ndarray       # (Nb x L) selects bypass flows
    Omega_bypass: np.ndarray          # (Nb x Nb) diagonal bypass resistance
    S_bypass_pump: np.ndarray         # (Nb x Pu) bypass -> controlling pump
    # pressure-reducing valves (PRVs); empty for most networks
    valve_index: np.ndarray           # (Nv,) 0-based link indices
    Pi_prime_valve: np.ndarray        # (Nv x L) selects valve flows
    valve_up_sel: np.ndarray          # (Nv x N) picks the upstream node head
    valve_down_sel: np.ndarray        # (Nv x N) picks the downstream node head


def build_matrices(
    raw: RawNetwork,
    bypass_index: np.ndarray = None,
    bypass_pump_pos: np.ndarray = None,
    valve_index: np.ndarray = None,
) -> Matrices:
    N = raw.n_nodes
    L = raw.n_links
    pump_index = raw.link_pump_index
    tank_index = raw.tank_index
    reservoir_index = raw.reservoir_index
    junction_index = raw.junction_index

    bypass_index = (np.asarray(bypass_index, dtype=int) if bypass_index is not None
                    else np.array([], dtype=int))
    bypass_pump_pos = (np.asarray(bypass_pump_pos, dtype=int) if bypass_pump_pos is not None
                       else np.array([], dtype=int))
    valve_index = (np.asarray(valve_index, dtype=int) if valve_index is not None
                   else np.array([], dtype=int))

    # pipes = links that are not pumps, not permanently closed, not switched
    # bypasses, and not valves (each of those gets its own gated head constraint).
    # Closed pipes carry no head loss and are pinned to zero flow separately.
    excluded = np.zeros(L, dtype=bool)
    excluded[pump_index] = True
    excluded[raw.closed_pipe_index] = True
    if bypass_index.size:
        excluded[bypass_index] = True
    if valve_index.size:
        excluded[valve_index] = True
    pipe_index = np.where(~excluded)[0]

    # Pi (N x L): +1 at from-node, -1 at to-node
    Pi = np.zeros((N, L))
    for l in range(L):
        Pi[raw.from_node[l], l] += 1.0
        Pi[raw.to_node[l], l] += -1.0

    Pi_telda = Pi.T[pipe_index, :]            # (P x N)
    Pi_prime = np.eye(L)[pipe_index, :]       # (P x L)
    Pi_reduced = Pi[junction_index, :]        # (J x L)

    Lambda = np.zeros((len(pump_index), L))   # (Pu x L)
    for p, l in enumerate(pump_index):
        Lambda[p, l] = 1.0

    Theta = np.zeros((len(reservoir_index), N))  # (R x N)
    for r, n in enumerate(reservoir_index):
        Theta[r, n] = 1.0

    Tau = np.zeros((N, len(tank_index)))         # (N x Tk)
    for t, n in enumerate(tank_index):
        Tau[n, t] = 1.0

    Kappa = np.zeros((len(junction_index), N))   # (J x N)
    for j, n in enumerate(junction_index):
        Kappa[j, n] = 1.0

    Omega = np.diag(raw.link_resistance[pipe_index])   # (P x P) Hazen-Williams
    Delta = np.diag(raw.tank_area)                     # (Tk x Tk)

    # switched-bypass matrices
    n_pumps = len(pump_index)
    Pi_telda_bypass = Pi.T[bypass_index, :] if bypass_index.size else np.zeros((0, N))
    Pi_prime_bypass = np.eye(L)[bypass_index, :] if bypass_index.size else np.zeros((0, L))
    Omega_bypass = np.diag(raw.link_resistance[bypass_index]) if bypass_index.size else np.zeros((0, 0))
    S_bypass_pump = np.zeros((bypass_index.size, n_pumps))
    for i, p in enumerate(bypass_pump_pos):
        S_bypass_pump[i, p] = 1.0

    # PRV matrices: select the valve flow and the up/down node heads. The valve's
    # own head relation is the big-M PRV model (not the pipe energy equation), so
    # valves are excluded from pipe_index above but kept in Pi (mass balance).
    Pi_prime_valve = np.eye(L)[valve_index, :] if valve_index.size else np.zeros((0, L))
    valve_up_sel = np.zeros((valve_index.size, N))
    valve_down_sel = np.zeros((valve_index.size, N))
    for i, lk in enumerate(valve_index):
        valve_up_sel[i, raw.from_node[lk]] = 1.0
        valve_down_sel[i, raw.to_node[lk]] = 1.0

    return Matrices(
        Pi=Pi, Pi_telda=Pi_telda, Pi_prime=Pi_prime, Pi_reduced=Pi_reduced,
        Lambda=Lambda, Theta=Theta, Tau=Tau, Kappa=Kappa, Omega=Omega,
        Delta=Delta, pipe_index=pipe_index,
        bypass_index=bypass_index, Pi_telda_bypass=Pi_telda_bypass,
        Pi_prime_bypass=Pi_prime_bypass, Omega_bypass=Omega_bypass,
        S_bypass_pump=S_bypass_pump,
        valve_index=valve_index, Pi_prime_valve=Pi_prime_valve,
        valve_up_sel=valve_up_sel, valve_down_sel=valve_down_sel,
    )
