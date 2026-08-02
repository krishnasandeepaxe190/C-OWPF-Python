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


def build_matrices(raw: RawNetwork) -> Matrices:
    N = raw.n_nodes
    L = raw.n_links
    pump_index = raw.link_pump_index
    tank_index = raw.tank_index
    reservoir_index = raw.reservoir_index
    junction_index = raw.junction_index

    # pipes = links that are not pumps (valves are ignored in the FSP model)
    is_pump = np.zeros(L, dtype=bool)
    is_pump[pump_index] = True
    pipe_index = np.where(~is_pump)[0]

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

    Omega = np.diag(raw.link_resistance[pipe_index])   # (P x P)
    Delta = np.diag(raw.tank_area)                     # (Tk x Tk)

    return Matrices(
        Pi=Pi, Pi_telda=Pi_telda, Pi_prime=Pi_prime, Pi_reduced=Pi_reduced,
        Lambda=Lambda, Theta=Theta, Tau=Tau, Kappa=Kappa, Omega=Omega,
        Delta=Delta, pipe_index=pipe_index,
    )
