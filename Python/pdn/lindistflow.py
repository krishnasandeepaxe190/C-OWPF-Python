"""Linear (LinDistFlow / Kekatos) distribution-network voltage model.

For a radial single-phase feeder the *squared* voltage magnitudes at the N
non-slack buses are an **affine** function of the bus power injections::

    v2 = R @ (-p_net) + X @ (-q_net) + V_k                     (pu^2)

where ``p_net`` / ``q_net`` are the net active / reactive *loads* (positive =
consumption) at each bus, and

    A_tilde   full branch-bus incidence  (N x N+1),  A_tilde[i, parent(i)] = +1,
                                                     A_tilde[i, i+1]       = -1
    A         reduced incidence          = A_tilde[:, 1:]        (N x N)
    F         = -inv(A)                                          (N x N)
    R         = 2 F diag(r) F^T          X = 2 F diag(x) F^T
    V_k       = F a_0 v0^2   (a_0 = A_tilde[:, 0]);  since F a_0 = 1_N, V_k = v0^2

This is the standard loss-less LinDistFlow relaxation (a.k.a. the Kekatos linear
model).  It is **exact to first order** in the branch flows and matches a full
nonlinear Z-bus solve to a few 1e-3 pu on lightly/moderately loaded feeders; the
residual (line-loss term, dropped here) is what the nonlinear check in
``pdn.powerflow`` quantifies.

The affine ``v2 = M_p @ p + M_q @ q + c`` form (with sign folded in) is what the
coupled LP consumes: pump loads and PV injections just move ``p_net`` / ``q_net``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LinDistModel:
    """Prebuilt affine voltage model for one feeder (all pu)."""

    N: int
    A_tilde: np.ndarray      # (N, N+1) full incidence
    F: np.ndarray            # (N, N)  = -inv(reduced incidence)
    R: np.ndarray            # (N, N)  active-power -> v2 sensitivity (=2 F diag(r) F^T)
    X: np.ndarray            # (N, N)  reactive-power -> v2 sensitivity
    V_k: np.ndarray          # (N,)    constant term (= v0^2 * 1_N)
    r: np.ndarray            # (N,) pu branch resistance (bus = its feeding branch)
    x: np.ndarray            # (N,) pu branch reactance
    p_load: np.ndarray       # (N,) pu nominal active load
    q_load: np.ndarray       # (N,) pu nominal reactive load
    b_shunt: np.ndarray      # (N,) pu shunt-cap susceptance injection (+Q)
    pv_mask: np.ndarray      # (N,) bool, PV-hosting buses
    v0_sq: float             # slack squared voltage (pu^2)
    parent: np.ndarray       # (N,) parent bus index (0 = slack) of each non-slack bus

    def v2(self, p_net: np.ndarray, q_net: np.ndarray) -> np.ndarray:
        """Squared bus voltages for given net loads (pu).

        ``q_net`` is the net reactive load EXCLUDING shunt caps -- the caps are
        voltage-dependent (qˢ·v) and already folded into ``R``/``X``/``V_k`` via the
        Kekatos (I − X·diag(qˢ))⁻¹ correction, matching paper eq. (1b)/(2b).
        """
        return self.R @ (-p_net) + self.X @ (-q_net) + self.V_k

    def voltage(self, p_net: np.ndarray, q_net: np.ndarray) -> np.ndarray:
        return np.sqrt(np.maximum(self.v2(p_net, q_net), 0.0))


def build_lindist(feeder: dict) -> LinDistModel:
    """Assemble the affine voltage model from a ``pdn.feeders`` entry."""
    N = int(feeder["N"])
    parent = np.asarray(feeder["parent"], int)
    r = np.asarray(feeder["r"], float)
    x = np.asarray(feeder["x"], float)
    p = np.asarray(feeder["p"], float)
    q = np.asarray(feeder["q"], float)
    pv = np.asarray(feeder["pv"], int).astype(bool)
    v0_sq = float(feeder["v0_sq"])

    A_t = np.zeros((N, N + 1))
    for i in range(N):
        A_t[i, parent[i]] = 1.0
        A_t[i, i + 1] = -1.0
    a0 = A_t[:, 0]
    A = A_t[:, 1:]
    F = -np.linalg.inv(A)
    R = 2.0 * F @ np.diag(r) @ F.T
    X = 2.0 * F @ np.diag(x) @ F.T
    V_k = F @ a0 * v0_sq

    b_shunt = np.zeros(N)
    for k, val in (feeder.get("caps") or {}).items():
        b_shunt[int(k)] = float(val)

    # Voltage-dependent shunt caps (paper eq. 1b/2b): the cap injects qˢ·v, so
    #   (I − X·diag(qˢ)) v = R(−p) + X(−q) + V_k   →   fold (I−Qsx)⁻¹ into R,X,V_k.
    # After this, ``q`` passed to the model EXCLUDES caps; they act through the
    # correction, matching func_branchbus.m's V_nsh (Kekatos with shunt).
    if b_shunt.any():
        Minv = np.linalg.inv(np.eye(N) - X @ np.diag(b_shunt))
        R, X, V_k = Minv @ R, Minv @ X, Minv @ V_k

    return LinDistModel(
        N=N, A_tilde=A_t, F=F, R=R, X=X, V_k=V_k, r=r, x=x,
        p_load=p, q_load=q, b_shunt=b_shunt, pv_mask=pv, v0_sq=v0_sq, parent=parent,
    )


def branch_flow_matrix(model: LinDistModel) -> np.ndarray:
    """Subtree-membership matrix T (N x N): ``P_branch = T @ p_net``.

    Row i (the branch feeding bus i) sums the net injections of bus i and all its
    descendants -- the lossless LinDistFlow branch flow. With per-unit branch flows
    the network loss is ``sum_i r_i (P_branch_i^2 + Q_branch_i^2)``.
    """
    N = model.N
    parent = np.asarray(model.parent, int)
    T = np.zeros((N, N))
    for k in range(N):
        node = k
        while True:
            T[node, k] = 1.0            # node is ancestor-or-self of k
            pg = parent[node]           # global parent index (0 = slack)
            if pg == 0:
                break
            node = pg - 1               # global -> 0-based non-slack
    return T
