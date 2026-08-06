"""Nonlinear single-phase Z-bus (fixed-point) power flow -- the *validator*.

This is a direct port of the MATLAB ``func_zbussan.m`` used in the IEEE Access
work.  The optimizer works on the *linear* LinDistFlow model (``pdn.lindistflow``)
because that stays convex; after a schedule is fixed we replay the resulting bus
injections through this exact nonlinear solve and report the voltage error, the
same two-run validation idea used on the water side.

Given the full incidence ``A_tilde`` and per-branch series admittance ``y = 1/z``::

    Ybus = A_tilde^T diag(y) A_tilde                         (N+1 x N+1)
    Y    = Ybus[1:, 1:] + j diag(b_shunt)                    (drop slack row/col)
    w    = -Y^{-1} (Ybus[1:, 0] * Vs)
    loop: I = conj(S)/conj(V);  V = Y^{-1} I + w   until |V| converges

Shunt capacitors are modeled the physically-correct way -- as a fixed susceptance
in the Y-bus (their reactive injection Q = b V^2 tracks voltage), matching the
MATLAB ``func_zbussan``.  The linear LinDistFlow model instead folds caps in as a
*constant* Q injection (b at V=1), so ``q_net`` passed here must EXCLUDE caps; the
small resulting difference is part of the genuine linearization error this solve
is meant to expose.
"""
from __future__ import annotations

import numpy as np

from .lindistflow import LinDistModel


def zbus_solve(model: LinDistModel, p_net: np.ndarray, q_net: np.ndarray,
               tol: float = 1e-10, max_iter: int = 1000):
    """Full nonlinear Z-bus solve.

    Returns ``(vmag, vcomplex, loss)``: non-slack voltage magnitudes (pu), their
    complex values, and the total real network loss (pu) from the exact branch
    currents ``I_br = y (A_tilde V_full)`` as ``sum |I_br|^2 r``.

    ``p_net`` / ``q_net`` are net loads in pu (positive = consumption) with PV and
    pump loads already folded in, but **excluding** shunt caps -- those are added
    here as a Y-bus susceptance (the physically-correct, voltage-dependent model).
    """
    N = model.N
    z = model.r + 1j * model.x
    y = 1.0 / z
    Ybus = model.A_tilde.T @ np.diag(y) @ model.A_tilde
    Y = Ybus[1:, 1:] + 1j * np.diag(model.b_shunt)
    YNS = Ybus[1:, 0]
    Vs = float(np.sqrt(model.v0_sq))

    P = -np.asarray(p_net, float)          # injections (negative of load)
    Q = -np.asarray(q_net, float)
    w = -np.linalg.solve(Y, YNS * Vs)
    V = np.ones(N, dtype=complex)
    for _ in range(max_iter):
        I = np.conj(P + 1j * Q) / np.conj(V)
        Vn = np.linalg.solve(Y, I) + w
        if np.max(np.abs(np.abs(Vn) - np.abs(V))) < tol:
            V = Vn
            break
        V = Vn

    V_full = np.concatenate(([Vs + 0j], V))            # prepend slack bus
    I_br = y * (model.A_tilde @ V_full)                # per-branch current (pu)
    loss = float(np.sum(np.abs(I_br) ** 2 * model.r))  # sum |I|^2 r  (pu)
    return np.abs(V), V, loss


def zbus_powerflow(model: LinDistModel, p_net: np.ndarray, q_net: np.ndarray,
                   tol: float = 1e-10, max_iter: int = 1000) -> np.ndarray:
    """Nonlinear bus voltage magnitudes (pu) -- thin wrapper over ``zbus_solve``."""
    vmag, _, _ = zbus_solve(model, p_net, q_net, tol, max_iter)
    return vmag
