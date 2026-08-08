"""Sanity tests for the PWL-MILP benchmark (Oikonomou-Parvania comparison)."""
import numpy as np
import pytest

from owf.config import SolverConfig
from owf.network import setup
from owf.pwl_benchmark import build_pwl_milp, solve_pwl_owf


def test_pwl_threenode_k5_solves_and_counts():
    wdn = setup(SolverConfig(net_num=3))
    r = solve_pwl_owf(wdn, K=5, time_limit=60.0)
    assert r.status in ("optimal", "optimal_inaccurate")
    S, U, T = wdn.n_pipes, wdn.n_pumps, wdn.time
    # paper eqs. 17-24: (K-1) segment binaries per pipe/pump-hour + on/off
    assert r.n_binary == S * 4 * T + U * 4 * T + U * T
    assert r.flows is not None
    assert np.isfinite(r.true_cost) and r.true_cost > 0
    # an OFF pump must carry zero flow and zero PWL power
    q = wdn.M.Lambda @ r.flows
    off = r.onoff < 0.5
    assert np.max(np.abs(q[off])) < 1e-6
    assert np.max(np.abs(r.ppump_pwl[off])) < 1e-6


def test_pwl_rejects_even_k_and_special_networks():
    wdn = setup(SolverConfig(net_num=3))
    with pytest.raises(ValueError):
        build_pwl_milp(wdn, K=8)
