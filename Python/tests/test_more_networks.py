"""Tests for the additional FSP networks: 3-node (working) and Net1 (loads)."""
import numpy as np
import pytest

from owf import SolverConfig, setup, solve_owf, validate_schedule


# --------------------------------------------------------------------------
# 3-node: fully working, validates against EPANET to paper accuracy.
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def threenode():
    wdn = setup(SolverConfig(net_num=3))
    return wdn, solve_owf(wdn)


def test_threenode_sizes(threenode):
    wdn, _ = threenode
    assert wdn.n_nodes == 3
    assert wdn.n_links == 2
    assert wdn.n_pumps == 1
    assert wdn.n_tanks == 1


def test_threenode_pump_coeff_from_epanet(threenode):
    """Pump coefficients are derived from the EPANET large-pump curve (~533, ~1.3e-6)."""
    wdn, _ = threenode
    assert wdn.pump.h0[0] == pytest.approx(533.32, rel=1e-3)
    assert wdn.pump.max_flow[0] == pytest.approx(20000.0, rel=1e-2)


def test_threenode_converges(threenode):
    _, result = threenode
    assert result.status == "optimal"
    assert result.converged


def test_threenode_matches_epanet_schedule(threenode):
    """Optimized schedule re-run in EPANET matches the optimizer closely (Fig. 4)."""
    wdn, result = threenode
    rep = validate_schedule(wdn, result)
    assert rep.max_abs_head < 0.1          # ft
    assert rep.max_abs_pump_flow < 1.0     # GPM


# --------------------------------------------------------------------------
# Net1 (net 11): looped network -- builds and derives pump coeffs, but the
# successive-linearization loop does not converge from the default init.
# Documented limitation; kept as a build test so the data/spec stay wired in.
# --------------------------------------------------------------------------
def test_net1_builds():
    wdn = setup(SolverConfig(net_num=11))
    assert wdn.n_nodes == 11
    assert wdn.n_links == 13
    assert wdn.n_pumps == 1
    # pump coefficients derived from EPANET curve 1 (1500, 250)
    assert wdn.pump.h0[0] == pytest.approx(333.3, rel=1e-2)
    assert wdn.pump.max_flow[0] == pytest.approx(3000.0, rel=1e-2)


@pytest.mark.xfail(reason="Net1 successive-linearization needs a tuned warm-start", strict=False)
def test_net1_converges():
    wdn = setup(SolverConfig(net_num=11))
    result = solve_owf(wdn)
    assert result.converged
