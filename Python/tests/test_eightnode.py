"""Regression tests for the 8-node FSP OWF problem."""
import numpy as np
import pytest

from owf import SolverConfig, setup, solve_owf, validate


@pytest.fixture(scope="module")
def solved():
    wdn = setup(SolverConfig(net_num=8, price_choice=1, choice=1))
    result = solve_owf(wdn)
    return wdn, result


def test_network_sizes():
    wdn = setup(SolverConfig(net_num=8))
    assert wdn.n_nodes == 8
    assert wdn.n_links == 9
    assert wdn.n_pumps == 1
    assert wdn.n_tanks == 1
    assert wdn.n_reservoirs == 1
    assert wdn.n_junctions == 6
    assert wdn.n_pipes == 8
    assert wdn.time == 12


def test_converges(solved):
    _, result = solved
    assert result.status == "optimal"
    assert result.converged
    assert result.n_iter >= 1


def test_onoff_binary(solved):
    _, result = solved
    onoff = result.onoff
    assert np.all((np.abs(onoff) < 1e-6) | (np.abs(onoff - 1) < 1e-6))


def test_mass_balance_holds(solved):
    """Pi_reduced Q == -demand at junctions (a model equality constraint)."""
    wdn, result = solved
    residual = wdn.M.Pi_reduced @ result.flows + wdn.junction_demand_profile
    assert np.linalg.norm(residual) < 1e-4


def test_reservoir_head_fixed(solved):
    """Reservoir head is pinned to its elevation."""
    wdn, result = solved
    r = wdn.raw.reservoir_index[0]
    assert np.allclose(result.heads[r, :], wdn.raw.node_elevations[r], atol=1e-4)


def test_pump_flow_within_bounds(solved):
    wdn, result = solved
    pump_flows = wdn.M.Lambda @ result.flows
    assert np.all(pump_flows >= -1e-4)
    assert np.all(pump_flows <= wdn.pump.max_flow[:, None] + 1e-3)


def test_pump_off_implies_zero_flow(solved):
    """When OnOff == 0 the pump flow must be (near) zero."""
    wdn, result = solved
    pump_flows = wdn.M.Lambda @ result.flows
    off = np.abs(result.onoff) < 1e-6
    assert np.all(np.abs(pump_flows[off]) < 1e-3)


def test_power_matches_epanet_aggregate(solved):
    """Total true FSP power should be in the right ballpark vs EPANET."""
    wdn, result = solved
    report = validate(wdn, result)
    rel = abs(report.pump_power_opt_true.sum() - report.pump_power_epanet.sum()) / abs(
        report.pump_power_epanet.sum()
    )
    assert rel < 0.05
