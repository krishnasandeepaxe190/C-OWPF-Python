"""Tests for pump-schedule optimization (savings vs EPANET's own operation)."""
import numpy as np
import pytest

from owf import SolverConfig, setup, validate_schedule
from owf.warmstart import optimize_schedule, price_threshold_schedules, true_energy_cost


@pytest.fixture(scope="module")
def optimized_net2():
    wdn = setup(SolverConfig(net_num=36))
    best, info = optimize_schedule(wdn, inner_iter=12, polish=False, verbose=False)
    return wdn, best, info


def test_price_threshold_family_shape():
    wdn = setup(SolverConfig(net_num=8))
    scheds = price_threshold_schedules(wdn)
    assert "all_on" in scheds
    for s in scheds.values():
        assert s.shape == (wdn.n_pumps, wdn.time)
        assert set(np.unique(s)) <= {0.0, 1.0}


def test_true_cost_positive():
    wdn = setup(SolverConfig(net_num=8))
    from owf import solve_owf
    r = solve_owf(wdn)
    assert true_energy_cost(wdn, r.flows) > 0


def test_net2_optimization_saves(optimized_net2):
    """The optimizer should beat EPANET's operation on Net2, feasibly."""
    _, _, info = optimized_net2
    assert info["best_cost"] < info["baseline_cost"]
    assert info["savings_pct"] > 10.0
    assert info["best_slack"] < 1.0


def test_net2_optimized_schedule_feasible_in_epanet(optimized_net2):
    """Re-simulating the optimized schedule in EPANET gives positive pressures
    and in-bounds tanks -- i.e. the savings are physically real."""
    wdn, best, _ = optimized_net2
    rep = validate_schedule(wdn, best)
    junction_head = wdn.M.Kappa @ rep.heads_epanet
    elev = (wdn.M.Kappa @ wdn.raw.node_elevations[:, None]).ravel()
    assert (junction_head - elev[:, None]).min() > 0.0
    tank_head = wdn.M.Tau.T @ rep.heads_epanet
    for i in range(wdn.n_tanks):
        assert tank_head[i].min() >= wdn.tank.min_head[i] - 1.0
        assert tank_head[i].max() <= wdn.tank.max_head[i] + 1.0


def test_availability_respected():
    """Net3's Lake pump must stay off outside its availability window."""
    wdn = setup(SolverConfig(net_num=97))
    from owf.warmstart import _apply_availability
    sched = np.ones((wdn.n_pumps, wdn.time))
    out = _apply_availability(wdn, sched)
    for pos, (start, end) in wdn.pump_avail.items():
        for t in range(wdn.time):
            if not (start <= t < end):
                assert out[pos, t] == 0.0
