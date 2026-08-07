"""Pressure-reducing valve (PRV) tests.

Uses a self-contained 8-node network with a PRV (junction 6 -> 9, setting 20 psi),
written to a temp file at runtime -- no network fixture is committed. The PRV
regulates its downstream junction to h_set = E_down + P_set = 720 + 20*2.307 =
766.1 ft, matching EPANET's own three-state PRV logic.
"""
import numpy as np
import pytest

from owf import config as cfg
from owf.config import NetworkSpec, SolverConfig
from owf.network import setup
from owf.solver import solve_owf
from owf.warmstart import epanet_default_onoff
from owf.validation import validate_schedule

PRV_INP = """[TITLE]

[JUNCTIONS]
 2   700.0
 3   710.0
 4   700.0
 5   650.0
 6   700.0
 7   700.0
 9   720.0
 10  720.0

[RESERVOIRS]
 1   700.0

[TANKS]
 8   830.0   10.0   0.0   20.0   60.0   0.0

[PIPES]
 1   3   2    3000.0   14.0   100.0   0.0
 2   3   7    5000.0   12.0   100.0   0.0
 3   3   4    5000.0    8.0   100.0   0.0
 4   4   6    5000.0    8.0   100.0   0.0
 5   6   7    5000.0    8.0   100.0   0.0
 7   4   5    5000.0    6.0   100.0   0.0
 8   5   6    7000.0    6.0   100.0   0.0
 6   7   8    7000.0   10.0   100.0   0.0
 11  9   10   1000.0   12.0   100.0   0.0

[PUMPS]
 9   1   2   HEAD 2

[VALVES]
 10   6   9   12.0   PRV   20.0   10.0

[DEMANDS]
 3   150.0   2
 4   150.0   2
 5   200.0   2
 6   150.0   2
 9   150.0   2
 10  200.0   2

[PATTERNS]
 1   0.8 0.8 0.7 0.6 0.6 0.5
 1   0.6 0.7 0.7 0.9 0.9 1.0
 2   1.0 0.8 0.7 0.6 0.6 0.5
 2   0.6 0.7 0.7 0.9 0.9 1.0

[CURVES]
 1   0.0    0.0
 1   300.0  35.94
 1   700.0  67.09
 1   1200.0 81.0
 1   1500.0 75.99
 1   2000.0 45.40
 1   2400.0 1.0
 2   1200.0 200.0

[CONTROLS]
 LINK 9 0.0 AT CLOCKTIME 0:00:00
 LINK 9 1.0 AT CLOCKTIME 1:00:00
 LINK 9 0.0 AT CLOCKTIME 2:00:00
 LINK 9 1.0 AT CLOCKTIME 3:00:00
 LINK 9 0.0 AT CLOCKTIME 4:00:00
 LINK 9 1.0 AT CLOCKTIME 7:00:00
 LINK 9 0.0 AT CLOCKTIME 9:00:00

[TIMES]
 DURATION            11:00:00
 HYDRAULIC TIMESTEP  1:00:00
 PATTERN TIMESTEP    1:00:00
 START CLOCKTIME     0:00:00

[OPTIONS]
 UNITS      GPM
 PRESSURE   PSI
 HEADLOSS   H-W
 UNBALANCED CONTINUE 10
 TRIALS     40
 ACCURACY   0.001

[COORDINATES]
 2  2212 7492
 3  3185 7492
 4  3215 6715
 5  3254 5909
 6  4424 6735
 7  4385 7512
 9  6007 6755
 10 6892 6794
 1  1307 7522
 8  6853 7522

[END]
"""


@pytest.fixture
def prv_net(tmp_path):
    p = tmp_path / "prv8.inp"
    p.write_text(PRV_INP)
    net = 9208
    cfg.NETWORKS[net] = NetworkSpec(net_num=net, name="prv8test", inp_relpath=str(p))
    yield net
    cfg.NETWORKS.pop(net, None)


def test_prv_setup(prv_net):
    """The PRV is parsed, excluded from the pipe set, and h_set is E_down + P_set."""
    w = setup(SolverConfig(net_num=prv_net, time=12))
    assert w.n_valves == 1
    assert w.M.valve_index.size == 1
    assert abs(w.valve_hset[0] - (720.0 + 20.0 * 2.30724939)) < 1e-3
    assert not (set(w.M.valve_index.tolist()) & set(w.M.pipe_index.tolist()))
    assert w.M.Pi_prime_valve.shape == (1, w.n_links)


def test_prv_regulates_and_matches_epanet(prv_net):
    """The PRV three-state model reproduces EPANET's downstream regulation."""
    w0 = setup(SolverConfig(net_num=prv_net, time=12))
    sched = epanet_default_onoff(w0)
    w = setup(SolverConfig(net_num=prv_net, time=12, fixed_schedule=sched,
                           soft_bounds=True, max_iter=25, penalty_growth=1.3,
                           feas_tol=1.0))
    r = solve_owf(w)
    assert r.flows is not None
    d9 = w.raw.node_name_id.index("9")
    rep = validate_schedule(w, r)
    # matches EPANET across the whole network
    assert rep.max_abs_head < 2.0
    # the PRV pins the downstream junction near its setpoint for several hours
    near_setpoint = np.abs(r.heads[d9] - w.valve_hset[0]) < 1.0
    assert near_setpoint.sum() >= 3
    # downstream head never exceeds the setpoint by more than a hair (regulation)
    assert float(np.max(r.heads[d9])) <= w.valve_hset[0] + 1.0
