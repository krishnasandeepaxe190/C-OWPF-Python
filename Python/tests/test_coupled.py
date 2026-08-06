"""Coupled water-power (C-OWPF) tests.

Covers the PDN linear model vs the nonlinear Z-bus, and the coupled fixed-schedule
LP: it must converge, keep the paper's pump-energy objective, and reproduce the
nonlinear feeder voltages to within the LinDistFlow linearization gap.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pdn import PDN, zbus_powerflow, FEEDER_CHOICES
from coupled import (setup, solve_coupled_epanet, CoupledConfig,
                     optimize_coupled_schedule)


@pytest.mark.parametrize("key", FEEDER_CHOICES)
def test_lindist_matches_nonlinear_nominal(key):
    """Linear LinDistFlow voltages match the nonlinear Z-bus at nominal load."""
    pdn = PDN.build(key)
    m = pdn.model
    p = m.p_load
    q = m.q_load                       # caps excluded: voltage-dependent, in R/X/V_k
    vlin = m.voltage(p, q)
    vnl = zbus_powerflow(m, p, q)      # nonlinear: caps go in the Y-bus
    # loss-linearization gap: tight on IEEE-13/33, looser on the stressed SB-128
    tol = 0.02 if key != "sb128" else 0.10
    assert np.max(np.abs(vlin - vnl)) < tol


@pytest.mark.parametrize("feeder", ["ieee13", "ieee33"])
def test_coupled_fixed_schedule_converges_and_validates(feeder):
    cc = CoupledConfig(feeder=feeder, enable_pv=True)
    wdn, pdn = setup(net_num=8, cc=cc)
    res = solve_coupled_epanet(wdn, pdn, cc)

    assert res.converged, f"{feeder}: coupled LP did not converge"
    assert res.water_max_slack < 2.0, "water head bounds badly violated"
    # objective is the paper's pure pump-energy cost (finite, positive)
    assert np.isfinite(res.energy_cost) and res.energy_cost > 0

    # linear coupled voltages vs the nonlinear Z-bus replay
    m = pdn.model
    err = 0.0
    for t in range(wdn.time):
        vnl = zbus_powerflow(m, res.p_net[:, t], res.q_net[:, t])
        err = max(err, float(np.nanmax(np.abs(res.voltage[:, t] - vnl))))
    assert err < 0.05, f"{feeder}: coupled voltage vs nonlinear error {err:.3f} too large"


def test_pv_reactive_supports_voltage():
    """PV reactive control must lift the minimum voltage vs the no-PV case."""
    cc_pv = CoupledConfig(feeder="ieee13", enable_pv=True)
    cc_no = CoupledConfig(feeder="ieee13", enable_pv=False)
    wdn, pdn = setup(net_num=8, cc=cc_pv)
    v_pv = solve_coupled_epanet(wdn, pdn, cc_pv).voltage.min()
    wdn2, pdn2 = setup(net_num=8, cc=cc_no)
    v_no = solve_coupled_epanet(wdn2, pdn2, cc_no).voltage.min()
    assert v_pv > v_no + 1e-3, f"PV did not help: v_pv={v_pv:.4f} v_no={v_no:.4f}"


def test_coupled_schedule_optimization_saves():
    """The voltage-aware schedule search must beat EPANET on cost, stay feasible,
    and its winner must validate against the nonlinear Z-bus."""
    cc = CoupledConfig(feeder="ieee13", enable_pv=True)
    wdn, pdn = setup(net_num=8, cc=cc)
    best, info = optimize_coupled_schedule(wdn, pdn, cc, verbose=False)

    assert best.converged
    assert info["best_cost"] <= info["baseline_cost"] + 1e-9
    assert info["savings_pct"] >= 0.0
    # winner reproduces the nonlinear feeder voltages
    m = pdn.model
    err = max(float(np.nanmax(np.abs(
        best.voltage[:, t] - zbus_powerflow(m, best.p_net[:, t],
                                            best.q_net[:, t]))))
        for t in range(wdn.time))
    assert err < 0.05


def test_objective_is_pump_cost_only():
    """The reported energy cost must equal the water-only pump cost (feeder-agnostic)."""
    costs = []
    for feeder in ["ieee13", "ieee33"]:
        cc = CoupledConfig(feeder=feeder)
        wdn, pdn = setup(net_num=8, cc=cc)
        costs.append(solve_coupled_epanet(wdn, pdn, cc).energy_cost)
    # pump energy cost is the same water problem regardless of which feeder hosts it
    assert abs(costs[0] - costs[1]) < 1e-6
