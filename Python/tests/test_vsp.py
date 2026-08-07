"""Variable-speed pump (VSP) tests: FSP-equivalence, speed reduction, EPANET
feasibility, and that the FSP path is untouched when no pump is variable-speed."""
import numpy as np

from owf.config import SolverConfig
from owf.network import setup
from owf.solver import solve_owf, _build_model
from owf.epanet_io import simulate_with_schedule
from owf.warmstart import true_energy_cost


def _epanet_replay(w, r):
    pl = (w.raw.link_pump_index + 1).tolist()
    return simulate_with_schedule(w.raw.inp_path, pl, np.round(r.onoff), w.time,
                                  w.raw.n_nodes, w.raw.n_links, pump_speeds=r.speed)


def test_fsp_path_unchanged_without_vsp():
    """No VSP -> no Speed/WW variables; the fixed-speed model is preserved."""
    w = setup(SolverConfig(net_num=8, time=12))
    m = _build_model(w)
    assert m.Speed is None and m.WW is None and m.DPrime is None
    assert w.n_vsp == 0
    r = solve_owf(w)
    assert r.converged and r.speed is None


def test_vsp_speed1_equals_fsp():
    """A VSP pinned to omega == 1 reproduces the FSP solution exactly."""
    rf = solve_owf(setup(SolverConfig(net_num=8, time=12)))
    sched = np.round(rf.onoff)
    rv = solve_owf(setup(SolverConfig(net_num=8, time=12,
                                      vsp_pumps={"9": (1.0, 1.0)},
                                      fixed_schedule=sched)))
    assert rv.converged
    assert abs(rv.objective - rf.objective) < 1e-4


def test_vsp_reduces_cost_and_is_feasible():
    """Variable speed cuts cost vs FSP and stays feasible when replayed in EPANET."""
    rf = solve_owf(setup(SolverConfig(net_num=8, time=12)))
    w = setup(SolverConfig(net_num=8, time=12, vsp_pumps={"9": (0.8, 1.0)},
                           soft_bounds=True, damping=0.5, max_iter=80))
    r = solve_owf(w)
    assert r.converged and r.speed is not None
    on = r.onoff[0] > 0.5
    assert r.speed[0][on].max() <= 1.0 + 1e-6
    assert r.speed[0][on].min() >= 0.8 - 1e-6
    assert r.speed[0][on].mean() < 0.99                      # actually runs slower
    assert true_energy_cost(w, r.flows, r.speed) < true_energy_cost(w, rf.flows)
    hep, _ = _epanet_replay(w, r)
    ji = w.raw.junction_index
    min_pressure = float(np.min(hep[ji, :] - w.raw.node_elevations[ji, None]))
    assert min_pressure > 0                                   # hydraulically feasible


def test_epanet_setting_speed_is_read(tmp_path):
    """The baseline reads the pump relative speed EPANET applies (SETTING/SPEED)."""
    from owf.config import NETWORKS
    from owf.epanet_io import read_inp, epanet_pump_speeds
    src = str(NETWORKS[8].inp_path)
    txt = open(src, errors="ignore").read()
    # native 8-node runs at full speed on its running hours
    sp1 = epanet_pump_speeds(read_inp(src), 12)
    assert np.all(sp1 >= 0.999)
    # a relative speed of 0.85 is read back on running hours
    assert "HEAD 2\tSPEED 1" in txt
    p = tmp_path / "eight_085.inp"
    p.write_text(txt.replace("HEAD 2\tSPEED 1", "HEAD 2\tSPEED 0.85"))
    sp2 = epanet_pump_speeds(read_inp(str(p)), 12)
    on = sp2[0] < 0.999
    assert on.any() and np.allclose(sp2[0][on], 0.85, atol=1e-3)


def test_vsp_run_case_reports_savings():
    """End-to-end run_case with VSP reports a saving and a feasible EPANET replay."""
    from main_owf import run_case
    case, _, r = run_case(8, "direct", 1, 12, plot=False, outdir="outputs",
                          verbose=False, vsp={"9": (0.8, 1.0)})
    assert case.converged
    assert case.savings_pct > 5.0
    assert case.min_pressure > 0
    assert np.isfinite(case.max_dhead) and case.max_dhead < 10.0
