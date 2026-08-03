"""Smoke tests for the diagnostic plots."""
import pytest

from owf import SolverConfig, setup, solve_owf, validate_schedule

plt = pytest.importorskip("matplotlib")
from owf.plots import (  # noqa: E402
    plot_all,
    plot_convergence,
    plot_error_summary,
    plot_flows,
    plot_heads,
    plot_pump_schedule,
)


@pytest.fixture(scope="module")
def solved():
    wdn = setup(SolverConfig(net_num=8))
    result = solve_owf(wdn)
    report = validate_schedule(wdn, result)
    return wdn, result, report


def test_individual_plots(solved):
    wdn, result, report = solved
    assert plot_convergence(result) is not None
    assert plot_pump_schedule(wdn, result) is not None
    assert plot_flows(wdn, result, report) is not None
    assert plot_heads(wdn, result, report) is not None
    assert plot_error_summary(wdn, result, report) is not None


def test_plot_all_writes_files(solved, tmp_path):
    wdn, result, report = solved
    paths = plot_all(wdn, result, report, outdir=tmp_path)
    assert len(paths) == 5
    for p in paths:
        assert p.exists() and p.stat().st_size > 0
