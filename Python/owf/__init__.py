"""FSP Optimal Water Flow (OWF) — Python translation of the MATLAB C-OWPF WDN solver.

Solves the water-side Optimal Water Flow problem for fixed-speed pumps (FSPs) via
a successive linear-approximation loop, each iteration a MILP solved with HiGHS
through CVXPY. EPANET (.inp parsing and hydraulic ground truth) is handled by epyt.
"""
from .config import NETWORKS, SolverConfig
from .network import WDN, setup
from .solver import OWFResult, solve_owf
from .validation import (
    ScheduleValidationReport,
    ValidationReport,
    validate,
    validate_schedule,
)
from .warmstart import (
    candidate_schedules,
    solve_fixed_schedule,
    solve_from_epanet,
    solve_multistart,
    solve_warmstart,
)

# plots imports matplotlib lazily inside its functions, but the module itself is
# only pulled in on demand to keep matplotlib an optional dependency.
def __getattr__(name):  # pragma: no cover - thin lazy re-export
    if name in {"plot_all", "plot_convergence", "plot_flows", "plot_heads",
                "plot_pump_schedule", "plot_error_summary"}:
        from . import plots
        return getattr(plots, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "SolverConfig",
    "NETWORKS",
    "WDN",
    "setup",
    "OWFResult",
    "solve_owf",
    "ValidationReport",
    "validate",
    "ScheduleValidationReport",
    "validate_schedule",
    "candidate_schedules",
    "solve_multistart",
    "solve_fixed_schedule",
    "solve_from_epanet",
    "solve_warmstart",
    "plot_all",
    "plot_convergence",
    "plot_flows",
    "plot_heads",
    "plot_pump_schedule",
    "plot_error_summary",
]
