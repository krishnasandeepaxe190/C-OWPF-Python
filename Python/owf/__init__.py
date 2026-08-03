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
    solve_multistart,
    solve_warmstart,
)

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
    "solve_warmstart",
]
