"""FSP Optimal Water Flow (OWF) — Python translation of the MATLAB C-OWPF WDN solver.

Solves the water-side Optimal Water Flow problem for fixed-speed pumps (FSPs) via
a successive linear-approximation loop, each iteration a MILP solved with HiGHS
through CVXPY. EPANET (.inp parsing and hydraulic ground truth) is handled by epyt.
"""
from .config import NETWORKS, SolverConfig
from .network import WDN, setup
from .solver import OWFResult, solve_owf
from .validation import ValidationReport, validate

__all__ = [
    "SolverConfig",
    "NETWORKS",
    "WDN",
    "setup",
    "OWFResult",
    "solve_owf",
    "ValidationReport",
    "validate",
]
