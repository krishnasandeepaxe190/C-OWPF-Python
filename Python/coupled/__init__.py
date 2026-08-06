"""Coupled Optimal Water-Power Flow (C-OWPF).

Ties the water OWF (``owf``) to a distribution feeder (``pdn``) through the pumps'
electrical load, co-optimizing the pump schedule with PV active/reactive dispatch
under voltage limits.  The schedule's binaries are the only integers; the feeder
layer is linear.
"""
from .config import CoupledConfig, LOAD_PROFILE_24
from .coupled_lp import solve_coupled, build_coupled_problem, CoupledResult, default_pump_buses
from .runner import setup, solve_coupled_schedule, solve_coupled_epanet
from .schedule import optimize_coupled_schedule
from .validation import validate_coupled, CoupledValidationReport, coupled_loss_kwh

__all__ = [
    "CoupledConfig", "LOAD_PROFILE_24", "solve_coupled", "build_coupled_problem",
    "CoupledResult", "default_pump_buses", "setup", "solve_coupled_schedule",
    "solve_coupled_epanet", "optimize_coupled_schedule", "validate_coupled",
    "CoupledValidationReport", "coupled_loss_kwh",
]
