"""Validate the OWF solution against EPANET hydraulics (ports the tail of
Main_OWF_IEEE_ACCESS.m): error norms on heads, pipe flows and pump flows, plus
a true-power comparison.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .epanet_io import run_epanet
from .network import WDN
from .solver import OWFResult, _true_pump_power


@dataclass
class ValidationReport:
    error_heads: float
    error_pipe_flows: float
    error_pump_flows: float
    pump_power_epanet: np.ndarray
    pump_power_opt_true: np.ndarray
    heads_epanet: np.ndarray
    flows_epanet: np.ndarray

    def summary(self) -> str:
        return (
            "EPANET validation (Euclidean error norms):\n"
            f"  heads      : {self.error_heads:.6g}\n"
            f"  pipe flows : {self.error_pipe_flows:.6g}\n"
            f"  pump flows : {self.error_pump_flows:.6g}\n"
            f"  total pump power (EPANET) : {self.pump_power_epanet.sum():.4f} kW\n"
            f"  total pump power (opt,true): {self.pump_power_opt_true.sum():.4f} kW"
        )


def validate(wdn: WDN, result: OWFResult) -> ValidationReport:
    flows_ep, heads_ep, _ = run_epanet(wdn.raw)
    T = wdn.time
    steps = min(T, flows_ep.shape[0])

    flows_ep_t = flows_ep[:steps, :].T   # (L x steps)
    heads_ep_t = heads_ep[:steps, :].T   # (N x steps)

    flows_opt = result.flows[:, :steps]
    heads_opt = result.heads[:, :steps]

    pipe_ep = wdn.M.Pi_prime @ flows_ep_t
    pump_ep = wdn.M.Lambda @ flows_ep_t
    pipe_opt = wdn.M.Pi_prime @ flows_opt
    pump_opt = wdn.M.Lambda @ flows_opt

    return ValidationReport(
        error_heads=float(np.linalg.norm(heads_ep_t - heads_opt)),
        error_pipe_flows=float(np.linalg.norm(pipe_ep - pipe_opt)),
        error_pump_flows=float(np.linalg.norm(pump_ep - pump_opt)),
        pump_power_epanet=_true_pump_power(wdn, flows_ep_t),
        pump_power_opt_true=_true_pump_power(wdn, flows_opt),
        heads_epanet=heads_ep_t,
        flows_epanet=flows_ep_t,
    )
