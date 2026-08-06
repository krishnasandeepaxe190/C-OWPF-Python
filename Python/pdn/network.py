"""PDN object: a feeder's linear voltage model plus its DER / limit settings.

``PDN.build(key)`` gives a ready-to-use power network -- the LinDistFlow model,
per-bus PV inverter ratings, a 24-hour normalized solar-availability profile,
shunt caps, and voltage limits -- everything the coupled LP needs on the power
side.  Nothing here knows about water; the pump->bus coupling lives in
``coupled/`` so the same PDN can host different water pumps.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .feeders import FEEDERS, FEEDER_CHOICES
from .lindistflow import LinDistModel, build_lindist

# Typical clear-day normalized solar availability (fraction of inverter kW-rating
# available as *active* PV power), hour 0..23.  Zero overnight, bell-shaped, peak
# ~1.0 near solar noon.  Scaled per PV bus by its inverter rating.
SOLAR_PROFILE_24 = np.array([
    0.00, 0.00, 0.00, 0.00, 0.00, 0.02, 0.08, 0.20,
    0.38, 0.58, 0.75, 0.88, 0.97, 1.00, 0.97, 0.88,
    0.75, 0.58, 0.38, 0.20, 0.08, 0.02, 0.00, 0.00,
])

DEFAULT_VMIN = 0.95   # pu, ANSI C84.1 range-A lower
DEFAULT_VMAX = 1.05   # pu, ANSI C84.1 range-A upper


@dataclass
class PDN:
    """A distribution feeder ready for the coupled optimization (all pu)."""

    key: str
    label: str
    model: LinDistModel
    pv_rating: np.ndarray            # (N,) inverter apparent-power rating S_pv (pu); 0 where no PV
    solar_profile: np.ndarray        # (T,) fraction of rating available as active PV power
    vmin: float = DEFAULT_VMIN
    vmax: float = DEFAULT_VMAX
    orig_id: list = field(default_factory=list)

    @property
    def N(self) -> int:
        return self.model.N

    @property
    def pv_buses(self) -> np.ndarray:
        return np.where(self.pv_rating > 0)[0]

    def limit_pv(self, k: int) -> "PDN":
        """Keep only the ``k`` largest-rated PV sites active (zero the rest)."""
        pv = self.pv_buses
        if 0 <= k < pv.size:
            keep = pv[np.argsort(-self.pv_rating[pv])[:k]]
            mask = np.zeros(self.N, bool)
            mask[keep] = True
            self.pv_rating = np.where(mask, self.pv_rating, 0.0)
        return self

    def solar(self, T: int) -> np.ndarray:
        """Solar availability tiled/truncated to a T-hour horizon."""
        prof = self.solar_profile
        if T == len(prof):
            return prof.copy()
        reps = int(np.ceil(T / len(prof)))
        return np.tile(prof, reps)[:T]

    @staticmethod
    def build(key: str, pv_sizing: float = 1.2, vmin: float = DEFAULT_VMIN,
              vmax: float = DEFAULT_VMAX) -> "PDN":
        """Construct a PDN from the feeder registry.

        Each PV inverter's apparent-power rating is ``pv_sizing`` x the bus's
        nominal active load (the paper places PV on load-hosting buses).  Sizing
        strictly to load -- not a fixed per-unit floor -- keeps PV proportional to
        the feeder regardless of its VA base, avoiding artificial reverse-flow
        overvoltage on high-base feeders (e.g. IEEE-33 at 100 MVA).
        """
        if key not in FEEDERS:
            raise ValueError(f"Unknown feeder '{key}'. Available: {FEEDER_CHOICES}")
        f = FEEDERS[key]
        model = build_lindist(f)
        pv_mask = model.pv_mask
        rating = np.zeros(model.N)
        nameplate = f.get("pv_nameplate")
        if nameplate:
            # Feeders with explicit PV nameplate ratings (pu): inverter apparent
            # rating S = pv_sizing * nameplate (so pv_sizing is the S/Pmax factor).
            for idx, cap in nameplate.items():
                rating[int(idx)] = pv_sizing * float(cap)
        else:
            rating[pv_mask] = pv_sizing * model.p_load[pv_mask]
        return PDN(
            key=key, label=f["label"], model=model, pv_rating=rating,
            solar_profile=SOLAR_PROFILE_24.copy(), vmin=vmin, vmax=vmax,
            orig_id=list(f.get("orig_id", [])),
        )


# Convenience registry mirror so callers can do PDN_SPECS[key] like the water side.
PDN_SPECS = {k: FEEDERS[k]["label"] for k in FEEDER_CHOICES}
