"""Two SCE feeders transcribed from the paper's Table I (Fig. 5, 47-bus) and
Table II (Fig. 6, 56/57-bus).

Loads are peak APPARENT power in MVA at power factor 0.9 (P = 0.9 S, Q = 0.436 S).
PV and shunt caps use the tables' explicit nameplate ratings.  Tie lines (Fig. 5:
1-12, 1-30; Fig. 6: 1-32) are omitted for the base radial configuration, and
zero-impedance jumpers are floored to a tiny finite impedance.  Fig. 5's bus-1
load is modelled by adding a virtual substation slack (node 0 -> bus 1).

NOTE: transcribed from the table image -- spot-check values against the paper.
Both reproduce a full nonlinear Z-bus solve to < 0.01 pu (see tests).
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

_PF = 0.9
_SIN = float(np.sqrt(1 - _PF ** 2))     # 0.4359

# ---- Fig. 5 : 47-bus, Vbase 12.35 kV, Sbase 1 MVA -----------------------------
_F5_V, _F5_S = 12.35e3, 1e6
_F5_EDGES = [
 (1,2,0.259,0.808),(2,13,0,0),(2,3,0.031,0.092),(3,4,0.046,0.092),(3,14,0.092,0.031),
 (3,15,0.214,0.046),(4,20,0.336,0.061),(4,5,0.107,0.183),(5,26,0.061,0.015),(5,6,0.015,0.031),
 (6,27,0.168,0.061),(6,7,0.031,0.046),(7,32,0.076,0.015),(7,8,0.015,0.015),(8,40,0.046,0.015),
 (8,39,0.244,0.046),(8,41,0.107,0.031),(8,35,0.076,0.015),(8,9,0.031,0.031),(9,10,0.015,0.015),
 (9,42,0.153,0.046),(10,11,0.107,0.076),(10,46,0.229,0.122),(11,47,0.031,0.015),(11,12,0.076,0.046),
 (15,18,0.046,0.015),(15,16,0.107,0.015),(16,17,0,0),(18,19,0,0),(20,21,0.122,0.092),
 (20,25,0.214,0.046),(21,24,0,0),(21,22,0.198,0.046),(22,23,0,0),(27,31,0.046,0.015),
 (27,28,0.107,0.031),(28,29,0.107,0.031),(29,30,0.061,0.015),(32,33,0.046,0.015),(33,34,0.031,0.015),
 (35,36,0.076,0.015),(35,37,0.076,0.046),(35,38,0.107,0.015),(42,43,0.061,0.015),(43,44,0.061,0.015),
 (43,45,0.061,0.015),
]
_F5_LOADS = {1:30,11:0.67,12:0.45,14:0.89,16:0.07,18:0.67,21:0.45,22:2.23,25:0.45,26:0.2,
 28:0.13,29:0.13,30:0.2,31:0.07,32:0.13,33:0.27,34:0.2,36:0.27,38:0.45,39:1.34,
 40:0.13,41:0.67,42:0.13,44:0.45,45:0.2,46:0.45}
_F5_PV = {13:1.5,17:0.4,19:1.5,23:1.0,24:2.0}      # bus : nameplate MW
_F5_CAPS = {1:6.0,3:1.2,37:1.8,47:1.8}             # bus : Mvar

# ---- Fig. 6 : 57-bus, Vbase 12 kV, Sbase 1 MVA --------------------------------
_F6_V, _F6_S = 12e3, 1e6
_F6_EDGES = [
 (1,2,0.160,0.388),(2,3,0.824,0.315),(2,4,0.144,0.349),(4,5,1.026,0.421),(4,6,0.741,0.466),
 (4,7,0.528,0.468),(7,8,0.358,0.314),(8,9,2.032,0.798),(8,10,0.502,0.441),(10,11,0.372,0.327),
 (11,12,1.431,0.999),(11,13,0.429,0.377),(13,14,0.671,0.257),(13,15,0.457,0.401),(15,16,1.008,0.385),
 (15,17,0.153,0.134),(17,18,0.971,0.722),(18,19,1.885,0.721),(4,20,0.138,0.334),(19,58,0.09,0.2),
 (20,21,0.251,0.096),(21,22,1.818,0.695),(20,23,0.225,0.542),(23,24,0.127,0.028),(23,25,0.284,0.687),
 (25,26,0.171,0.414),(26,27,0.414,0.386),(27,28,0.210,0.196),(28,29,0.395,0.369),(29,30,0.248,0.232),
 (30,31,0.279,0.260),(26,32,0.205,0.495),(32,33,0.263,0.073),(32,34,0.071,0.171),(34,35,0.625,0.273),
 (34,36,0.510,0.209),(36,37,2.018,0.829),(34,38,1.062,0.406),(38,39,0.610,0.238),(39,40,2.349,0.964),
 (34,41,0.115,0.278),(41,42,0.159,0.384),(42,43,0.934,0.383),(42,44,0.506,0.163),(42,45,0.095,0.195),
 (42,46,1.915,0.769),(41,47,0.157,0.379),(47,48,1.641,0.670),(47,49,0.081,0.196),(49,50,1.727,0.709),
 (49,51,0.112,0.270),(51,52,0.674,0.275),(51,53,0.070,0.170),(53,54,2.041,0.780),(53,55,0.813,0.334),
 (53,56,0.141,0.340),(53,57,0.1,0.3),
]
_F6_LOADS = {3:0.057,5:0.121,6:0.049,7:0.053,8:0.047,9:0.068,10:0.048,11:0.067,12:0.094,14:0.057,
 16:0.053,17:0.057,18:0.112,19:0.087,22:0.063,24:0.135,25:0.100,27:0.048,28:0.038,29:0.044,
 31:0.053,32:0.223,33:0.123,34:0.067,35:0.094,36:0.097,37:0.281,38:0.117,39:0.131,40:0.030,
 41:0.046,42:0.054,43:0.083,44:0.057,46:0.134,47:0.045,48:0.196,50:0.045,52:0.315,54:0.061,
 55:0.055,56:0.130}
_F6_PV = {45:5.0}
_F6_CAPS = {19:0.6,21:0.6,30:0.6,53:0.6}


def _build(label, desc, edges, loads, pv_mw, caps_mvar, vbase, sbase,
           v0_kv, virtual_slack_load=None) -> dict:
    zbase = vbase ** 2 / sbase
    edges = list(edges)
    if virtual_slack_load is not None:
        edges = [(0, virtual_slack_load, 0.01, 0.01)] + edges

    to_set = {e[1] for e in edges}
    slack = next(e[0] for e in edges if e[0] not in to_set)
    adj = defaultdict(list)
    for f, t, r, x in edges:
        adj[f].append((t, r, x))
    idx = {slack: 0}; parent_of = {}; rx = {}
    q = deque([slack])
    while q:
        u = q.popleft()
        for (t, r, x) in adj[u]:
            if t in idx:
                continue                       # tie line -> drop
            if r == 0 and x == 0:
                r, x = 1e-3, 1e-3              # jumper floor
            idx[t] = len(idx); parent_of[t] = u; rx[t] = (r, x)
            q.append(t)
    N = len(idx) - 1

    parent = [0]*N; r = [0.0]*N; x = [0.0]*N; p = [0.0]*N; qq = [0.0]*N
    pv = [0]*N; orig = [0]*N; caps = {}; nameplate = {}
    for bus, i in idx.items():
        if i == 0:
            continue
        j = i - 1
        parent[j] = idx[parent_of[bus]]
        r[j] = round(rx[bus][0]/zbase, 8); x[j] = round(rx[bus][1]/zbase, 8)
        orig[j] = int(bus)
        if bus in loads:
            S = loads[bus]
            p[j] = round(_PF*S*1e6/sbase, 8); qq[j] = round(_SIN*S*1e6/sbase, 8)
        if bus in pv_mw:
            pv[j] = 1; nameplate[j] = round(pv_mw[bus]*1e6/sbase, 8)
        if bus in caps_mvar:
            caps[j] = round(caps_mvar[bus]*1e6/sbase, 8)
    return dict(label=label, description=desc, N=N, SBase=float(sbase),
                VBase=round(float(vbase), 4), ZBase=round(float(zbase), 6),
                v0_sq=round((v0_kv*1e3/vbase)**2, 8), slack_id=int(slack),
                parent=parent, r=r, x=x, p=p, q=qq, pv=pv, caps=caps,
                pv_nameplate=nameplate, orig_id=orig)


PAPER_FEEDERS = {
    "sce47": _build("SCE-47",
                    "SCE 47-bus feeder (Table I, Fig. 5; 12.35 kV, 1 MVA), MVA loads "
                    "@pf0.9, explicit PV/cap nameplates.",
                    _F5_EDGES, _F5_LOADS, _F5_PV, _F5_CAPS, _F5_V, _F5_S, 12.35,
                    virtual_slack_load=1),
    "sce56": _build("SCE-56",
                    "SCE 56-bus feeder (Table II, Fig. 6; 12 kV, 1 MVA), MVA loads "
                    "@pf0.9, 5 MW PV at bus 45, four 0.6 MVAr caps.",
                    _F6_EDGES, _F6_LOADS, _F6_PV, _F6_CAPS, _F6_V, _F6_S, 12.0),
}
