"""Fiber-separable Solve (Move 1) — no LP when constraints don't couple cells."""

from __future__ import annotations

import numpy as np

from .formulate import LPData


def solve_bounds_fiber(lp: LPData) -> tuple[float, float, np.ndarray, np.ndarray]:
    """Per-cell min/max of c on the allowed fiber; width = sum P(v) Delta_v.

    Returns lower, upper, q_lower witness, q_upper witness.
    """
    ts = lp.type_space
    n = ts.n_types
    q_lo = np.zeros(n)
    q_hi = np.zeros(n)
    lower = 0.0
    upper = 0.0

    for i, cell in enumerate(lp.cells):
        mass = lp.P_vec[i]
        if mass <= 0:
            continue
        members = [t for t in ts.fiber[cell] if lp.allowed_types[t]]
        if not members:
            raise ValueError(f"empty allowed fiber for cell {cell} (assumption vs data)")
        # choose type minimizing / maximizing c within fiber
        t_min = min(members, key=lambda t: lp.c[t])
        t_max = max(members, key=lambda t: lp.c[t])
        c_min = lp.c[t_min]
        c_max = lp.c[t_max]
        lower += mass * c_min
        upper += mass * c_max
        q_lo[t_min] += mass
        q_hi[t_max] += mass

    return lower, upper, q_lo, q_hi


def is_fiber_separable(lp: LPData) -> bool:
    """True when Move-1 per-cell scan is exact."""
    if lp.method_hint in ("mip", "convex", "fractional"):
        return False
    if getattr(lp, "c_den", None) is not None:
        return False
    if lp.A_ub is not None:
        return False
    if lp.support_k is not None or lp.f_sensitivity is not None:
        return False
    if any(
        "instrument LP" in n or "root-block lift" in n or "pinning failed" in n
        for n in lp.notes
    ):
        return False
    # An explicit empty model is the equality 0 = 1, which a per-cell scan
    # never sees (it has no observational cells).
    if any(str(n).startswith("infeasible:") for n in lp.notes):
        return False
    # Stacked / block-diagonal LPs concatenate several type spaces; a
    # per-cell scan on the first block is not the joint optimum.
    if any("stacked" in n for n in lp.notes):
        return False
    ts = getattr(lp, "type_space", None)
    if ts is not None and len(np.asarray(lp.c)) != int(ts.n_types):
        return False
    return True
