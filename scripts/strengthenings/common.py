"""Small exact helpers shared by the strengthening scripts (binary variables)."""

from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import linprog

MAPS = list(itertools.product((0, 1), repeat=2))  # r = (r(0), r(1)) : {0,1} -> {0,1}


def lp(c, A_eq, b_eq, A_ub=None, b_ub=None, sense="max"):
    """Solve min/max c^T z s.t. A_eq z = b_eq, A_ub z <= b_ub, z >= 0 (HiGHS)."""
    c = np.asarray(c, float)
    res = linprog(
        -c if sense == "max" else c,
        A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
        bounds=(0, None), method="highs",
    )
    if res.status != 0:
        return None, None
    val = -res.fun if sense == "max" else res.fun
    return float(val), res.x


def interval(c, A_eq, b_eq, A_ub=None, b_ub=None):
    lo, xlo = lp(c, A_eq, b_eq, A_ub, b_ub, "min")
    hi, xhi = lp(c, A_eq, b_eq, A_ub, b_ub, "max")
    return lo, hi, xlo, xhi
