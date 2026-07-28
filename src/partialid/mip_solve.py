"""MIP path for support_bound(k): at most k positive latent types."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix, eye, hstack

from .backends import SolveResult
from .formulate import LPData


@dataclass
class MIPBounds:
    lower: SolveResult
    upper: SolveResult


def solve_bounds_mip(lp: LPData) -> MIPBounds:
    """Min/max c'q with |supp(q)| ≤ k via binary indicators (HiGHS MILP)."""
    if lp.support_k is None:
        raise ValueError("solve_bounds_mip requires support_k on LPData")
    k = int(lp.support_k)
    n = len(lp.c)
    # Decision vector x = [q (n), z (n)]; z binary, q ≤ z, 1'z ≤ k
    c_base = np.concatenate([lp.c, np.zeros(n)])

    # Bounds: q≥0 free upper; z in [0,1]; forbidden types: q=0,z=0
    lb = np.zeros(2 * n)
    ub = np.full(2 * n, np.inf)
    ub[n:] = 1.0
    for t in range(n):
        if not lp.allowed_types[t]:
            ub[t] = 0.0
            ub[n + t] = 0.0
    bounds = Bounds(lb, ub)
    integrality = np.concatenate([np.zeros(n), np.ones(n)])

    constraints = []
    # A_eq q = b_eq
    if lp.A_eq is not None and lp.A_eq.shape[0] > 0:
        Aeq = hstack([lp.A_eq.tocsr(), csr_matrix((lp.A_eq.shape[0], n))])
        constraints.append(LinearConstraint(Aeq, lp.b_eq, lp.b_eq))
    # A_ub q ≤ b_ub
    if lp.A_ub is not None and lp.b_ub is not None:
        Aub = hstack([lp.A_ub.tocsr(), csr_matrix((lp.A_ub.shape[0], n))])
        constraints.append(LinearConstraint(Aub, np.full(len(lp.b_ub), -np.inf), lp.b_ub))
    # q - z ≤ 0
    QZ = hstack([eye(n, format="csr"), -eye(n, format="csr")])
    constraints.append(LinearConstraint(QZ, np.full(n, -np.inf), np.zeros(n)))
    # 1'z ≤ k
    zsum = csr_matrix(
        (np.ones(n), (np.zeros(n, dtype=int), np.arange(n, 2 * n))),
        shape=(1, 2 * n),
    )
    constraints.append(LinearConstraint(zsum, -np.inf, float(k)))

    def _one(maximize: bool) -> SolveResult:
        c = -c_base if maximize else c_base
        res = milp(
            c=c,
            integrality=integrality,
            bounds=bounds,
            constraints=constraints,
            options={"disp": False},
        )
        if not res.success or res.x is None:
            return SolveResult(
                value=float("nan"),
                q=np.zeros(n),
                success=False,
                message=res.message,
            )
        q = np.asarray(res.x[:n], dtype=float)
        val = float(lp.c @ q)
        return SolveResult(value=val, q=q, success=True, message=res.message)

    return MIPBounds(lower=_one(False), upper=_one(True))
