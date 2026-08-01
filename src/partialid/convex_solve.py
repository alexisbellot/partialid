"""Convex sensitivity programs (anchored f-divergence; optional CVXPY).

Uses the same conditional / categorical propensity API as Tan MSM:
``π(·|r,pa)`` vs ``π₀(·|pa)``. Binary root treatment recovers the classical
Jin–Ren–Candès anchored f-sensitivity constraints.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .backends import SolveResult
from .formulate import LPData
from .sensitivity_budgets import rest_type_groups
from .types_canon import TypeSpace


def cvxpy_available() -> bool:
    try:
        import cvxpy  # noqa: F401

        return True
    except ImportError:
        return False


@dataclass
class ConvexBounds:
    lower: SolveResult
    upper: SolveResult


def _stratum_masses(q, ts: TypeSpace, allow: np.ndarray, treatment: str, pi0: dict):
    """Yield (mass_by_x dict of affine exprs, pi_pa) for each (r, pa) stratum."""
    parents = list(ts.diagram.parents(treatment))
    domain_x = list(ts.variables.domains[treatment])
    groups = rest_type_groups(ts, allow, treatment)
    for members in groups.values():
        for pa, pi_pa in pi0.items():
            pa_map = {p: pa[i] for i, p in enumerate(parents)}
            by_x: dict[object, list] = {x: [] for x in domain_x}
            for t in members:
                x = ts.response_on(t, treatment, pa_map)
                by_x[x].append(q[int(t)])
            masses = {x: (sum(by_x[x]) if by_x[x] else 0 * q[0]) for x in domain_x}
            yield masses, pi_pa, domain_x


def solve_bounds_convex(lp: LPData) -> ConvexBounds:
    """Min/max c'q under LP equalities/inequalities plus anchored f-sensitivity."""
    if lp.f_sensitivity is None:
        raise ValueError("solve_bounds_convex requires f_sensitivity on LPData")
    if not cvxpy_available():
        raise ImportError(
            "f-sensitivity (kl/chi2/tv) requires cvxpy. "
            "Install with: pip install partialid"
        )

    import cvxpy as cp

    fs = lp.f_sensitivity
    n = len(lp.c)
    ts = lp.type_space
    if not isinstance(ts, TypeSpace):
        raise ValueError("f-sensitivity requires a standard TypeSpace")
    if lp.pi0 is None:
        from .sensitivity_budgets import build_pi0_for_treatment

        pi0 = build_pi0_for_treatment(ts, lp.cells, lp.P_vec, fs.treatment)
    else:
        pi0 = lp.pi0
    rho = float(fs.rho)

    def _one(maximize: bool) -> SolveResult:
        q = cp.Variable(n, nonneg=True)
        cons = []
        for t in range(n):
            if not lp.allowed_types[t]:
                cons.append(q[t] == 0)
        if lp.A_eq is not None and lp.A_eq.shape[0] > 0:
            cons.append(lp.A_eq @ q == lp.b_eq)
        if lp.A_ub is not None and lp.b_ub is not None:
            cons.append(lp.A_ub @ q <= lp.b_ub)

        strata = list(_stratum_masses(q, ts, lp.allowed_types, fs.treatment, pi0))
        if fs.f == "kl":
            # Σ_{r,pa} Σ_x rel_entr(q_x, m π0(x|pa)) ≤ ρ
            div = 0
            for masses, pi_pa, domain_x in strata:
                m = sum(masses[x] for x in domain_x)
                for x in domain_x:
                    p0 = float(pi_pa.get(x, 0.0))
                    if p0 <= 1e-15:
                        continue
                    div += cp.rel_entr(masses[x], m * p0)
            cons.append(div <= rho)
        elif fs.f == "chi2":
            # Σ_{r,pa} Σ_x (q_x - π0 m)^2 / m ≤ ρ
            div = 0
            for masses, pi_pa, domain_x in strata:
                m = sum(masses[x] for x in domain_x)
                for x in domain_x:
                    p0 = float(pi_pa.get(x, 0.0))
                    div += cp.quad_over_lin(masses[x] - p0 * m, m + 1e-12)
            cons.append(div <= rho)
        elif fs.f == "tv":
            # Σ_{r,pa} (1/2) Σ_x |q_x - π0 m| ≤ ρ
            # Binary root recovers Σ_r |q1 - e0 m| (the two TV terms coincide).
            div = 0
            for masses, pi_pa, domain_x in strata:
                m = sum(masses[x] for x in domain_x)
                tv = 0
                for x in domain_x:
                    p0 = float(pi_pa.get(x, 0.0))
                    tv += cp.abs(masses[x] - p0 * m)
                div += 0.5 * tv
            cons.append(div <= rho)
        else:
            raise ValueError(f"unknown f {fs.f!r}")

        obj = cp.Maximize(lp.c @ q) if maximize else cp.Minimize(lp.c @ q)
        prob = cp.Problem(obj, cons)
        solved = False
        for solver in (
            getattr(cp, "CLARABEL", None),
            getattr(cp, "ECOS", None),
            getattr(cp, "OSQP", None),
            cp.SCS,
        ):
            if solver is None:
                continue
            try:
                prob.solve(solver=solver, verbose=False)
                if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                    solved = True
                    break
            except Exception:
                continue
        if not solved:
            try:
                prob.solve(verbose=False)
            except Exception as exc:
                return SolveResult(
                    value=float("nan"),
                    q=np.zeros(n),
                    success=False,
                    message=str(exc),
                )

        if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return SolveResult(
                value=float("nan"),
                q=np.zeros(n),
                success=False,
                message=str(prob.status),
            )
        return SolveResult(
            value=float(prob.value),
            q=np.asarray(q.value, dtype=float).ravel(),
            success=True,
            message=str(prob.status),
        )

    return ConvexBounds(lower=_one(False), upper=_one(True))
