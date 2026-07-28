"""C-canonical finite-latent parameterization (Zhang–Bareinboim NeurIPS 2022 Thm 2).

For interventional queries on a single c-component, exogenous cardinality
``d_U = |Ω_Pa(C)|`` suffices to represent the relevant c-factors. Here we take
``C`` = all endogenous nodes (maximal confounding / one block), so
``d = Π_v |Ω_v|``.

When the response-type count ``N`` exceeds ``d``, we restrict to at most ``d``
positive types (support bound) — equivalent to a C-canonical mixture of at
most ``d`` deterministic mechanisms. When the LP is fiber-separable, Move-1
already uses an O(|cells|) parameterization; we route there instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Mapping

import numpy as np

from .assumptions import AssumptionSet, SupportBoundAssumption, as_assumption_set
from .backends import solve_bounds_lp
from .fiber_solve import is_fiber_separable, solve_bounds_fiber
from .formulate import formulate
from .lazy_types import c_canonical_cardinality
from .mip_solve import solve_bounds_mip
from .query import Query
from .variables import Variables


@dataclass
class CCanonicalResult:
    lower: float
    upper: float
    width: float
    success: bool
    message: str
    method: str
    formulate_time: float
    solve_time: float
    notes: list[str]
    q_lower: np.ndarray | None = None
    q_upper: np.ndarray | None = None
    d_canonical: int = 0
    n_response_types: int = 0


def bound_c_canonical(
    variables: Variables,
    data: Mapping,
    query: Query,
    assumptions: AssumptionSet | None = None,
    *,
    optimize: bool = True,
) -> CCanonicalResult:
    """Bounds under C-canonical latent cardinality (interventional focus)."""
    assumptions = as_assumption_set(assumptions)
    notes: list[str] = ["parameterization=c_canonical"]

    if query.kind != "interventional":
        # Fall back to ordinary formulate; cardinality cap still applied if helpful
        notes.append("non-interventional: applying support cap when N>d")

    diagram = assumptions.diagram()
    if diagram is None:
        return CCanonicalResult(
            float("nan"),
            float("nan"),
            float("nan"),
            False,
            "c_canonical requires order/diagram",
            "c_canonical",
            0.0,
            0.0,
            notes,
        )

    d = c_canonical_cardinality(variables, diagram)
    notes.append(f"d_U=|Ω_Pa(C)|={d}")

    t0 = perf_counter()
    # If support_bound already present, keep the tighter k
    existing = assumptions.of_type(SupportBoundAssumption)
    if existing:
        k = min(existing[0].k, d)
        base = AssumptionSet(
            tuple(a for a in assumptions if not isinstance(a, SupportBoundAssumption))
        )
        assumptions = base & AssumptionSet((SupportBoundAssumption(k=k),))
        notes.append(f"merged support_bound k={k}")
    else:
        assumptions = assumptions  # may add below

    lp = formulate(variables, data, query, assumptions, optimize=optimize)
    t1 = perf_counter()
    n = lp.type_space.n_types
    notes.extend(lp.notes)
    notes.append(f"n_response_types={n}")

    # Fiber-separable interventional: O(|cells|) — already tighter than C-canonical MIP
    if is_fiber_separable(lp) and query.kind == "interventional":
        lo, hi, q_lo, q_hi = solve_bounds_fiber(lp)
        t2 = perf_counter()
        notes.append("routed to fiber (separable; C-canonical non-binding)")
        return CCanonicalResult(
            lower=lo,
            upper=hi,
            width=hi - lo,
            success=True,
            message="fiber",
            method="fiber",
            formulate_time=t1 - t0,
            solve_time=t2 - t1,
            notes=notes,
            q_lower=q_lo,
            q_upper=q_hi,
            d_canonical=d,
            n_response_types=n,
        )

    # Apply cardinality cap when response types exceed C-canonical size
    if n > d and lp.support_k is None:
        notes.append(f"imposing support_bound k={d} (N={n}>d)")
        # Re-formulate with support bound
        assumptions = assumptions & AssumptionSet((SupportBoundAssumption(k=d),))
        lp = formulate(variables, data, query, assumptions, optimize=optimize)
        t1 = perf_counter()
        notes.extend([n for n in lp.notes if n not in notes])

    if lp.support_k is not None or lp.method_hint == "mip":
        mb = solve_bounds_mip(lp)
        t2 = perf_counter()
        success = mb.lower.success and mb.upper.success
        lo = mb.lower.value if mb.lower.success else float("nan")
        hi = mb.upper.value if mb.upper.success else float("nan")
        return CCanonicalResult(
            lower=lo,
            upper=hi,
            width=(hi - lo) if success else float("nan"),
            success=success,
            message=f"lo:{mb.lower.message}; hi:{mb.upper.message}",
            method="mip",
            formulate_time=t1 - t0,
            solve_time=t2 - t1,
            notes=notes,
            q_lower=mb.lower.q if mb.lower.success else None,
            q_upper=mb.upper.q if mb.upper.success else None,
            d_canonical=d,
            n_response_types=n,
        )

    lo_r, hi_r = solve_bounds_lp(lp)
    t2 = perf_counter()
    success = lo_r.success and hi_r.success
    lo = lo_r.value if lo_r.success else float("nan")
    hi = hi_r.value if hi_r.success else float("nan")
    notes.append("N<=d: full response LP (C-canonical non-binding)")
    return CCanonicalResult(
        lower=lo,
        upper=hi,
        width=(hi - lo) if success else float("nan"),
        success=success,
        message=f"lo:{lo_r.message}; hi:{hi_r.message}",
        method="lp",
        formulate_time=t1 - t0,
        solve_time=t2 - t1,
        notes=notes,
        q_lower=lo_r.q if lo_r.success else None,
        q_upper=hi_r.q if hi_r.success else None,
        d_canonical=d,
        n_response_types=n,
    )
