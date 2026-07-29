"""MSM / c-dependence aggregate solver (sensitivity-native LP).

For discrete treatment ``X`` and outcome ``Y`` of arbitrary finite cardinality,
response types are maps ``x ↦ y`` and Tan MSM / c-dependence act per map:

1. **Binary XY + Tan MSM / anchored Γ (no MTR):** exact O(1) margin formulas
   (THEORY §8.7.3 generalized to arbitrary ``P`` and ``θ0=e0/(1-e0)``).
2. **General |X|,|Y| (± MTR):** transportation LP in ``q[r, x]`` — relative-
   propensity LR (Tan when binary) or ℓ∞ c-dependence.
3. **Inert prune:** order-only chains reduce to the active ``{X,Y}`` district
   before the aggregate (Move 2).
4. **IV + MSM:** instrument R-side LP with anchored budgets (no column aggregation);
   ``msm_agg`` routes there when ``WR={X,Y}``.
5. **Exact breakpoints / breakdown** for the binary-Y MSM path (no MTR).

This is the sensitivity analogue of hyperarc aggregation: a specialized exact
program for budgeted faces, not a new LP kernel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Mapping, Sequence

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from .assumptions import (
    AssumptionSet,
    CDependenceAssumption,
    DiagramAssumption,
    MonotonicityAssumption,
    NoDefiersAssumption,
    OrderAssumption,
    RosenbaumAssumption,
    TanMSMAssumption,
    as_assumption_set,
)
from .formulate import _normalize_P
from .query import Query
from .variables import Variables

BudgetKind = Literal["Lambda", "Gamma", "c"]
AggMode = Literal["xy_transport", "iv_instrument"]


@dataclass
class MSMAggResult:
    lower: float
    upper: float
    width: float
    success: bool
    message: str = ""
    formula_id: str | None = None
    n_agg: int = 0
    notes: list[str] = field(default_factory=list)
    closed_form: bool = False


def _xy_table(
    variables: Variables,
    data: Mapping,
    treatment: str,
    outcome: str,
) -> tuple[np.ndarray, list, list] | None:
    """Return joint table P[x, y] with domain lists, or None if not applicable.

    Supports arbitrary finite treatment / outcome cardinalities.
    """
    if treatment not in variables.names or outcome not in variables.names:
        return None
    if set(variables.names) != {treatment, outcome}:
        return None
    dx = list(variables.domains[treatment])
    dy = list(variables.domains[outcome])
    order = (treatment, outcome)
    cells, Pvec = _normalize_P(variables, order, data)
    table = np.zeros((len(dx), len(dy)), dtype=float)
    x_index = {v: i for i, v in enumerate(dx)}
    y_index = {v: i for i, v in enumerate(dy)}
    for cell, p in zip(cells, Pvec):
        table[x_index[cell[0]], y_index[cell[1]]] = float(p)
    return table, list(dx), list(dy)


def _flip_x(table: np.ndarray) -> np.ndarray:
    return table[::-1, :].copy()


def _msm_binary_y_bounds_from_table(
    table: np.ndarray,
    Lam: float,
    *,
    y_star_index: int,
) -> tuple[float, float, float]:
    """Sharp bounds on P(Y=y*|do(X=1)) under Tan MSM for binary Y."""
    if Lam < 1.0 - 1e-15:
        raise ValueError("Tan MSM requires Λ ≥ 1")
    if table.shape[1] != 2:
        raise ValueError("closed margin formula requires binary Y")
    if y_star_index == 0:
        table = table[:, ::-1]
    p00, p01 = float(table[0, 0]), float(table[0, 1])
    p10, p11 = float(table[1, 0]), float(table[1, 1])
    px0 = p00 + p01
    e0 = p10 + p11
    if not (0 < e0 < 1):
        raise ValueError(f"e0={e0} must lie in (0,1)")
    th = e0 / (1.0 - e0)
    Smax = min(px0, (Lam / th) * p11, px0 - p10 / (Lam * th))
    Smin = max(0.0, p11 / (Lam * th), px0 - (Lam / th) * p10)
    Smax = float(min(max(Smax, Smin), px0))
    Smin = float(max(min(Smin, Smax), 0.0))
    lo, hi = p11 + Smin, p11 + Smax
    return lo, hi, hi - lo


def msm_binary_xy_bounds(
    P: Mapping[tuple, float],
    Lam: float,
    *,
    x_star: int = 1,
    y_star: int = 1,
) -> tuple[float, float, float]:
    """Closed-form Tan MSM bounds for binary XY (arbitrary table)."""
    table = np.zeros((2, 2))
    for (x, y), v in P.items():
        table[int(x), int(y)] += float(v)
    table = table / table.sum()
    if int(x_star) == 0:
        table = _flip_x(table)
    return _msm_binary_y_bounds_from_table(table, float(Lam), y_star_index=int(y_star))


def msm_binary_xy_breakpoints(
    P: Mapping[tuple, float],
    *,
    x_star: int = 1,
    y_star: int = 1,
    lam_max: float = 50.0,
) -> list[float]:
    """Exact Λ knots where the active S_min / S_max piece switches (≥1)."""
    table = np.zeros((2, 2))
    for (x, y), v in P.items():
        table[int(x), int(y)] += float(v)
    table = table / table.sum()
    if int(x_star) == 0:
        table = _flip_x(table)
    if int(y_star) == 0:
        table = table[:, ::-1]
    p10, p11 = float(table[1, 0]), float(table[1, 1])
    px0 = float(table[0, 0] + table[0, 1])
    e0 = p10 + p11
    th = e0 / (1.0 - e0)
    knots = {1.0, float(lam_max)}

    def _quad_roots(a: float, b: float, c: float) -> list[float]:
        if abs(a) < 1e-18:
            if abs(b) < 1e-18:
                return []
            return [float(-c / b)]
        disc = b * b - 4 * a * c
        if disc < -1e-12:
            return []
        disc = max(disc, 0.0)
        s = np.sqrt(disc)
        return [float((-b + s) / (2 * a)), float((-b - s) / (2 * a))]

    for r in _quad_roots(p10, -px0 * th, p11):
        if r >= 1.0 - 1e-12:
            knots.add(r)
    for r in _quad_roots(p11, -px0 * th, p10):
        if r >= 1.0 - 1e-12:
            knots.add(r)
    if p10 > 1e-15:
        knots.add(px0 * th / p10)
    if p11 > 1e-15:
        knots.add(px0 * th / p11)
    out = sorted(k for k in knots if 1.0 - 1e-12 <= k <= lam_max + 1e-12)
    cleaned: list[float] = []
    for k in out:
        if not cleaned or abs(k - cleaned[-1]) > 1e-9:
            cleaned.append(float(k))
    return cleaned


def msm_binary_xy_breakdown(
    P: Mapping[tuple, float],
    *,
    x_star: int = 1,
    y_star: int = 1,
    null: float = 0.0,
    which: Literal["lower", "upper", "interval"] = "interval",
    lam_max: float = 1e6,
    tol: float = 1e-10,
) -> float | None:
    """Smallest Λ≥1 at which the null enters the identification interval."""
    lo0, hi0, _ = msm_binary_xy_bounds(P, 1.0, x_star=x_star, y_star=y_star)

    def hits(Lam: float) -> bool:
        lo, hi, _ = msm_binary_xy_bounds(P, Lam, x_star=x_star, y_star=y_star)
        if which == "interval":
            return lo - tol <= null <= hi + tol
        if which == "lower":
            return lo <= null + tol
        return hi >= null - tol

    if hits(1.0):
        return 1.0
    if not hits(lam_max):
        return None
    lo_l, hi_l = 1.0, float(lam_max)
    for _ in range(80):
        mid = 0.5 * (lo_l + hi_l)
        if hits(mid):
            hi_l = mid
        else:
            lo_l = mid
    return float(hi_l)


def _mtr_response_ok(resp: tuple[int, ...], direction: str | None) -> bool:
    """Monotone treatment response on an ordered treatment domain."""
    if direction is None:
        return True
    if direction == "increasing":
        return all(resp[i] <= resp[i + 1] for i in range(len(resp) - 1))
    if direction == "decreasing":
        return all(resp[i] >= resp[i + 1] for i in range(len(resp) - 1))
    raise ValueError(f"bad MTR direction {direction!r}")


def _transport_lp_bounds(
    table: np.ndarray,
    *,
    x_star_index: int,
    y_star_index: int,
    param: BudgetKind,
    value: float,
    mtr_direction: str | None = None,
) -> tuple[bool, float, float, str, int]:
    """LP in q[r, x] for categorical X and Y under relative-propensity LR / ℓ∞ c-dep.

    Each outcome response ``r`` is a tuple ``(y_0,…,y_{m-1})``; variables are
    ``q[r, x]`` (mass on treatment level x with response r). Binary X recovers
    the classical ``q[x,y0,y1]`` transport LP.
    """
    from itertools import product as _product

    n_x, n_y = table.shape
    if not (0 <= x_star_index < n_x):
        return False, float("nan"), float("nan"), "bad x*", 0
    responses = [
        resp for resp in _product(range(n_y), repeat=n_x) if _mtr_response_ok(resp, mtr_direction)
    ]
    if not responses:
        return False, float("nan"), float("nan"), "MTR removes all responses", 0
    n = n_x * len(responses)
    pi0 = np.asarray([float(table[x].sum()) for x in range(n_x)], dtype=float)
    if np.any(pi0 < -1e-15) or abs(pi0.sum() - 1.0) > 1e-6:
        # allow slight renormalization noise
        s = float(pi0.sum())
        if s <= 1e-15:
            return False, float("nan"), float("nan"), "degenerate P(X)", n
        pi0 = pi0 / s

    def idx(j: int, x: int) -> int:
        return j * n_x + x

    A_eq = []
    b_eq = []
    for x in range(n_x):
        for y in range(n_y):
            row = np.zeros(n)
            for j, resp in enumerate(responses):
                if resp[x] == y:
                    row[idx(j, x)] = 1.0
            A_eq.append(row)
            b_eq.append(float(table[x, y]))

    A_ub: list[np.ndarray] = []
    b_ub: list[float] = []
    levels = [x for x in range(n_x) if pi0[x] > 1e-15]
    if param in ("Lambda", "Gamma"):
        Lam = float(value) if param == "Lambda" else float(np.sqrt(value))
        if Lam < 1.0 - 1e-15:
            return False, float("nan"), float("nan"), "Λ<1", n
        if len(levels) < 2:
            return False, float("nan"), float("nan"), "need ≥2 positive P(X=x)", n
        for j in range(len(responses)):
            for x, xp in _product(levels, repeat=2):
                if x == xp:
                    continue
                row = np.zeros(n)
                row[idx(j, x)] += pi0[xp]
                row[idx(j, xp)] -= Lam * pi0[x]
                A_ub.append(row)
                b_ub.append(0.0)
    elif param == "c":
        c = float(value)
        for j in range(len(responses)):
            for x in range(n_x):
                lo = float(pi0[x] - c)
                hi = float(pi0[x] + c)
                r_hi = np.zeros(n)
                for xp in range(n_x):
                    r_hi[idx(j, xp)] -= hi
                r_hi[idx(j, x)] += 1.0
                A_ub.append(r_hi)
                b_ub.append(0.0)
                r_lo = np.zeros(n)
                for xp in range(n_x):
                    r_lo[idx(j, xp)] += lo
                r_lo[idx(j, x)] -= 1.0
                A_ub.append(r_lo)
                b_ub.append(0.0)
    else:
        return False, float("nan"), float("nan"), f"bad param {param}", n

    c_obj = np.zeros(n)
    for j, resp in enumerate(responses):
        if resp[x_star_index] == y_star_index:
            for x in range(n_x):
                c_obj[idx(j, x)] = 1.0

    A_eq_m = sparse.csr_matrix(np.asarray(A_eq, dtype=float))
    b_eq_a = np.asarray(b_eq, dtype=float)
    A_ub_m = sparse.csr_matrix(np.asarray(A_ub, dtype=float))
    b_ub_a = np.asarray(b_ub, dtype=float)
    bounds = [(0.0, None)] * n

    def _solve(maximize: bool) -> tuple[bool, float, str]:
        c = -c_obj if maximize else c_obj
        res = linprog(
            c,
            A_ub=A_ub_m,
            b_ub=b_ub_a,
            A_eq=A_eq_m,
            b_eq=b_eq_a,
            bounds=bounds,
            method="highs",
        )
        if not res.success:
            return False, float("nan"), res.message
        val = float(-res.fun if maximize else res.fun)
        return True, val, res.message

    ok_lo, lo, msg_lo = _solve(False)
    ok_hi, hi, msg_hi = _solve(True)
    if not (ok_lo and ok_hi):
        return False, lo, hi, f"lo:{msg_lo}; hi:{msg_hi}", n
    return True, lo, hi, "ok", n


def _budget_from_assumptions(
    a: AssumptionSet, treatment: str
) -> tuple[str, BudgetKind, float] | tuple[None, None, None]:
    msm = list(a.of_type(TanMSMAssumption))
    ros = list(a.of_type(RosenbaumAssumption))
    cdep = list(a.of_type(CDependenceAssumption))
    n_budgets = sum(1 for b in (msm, ros, cdep) if b)
    if n_budgets != 1:
        return None, None, None
    if msm:
        if msm[0].treatment != treatment or not msm[0].anchored:
            return None, None, None
        return "tan_msm", "Lambda", float(msm[0].Lambda)
    if ros:
        if ros[0].treatment != treatment or not ros[0].anchored:
            return None, None, None
        return "anchored_rosenbaum", "Gamma", float(ros[0].Gamma)
    if cdep[0].treatment != treatment:
        return None, None, None
    return "c_dependence", "c", float(cdep[0].c)


def _mtr_direction(
    a: AssumptionSet, *, treatment: str, outcome: str
) -> str | None | Literal[False]:
    """Return MTR direction, None if absent, False if incompatible MTR atoms."""
    monos = list(a.of_type(MonotonicityAssumption))
    if not monos:
        return None
    dirs = set()
    for m in monos:
        if m.outcome != outcome or m.treatment != treatment:
            return False
        dirs.add(m.direction)
    if len(dirs) != 1:
        return False
    return next(iter(dirs))


def applicable_msm_aggregate(
    variables: Variables,
    query: Query,
    assumptions: AssumptionSet | None,
    *,
    treatment: str = "X",
) -> tuple[bool, str, BudgetKind | None, float | None, AggMode | None]:
    """Return (ok, reason, param_kind, value, mode)."""
    a = as_assumption_set(assumptions)
    if query.kind != "interventional":
        return False, "query not interventional", None, None, None
    outcome = query.outcome
    do = dict(query.intervention)
    if set(do) != {treatment} or treatment not in variables.names:
        return False, "need do(treatment) only", None, None, None
    if outcome not in variables.names:
        return False, "bad outcome", None, None, None
    if variables.size(treatment) < 2:
        return False, "treatment needs ≥2 levels", None, None, None

    reason, kind, value = _budget_from_assumptions(a, treatment)
    if kind is None or value is None or reason is None:
        return False, "need exactly one anchored sensitivity budget", None, None, None

    mtr = _mtr_direction(a, treatment=treatment, outcome=outcome)
    if mtr is False:
        return False, "incompatible MTR atoms", None, None, None

    allowed_extra = (
        OrderAssumption,
        DiagramAssumption,
        TanMSMAssumption,
        RosenbaumAssumption,
        CDependenceAssumption,
        MonotonicityAssumption,
        NoDefiersAssumption,
    )
    other = [atom for atom in a.atoms if not isinstance(atom, allowed_extra)]
    if other:
        return False, "extra non-supported atoms", None, None, None

    orders = a.of_type(OrderAssumption)
    diagrams = a.of_type(DiagramAssumption)
    if orders and diagrams:
        return False, "mixed order+diagram", None, None, None
    if not orders and not diagrams:
        return False, "need order or diagram", None, None, None

    # IV path: WR = {treatment, outcome}
    if diagrams:
        from .instrument import detect_wl_wr_cut

        g = diagrams[0].diagram
        if treatment not in g.nodes or outcome not in g.nodes:
            return False, "diagram nodes missing treatment/outcome", None, None, None
        split = detect_wl_wr_cut(g, treatments=[treatment], outcome=outcome)
        if split is not None:
            _wl, wr = split
            if set(wr) == {treatment, outcome}:
                tag = reason + ("_mtr" if mtr else "") + "_iv"
                return True, tag, kind, value, "iv_instrument"
        # Single-block XY diagram (no instrument cut)
        if set(g.nodes) == {treatment, outcome}:
            tag = reason + ("_mtr" if mtr else "")
            return True, tag, kind, value, "xy_transport"
        return False, "diagram not XY and not instrument WR={X,Y}", None, None, None

    # Order-only: allow inert extras; active set must be {treatment, outcome}
    if len(orders) != 1:
        return False, "need a single order atom", None, None, None
    ord_names = list(orders[0].order)
    if treatment not in ord_names or outcome not in ord_names:
        return False, "order missing treatment/outcome", None, None, None
    # No-defiers needs an instrument — not meaningful on pure order-only XY
    if a.of_type(NoDefiersAssumption) and set(ord_names) != {treatment, outcome}:
        # still ok if IV-like order? order-only has no instrument structure
        return False, "no_defiers requires a diagram (IV)", None, None, None
    if a.of_type(NoDefiersAssumption):
        return False, "no_defiers requires a diagram (IV)", None, None, None
    tag = reason + ("_mtr" if mtr else "")
    if set(ord_names) - {treatment, outcome}:
        tag = tag + "_inert"
    return True, tag, kind, value, "xy_transport"


def _prepare_xy_context(
    variables: Variables,
    data: Mapping,
    query: Query,
    assumptions: AssumptionSet,
    *,
    treatment: str,
) -> tuple[Variables, Mapping, list[str]]:
    """Inert-prune order-only problems down to {treatment, outcome}."""
    from .formulate import _inert_prune_order_only
    from .graph import CausalDiagram

    notes: list[str] = []
    diagram = assumptions.diagram()
    if diagram is None:
        orders = assumptions.of_type(OrderAssumption)
        if orders:
            diagram = CausalDiagram.from_order(orders[0].order, maximal_confounding=True)
        else:
            diagram = CausalDiagram.from_order(variables.names, maximal_confounding=True)
    V2, D2, _g2 = _inert_prune_order_only(variables, data, query, diagram, notes)
    return V2, D2, notes


def _bound_via_instrument(
    variables: Variables,
    data: Mapping,
    query: Query,
    assumptions: AssumptionSet,
    *,
    reason: str,
    kind: BudgetKind,
    value: float,
) -> MSMAggResult:
    """Solve instrument R-side LP with sensitivity (and optional MTR / no-defiers)."""
    from .backends import solve_bounds_lp
    from .formulate import formulate

    notes = [f"msm_agg/{reason}", f"budget={kind}:{value}", "iv_instrument"]
    lp = formulate(variables, data, query, assumptions, optimize=True)
    notes.extend(list(lp.notes))
    if lp.A_ub is None:
        return MSMAggResult(
            float("nan"),
            float("nan"),
            float("nan"),
            False,
            message="instrument LP missing sensitivity rows",
            notes=notes,
        )
    lo_r, hi_r = solve_bounds_lp(lp)
    if not (lo_r.success and hi_r.success):
        return MSMAggResult(
            float("nan"),
            float("nan"),
            float("nan"),
            False,
            message=f"lo:{lo_r.message}; hi:{hi_r.message}",
            notes=notes,
            n_agg=int(lp.A_eq.shape[1]) if lp.A_eq is not None else 0,
        )
    lo, hi = float(lo_r.value), float(hi_r.value)
    return MSMAggResult(
        lo,
        hi,
        hi - lo,
        True,
        message="instrument+sensitivity",
        formula_id="msm_instrument_lp",
        n_agg=int(lp.A_eq.shape[1]),
        notes=notes + [f"raw_cols={lp.A_eq.shape[1]}"],
        closed_form=False,
    )


def bound_msm_aggregate(
    variables: Variables,
    data: Mapping,
    query: Query,
    assumptions: AssumptionSet | None = None,
    *,
    treatment: str = "X",
) -> MSMAggResult:
    """Exact MSM/c-dep bounds via margin formula, transport LP, or IV R-side LP."""
    ok, reason, kind, value, mode = applicable_msm_aggregate(
        variables, query, assumptions, treatment=treatment
    )
    if not ok or kind is None or value is None or mode is None:
        return MSMAggResult(
            float("nan"),
            float("nan"),
            float("nan"),
            False,
            message=reason,
            notes=["msm_agg_inapplicable"],
        )

    a = as_assumption_set(assumptions)
    if mode == "iv_instrument":
        return _bound_via_instrument(
            variables, data, query, a, reason=reason, kind=kind, value=value
        )

    V2, D2, prune_notes = _prepare_xy_context(variables, data, query, a, treatment=treatment)
    if set(V2.names) != {treatment, query.outcome}:
        return MSMAggResult(
            float("nan"),
            float("nan"),
            float("nan"),
            False,
            message="inert prune did not reduce to treatment+outcome",
            notes=prune_notes,
        )

    parsed = _xy_table(V2, D2, treatment, query.outcome)
    if parsed is None:
        return MSMAggResult(
            float("nan"),
            float("nan"),
            float("nan"),
            False,
            message="not an XY table",
            notes=prune_notes,
        )
    table, dx, dy = parsed
    x_star = dict(query.intervention)[treatment]
    y_star = query.value
    if x_star not in dx:
        return MSMAggResult(float("nan"), float("nan"), float("nan"), False, message="bad x*")
    if y_star not in dy:
        return MSMAggResult(float("nan"), float("nan"), float("nan"), False, message="bad y*")
    x_idx = dx.index(x_star)
    y_idx = dy.index(y_star)

    mtr = _mtr_direction(a, treatment=treatment, outcome=query.outcome)
    assert mtr is not False
    notes = [f"msm_agg/{reason}", f"budget={kind}:{value}"] + prune_notes
    if mtr:
        notes.append(f"MTR_{mtr}")
    notes.append(f"|X|={len(dx)},|Y|={len(dy)}")

    # Closed margin only for binary XY, Tan/Γ, no MTR
    if mtr is None and len(dx) == 2 and len(dy) == 2 and kind in ("Lambda", "Gamma"):
        table_cf = table if x_idx == 1 else _flip_x(table)
        Lam = float(value) if kind == "Lambda" else float(np.sqrt(value))
        try:
            lo, hi, w = _msm_binary_y_bounds_from_table(table_cf, Lam, y_star_index=y_idx)
        except ValueError as exc:
            return MSMAggResult(float("nan"), float("nan"), float("nan"), False, message=str(exc))
        fid = "msm_margin_binary_xy" if kind == "Lambda" else "msm_margin_binary_xy_gamma"
        return MSMAggResult(
            lo,
            hi,
            w,
            True,
            message="closed margin",
            formula_id=fid,
            n_agg=1,
            notes=notes + ["closed_form_margin", "THEORY §8.7.3 generalized"],
            closed_form=True,
        )

    ok_lp, lo, hi, msg, n_agg = _transport_lp_bounds(
        table,
        x_star_index=x_idx,
        y_star_index=y_idx,
        param=kind,
        value=float(value),
        mtr_direction=mtr,
    )
    if not ok_lp:
        return MSMAggResult(
            float("nan"),
            float("nan"),
            float("nan"),
            False,
            message=msg,
            n_agg=n_agg,
            notes=notes,
        )
    fid = "msm_transport_lp_mtr" if mtr else "msm_transport_lp"
    return MSMAggResult(
        lo,
        hi,
        hi - lo,
        True,
        message=msg,
        formula_id=fid,
        n_agg=n_agg,
        notes=notes + [f"transport_vars={n_agg}"],
        closed_form=False,
    )


def msm_agg_path_binary_xy(
    P: Mapping[tuple, float],
    *,
    param: BudgetKind = "Lambda",
    x_star: int = 1,
    y_star: int = 1,
    theta_min: float = 1.0,
    theta_max: float = 20.0,
    extra_values: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Exact PWL path for binary XY Tan MSM / anchored Γ."""
    if param == "c":
        raise ValueError("use transport grid for c-dependence path")
    if param == "Lambda":
        lam_min, lam_max = float(theta_min), float(theta_max)
        to_theta = lambda lam: lam
        from_theta = lambda th: th
    else:
        lam_min, lam_max = float(np.sqrt(theta_min)), float(np.sqrt(theta_max))
        to_theta = lambda lam: lam * lam
        from_theta = lambda th: float(np.sqrt(th))

    raw_bps = msm_binary_xy_breakpoints(P, x_star=x_star, y_star=y_star, lam_max=lam_max)
    knots_lam = [lam_min, lam_max]
    for b in raw_bps:
        if lam_min < b < lam_max:
            knots_lam.append(b)
    if extra_values:
        for v in extra_values:
            lam = from_theta(float(v)) if param == "Gamma" else float(v)
            if lam_min - 1e-15 <= lam <= lam_max + 1e-15:
                knots_lam.append(lam)
    knots_lam = sorted(set(round(x, 12) for x in knots_lam))
    vals = np.asarray([to_theta(lam) for lam in knots_lam], dtype=float)
    lower = np.empty(len(knots_lam))
    upper = np.empty(len(knots_lam))
    width = np.empty(len(knots_lam))
    for i, lam in enumerate(knots_lam):
        lo, hi, w = msm_binary_xy_bounds(P, lam, x_star=x_star, y_star=y_star)
        lower[i], upper[i], width[i] = lo, hi, w
    bps_lam = [lam_min, lam_max]
    for b in raw_bps:
        if lam_min < b < lam_max:
            bps_lam.append(b)
    bps = np.asarray(sorted({to_theta(b) for b in bps_lam}), dtype=float)
    notes = [
        "msm_agg exact binary-XY path",
        f"n_knots={len(knots_lam)}",
    ]
    return vals, lower, upper, width, bps, notes
