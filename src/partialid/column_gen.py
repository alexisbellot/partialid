"""Column generation — including matrix-free pricing (no full type scan).

Track A3: when budget rows ``A_ub`` are present (Tan MSM / c-dep / κ), pricing
uses reduced costs ``c - A_eq'π - A_ub'μ`` with a single sparse matvec for
``A_ub'μ``, and seeds partner columns from MSM groups when available.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from .formulate import LPData
from .lazy_types import LazyTypeSpace, TypeKey
from .query import Query
from .sensitivity_grid import MSMGroup, _msm_groups


@dataclass
class ColGenResult:
    lower: float
    upper: float
    q_lower: np.ndarray
    q_upper: np.ndarray
    n_columns: int
    iterations: int
    success: bool
    message: str
    matrix_free: bool = False


def _master(
    lp: LPData,
    columns: list[int],
    maximize: bool,
) -> tuple[bool, float, np.ndarray, np.ndarray | None, np.ndarray | None, str]:
    if not columns:
        return False, float("nan"), np.zeros(lp.type_space.n_types), None, None, "no columns"

    idx = np.asarray(columns, dtype=int)
    c = lp.c[idx]
    if maximize:
        c = -c

    A_eq = lp.A_eq[:, idx]
    b_eq = lp.b_eq
    A_ub = lp.A_ub[:, idx] if lp.A_ub is not None else None
    b_ub = lp.b_ub

    bounds = [(0.0, None) if lp.allowed_types[t] else (0.0, 0.0) for t in idx]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if not res.success:
        return False, float("nan"), np.zeros(lp.type_space.n_types), None, None, res.message

    q_full = np.zeros(lp.type_space.n_types)
    q_full[idx] = res.x
    val = float(-res.fun if maximize else res.fun)

    eq_duals = None
    ub_duals = None
    if hasattr(res, "eqlin") and res.eqlin is not None:
        eq_duals = np.asarray(res.eqlin.marginals, dtype=float)
    if hasattr(res, "ineqlin") and res.ineqlin is not None and lp.A_ub is not None:
        ub_duals = np.asarray(res.ineqlin.marginals, dtype=float)

    return True, val, q_full, eq_duals, ub_duals, res.message


def _ub_contrib(lp: LPData, ub_duals: np.ndarray | None) -> np.ndarray | None:
    """Per-type budget dual contribution ``(A_ub' μ)_t`` (Track A3)."""
    if lp.A_ub is None or ub_duals is None:
        return None
    mu = np.asarray(ub_duals, dtype=float).ravel()
    # A_ub is (n_ub, n); A_ub.T @ μ is length-n (sparse matvec).
    return np.asarray(lp.A_ub.T @ mu, dtype=float).ravel()


def _price_matrix_free_materialized(
    lp: LPData,
    eq_duals: np.ndarray,
    ub_duals: np.ndarray | None,
    maximize: bool,
    in_set: set[int],
    tol: float,
) -> int | None:
    """Price by scanning fibers only (not a flat 0..N loop with dense rc).

    Reduced cost for min: ``rc_t = c_t - π_{cell(t)} - (A_ub'μ)_t``.
    """
    ts = lp.type_space
    ub_rc = _ub_contrib(lp, ub_duals)
    best_t = None
    best_rc = 0.0
    for i, cell in enumerate(lp.cells):
        pi = float(eq_duals[i])
        for t in ts.fiber[cell]:
            if not lp.allowed_types[t] or t in in_set:
                continue
            rc = float(lp.c[t]) - pi
            if ub_rc is not None:
                rc -= float(ub_rc[t])
            if maximize:
                rc = -rc
            if rc < -tol and (best_t is None or rc < best_rc):
                best_t = t
                best_rc = rc
    return best_t


def _msm_groups_for_lp(lp: LPData) -> list[MSMGroup] | None:
    """Rebuild MSM/c-dep groups from the type space when budget rows exist."""
    if lp.A_ub is None:
        return None
    ts = lp.type_space
    # Prefer named treatment X when present; else first variable that admits groups.
    candidates: list[str] = []
    if "X" in ts.order:
        candidates.append("X")
    candidates.extend([v for v in ts.order if v not in candidates])
    for treatment in candidates:
        try:
            groups = _msm_groups(ts, lp.allowed_types, treatment)
        except Exception:
            continue
        if groups:
            return groups
    return None


def _msm_partner_map(groups: list[MSMGroup]) -> dict[int, list[int]]:
    """Map type index → all types in its MSM/c-dep group."""
    out: dict[int, list[int]] = {}
    for g in groups:
        members = [int(i) for i in g.all_members()]
        for t in members:
            out[t] = members
    return out


def _msm_partners(
    lp: LPData,
    t: int,
    partner_map: dict[int, list[int]] | None = None,
) -> list[int]:
    """Types that share an MSM/c-dep group with ``t`` (needed for coupled budgets)."""
    if partner_map is not None and t in partner_map:
        return [j for j in partner_map[t] if j != t and lp.allowed_types[j]]
    # Fallback: flip root binary coordinates (old heuristic)
    ts = lp.type_space
    partners = []
    type_ids = list(ts.joint_types[t])
    for v_i, v in enumerate(ts.order):
        if ts.diagram.parents(v):
            continue
        if len(ts.response_fns[v]) != 2:
            continue
        flipped = list(type_ids)
        flipped[v_i] = 1 - flipped[v_i]
        j = ts.type_index.get(tuple(flipped))
        if j is not None and lp.allowed_types[j]:
            partners.append(j)
    return partners


def _eq_only_support_seed(lp: LPData, partner_map: dict[int, list[int]] | None) -> list[int] | None:
    """Feasible observational support ignoring budgets, then close MSM groups.

    Needed because a sparse min/max-c fiber seed is often MSM-infeasible
    (Track A3).
    """
    n = lp.type_space.n_types
    bounds = [(0.0, None) if lp.allowed_types[t] else (0.0, 0.0) for t in range(n)]
    # Any feasible point of A_eq q = P, q≥0 (ignore A_ub).
    res = linprog(
        c=np.zeros(n),
        A_eq=lp.A_eq,
        b_eq=lp.b_eq,
        bounds=bounds,
        method="highs",
    )
    if not res.success or res.x is None:
        return None
    cols = [t for t, x in enumerate(res.x) if x > 1e-12 and lp.allowed_types[t]]
    if not cols:
        return None
    extra: list[int] = []
    for t in cols:
        extra.extend(_msm_partners(lp, t, partner_map))
    seen: set[int] = set()
    out: list[int] = []
    for t in cols + extra:
        if t not in seen and lp.allowed_types[t]:
            seen.add(t)
            out.append(t)
    return out


def _initial_columns(lp: LPData, partner_map: dict[int, list[int]] | None = None) -> list[int]:
    cols: list[int] = []
    if any("instrument LP" in n for n in lp.notes):
        return [t for t in range(lp.type_space.n_types) if lp.allowed_types[t]]

    # Budget LPs: start from an eq-only feasible support (then MSM-close).
    if lp.A_ub is not None:
        seeded = _eq_only_support_seed(lp, partner_map)
        if seeded:
            return seeded

    for i, cell in enumerate(lp.cells):
        if lp.P_vec[i] <= 0:
            continue
        members = [t for t in lp.type_space.fiber[cell] if lp.allowed_types[t]]
        if not members:
            raise ValueError(f"no allowed type in fiber {cell}")
        cols.append(members[0])
        cols.append(min(members, key=lambda t: lp.c[t]))
        cols.append(max(members, key=lambda t: lp.c[t]))

    if lp.A_ub is not None:
        extra = []
        for t in cols:
            extra.extend(_msm_partners(lp, t, partner_map))
        cols.extend(extra)

    seen: set[int] = set()
    out: list[int] = []
    for t in cols:
        if t not in seen and lp.allowed_types[t]:
            seen.add(t)
            out.append(t)
    return out


def solve_bounds_colgen(
    lp: LPData,
    *,
    max_iter: int = 500,
    tol: float = 1e-9,
    matrix_free: bool = True,
) -> ColGenResult:
    """Column generation for min and max.

    With ``matrix_free=True`` (default), pricing walks fibers instead of
    allocating a dense reduced-cost vector of length ``n_types``.
    Budget rows (MSM / c-dep / κ) enter reduced costs via ``A_ub'μ`` (Track A3).
    """
    groups = _msm_groups_for_lp(lp) if lp.A_ub is not None else None
    partner_map = _msm_partner_map(groups) if groups else None

    def optimize(maximize: bool) -> tuple[bool, float, np.ndarray, int, str]:
        columns = _initial_columns(lp, partner_map)
        q = np.zeros(lp.type_space.n_types)
        val = float("nan")
        msg = ""
        expanded = False
        for it in range(1, max_iter + 1):
            ok, val, q, eq_d, ub_d, msg = _master(lp, columns, maximize)
            if not ok:
                if lp.A_ub is not None and not expanded:
                    # Last resort: dense column set (still correct; loses sparsity)
                    columns = [t for t in range(lp.type_space.n_types) if lp.allowed_types[t]]
                    expanded = True
                    continue
                return False, val, q, it, msg
            if eq_d is None:
                return False, val, q, it, msg
            in_set = set(columns)
            if matrix_free:
                best_t = _price_matrix_free_materialized(lp, eq_d, ub_d, maximize, in_set, tol)
            else:
                # legacy dense scan (still uses one A_ub'μ matvec)
                ub_rc = _ub_contrib(lp, ub_d)
                rc = lp.c - lp.A_eq.T @ eq_d
                if ub_rc is not None:
                    rc = rc - ub_rc
                if maximize:
                    rc = -rc
                best_t = None
                best_rc = 0.0
                for t in range(lp.type_space.n_types):
                    if not lp.allowed_types[t] or t in in_set:
                        continue
                    if rc[t] < -tol and (best_t is None or rc[t] < best_rc):
                        best_t = t
                        best_rc = rc[t]
            if best_t is None:
                return True, val, q, it, msg
            columns.append(best_t)
            for p in _msm_partners(lp, best_t, partner_map):
                if p not in in_set:
                    columns.append(p)
        return False, val, q, it, "max_iter"

    ok_lo, lo, q_lo, it_lo, msg_lo = optimize(False)
    ok_hi, hi, q_hi, it_hi, msg_hi = optimize(True)
    success = ok_lo and ok_hi
    return ColGenResult(
        lower=lo,
        upper=hi,
        q_lower=q_lo,
        q_upper=q_hi,
        n_columns=int(np.count_nonzero(q_lo) + np.count_nonzero(q_hi)),
        iterations=it_lo + it_hi,
        success=success,
        message=f"lo:{msg_lo}; hi:{msg_hi}",
        matrix_free=matrix_free,
    )


# ---------------------------------------------------------------------------
# Fully lazy colgen (never materializes TypeSpace.joint_types)
# ---------------------------------------------------------------------------


@dataclass
class LazyColGenResult:
    lower: float
    upper: float
    n_columns: int
    iterations: int
    success: bool
    message: str
    notes: list[str]


def _lazy_master(
    keys: list[TypeKey],
    c_cols: np.ndarray,
    A_eq: np.ndarray,
    b_eq: np.ndarray,
    maximize: bool,
) -> tuple[bool, float, np.ndarray, np.ndarray | None, str]:
    if not keys:
        return False, float("nan"), np.zeros(0), None, "no columns"
    c = -c_cols if maximize else c_cols
    res = linprog(
        c,
        A_eq=sparse.csr_matrix(A_eq),
        b_eq=b_eq,
        bounds=[(0.0, None)] * len(keys),
        method="highs",
    )
    if not res.success:
        return False, float("nan"), np.zeros(len(keys)), None, res.message
    val = float(-res.fun if maximize else res.fun)
    eq_d = None
    if hasattr(res, "eqlin") and res.eqlin is not None:
        eq_d = np.asarray(res.eqlin.marginals, dtype=float)
    return True, val, res.x, eq_d, res.message


def solve_bounds_colgen_lazy(
    lazy: LazyTypeSpace,
    cells: list[tuple],
    P_vec: np.ndarray,
    query: Query,
    *,
    max_iter: int = 500,
    tol: float = 1e-9,
    max_per_fiber: int = 128,
) -> LazyColGenResult:
    """Matrix-free colgen on a LazyTypeSpace (equality-only interventional LP)."""
    if query.kind != "interventional":
        return LazyColGenResult(
            float("nan"),
            float("nan"),
            0,
            0,
            False,
            "lazy colgen supports interventional queries",
            [],
        )

    notes = [
        f"lazy TypeSpace n_types≈{lazy.n_types_est}",
        f"n_cells={len(cells)}",
        "matrix-free pricing",
    ]
    cell_index = {cell: i for i, cell in enumerate(cells)}
    intervention = dict(query.intervention)
    outcome, value = query.outcome, query.value

    def seed_keys() -> list[TypeKey]:
        keys: list[TypeKey] = []
        seen: set[TypeKey] = set()
        for cell in cells:
            k_lo, k_hi, _, _ = lazy.fiber_extremes_for_query(
                cell,
                outcome,
                value,
                intervention,
                max_list=max_per_fiber,
            )
            for k in (k_lo, k_hi):
                if k is not None and k not in seen:
                    seen.add(k)
                    keys.append(k)
            one = lazy.one_fiber_member(cell)
            if one is not None and one not in seen:
                seen.add(one)
                keys.append(one)
        return keys

    def build_ca(keys: list[TypeKey]) -> tuple[np.ndarray, np.ndarray]:
        m, n = len(cells), len(keys)
        A = np.zeros((m, n))
        c = np.zeros(n)
        for j, key in enumerate(keys):
            cell = lazy.observational_cell(key)
            A[cell_index[cell], j] = 1.0
            c[j] = lazy.query_coefficient(key, outcome, value, intervention)
        return c, A

    def optimize(maximize: bool) -> tuple[bool, float, int, str]:
        keys = seed_keys()
        val = float("nan")
        msg = ""
        for it in range(1, max_iter + 1):
            c_cols, A = build_ca(keys)
            ok, val, _x, eq_d, msg = _lazy_master(keys, c_cols, A, P_vec, maximize)
            if not ok or eq_d is None:
                return False, val, it, msg
            active = set(keys)
            best = lazy.price_best_keys(
                cells,
                P_vec,
                eq_d,
                outcome=outcome,
                value=value,
                intervention=intervention,
                maximize=maximize,
                active=active,
                tol=tol,
                max_per_fiber=max_per_fiber,
            )
            if best is None:
                return True, val, it, msg
            keys.append(best)
        return False, val, max_iter, "max_iter"

    ok_lo, lo, it_lo, msg_lo = optimize(False)
    ok_hi, hi, it_hi, msg_hi = optimize(True)
    return LazyColGenResult(
        lower=lo,
        upper=hi,
        n_columns=0,
        iterations=it_lo + it_hi,
        success=ok_lo and ok_hi,
        message=f"lo:{msg_lo}; hi:{msg_hi}",
        notes=notes,
    )
