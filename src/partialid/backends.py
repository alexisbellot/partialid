"""LP backends (HiGHS via SciPy).

Equality-only programs ``min/max c'q s.t. A q = P, q≥0`` are solved in the
**dual** when ``n_vars ≫ n_eq`` (Track A): dual dimension equals the number of
observational cells, independent of the type count. Applies to any such LP
(instrument R-side, full-type, aggregated columns)—not IV-specific.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from .formulate import LPData

# Dual when n_vars > max(DUAL_MIN_VARS, DUAL_RATIO * n_eq)
DUAL_MIN_VARS = 64
DUAL_RATIO = 4
# After formulate, auto may prefer matrix-free colgen at this type count
# when the dual gate does not fire (N large but N ≯ 4 m).
AUTO_COLGEN_TYPES = 4096


@dataclass
class SolveResult:
    value: float
    q: np.ndarray
    success: bool
    message: str
    fun_time: float = 0.0
    used_dual: bool = False
    recovered_primal: bool = False


def should_use_dual(
    lp: LPData,
    *,
    force: bool = False,
    min_vars: int = DUAL_MIN_VARS,
    ratio: int = DUAL_RATIO,
) -> bool:
    """True when the eq-only dual shortcut is applicable / preferred.

    Parameters
    ----------
    force :
        If True, use dual whenever the LP is equality-only (no size gate).
    """
    if lp.A_eq is None or lp.A_eq.shape[0] == 0:
        return False
    if lp.A_ub is not None:
        return False
    n_vars = int(lp.A_eq.shape[1])
    n_eq = int(lp.A_eq.shape[0])
    if force:
        return n_vars > 0 and n_eq > 0
    return n_vars > max(min_vars, ratio * n_eq)


def prefer_colgen_auto(lp: LPData, *, threshold: int = AUTO_COLGEN_TYPES) -> bool:
    """True when ``auto`` should run matrix-free colgen instead of dense primal.

    Used when the dual size-gate does not fire but the type count is still large.
    Track A3: also allowed with budget rows (MSM / c-dep / κ); pricing uses
    ``A_ub'μ``. Still blocked for MIP/convex hints.
    """
    if lp.A_eq is None:
        return False
    if lp.method_hint in ("mip", "convex", "gpp", "fractional"):
        return False
    if getattr(lp, "c_den", None) is not None:
        return False
    if should_use_dual(lp):
        return False
    n = int(lp.A_eq.shape[1])
    return n > threshold


def _solve_primal(lp: LPData, c_primal: np.ndarray) -> SolveResult:
    bounds = [(0.0, None) if lp.allowed_types[t] else (0.0, 0.0) for t in range(len(c_primal))]
    res = linprog(
        c=c_primal,
        A_ub=lp.A_ub,
        b_ub=lp.b_ub,
        A_eq=lp.A_eq,
        b_eq=lp.b_eq,
        bounds=bounds,
        method="highs",
    )
    if not res.success:
        return SolveResult(
            value=float("nan"),
            q=np.zeros_like(c_primal),
            success=False,
            message=res.message,
        )
    return SolveResult(
        value=float(res.fun),
        q=np.asarray(res.x, dtype=float),
        success=True,
        message=res.message,
        used_dual=False,
    )


def _primal_feasible(
    lp: LPData,
    q: np.ndarray,
    c_primal: np.ndarray,
    value: float,
    *,
    tol: float = 1e-6,
) -> bool:
    if q is None or not np.all(np.isfinite(q)):
        return False
    if np.any(q < -tol):
        return False
    q_clip = np.maximum(q, 0.0)
    resid = lp.A_eq @ q_clip - lp.b_eq
    if float(np.max(np.abs(resid))) > tol * (1.0 + float(np.max(np.abs(lp.b_eq)))):
        return False
    obj = float(c_primal @ q_clip)
    if abs(obj - value) > tol * (1.0 + abs(value)):
        return False
    return True


def _recover_from_dual_marginals(
    lp: LPData,
    free: np.ndarray,
    dual_res,
) -> np.ndarray | None:
    """Inequality duals of the dual LP are the primal x on free columns."""
    if not hasattr(dual_res, "ineqlin") or dual_res.ineqlin is None:
        return None
    marg = getattr(dual_res.ineqlin, "marginals", None)
    if marg is None:
        return None
    mu = np.asarray(marg, dtype=float).ravel()
    if mu.size != free.size:
        return None
    # SciPy may return signed multipliers; primal mass is nonnegative
    mu = np.maximum(mu, 0.0)
    q = np.zeros(lp.A_eq.shape[1], dtype=float)
    q[free] = mu
    return q


def _recover_restricted_primal(
    lp: LPData,
    c_primal: np.ndarray,
    y: np.ndarray,
    free: np.ndarray,
    *,
    tol: float = 1e-7,
) -> np.ndarray | None:
    """Solve primal on columns nearly tight at the dual optimum y."""
    A = lp.A_eq.tocsr()
    # reduced costs: c - A'y; tight => active basis candidates
    At_y = A.T @ y
    rc = c_primal[free] - At_y[free]
    scale = 1.0 + float(np.max(np.abs(c_primal[free])))
    active = free[np.abs(rc) <= tol * scale]
    if active.size == 0:
        active = free[np.abs(rc) <= 1e-5 * scale]
    if active.size == 0:
        return None

    A_S = A[:, active]
    c_S = c_primal[active]
    res = linprog(
        c_S,
        A_eq=A_S,
        b_eq=lp.b_eq,
        bounds=[(0.0, None)] * active.size,
        method="highs",
    )
    if not res.success:
        return None
    q = np.zeros(A.shape[1], dtype=float)
    q[active] = np.asarray(res.x, dtype=float)
    return q


def _solve_dual_eq_only(
    lp: LPData,
    c_primal: np.ndarray,
    *,
    recover_primal: bool = True,
) -> SolveResult:
    """Solve min c'x s.t. A x = b, x>=0 via dual when n_vars ≫ n_eq."""
    A = lp.A_eq.tocsr()
    b = lp.b_eq
    free = np.where(lp.allowed_types)[0]
    A_ub = A.T.tocsr()[free]
    b_ub = c_primal[free]
    m = A.shape[0]
    res = linprog(
        -b,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=[(None, None)] * m,
        method="highs",
    )
    if not res.success:
        return SolveResult(
            value=float("nan"),
            q=np.zeros(A.shape[1]),
            success=False,
            message=f"dual:{res.message}",
            used_dual=True,
        )

    value = float(-res.fun)
    q = np.full(A.shape[1], np.nan)
    recovered = False
    msg = res.message + " [dual]"

    if recover_primal:
        y = np.asarray(res.x, dtype=float)
        q_try = _recover_from_dual_marginals(lp, free, res)
        if q_try is not None and _primal_feasible(lp, q_try, c_primal, value):
            q, recovered = q_try, True
            msg += "+witness"
        else:
            q_try = _recover_restricted_primal(lp, c_primal, y, free)
            if q_try is not None and _primal_feasible(lp, q_try, c_primal, value):
                q, recovered = q_try, True
                msg += "+witness/restricted"
            else:
                msg += "+witness_failed"

    return SolveResult(
        value=value,
        q=q,
        success=True,
        message=msg,
        used_dual=True,
        recovered_primal=recovered,
    )


def _solve_one(
    lp: LPData,
    maximize: bool,
    *,
    recover_primal: bool = True,
    force_dual: bool = False,
) -> SolveResult:
    c_primal = -lp.c if maximize else lp.c
    use_dual = should_use_dual(lp, force=force_dual)
    if use_dual:
        res = _solve_dual_eq_only(lp, c_primal, recover_primal=recover_primal)
        if not res.success:
            res = _solve_primal(lp, c_primal)
        elif recover_primal and not res.recovered_primal:
            # Dual value ok but witness missing — fall back to full primal once
            prim = _solve_primal(lp, c_primal)
            if prim.success:
                return SolveResult(
                    value=-prim.value if maximize else prim.value,
                    q=prim.q,
                    success=True,
                    message=prim.message + " [primal_fallback]",
                    used_dual=False,
                    recovered_primal=True,
                )
    else:
        res = _solve_primal(lp, c_primal)

    if not res.success:
        return res
    val = -res.value if maximize else res.value
    return SolveResult(
        value=float(val),
        q=res.q,
        success=True,
        message=res.message,
        used_dual=res.used_dual,
        recovered_primal=res.recovered_primal or (not res.used_dual),
    )


def _solve_fractional_one(lp: LPData, *, maximize: bool) -> SolveResult:
    """Charnes–Cooper transform for ``(cᵀq)/(c_denᵀq)``.

    Set ``y = t q`` with ``t = 1/(c_denᵀq)``, so ``c_denᵀ y = 1``,
    ``A_eq y = b_eq t``, ``A_ub y ≤ b_ub t``, ``y,t ≥ 0``. Optimize ``cᵀ y``.
    """
    if lp.c_den is None:
        raise ValueError("fractional solve requires lp.c_den")
    n = len(lp.c)
    # Variables: [y_0..y_{n-1}, t]
    c_obj = np.zeros(n + 1)
    c_obj[:n] = -lp.c if maximize else lp.c

    # A_eq y - b_eq t = 0
    Aeq = lp.A_eq.tocsr()
    m = Aeq.shape[0]
    A_eq_ext = sparse.hstack(
        [Aeq, -sparse.csr_matrix(lp.b_eq.reshape(-1, 1))],
        format="csr",
    )
    b_eq = np.zeros(m)

    # c_denᵀ y = 1
    den_row = sparse.csr_matrix(np.append(lp.c_den, 0.0).reshape(1, -1))
    A_eq_ext = sparse.vstack([A_eq_ext, den_row], format="csr")
    b_eq = np.append(b_eq, 1.0)

    A_ub_ext = None
    b_ub = None
    if lp.A_ub is not None:
        Aub = lp.A_ub.tocsr()
        A_ub_ext = sparse.hstack(
            [Aub, -sparse.csr_matrix(lp.b_ub.reshape(-1, 1))],
            format="csr",
        )
        b_ub = np.zeros(Aub.shape[0])

    bounds = []
    for t in range(n):
        bounds.append((0.0, None) if lp.allowed_types[t] else (0.0, 0.0))
    bounds.append((0.0, None))  # t

    res = linprog(
        c=c_obj,
        A_ub=A_ub_ext,
        b_ub=b_ub,
        A_eq=A_eq_ext,
        b_eq=b_eq,
        bounds=bounds,
        method="highs",
    )
    if not res.success:
        return SolveResult(
            value=float("nan"),
            q=np.zeros(n),
            success=False,
            message=f"fractional:{res.message}",
        )
    y = np.asarray(res.x[:n], dtype=float)
    t = float(res.x[n])
    if t <= 1e-15:
        return SolveResult(
            value=float("nan"),
            q=np.zeros(n),
            success=False,
            message="fractional: t≈0 (denom vanished)",
        )
    q = y / t
    val = float(lp.c @ q)
    return SolveResult(
        value=val,
        q=q,
        success=True,
        message=res.message + " [Charnes-Cooper]",
        used_dual=False,
        recovered_primal=True,
    )


def solve_bounds_lp(
    lp: LPData,
    *,
    recover_primal: bool = True,
    force_dual: bool = False,
) -> tuple[SolveResult, SolveResult]:
    """Return (lower, upper) SolveResults.

    When the dual shortcut is used, ``recover_primal=True`` (default) restores
    feasible witness vectors ``q`` via dual inequality multipliers (or a
    restricted primal on the active dual face). Set False to skip witnesses
    and keep dual-only speed.

    ``force_dual=True`` attempts the dual on any equality-only LP (no size gate).

    Linear-fractional objectives (``lp.c_den`` set) use Charnes–Cooper.
    """
    if lp.c_den is not None or lp.method_hint == "fractional":
        return (
            _solve_fractional_one(lp, maximize=False),
            _solve_fractional_one(lp, maximize=True),
        )
    return (
        _solve_one(lp, maximize=False, recover_primal=recover_primal, force_dual=force_dual),
        _solve_one(lp, maximize=True, recover_primal=recover_primal, force_dual=force_dual),
    )
