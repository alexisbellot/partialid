"""Statistical inference on identification bounds (P3).

Bootstrap percentile intervals for the identification endpoints and for
a scalar functional of the interval (default: midpoint). The observational
law is resampled as a multinomial with the estimated cell probabilities.

Population object first: given a multinomial sample from P, bootstrap the
bound map P |-> (ell(P), u(P)) and form Imbens-Manski-style intervals
for the partially identified parameter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Literal, Mapping, Sequence

import numpy as np

from .api import bound
from .assumptions import AssumptionSet, as_assumption_set
from .query import Query
from .variables import Variables


@dataclass
class BoundCIResult:
    """Confidence interval for a partially identified scalar."""

    lower: float
    upper: float
    width: float
    point_lower: float
    point_upper: float
    level: float
    method: str
    n_boot: int
    success: bool = True
    message: str = ""
    notes: list[str] = field(default_factory=list)
    formulate_time: float = 0.0
    solve_time: float = 0.0

    def __repr__(self) -> str:
        return (
            f"BoundCIResult(ci=[{self.lower:.4g},{self.upper:.4g}], "
            f"point=[{self.point_lower:.4g},{self.point_upper:.4g}], "
            f"level={self.level}, method={self.method!r})"
        )


def _as_joint_table(variables: Variables, data: Mapping) -> tuple[list[tuple], np.ndarray]:
    order = tuple(variables.names)
    cells = variables.cells(order)
    p = np.zeros(len(cells), dtype=float)
    for key, val in data.items():
        if isinstance(key, dict):
            cell = tuple(key[v] for v in order)
        else:
            cell = tuple(key)
        p[cells.index(cell)] += float(val)
    s = float(p.sum())
    if s <= 0:
        raise ValueError("data has total mass 0")
    p /= s
    return cells, p


def _counts_to_data(cells: Sequence[tuple], counts: np.ndarray) -> dict[tuple, float]:
    total = float(counts.sum())
    if total <= 0:
        raise ValueError("bootstrap draw has zero counts")
    return {c: float(k) / total for c, k in zip(cells, counts) if k > 0}


def bound_ci(
    variables: Variables,
    data: Mapping,
    query: Query,
    assumptions: AssumptionSet | None = None,
    *,
    n: int | None = None,
    n_boot: int = 200,
    level: float = 0.95,
    seed: int = 0,
    method: str = "auto",
    optimize: bool = True,
    style: Literal["percentile", "imbens_manski"] = "imbens_manski",
) -> BoundCIResult:
    """Bootstrap CI for the identification interval.

    Parameters
    ----------
    n :
        Sample size for multinomial resampling. If omitted, uses a default of
        1000 (population probabilities are treated as the true multinomial).
    style :
        ``percentile`` — percentile CI on bootstrapped \\(\\ell\\) and \\(u\\)
        separately (conservative union).
        ``imbens_manski`` — Imbens–Manski (2004) style: expand the point
        interval by the one-sided normal critical value times bootstrap SEs
        of the endpoints (simple plug-in; adequate for smoke / workflow use).
    """
    assumptions = as_assumption_set(assumptions)
    cells, p = _as_joint_table(variables, data)
    n_obs = int(n) if n is not None else 1000
    if not (0.0 < level < 1.0):
        raise ValueError("level must lie in (0,1)")

    t0 = perf_counter()
    point = bound(
        variables,
        data,
        query,
        assumptions,
        method=method,
        optimize=optimize,  # type: ignore[arg-type]
    )
    t1 = perf_counter()
    if not point.success:
        return BoundCIResult(
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            level,
            style,
            n_boot,
            success=False,
            message=point.message,
            notes=list(point.notes),
            formulate_time=t1 - t0,
        )

    rng = np.random.default_rng(seed)
    los = np.empty(n_boot)
    his = np.empty(n_boot)
    t_s0 = perf_counter()
    for b in range(n_boot):
        counts = rng.multinomial(n_obs, p)
        Pb = _counts_to_data(cells, counts)
        rb = bound(
            variables,
            Pb,
            query,
            assumptions,
            method=method,
            optimize=optimize,  # type: ignore[arg-type]
        )
        if not rb.success:
            los[b] = float("nan")
            his[b] = float("nan")
        else:
            los[b] = rb.lower
            his[b] = rb.upper
    t_s1 = perf_counter()

    ok = np.isfinite(los) & np.isfinite(his)
    if ok.sum() < max(10, n_boot // 5):
        return BoundCIResult(
            float("nan"),
            float("nan"),
            float("nan"),
            point.lower,
            point.upper,
            level,
            style,
            n_boot,
            success=False,
            message="too many infeasible bootstrap draws",
            notes=[f"n_ok={int(ok.sum())}/{n_boot}"],
            formulate_time=t1 - t0,
            solve_time=t_s1 - t_s0,
        )

    los_ok, his_ok = los[ok], his[ok]
    alpha = 1.0 - level
    notes = [f"n_obs={n_obs}", f"n_ok={int(ok.sum())}/{n_boot}", f"style={style}"]

    if style == "percentile":
        # Conservative: lower endpoint = α quantile of ℓ, upper = 1-α of u
        ci_lo = float(np.quantile(los_ok, alpha))
        ci_hi = float(np.quantile(his_ok, 1.0 - alpha))
    else:
        # Imbens–Manski plug-in with bootstrap SEs and one-sided z
        try:
            from scipy.stats import norm

            z = float(norm.ppf(1.0 - alpha))
        except Exception:
            z = 1.64 if abs(alpha - 0.05) < 1e-9 else 1.96
        se_lo = float(np.std(los_ok, ddof=1))
        se_hi = float(np.std(his_ok, ddof=1))
        ci_lo = float(point.lower - z * se_lo)
        ci_hi = float(point.upper + z * se_hi)
        notes += [f"z={z:.4g}", f"se_lo={se_lo:.4g}", f"se_hi={se_hi:.4g}"]

    ci_lo = (
        max(0.0, min(1.0, ci_lo)) if query.kind in ("interventional", "pn", "ps", "pns") else ci_lo
    )
    ci_hi = (
        max(0.0, min(1.0, ci_hi)) if query.kind in ("interventional", "pn", "ps", "pns") else ci_hi
    )
    if ci_lo > ci_hi:
        ci_lo, ci_hi = ci_hi, ci_lo

    return BoundCIResult(
        ci_lo,
        ci_hi,
        ci_hi - ci_lo,
        point.lower,
        point.upper,
        level,
        style,
        n_boot,
        success=True,
        message="ok",
        notes=notes,
        formulate_time=t1 - t0,
        solve_time=t_s1 - t_s0,
    )
