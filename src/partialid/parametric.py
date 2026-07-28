"""Track B — budget-parametric identification paths.

Treats a sensitivity budget θ (Tan Λ, Rosenbaum Γ, c-dependence, margin κ)
as a **path**: one template LP, many θ, optional PWL breakpoints / closed form.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Callable, Literal, Mapping, Sequence

import numpy as np

from .assumptions import Assumptions, AssumptionSet, as_assumption_set
from .backends import solve_bounds_lp
from .closed_forms import _xy_binary_p
from .msm_aggregate import (
    msm_agg_path_binary_xy,
    msm_binary_xy_bounds,
    msm_binary_xy_breakdown,
)
from .query import Query
from .sensitivity_grid import SensitivityTemplate
from .variables import Variables

Param = Literal["Lambda", "Gamma", "c", "kappa"]


@dataclass
class BoundPathResult:
    """Bounds as a function of a sensitivity budget.

    Attributes
    ----------
    param :
        Budget name (``Lambda``, ``Gamma``, ``c``, ``kappa``).
    values :
        Evaluation abscissae (sorted). When ``closed_form`` or breakpoints
        mode succeeded, these are the PWL knots; otherwise a dense grid.
    lower, upper, width :
        Arrays aligned with ``values``.
    breakpoints :
        Knots where the active regime / slope is believed to change.
        Always includes the endpoints of ``values`` when nonempty.
    closed_form :
        True when binary-XY Tan / Rosenbaum analytic pieces were used (no LP).
    """

    param: str
    values: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    width: np.ndarray
    breakpoints: np.ndarray
    closed_form: bool = False
    formula_id: str | None = None
    notes: list[str] = field(default_factory=list)
    template_time: float = 0.0
    solve_time: float = 0.0
    # Optional analytic path context for ``at`` / ``breakdown``
    _Pxy: dict | None = field(default=None, repr=False)
    _x_star: int = field(default=1, repr=False)
    _y_star: int = field(default=1, repr=False)
    # Optional exact evaluator for bisection breakdown (IV / conditional / categorical)
    _eval_fn: Callable[[float], tuple[float, float]] | None = field(default=None, repr=False)

    def at(self, theta: float) -> tuple[float, float, float]:
        """Return ``(lower, upper, width)`` at ``theta`` (PWL / analytic)."""
        th = float(theta)
        if self.closed_form and self.param in ("Lambda", "Gamma") and self._Pxy is not None:
            lam = th if self.param == "Lambda" else float(np.sqrt(th))
            if lam < 1.0 - 1e-15:
                raise ValueError("Tan MSM / Rosenbaum requires Λ≥1 (Γ≥1)")
            return msm_binary_xy_bounds(self._Pxy, lam, x_star=self._x_star, y_star=self._y_star)
        if self._eval_fn is not None:
            lo, hi = self._eval_fn(th)
            return float(lo), float(hi), float(hi - lo)
        v = self.values
        if v.size == 0:
            raise ValueError("empty path")
        if th <= v[0]:
            return float(self.lower[0]), float(self.upper[0]), float(self.width[0])
        if th >= v[-1]:
            return float(self.lower[-1]), float(self.upper[-1]), float(self.width[-1])
        i = int(np.searchsorted(v, th) - 1)
        t0, t1 = float(v[i]), float(v[i + 1])
        a = 0.0 if t1 <= t0 else (th - t0) / (t1 - t0)
        lo = (1 - a) * float(self.lower[i]) + a * float(self.lower[i + 1])
        hi = (1 - a) * float(self.upper[i]) + a * float(self.upper[i + 1])
        return lo, hi, hi - lo

    def to_dict(self) -> dict:
        return {
            "param": self.param,
            "values": [float(v) for v in self.values],
            "lower": [float(v) for v in self.lower],
            "upper": [float(v) for v in self.upper],
            "width": [float(v) for v in self.width],
            "breakpoints": [float(v) for v in self.breakpoints],
            "closed_form": bool(self.closed_form),
            "formula_id": self.formula_id,
            "notes": list(self.notes),
        }

    def summary(self, *, at: float | None = None) -> str:
        """Human-readable path summary; optional evaluation at ``at``."""
        lines = [
            f"sensitivity path in {self.param}",
            f"range: [{float(self.values[0]):.6g}, {float(self.values[-1]):.6g}]"
            if self.values.size
            else "range: (empty)",
            f"closed_form: {self.closed_form}",
        ]
        if self.values.size:
            lines.append(
                f"width @ endpoints: {float(self.width[0]):.6g} → {float(self.width[-1]):.6g}"
            )
        if at is not None:
            lo, hi, w = self.at(at)
            lines.append(f"at {self.param}={at:g}: [{lo:.6g}, {hi:.6g}] (width {w:.6g})")
        return "\n".join(lines)

    def breakdown(
        self,
        null: float = 0.0,
        *,
        which: Literal["lower", "upper", "interval"] = "interval",
        lam_max: float = 1e6,
        tol: float = 1e-8,
    ) -> float | None:
        """Budget at which ``null`` enters the bound interval.

        For analytic binary-XY Tan/Γ paths, uses the exact margin formula.
        When an ``_eval_fn`` is attached (IV / conditional / categorical grid
        paths), uses bisection with fresh solves. Else PWL interpolation on
        stored knots (approximate).
        """
        if self.param in ("Lambda", "Gamma") and self.closed_form and self._Pxy is not None:
            lam_star = msm_binary_xy_breakdown(
                self._Pxy,
                x_star=self._x_star,
                y_star=self._y_star,
                null=float(null),
                which=which,
                lam_max=float(lam_max) if self.param == "Lambda" else float(np.sqrt(lam_max)),
            )
            if lam_star is None:
                return None
            return float(lam_star if self.param == "Lambda" else lam_star * lam_star)

        def hits_bounds(lo: float, hi: float) -> bool:
            if not (np.isfinite(lo) and np.isfinite(hi)):
                return False
            if which == "interval":
                return lo - tol <= null <= hi + tol
            if which == "lower":
                return lo <= null + tol
            return hi >= null - tol

        if self._eval_fn is not None and len(self.values) > 0:
            t_lo = float(self.values[0])
            t_hi = (
                float(min(self.values[-1], lam_max))
                if self.param
                in (
                    "Lambda",
                    "Gamma",
                )
                else float(self.values[-1])
            )
            lo0, hi0 = self._eval_fn(t_lo)
            if hits_bounds(lo0, hi0):
                return t_lo
            lo1, hi1 = self._eval_fn(t_hi)
            if not hits_bounds(lo1, hi1):
                return None
            a, b = t_lo, t_hi
            for _ in range(60):
                mid = 0.5 * (a + b)
                lo_m, hi_m = self._eval_fn(mid)
                if hits_bounds(lo_m, hi_m):
                    b = mid
                else:
                    a = mid
            return float(b)

        # Approximate on stored knots (all params, including c / kappa)
        def hits(i: int) -> bool:
            return hits_bounds(float(self.lower[i]), float(self.upper[i]))

        if len(self.values) == 0:
            return None
        if hits(0):
            return float(self.values[0])
        for i in range(1, len(self.values)):
            if hits(i):
                lo0, hi0 = float(self.lower[i - 1]), float(self.upper[i - 1])
                lo1, hi1 = float(self.lower[i]), float(self.upper[i])
                t0, t1 = float(self.values[i - 1]), float(self.values[i])
                if which == "lower" and lo0 > null >= lo1 - 1e-15:
                    if abs(lo0 - lo1) < 1e-15:
                        return t1
                    a = (lo0 - null) / (lo0 - lo1)
                    return float(t0 + a * (t1 - t0))
                if which == "upper" and hi0 < null <= hi1 + 1e-15:
                    if abs(hi1 - hi0) < 1e-15:
                        return t1
                    a = (null - hi0) / (hi1 - hi0)
                    return float(t0 + a * (t1 - t0))
                if which == "interval":
                    return float(t1)
                return float(t1)
        return None

    def __repr__(self) -> str:
        n = len(self.values)
        cf = ", closed_form" if self.closed_form else ""
        return (
            f"BoundPathResult(param={self.param!r}, n={n}, breakpoints={len(self.breakpoints)}{cf})"
        )


def _eval_template_grid(
    tmpl: SensitivityTemplate,
    param: Param,
    values: Sequence[float],
    *,
    warm_start: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, list[str]]:
    """Evaluate bounds on an explicit grid; returns values, lo, hi, width, solve_s, notes."""
    from .highs_grid import highspy_available, solve_sensitivity_grid_highspy

    vals = np.asarray([float(v) for v in values], dtype=float)
    notes = list(tmpl.notes)
    t0 = perf_counter()

    # highspy coeff updates assume binary-root ones/zeros row pattern
    use_highspy = (
        warm_start
        and highspy_available()
        and len(vals) > 0
        and not getattr(tmpl, "use_general_budget", False)
    )
    if use_highspy:
        order = np.argsort(vals)
        ordered = vals[order].tolist()
        points = solve_sensitivity_grid_highspy(tmpl, param, ordered)
        lo = np.empty(len(vals))
        hi = np.empty(len(vals))
        for ord_i, pt in enumerate(points):
            idx = int(order[ord_i])
            lo[idx] = pt.lower.value if pt.lower.success else np.nan
            hi[idx] = pt.upper.value if pt.upper.success else np.nan
        notes.append("highspy_persistent")
    else:
        lo = np.empty(len(vals))
        hi = np.empty(len(vals))
        for i, v in enumerate(vals):
            lp = tmpl.lp_for(param, float(v))
            lo_r, hi_r = solve_bounds_lp(lp, recover_primal=False)
            lo[i] = lo_r.value if lo_r.success else np.nan
            hi[i] = hi_r.value if hi_r.success else np.nan
        notes.append(
            "scipy_template_grid"
            + ("_general_budget" if getattr(tmpl, "use_general_budget", False) else "")
        )

    solve_s = perf_counter() - t0
    width = hi - lo
    return vals, lo, hi, width, solve_s, notes


def _slope_kinks(
    values: np.ndarray,
    y: np.ndarray,
    *,
    rel_tol: float = 0.05,
    abs_tol: float = 1e-4,
) -> list[int]:
    """Indices i (1..n-2) where discrete slope of y changes."""
    n = len(values)
    if n < 3:
        return []
    kinks: list[int] = []
    for i in range(1, n - 1):
        dt0 = values[i] - values[i - 1]
        dt1 = values[i + 1] - values[i]
        if dt0 <= 0 or dt1 <= 0:
            continue
        s0 = (y[i] - y[i - 1]) / dt0
        s1 = (y[i + 1] - y[i]) / dt1
        scale = abs_tol + rel_tol * max(abs(s0), abs(s1), 1e-12)
        if abs(s1 - s0) > scale:
            kinks.append(i)
    return kinks


def _refine_breakpoints(
    tmpl: SensitivityTemplate,
    param: Param,
    values: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    warm_start: bool,
    max_refine: int = 6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[float]]:
    """Bisect intervals around slope kinks; return denser path + breakpoint list."""
    bps = {float(values[0]), float(values[-1])}
    kink_idx = set(_slope_kinks(values, lower)) | set(_slope_kinks(values, upper))
    for i in kink_idx:
        bps.add(float(values[i]))

    # Bisect neighboring segments that contain a kink index
    extras: list[float] = []
    for i in sorted(kink_idx):
        for a, b in ((i - 1, i), (i, i + 1)):
            if a < 0 or b >= len(values):
                continue
            lo_t, hi_t = float(values[a]), float(values[b])
            for _ in range(max_refine):
                mid = 0.5 * (lo_t + hi_t)
                extras.append(mid)
                # probe mid to decide which side keeps the kink (heuristic: keep both halves)
                lo_t, hi_t = mid, hi_t  # one-sided refine toward right; also add left mid once
                break
            extras.append(0.5 * (float(values[a]) + float(values[b])))

    if extras:
        new_vals = np.unique(np.concatenate([values, np.asarray(extras, dtype=float)]))
        new_vals, lo, hi, width, _, _ = _eval_template_grid(
            tmpl, param, new_vals.tolist(), warm_start=warm_start
        )
        kink_idx2 = set(_slope_kinks(new_vals, lo)) | set(_slope_kinks(new_vals, hi))
        for i in kink_idx2:
            bps.add(float(new_vals[i]))
        return new_vals, lo, hi, width, sorted(bps)

    return values, lower, upper, upper - lower, sorted(bps)


def bound_path(
    variables: Variables,
    data: Mapping,
    query: Query,
    base_assumptions: AssumptionSet | None = None,
    *,
    param: Param = "Lambda",
    treatment: str = "X",
    values: Sequence[float] | None = None,
    theta_min: float | None = None,
    theta_max: float | None = None,
    n_grid: int = 17,
    breakpoints: bool = True,
    warm_start: bool = True,
    optimize: bool = True,
    prefer_closed: bool = True,
) -> BoundPathResult:
    """Compute identification bounds as a path in a sensitivity budget.

    Modes
    -----
    1. **Exact binary-XY MSM path:** when ``prefer_closed`` and the instance is
       binary ``X,Y`` + interventional query, returns analytic PWL pieces with
       exact breakpoints (THEORY §8.7.3 generalized; RE1 kink at Λ=7/3).
    2. **Explicit ``values``:** evaluate the template LP on that grid (HiGHS
       persistent when available).
    3. **``breakpoints=True`` (default):** log-grid + slope-kink refinement to
       approximate the PWL identification-gap curve.

    Parameters
    ----------
    param :
        ``Lambda`` (Tan MSM), ``Gamma`` (anchored Rosenbaum), ``c``, or ``kappa``.
    theta_min, theta_max :
        Range when ``values`` is omitted. Defaults: Λ/Γ from 1→20, c from 0→0.5,
        κ from 0→0.5.
    """
    if param not in ("Lambda", "Gamma", "c", "kappa"):
        raise ValueError(f"unknown param {param!r}")

    base = as_assumption_set(base_assumptions)

    # Defaults for range
    if theta_min is None:
        theta_min = 1.0 if param in ("Lambda", "Gamma") else 0.0
    if theta_max is None:
        theta_max = 20.0 if param in ("Lambda", "Gamma") else 0.5

    # --- Exact binary-XY MSM / anchored-Γ path (general P, not only RE1) ---
    if prefer_closed and param in ("Lambda", "Gamma"):
        Pxy = _xy_binary_p(variables, data)
        if (
            Pxy is not None
            and query.kind == "interventional"
            and list(variables.names) == ["X", "Y"]
            and tuple(variables.domains["X"]) == (0, 1)
            and tuple(variables.domains["Y"]) == (0, 1)
            and set(dict(query.intervention)) == {treatment}
            and query.outcome == "Y"
            and query.value in (0, 1)
        ):
            extra = list(values) if values is not None else None
            x_star = int(dict(query.intervention)[treatment])
            y_star = int(query.value)
            vals, lo, hi, w, bps, notes = msm_agg_path_binary_xy(
                Pxy,
                param=param,  # type: ignore[arg-type]
                x_star=x_star,
                y_star=y_star,
                theta_min=float(theta_min),
                theta_max=float(theta_max),
                extra_values=extra,
            )
            return BoundPathResult(
                param=param,
                values=vals,
                lower=lo,
                upper=hi,
                width=w,
                breakpoints=bps,
                closed_form=True,
                formula_id="msm_margin_binary_xy",
                notes=notes,
                _Pxy=dict(Pxy),
                _x_star=x_star,
                _y_star=y_star,
            )

    t_form0 = perf_counter()
    try:
        tmpl = SensitivityTemplate.build(
            variables, data, query, base, treatment=treatment, optimize=optimize
        )
        template_time = perf_counter() - t_form0
        use_template = True
    except ValueError as exc:
        # Instrument / aggregated LPs: fall back to repeated ``bound`` solves
        tmpl = None
        template_time = perf_counter() - t_form0
        use_template = False
        fallback_reason = str(exc)

    if values is not None:
        grid = [float(v) for v in values]
    else:
        if param in ("Lambda", "Gamma"):
            lo = max(float(theta_min), 1.0)
            hi = max(float(theta_max), lo)
            grid = np.geomspace(lo, hi, num=max(n_grid, 3)).tolist()
        else:
            lo = float(theta_min)
            hi = max(float(theta_max), lo)
            grid = np.linspace(lo, hi, num=max(n_grid, 3)).tolist()

    def _make_eval_fn():
        from .api import bound as bound_fn

        def _eval(th: float) -> tuple[float, float]:
            if param == "Lambda":
                extra = Assumptions.tan_msm(treatment, float(th))
            elif param == "Gamma":
                extra = Assumptions.rosenbaum_gamma(treatment, float(th))
            elif param == "c":
                extra = Assumptions.c_dependence(treatment, float(th))
            else:
                extra = Assumptions.margin_budget(float(th))
            r = bound_fn(
                variables,
                data,
                query,
                base & extra,
                method="auto",
                optimize=optimize,
            )
            if not r.success:
                return float("nan"), float("nan")
            return float(r.lower), float(r.upper)

        return _eval

    if not use_template:
        eval_fn = _make_eval_fn()
        vals = np.asarray(grid, dtype=float)
        lo_a = np.empty(len(vals))
        hi_a = np.empty(len(vals))
        t_s0 = perf_counter()
        for i, th in enumerate(vals):
            lo_a[i], hi_a[i] = eval_fn(float(th))
        # Slope-kink refine with fresh solves (IV / instrument path)
        bps_list = [float(vals[0]), float(vals[-1])] if len(vals) else []
        if breakpoints and len(vals) >= 3:
            kink_idx = set(_slope_kinks(vals, lo_a)) | set(_slope_kinks(vals, hi_a))
            extras: list[float] = []
            for i in sorted(kink_idx):
                bps_list.append(float(vals[i]))
                for a, b in ((i - 1, i), (i, i + 1)):
                    if 0 <= a < b < len(vals):
                        extras.append(0.5 * (float(vals[a]) + float(vals[b])))
            if extras:
                new_vals = np.unique(np.concatenate([vals, np.asarray(extras, dtype=float)]))
                lo_n = np.empty(len(new_vals))
                hi_n = np.empty(len(new_vals))
                for i, th in enumerate(new_vals):
                    lo_n[i], hi_n[i] = eval_fn(float(th))
                vals, lo_a, hi_a = new_vals, lo_n, hi_n
                kink_idx2 = set(_slope_kinks(vals, lo_a)) | set(_slope_kinks(vals, hi_a))
                for i in kink_idx2:
                    bps_list.append(float(vals[i]))
                bps_list = sorted(set(bps_list))
        solve_s = perf_counter() - t_s0
        width = hi_a - lo_a
        notes = [
            "Track B bound_path via repeated bound (IV/conditional/instrument)",
            fallback_reason,
        ]
        if breakpoints:
            notes.append(f"breakpoint_refine n={len(vals)} bps={len(bps_list)}")
        return BoundPathResult(
            param=param,
            values=vals,
            lower=lo_a,
            upper=hi_a,
            width=width,
            breakpoints=np.asarray(bps_list, dtype=float),
            closed_form=False,
            formula_id="repeated_bound",
            notes=notes,
            template_time=template_time,
            solve_time=solve_s,
            _eval_fn=eval_fn,
        )

    vals, lo, hi, width, solve_s, notes = _eval_template_grid(
        tmpl, param, grid, warm_start=warm_start
    )
    notes = list(notes)
    notes.append("Track B bound_path template grid")

    bps_list = [float(vals[0]), float(vals[-1])] if len(vals) else []
    if breakpoints and len(vals) >= 3:
        vals, lo, hi, width, bps_list = _refine_breakpoints(
            tmpl, param, vals, lo, hi, warm_start=warm_start
        )
        notes.append(f"breakpoint_refine n={len(vals)} bps={len(bps_list)}")

    eval_fn = _make_eval_fn()
    return BoundPathResult(
        param=param,
        values=vals,
        lower=lo,
        upper=hi,
        width=width,
        breakpoints=np.asarray(bps_list, dtype=float),
        closed_form=False,
        formula_id=None,
        notes=notes,
        template_time=template_time,
        solve_time=solve_s,
        _eval_fn=eval_fn,
    )
