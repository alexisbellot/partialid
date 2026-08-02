"""Public high-level API."""

from __future__ import annotations

import warnings as py_warnings
from dataclasses import dataclass, field
from time import perf_counter
from typing import Literal, Mapping, Sequence

import numpy as np

from .assumptions import AssumptionSet, as_assumption_set
from .backends import prefer_colgen_auto, should_use_dual, solve_bounds_lp
from .c_canonical import bound_c_canonical
from .column_gen import solve_bounds_colgen, solve_bounds_colgen_lazy
from .convex_solve import solve_bounds_convex
from .fiber_solve import is_fiber_separable, solve_bounds_fiber
from .formulate import formulate
from .hyperarc import solve_hyperarc
from .identify import try_identify
from .lazy_types import LazyTypeSpace
from .mip_solve import solve_bounds_mip
from .msm_aggregate import applicable_msm_aggregate, bound_msm_aggregate
from .parametric import BoundPathResult, bound_path
from .query import Query
from .variables import Variables

# Re-export path API for `from partialid.api import bound_path`.
__all__ = [
    "BoundResult",
    "BoundPathResult",
    "bound",
    "bound_grid",
    "bound_grid_cold",
    "bound_path",
]


Method = Literal[
    "auto",
    "closed",
    "fiber",
    "lp",
    "dual",
    "colgen",
    "mip",
    "convex",
    "hyperarc",
    "c_canonical",
    "gpp",
    "msm_agg",
    "fractional",
]


def _select_auto_method(lp) -> str:
    """Track A2: fiber → dual-sized lp → colgen → dense lp."""
    if lp.method_hint in ("mip", "convex", "gpp", "gpp/latent", "gpp/poly", "fractional"):
        return "gpp" if lp.method_hint.startswith("gpp") else lp.method_hint
    if getattr(lp, "c_den", None) is not None:
        return "fractional"
    if is_fiber_separable(lp):
        return "fiber"
    if should_use_dual(lp):
        return "lp"  # backends dual shortcut; tagged via notes / method lp/dual
    if prefer_colgen_auto(lp):
        return "colgen"
    return "lp"


@dataclass
class BoundResult:
    lower: float
    upper: float
    width: float
    q_lower: np.ndarray | None = None
    q_upper: np.ndarray | None = None
    method: str = "lp"
    success: bool = True
    message: str = ""
    formulate_time: float = 0.0
    solve_time: float = 0.0
    notes: list[str] = field(default_factory=list)
    tight: bool = True
    warnings: list[str] = field(default_factory=list)

    @property
    def interval(self) -> tuple[float, float]:
        return (self.lower, self.upper)

    def to_dict(self) -> dict:
        """JSON-friendly summary for notebooks / logging."""
        return {
            "lower": float(self.lower),
            "upper": float(self.upper),
            "width": float(self.width),
            "interval": [float(self.lower), float(self.upper)],
            "method": self.method,
            "success": bool(self.success),
            "tight": bool(self.tight),
            "message": self.message,
            "notes": list(self.notes),
            "warnings": list(self.warnings),
            "formulate_time": float(self.formulate_time),
            "solve_time": float(self.solve_time),
        }

    def summary(self, query: Query | None = None) -> str:
        """Human-readable multi-line summary (interval, method, tightness)."""
        lines: list[str] = []
        if query is not None:
            lines.append(f"query: {query.describe()}")
        lines.append(f"interval: [{self.lower:.6g}, {self.upper:.6g}]")
        lines.append(f"width: {self.width:.6g}")
        lines.append(f"method: {self.method}")
        lines.append(f"tight: {self.tight}")
        lines.append(f"success: {self.success}")
        if self.message:
            lines.append(f"message: {self.message}")
        if self.warnings:
            lines.append("warnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")
        elif not self.tight:
            lines.append("note: tight=False (valid outer approximation; see docs/stability.md)")
        return "\n".join(lines)

    def __repr__(self) -> str:
        flag = "" if self.tight else ", tight=False"
        return (
            f"BoundResult(lower={self.lower:.6g}, upper={self.upper:.6g}, "
            f"width={self.width:.6g}, method={self.method!r}{flag})"
        )


def _emit_bound_warnings(result: BoundResult) -> BoundResult:
    for w in result.warnings:
        py_warnings.warn(w, UserWarning, stacklevel=3)
    return result


def _with_tightness(
    result: BoundResult,
    *,
    tight: bool | None = None,
    outer: bool = False,
    extra_warnings: Sequence[str] | None = None,
) -> BoundResult:
    warns = list(result.warnings)
    if extra_warnings:
        warns.extend(extra_warnings)
    is_tight = result.tight if tight is None else tight
    if outer:
        is_tight = False
        msg = (
            "Returned bounds may not be tight: multi c-component / merged-latent "
            "outer LP (valid outer approximation; unidentified factors not "
            "separated by ID)."
        )
        if msg not in warns:
            warns.append(msg)
        if "bounds_may_be_outer" not in result.notes:
            result.notes = list(result.notes) + ["bounds_may_be_outer"]
    result.tight = is_tight
    result.warnings = warns
    return _emit_bound_warnings(result)


def bound(
    variables: Variables,
    data: Mapping,
    query: Query,
    assumptions: AssumptionSet | None = None,
    *,
    method: Method = "auto",
    optimize: bool = True,
    recover_primal: bool = True,
    prefer_gpp: bool = True,
    prefer_poly: bool = False,
) -> BoundResult:
    """Compute tight numeric bounds on ``query`` under ``assumptions``.

    Parameters
    ----------
    variables :
        Discrete endogenous variables and domains.
    data :
        Observational distribution. Keys are tuples in the causal order
        (or declaration order if no order/diagram is given), values are
        probabilities (renormalized if needed).
    query :
        Causal query (see :class:`Query`).
    assumptions :
        Composable assumption set from :class:`Assumptions`.
    method :
        ``\"auto\"`` picks fiber when separable; else dual LP when
        ``n_vars ≫ n_eq``; else matrix-free colgen if types are huge; else LP.
        ``\"dual\"`` forces the equality-only dual shortcut (falls back to primal).
        ``\"colgen\"`` matrix-free column generation.
        ``\"msm_agg\"`` sensitivity-native MSM / c-dep aggregate (binary-X,
        arbitrary |Y|; MTR; IV R-side when ``WR={X,Y}``).
        ``\"hyperarc\"`` Shridharan–Iyengar hyperarc aggregation.
        ``\"c_canonical\"`` finite-latent cardinality (NeurIPS 2022 Thm 2).
    optimize :
        If False, disable inert pruning and instrument factorization (naive baseline).
    recover_primal :
        When the dual LP shortcut is used (large eq-only programs), restore
        witness vectors ``q_lower`` / ``q_upper``. Set False to skip recovery
        and keep dual-only speed (witnesses will be NaN).
    prefer_gpp :
        When classify yields bilinear factor/latent, build a GPP (default True).
        Set False to force the merged-latent outer LP.
    prefer_poly :
        When classify yields POLYNOMIAL (deg≥3), try nested multilinear
        ``gpp/poly`` (default False → outer LP with ``tight=False``).
    """
    assumptions = as_assumption_set(assumptions)
    notes_pre: list[str] = []

    # Pretreatment conditioning: slice P before closed/ID/formulate.
    # Skip for hyperarc — it consumes ``condition`` as VA context (Thm 8).
    if (
        query.kind == "interventional"
        and (query.meta or {}).get("condition")
        and method != "hyperarc"
    ):
        from .assumptions import OrderAssumption
        from .condition_slice import apply_pretreatment_condition
        from .graph import CausalDiagram

        diagram0 = assumptions.diagram()
        if diagram0 is None:
            orders = assumptions.of_type(OrderAssumption)
            if orders:
                diagram0 = CausalDiagram.from_order(orders[0].order)
        variables, data, query, assumptions = apply_pretreatment_condition(
            variables, data, query, diagram0, notes_pre, assumptions
        )

    def _tag_notes(result: BoundResult) -> BoundResult:
        if notes_pre:
            result.notes = list(notes_pre) + list(result.notes)
        return result

    # Analytic Moves (closed forms) before numeric solvers
    if method in ("auto", "closed") and optimize:
        from .closed_forms import try_closed_form

        t_cf0 = perf_counter()
        cfr = try_closed_form(variables, data, query, assumptions)
        t_cf1 = perf_counter()
        if cfr is not None:
            return _tag_notes(
                _with_tightness(
                    BoundResult(
                        lower=float(cfr.lower),
                        upper=float(cfr.upper),
                        width=float(cfr.upper - cfr.lower),
                        method=f"closed/{cfr.formula_id}",
                        success=True,
                        message="closed form",
                        formulate_time=t_cf1 - t_cf0,
                        solve_time=0.0,
                        notes=list(cfr.notes),
                        tight=True,
                    )
                )
            )
        if method == "closed":
            return BoundResult(
                lower=float("nan"),
                upper=float("nan"),
                width=float("nan"),
                method="closed",
                success=False,
                message="no matching closed-form pattern",
                notes=["closed_form_unavailable"],
                tight=False,
            )

    # ID reduction for explicit diagrams (before LP), when not forcing a special solver
    if optimize and method in ("auto", "fiber", "lp") and query.kind in (
        "interventional",
        "ate",
    ):
        t_id0 = perf_counter()
        idr = try_identify(variables, data, query, assumptions)
        t_id1 = perf_counter()
        if idr.identified and idr.value is not None:
            # criterion is already "id" or "id/ate"
            method_tag = (
                idr.criterion
                if idr.criterion.startswith("id")
                else f"id/{idr.criterion or 'id'}"
            )
            return _tag_notes(
                _with_tightness(
                    BoundResult(
                        lower=float(idr.value),
                        upper=float(idr.value),
                        width=0.0,
                        method=method_tag,
                        success=True,
                        message="point-identified",
                        formulate_time=t_id1 - t_id0,
                        solve_time=0.0,
                        notes=list(idr.notes),
                        tight=True,
                    )
                )
            )

    # Sensitivity-native aggregate (MSM / c-dep / anchored Γ on binary-X XY)
    if method in ("auto", "msm_agg") and optimize and query.kind == "interventional":
        from .assumptions import (
            CDependenceAssumption,
            RosenbaumAssumption,
            TanMSMAssumption,
        )

        treat = "X"
        for Atom in (TanMSMAssumption, RosenbaumAssumption, CDependenceAssumption):
            atoms = assumptions.of_type(Atom)
            if atoms:
                treat = atoms[0].treatment
                break
        ok_agg, _why, _kind, _val, _mode = applicable_msm_aggregate(
            variables, query, assumptions, treatment=treat
        )
        if ok_agg or method == "msm_agg":
            t_a0 = perf_counter()
            ar = bound_msm_aggregate(variables, data, query, assumptions, treatment=treat)
            t_a1 = perf_counter()
            if ar.success:
                return _tag_notes(
                    _with_tightness(
                        BoundResult(
                            lower=ar.lower,
                            upper=ar.upper,
                            width=ar.width,
                            method=f"msm_agg/{ar.formula_id or 'transport'}",
                            success=True,
                            message=ar.message,
                            formulate_time=t_a1 - t_a0,
                            solve_time=0.0,
                            notes=list(ar.notes) + ([f"n_agg={ar.n_agg}"] if ar.n_agg else []),
                            tight=True,
                        )
                    )
                )
            if method == "msm_agg":
                return BoundResult(
                    lower=float("nan"),
                    upper=float("nan"),
                    width=float("nan"),
                    method="msm_agg",
                    success=False,
                    message=ar.message,
                    notes=list(ar.notes),
                    tight=False,
                )

    if method == "c_canonical":
        cr = bound_c_canonical(variables, data, query, assumptions, optimize=optimize)
        return BoundResult(
            lower=cr.lower,
            upper=cr.upper,
            width=cr.width,
            q_lower=cr.q_lower,
            q_upper=cr.q_upper,
            method=f"c_canonical/{cr.method}",
            success=cr.success,
            message=cr.message,
            formulate_time=cr.formulate_time,
            solve_time=cr.solve_time,
            notes=cr.notes,
        )

    if method == "hyperarc":
        diagram = assumptions.diagram()
        if diagram is None:
            raise ValueError("hyperarc requires order/diagram assumptions")
        t0 = perf_counter()
        hr = solve_hyperarc(variables, data, query, diagram)
        t1 = perf_counter()
        return BoundResult(
            lower=hr.lower,
            upper=hr.upper,
            width=(hr.upper - hr.lower) if hr.success else float("nan"),
            q_lower=hr.q_lower,
            q_upper=hr.q_upper,
            method="hyperarc",
            success=hr.success,
            message=hr.message,
            formulate_time=t1 - t0,
            solve_time=0.0,
            notes=hr.notes
            + [
                f"|H|={hr.n_hyperarcs}",
                f"raw≈{hr.n_raw_types_est}",
                "closed_form" if hr.closed_form else "hyperarc_lp",
            ],
        )

    # Fully lazy colgen: avoid materializing TypeSpace when N is huge.
    # Apply inert prune first — otherwise order-only chains with N̂≫4096
    # take the lazy path on the *unpruned* diagram and can hang.
    if method == "colgen" and query.kind == "interventional" and optimize:
        from .assumptions import (
            CDependenceAssumption,
            FSensitivityAssumption,
            MarginBudgetAssumption,
            RosenbaumAssumption,
            SupportBoundAssumption,
            TanMSMAssumption,
        )
        from .formulate import _inert_prune_order_only, _normalize_P

        coupling = (
            assumptions.of_type(TanMSMAssumption)
            or assumptions.of_type(CDependenceAssumption)
            or assumptions.of_type(RosenbaumAssumption)
            or assumptions.of_type(FSensitivityAssumption)
            or assumptions.of_type(MarginBudgetAssumption)
            or assumptions.of_type(SupportBoundAssumption)
        )
        diagram = assumptions.diagram()
        if diagram is not None and not coupling:
            prune_notes: list[str] = []
            V_lazy, data_lazy, g_lazy = _inert_prune_order_only(
                variables, data, query, diagram, prune_notes
            )
            lazy = LazyTypeSpace.from_diagram(V_lazy, g_lazy)
            if lazy.n_types_est > 4096:
                t0 = perf_counter()
                cells, P = _normalize_P(V_lazy, lazy.order, data_lazy)
                cg = solve_bounds_colgen_lazy(lazy, cells, P, query)
                t1 = perf_counter()
                return BoundResult(
                    lower=cg.lower,
                    upper=cg.upper,
                    width=(cg.upper - cg.lower) if cg.success else float("nan"),
                    method="colgen/lazy",
                    success=cg.success,
                    message=cg.message,
                    formulate_time=0.0,
                    solve_time=t1 - t0,
                    notes=prune_notes + cg.notes + [f"colgen_iters={cg.iterations}"],
                )

    t0 = perf_counter()
    lp = formulate(
        variables,
        data,
        query,
        assumptions,
        optimize=optimize,
        prefer_gpp=prefer_gpp,
        prefer_poly=prefer_poly,
    )
    t1 = perf_counter()

    def _finish(result: BoundResult) -> BoundResult:
        return _tag_notes(_with_tightness(result, outer=bool(lp.outer_approximation)))

    use = method
    auto_routed_dual = False
    auto_routed_colgen = False
    if method == "auto":
        use = _select_auto_method(lp)
        auto_routed_dual = use == "lp" and should_use_dual(lp)
        auto_routed_colgen = use == "colgen"
    elif method == "dual":
        use = "lp"
    if not optimize and use == "fiber":
        use = "lp"
    force_dual = method == "dual"

    gpp_hint = lp.method_hint in ("gpp", "gpp/latent", "gpp/poly")
    if use == "gpp" or (gpp_hint and method in ("auto", "gpp", "lp")):
        from .gpp import solve_bounds_gpp
        from .poly_gpp import PolyGPPData, solve_bounds_poly

        if lp.gpp is None:
            raise ValueError("method='gpp' requires a factorized GPP formulation")
        if isinstance(lp.gpp, PolyGPPData) or lp.method_hint == "gpp/poly":
            gb = solve_bounds_poly(lp.gpp)
            method_tag = "gpp/poly"
        else:
            gb = solve_bounds_gpp(lp.gpp)
            method_tag = "gpp/latent" if lp.method_hint == "gpp/latent" else "gpp"
        t2 = perf_counter()
        if not gb.success:
            t_fb0 = perf_counter()
            lp_outer = formulate(
                variables,
                data,
                query,
                assumptions,
                optimize=optimize,
                prefer_gpp=False,
                prefer_poly=False,
            )
            t_fb1 = perf_counter()
            lo_r, hi_r = solve_bounds_lp(lp_outer, recover_primal=recover_primal)
            t2 = perf_counter()
            success = lo_r.success and hi_r.success
            lo = lo_r.value if lo_r.success else float("nan")
            hi = hi_r.value if hi_r.success else float("nan")
            return _with_tightness(
                BoundResult(
                    lower=lo,
                    upper=hi,
                    width=(hi - lo) if success else float("nan"),
                    q_lower=lo_r.q if lo_r.success else None,
                    q_upper=hi_r.q if hi_r.success else None,
                    method="lp",
                    success=success,
                    message=f"GPP failed ({gb.message}); outer LP fallback",
                    formulate_time=(t1 - t0) + (t_fb1 - t_fb0),
                    solve_time=t2 - t_fb1,
                    notes=list(lp.notes)
                    + list(gb.notes)
                    + list(lp_outer.notes)
                    + ["gpp_infeasible_outer_fallback"],
                ),
                outer=True,
            )
        return _with_tightness(
            BoundResult(
                lower=gb.lower,
                upper=gb.upper,
                width=(gb.upper - gb.lower) if gb.success else float("nan"),
                method=method_tag,
                success=gb.success,
                message=gb.message,
                formulate_time=t1 - t0,
                solve_time=t2 - t1,
                notes=list(lp.notes) + list(gb.notes),
                tight=bool(gb.certified and gb.success),
            ),
            tight=bool(gb.certified and gb.success),
            outer=False,
            extra_warnings=(
                None
                if gb.certified
                else [
                    "GPP spatial BB: attainable nested-LP bounds; "
                    "global certificate incomplete for this block size."
                ]
            ),
        )

    if use == "fiber":
        if not is_fiber_separable(lp):
            raise ValueError("fiber method requires no coupling inequalities (no MSM/IV/…)")
        lo, hi, q_lo, q_hi = solve_bounds_fiber(lp)
        t2 = perf_counter()
        return _finish(
            BoundResult(
                lower=lo,
                upper=hi,
                width=hi - lo,
                q_lower=q_lo,
                q_upper=q_hi,
                method="fiber",
                success=True,
                formulate_time=t1 - t0,
                solve_time=t2 - t1,
                notes=lp.notes,
            )
        )

    if use == "colgen":
        cg = solve_bounds_colgen(lp, matrix_free=True)
        t2 = perf_counter()
        notes = list(lp.notes) + [
            f"colgen_iters={cg.iterations}",
            "matrix_free_pricing" if cg.matrix_free else "dense_pricing",
        ]
        if lp.A_ub is not None:
            notes.append("budget_aware_pricing")  # Track A3: A_ub'μ in reduced costs
        if auto_routed_colgen:
            notes.append("auto_routed_colgen")
        return _finish(
            BoundResult(
                lower=cg.lower,
                upper=cg.upper,
                width=cg.upper - cg.lower,
                q_lower=cg.q_lower,
                q_upper=cg.q_upper,
                method="colgen",
                success=cg.success,
                message=cg.message,
                formulate_time=t1 - t0,
                solve_time=t2 - t1,
                notes=notes,
            )
        )

    if use == "mip" or lp.support_k is not None:
        mb = solve_bounds_mip(lp)
        t2 = perf_counter()
        success = mb.lower.success and mb.upper.success
        lo = mb.lower.value if mb.lower.success else float("nan")
        hi = mb.upper.value if mb.upper.success else float("nan")
        return _finish(
            BoundResult(
                lower=lo,
                upper=hi,
                width=(hi - lo) if success else float("nan"),
                q_lower=mb.lower.q if mb.lower.success else None,
                q_upper=mb.upper.q if mb.upper.success else None,
                method="mip",
                success=success,
                message=f"lo:{mb.lower.message}; hi:{mb.upper.message}",
                formulate_time=t1 - t0,
                solve_time=t2 - t1,
                notes=lp.notes,
            )
        )

    if use == "convex" or lp.f_sensitivity is not None:
        cb = solve_bounds_convex(lp)
        t2 = perf_counter()
        success = cb.lower.success and cb.upper.success
        lo = cb.lower.value if cb.lower.success else float("nan")
        hi = cb.upper.value if cb.upper.success else float("nan")
        return _finish(
            BoundResult(
                lower=lo,
                upper=hi,
                width=(hi - lo) if success else float("nan"),
                q_lower=cb.lower.q if cb.lower.success else None,
                q_upper=cb.upper.q if cb.upper.success else None,
                method="convex",
                success=success,
                message=f"lo:{cb.lower.message}; hi:{cb.upper.message}",
                formulate_time=t1 - t0,
                solve_time=t2 - t1,
                notes=lp.notes,
            )
        )

    # full LP (optional dual shortcut inside backends)
    lo_r, hi_r = solve_bounds_lp(lp, recover_primal=recover_primal, force_dual=force_dual)
    t2 = perf_counter()
    success = lo_r.success and hi_r.success
    lo = lo_r.value if lo_r.success else float("nan")
    hi = hi_r.value if hi_r.success else float("nan")
    notes = list(lp.notes)
    used_dual = bool(lo_r.used_dual or hi_r.used_dual)
    if used_dual:
        notes.append("dual_lp")
        if recover_primal and lo_r.recovered_primal and hi_r.recovered_primal:
            notes.append("primal_witnesses")
        elif not recover_primal:
            notes.append("primal_skipped")
    if auto_routed_dual and used_dual:
        notes.append("auto_routed_dual")
    if method == "dual":
        notes.append("force_dual")
    method_tag = "lp/dual" if used_dual else "lp"
    return _finish(
        BoundResult(
            lower=lo,
            upper=hi,
            width=(hi - lo) if success else float("nan"),
            q_lower=lo_r.q if lo_r.success else None,
            q_upper=hi_r.q if hi_r.success else None,
            method=method_tag,
            success=success,
            message=f"lo:{lo_r.message}; hi:{hi_r.message}",
            formulate_time=t1 - t0,
            solve_time=t2 - t1,
            notes=notes,
        )
    )


def bound_grid(
    variables: Variables,
    data: Mapping,
    query: Query,
    base_assumptions: AssumptionSet | None = None,
    *,
    param: str = "Lambda",
    values: Sequence[float],
    treatment: str = "X",
    method: Method = "lp",
    warm_start: bool = True,
    optimize: bool = True,
) -> list[tuple[float, BoundResult]]:
    """Sweep a sensitivity budget and return ``(value, BoundResult)`` pairs.

    Builds the type space / equality block once. With ``warm_start=True`` and
    optional ``highspy`` installed, keeps a persistent HiGHS model and only
    updates budget-dependent coefficients between points (P2).

    Parameters
    ----------
    param :
        ``\"Lambda\"`` (Tan MSM), ``\"Gamma\"`` (anchored Rosenbaum),
        ``\"c\"`` (c-dependence), or ``\"kappa\"`` (margin budget).
    """
    from .highs_grid import highspy_available, solve_sensitivity_grid_highspy
    from .sensitivity_grid import SensitivityTemplate

    if method not in ("auto", "lp"):
        method = "lp"
    if param not in ("Lambda", "Gamma", "c", "kappa"):
        raise ValueError(f"unknown param {param!r}")

    base = as_assumption_set(base_assumptions)
    t_form0 = perf_counter()
    tmpl = SensitivityTemplate.build(
        variables, data, query, base, treatment=treatment, optimize=optimize
    )
    template_time = perf_counter() - t_form0
    vals = [float(v) for v in values]

    use_highspy = warm_start and highspy_available()
    highspy_note = (
        None
        if use_highspy or not warm_start
        else "highspy_unavailable; install partialid for persistent HiGHS grids"
    )
    if use_highspy:
        # Ascending budgets keep the previous basis closer to feasible/optimal
        order = sorted(range(len(vals)), key=lambda i: vals[i])
        ordered_vals = [vals[i] for i in order]
        t0 = perf_counter()
        points = solve_sensitivity_grid_highspy(tmpl, param, ordered_vals)  # type: ignore[arg-type]
        _ = perf_counter() - t0
        out_slots: list[tuple[float, BoundResult] | None] = [None] * len(vals)
        for ord_i, pt in enumerate(points):
            idx = order[ord_i]
            success = pt.lower.success and pt.upper.success
            lo = pt.lower.value if pt.lower.success else float("nan")
            hi = pt.upper.value if pt.upper.success else float("nan")
            out_slots[idx] = (
                vals[idx],
                BoundResult(
                    lower=lo,
                    upper=hi,
                    width=(hi - lo) if success else float("nan"),
                    q_lower=pt.lower.q if pt.lower.success else None,
                    q_upper=pt.upper.q if pt.upper.success else None,
                    method="lp",
                    success=success,
                    message=f"lo:{pt.lower.message}; hi:{pt.upper.message}",
                    formulate_time=template_time if ord_i == 0 else 0.0,
                    solve_time=pt.solve_time + pt.update_time,
                    notes=list(tmpl.notes)
                    + [
                        f"template_build={template_time:.4g}s",
                        "highspy_persistent",
                    ],
                ),
            )
        return [s for s in out_slots if s is not None]  # type: ignore[misc]

    # Fallback: rebuild A_ub each point, SciPy HiGHS (no true warm start)
    out: list[tuple[float, BoundResult]] = []
    for i, v in enumerate(vals):
        t0 = perf_counter()
        lp = tmpl.lp_for(param, v)  # type: ignore[arg-type]
        t1 = perf_counter()
        lo_r, hi_r = solve_bounds_lp(lp)
        t2 = perf_counter()
        success = lo_r.success and hi_r.success
        lo = lo_r.value if lo_r.success else float("nan")
        hi = hi_r.value if hi_r.success else float("nan")
        out.append(
            (
                v,
                BoundResult(
                    lower=lo,
                    upper=hi,
                    width=(hi - lo) if success else float("nan"),
                    q_lower=lo_r.q if lo_r.success else None,
                    q_upper=hi_r.q if hi_r.success else None,
                    method="lp",
                    success=success,
                    message=f"lo:{lo_r.message}; hi:{hi_r.message}",
                    formulate_time=(t1 - t0) + (template_time if i == 0 else 0.0),
                    solve_time=t2 - t1,
                    notes=list(lp.notes)
                    + [
                        f"template_build={template_time:.4g}s",
                        "template_scipy",
                        *([highspy_note] if highspy_note else []),
                    ],
                ),
            )
        )
    return out


def bound_grid_cold(
    variables: Variables,
    data: Mapping,
    query: Query,
    base_assumptions: AssumptionSet | None = None,
    *,
    param: str = "Lambda",
    values: Sequence[float],
    treatment: str = "X",
    method: Method = "lp",
) -> list[tuple[float, BoundResult]]:
    """Legacy per-point formulate+solve (no template reuse). For benchmarks."""
    from .assumptions import Assumptions

    base = as_assumption_set(base_assumptions)
    out: list[tuple[float, BoundResult]] = []
    for v in values:
        if param == "Lambda":
            extra = Assumptions.tan_msm(treatment, v)
        elif param == "Gamma":
            extra = Assumptions.rosenbaum_gamma(treatment, v, anchored=True)
        elif param == "c":
            extra = Assumptions.c_dependence(treatment, v)
        elif param == "kappa":
            extra = Assumptions.margin_budget(v)
        else:
            raise ValueError(f"unknown param {param!r}")
        r = bound(variables, data, query, base & extra, method=method)
        out.append((float(v), r))
    return out
