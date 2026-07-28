"""Persistent HiGHS models for sensitivity budget sweeps (optional highspy)."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Literal

import numpy as np

from .backends import SolveResult
from .formulate import LPData
from .sensitivity_grid import MSMGroup, SensitivityTemplate

Param = Literal["Lambda", "Gamma", "c", "kappa"]


def highspy_available() -> bool:
    try:
        import highspy  # noqa: F401

        return True
    except ImportError:
        return False


def _build_model(lp: LPData, *, minimize: bool):
    import highspy

    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    n = len(lp.c)
    for j in range(n):
        ub = highspy.kHighsInf if lp.allowed_types[j] else 0.0
        h.addVar(0.0, ub)
    c = lp.c if minimize else -lp.c
    h.changeColsCost(n, list(range(n)), c.tolist())

    assert lp.A_eq is not None
    Aeq = lp.A_eq.tocsr()
    for i in range(Aeq.shape[0]):
        s, e = Aeq.indptr[i], Aeq.indptr[i + 1]
        b = float(lp.b_eq[i])
        h.addRow(b, b, e - s, Aeq.indices[s:e].tolist(), Aeq.data[s:e].tolist())

    n_eq = Aeq.shape[0]
    if lp.A_ub is not None:
        Aub = lp.A_ub.tocsr()
        for i in range(Aub.shape[0]):
            s, e = Aub.indptr[i], Aub.indptr[i + 1]
            h.addRow(
                -highspy.kHighsInf,
                float(lp.b_ub[i]),
                e - s,
                Aub.indices[s:e].tolist(),
                Aub.data[s:e].tolist(),
            )
    return h, n_eq


def _update_msm_coeffs(h, n_eq: int, groups: list[MSMGroup], theta0: float, Lam: float) -> None:
    r = n_eq
    for g in groups:
        for t in g.zeros:
            h.changeCoeff(r, int(t), -Lam * theta0)
        for t in g.ones:
            h.changeCoeff(r + 1, int(t), -Lam / theta0)
        r += 2


def _update_cdep_coeffs(h, n_eq: int, groups: list[MSMGroup], e0: float, c: float) -> None:
    lo = e0 - c
    hi = e0 + c
    r = n_eq
    for g in groups:
        # row r: -(e0+c) q0 + (1-e0-c) q1 <= 0
        for t in g.zeros:
            h.changeCoeff(r, int(t), -hi)
        for t in g.ones:
            h.changeCoeff(r, int(t), 1.0 - hi)
        # row r+1: (e0-c) q0 + (e0-c-1) q1 <= 0
        for t in g.zeros:
            h.changeCoeff(r + 1, int(t), lo)
        for t in g.ones:
            h.changeCoeff(r + 1, int(t), lo - 1.0)
        r += 2


def _update_kappa_bounds(h, n_eq: int, q0: float, kappa: float) -> None:
    import highspy

    # rows: c'q <= q0+kappa  and  -c'q <= -q0+kappa
    h.changeRowBounds(n_eq, -highspy.kHighsInf, q0 + kappa)
    h.changeRowBounds(n_eq + 1, -highspy.kHighsInf, -q0 + kappa)


def _run_one(h) -> SolveResult:
    import highspy

    h.run()
    info = h.getInfo()
    model_status = h.getModelStatus()
    success = model_status == highspy.HighsModelStatus.kOptimal
    if not success:
        return SolveResult(
            value=float("nan"),
            q=np.zeros(0),
            success=False,
            message=str(model_status),
        )
    sol = h.getSolution()
    q = np.asarray(sol.col_value, dtype=float)
    return SolveResult(
        value=float(info.objective_function_value),
        q=q,
        success=True,
        message="optimal [highspy]",
    )


@dataclass
class GridPointSolve:
    value: float
    lower: SolveResult
    upper: SolveResult
    update_time: float
    solve_time: float


def solve_sensitivity_grid_highspy(
    tmpl: SensitivityTemplate,
    param: Param,
    values: list[float],
) -> list[GridPointSolve]:
    """Persistent dual models: update budget coeffs, re-run (HiGHS hot-starts)."""
    if not values:
        return []

    # Seed LP at first budget (row sparsity pattern fixed)
    seed = float(values[0])
    if param == "Gamma":
        seed_lam = float(np.sqrt(seed))
        lp0 = tmpl.lp_for("Lambda", seed_lam)
    else:
        lp0 = tmpl.lp_for(param, seed)  # type: ignore[arg-type]

    h_lo, n_eq = _build_model(lp0, minimize=True)
    h_hi, _ = _build_model(lp0, minimize=False)

    out: list[GridPointSolve] = []
    for v in values:
        t0 = perf_counter()
        if param == "Lambda":
            _update_msm_coeffs(h_lo, n_eq, tmpl.groups, tmpl.theta0, float(v))
            _update_msm_coeffs(h_hi, n_eq, tmpl.groups, tmpl.theta0, float(v))
        elif param == "Gamma":
            Lam = float(np.sqrt(v))
            _update_msm_coeffs(h_lo, n_eq, tmpl.groups, tmpl.theta0, Lam)
            _update_msm_coeffs(h_hi, n_eq, tmpl.groups, tmpl.theta0, Lam)
        elif param == "c":
            _update_cdep_coeffs(h_lo, n_eq, tmpl.groups, tmpl.e0, float(v))
            _update_cdep_coeffs(h_hi, n_eq, tmpl.groups, tmpl.e0, float(v))
        elif param == "kappa":
            if tmpl.q0_margin is None:
                raise ValueError("margin budget requires a point reference")
            _update_kappa_bounds(h_lo, n_eq, tmpl.q0_margin, float(v))
            _update_kappa_bounds(h_hi, n_eq, tmpl.q0_margin, float(v))
        else:
            raise ValueError(f"unknown param {param!r}")
        t1 = perf_counter()

        lo = _run_one(h_lo)
        hi_raw = _run_one(h_hi)
        t2 = perf_counter()

        # hi model minimizes -c'q, so objective is -upper
        if hi_raw.success:
            hi = SolveResult(
                value=-hi_raw.value,
                q=hi_raw.q,
                success=True,
                message=hi_raw.message,
            )
        else:
            hi = hi_raw

        out.append(
            GridPointSolve(
                value=float(v),
                lower=lo,
                upper=hi,
                update_time=t1 - t0,
                solve_time=t2 - t1,
            )
        )
    return out
