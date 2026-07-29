"""Categorical msm_agg, IV/conditional bound_path, aligned f-sensitivity."""

from __future__ import annotations

import numpy as np
import pytest

from partialid import Assumptions, Query, Variables, bound, bound_path
from partialid.convex_solve import cvxpy_available


def test_msm_agg_ternary_matches_lp():
    V = Variables.from_mapping({"X": (0, 1, 2), "Y": (0, 1)})
    rng = np.random.default_rng(0)
    cells = list(V.cells())
    p = rng.dirichlet(np.ones(len(cells)))
    P = {c: float(v) for c, v in zip(cells, p)}
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    a = Assumptions.order("X", "Y") & Assumptions.tan_msm("X", 2.0)
    ra = bound(V, P, Q, a, method="msm_agg")
    rl = bound(V, P, Q, a, method="lp")
    assert ra.success and rl.success, (ra.message, rl.message)
    assert abs(ra.lower - rl.lower) + abs(ra.upper - rl.upper) < 1e-5
    assert any("|X|=3" in n for n in ra.notes)


def test_msm_agg_ternary_lambda1_point_id():
    V = Variables.from_mapping({"X": (0, 1, 2), "Y": (0, 1)})
    rng = np.random.default_rng(1)
    cells = list(V.cells())
    p = rng.dirichlet(np.ones(len(cells)))
    P = {c: float(v) for c, v in zip(cells, p)}
    Q = Query.interventional(outcome="Y", do={"X": 2}, value=1)
    a = Assumptions.order("X", "Y") & Assumptions.tan_msm("X", 1.0)
    r = bound(V, P, Q, a, method="msm_agg")
    assert r.success
    assert abs(r.width) < 1e-6
    # equals P(Y=1|X=2)
    px = sum(P[c] for c in cells if c[0] == 2)
    py = sum(P[c] for c in cells if c[0] == 2 and c[1] == 1)
    assert abs(r.lower - py / px) < 1e-5


def test_bound_path_iv_breakdown_bisection():
    from partialid.problems import binary_iv

    p = binary_iv(seed=2)
    path = bound_path(
        p.variables,
        p.data,
        p.query,
        p.assumptions,
        param="Lambda",
        values=[3.0, 5.0, 8.0, 12.0, 20.0],
        prefer_closed=False,
        breakpoints=True,
    )
    assert not path.closed_form
    assert path._eval_fn is not None
    assert any(
        "repeated_bound" in (path.formula_id or "") or "IV" in n or "repeated" in n
        for n in path.notes + [path.formula_id or ""]
    )
    # At large Λ, interval is wide; pick mid of first feasible knot as null probe
    finite = [
        (lo, hi) for lo, hi in zip(path.lower, path.upper) if np.isfinite(lo) and np.isfinite(hi)
    ]
    assert finite
    lo0, hi0 = finite[0]
    null = 0.5 * (lo0 + hi0)
    star = path.breakdown(null=null, which="interval")
    assert star is not None
    assert 3.0 - 1e-6 <= star <= 20.0 + 1e-6


def test_bound_path_ternary_template():
    V = Variables.from_mapping({"X": (0, 1, 2), "Y": (0, 1)})
    rng = np.random.default_rng(2)
    cells = list(V.cells())
    p = rng.dirichlet(np.ones(len(cells)))
    P = {c: float(v) for c, v in zip(cells, p)}
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    path = bound_path(
        V,
        P,
        Q,
        Assumptions.order("X", "Y"),
        param="Lambda",
        values=[1.0, 1.5, 2.0, 3.0],
        prefer_closed=False,
        breakpoints=False,
    )
    assert len(path.values) == 4
    assert np.isfinite(path.width[0]) and abs(path.width[0]) < 1e-5
    assert path.width[0] <= path.width[-1] + 1e-8


@pytest.mark.skipif(not cvxpy_available(), reason="cvxpy not installed")
def test_f_sensitivity_tv_ternary():
    V = Variables.from_mapping({"X": (0, 1, 2), "Y": (0, 1)})
    rng = np.random.default_rng(3)
    cells = list(V.cells())
    p = rng.dirichlet(np.ones(len(cells)))
    P = {c: float(v) for c, v in zip(cells, p)}
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    base = Assumptions.order("X", "Y")
    r0 = bound(V, P, Q, base & Assumptions.f_sensitivity("X", 0.0, f="tv"))
    r1 = bound(V, P, Q, base & Assumptions.f_sensitivity("X", 5.0, f="tv"))
    assert r0.success and r1.success
    assert abs(r0.width) < 1e-2
    assert r1.width + 1e-6 >= r0.width
