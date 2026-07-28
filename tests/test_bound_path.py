"""Track B: bound_path parametric sensitivity curves."""

from __future__ import annotations

import numpy as np
import pytest

from partialid import Assumptions, Query, Variables, bound_path
from partialid.closed_forms import msm_closed_form_re1

RE1_P = {(0, 0): 0.30, (0, 1): 0.20, (1, 0): 0.15, (1, 1): 0.35}


@pytest.fixture
def re1():
    V = Variables.from_binary("X", "Y")
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    a = Assumptions.order("X", "Y")
    return V, RE1_P, Q, a


def test_bound_path_re1_msm_closed(re1):
    V, P, Q, a = re1
    path = bound_path(V, P, Q, a, param="Lambda", theta_min=1.0, theta_max=7.0)
    assert path.closed_form
    assert path.formula_id == "msm_margin_binary_xy"
    # kink at 7/3
    assert any(abs(b - 7 / 3) < 1e-9 for b in path.breakpoints)
    for lam in (1.0, 1.5, 7 / 3, 3.0, 7.0):
        ell, u, w = path.at(lam)
        e2, u2, w2 = msm_closed_form_re1(lam)
        assert abs(ell - e2) < 1e-10
        assert abs(u - u2) < 1e-10
        assert abs(w - w2) < 1e-10


def test_bound_path_re1_gamma_closed(re1):
    V, P, Q, a = re1
    # Γ=9 → Λ=3
    path = bound_path(V, P, Q, a, param="Gamma", theta_min=1.0, theta_max=16.0)
    assert path.closed_form
    ell, u, w = path.at(9.0)
    e2, u2, w2 = msm_closed_form_re1(3.0)
    assert abs(ell - e2) < 1e-10
    assert abs(u - u2) < 1e-10
    assert abs(w - w2) < 1e-10


def test_bound_path_grid_matches_closed_re1(re1):
    """Force LP grid (prefer_closed=False) and match analytic at knots."""
    V, P, Q, a = re1
    vals = [1.0, 1.5, 7 / 3, 3.0, 5.0]
    path = bound_path(
        V, P, Q, a, param="Lambda", values=vals, prefer_closed=False, breakpoints=False
    )
    assert not path.closed_form
    assert len(path.values) == len(vals)
    for lam, lo, hi in zip(path.values, path.lower, path.upper):
        ell, u, _ = msm_closed_form_re1(float(lam))
        assert abs(lo - ell) < 1e-5
        assert abs(hi - u) < 1e-5


def test_bound_path_cdep_re1(re1):
    V, P, Q, a = re1
    path = bound_path(
        V,
        P,
        Q,
        a,
        param="c",
        theta_min=0.0,
        theta_max=0.4,
        n_grid=9,
        breakpoints=True,
        prefer_closed=False,
    )
    assert len(path.values) >= 3
    assert np.all(np.isfinite(path.lower))
    assert np.all(path.upper >= path.lower - 1e-9)
    # At c=0 should be point-identified near P(Y=1|X=1)=0.7
    lo, hi, w = path.at(0.0)
    assert abs(w) < 1e-4
    assert abs(lo - 0.70) < 1e-3


def test_bound_path_kappa_re1(re1):
    V, P, Q, a = re1
    path = bound_path(
        V,
        P,
        Q,
        a,
        param="kappa",
        values=[0.0, 0.1, 0.25, 0.5],
        breakpoints=False,
        prefer_closed=False,
    )
    assert len(path.values) == 4
    # κ=0 → width 0 around g-formula
    assert abs(path.width[0]) < 1e-6
    # width nondecreasing in κ
    assert path.width[1] <= path.width[2] + 1e-8
    assert path.width[2] <= path.width[3] + 1e-8


def test_bound_path_breakpoints_include_endpoints(re1):
    V, P, Q, a = re1
    path = bound_path(
        V,
        P,
        Q,
        a,
        param="Lambda",
        values=[1.0, 1.2, 1.5, 2.0, 3.0, 5.0],
        prefer_closed=False,
        breakpoints=True,
    )
    assert abs(path.breakpoints[0] - path.values[0]) < 1e-12
    assert abs(path.breakpoints[-1] - path.values[-1]) < 1e-12
