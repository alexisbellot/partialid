"""Relative-propensity LR MSM / ℓ∞ c-dependence (binary ≡ classical Tan)."""

from __future__ import annotations

import numpy as np

from partialid import Assumptions, Query, Variables, bound
from partialid.formulate import formulate
from partialid.sensitivity_budgets import msm_lr_rows

RE1_P = {(0, 0): 0.30, (0, 1): 0.20, (1, 0): 0.15, (1, 1): 0.35}


def test_binary_lr_matches_classical_tan_closed_form():
    V = Variables.from_binary("X", "Y")
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    a = Assumptions.order("X", "Y") & Assumptions.tan_msm("X", 3.0)
    r = bound(V, RE1_P, Q, a, method="lp")
    assert r.success
    # THEORY §8.7.3 at Λ=3
    assert abs(r.lower - 0.467) < 1e-2
    assert abs(r.upper - 0.80) < 1e-2


def test_binary_lr_rows_equivalent_to_odds_form():
    """q1*π0(0) - Λ π0(1) q0 ≤ 0  ⇔  q1 ≤ Λ θ0 q0 after clearing constants."""
    V = Variables.from_binary("X", "Y")
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    lp = formulate(V, RE1_P, Q, Assumptions.order("X", "Y"))
    ts, allow = lp.type_space, lp.allowed_types
    e0 = 0.5
    th = e0 / (1.0 - e0)
    Lam = 2.5
    pi0 = {(): {0: 1.0 - e0, 1: e0}}
    rows, _ = msm_lr_rows(ts, allow, "X", Lam, pi0)
    assert len(rows) >= 2
    for row in rows:
        # Scale so the positive treatment-level block has coefficient 1
        pos = np.where(row > 1e-12)[0]
        neg = np.where(row < -1e-12)[0]
        assert len(pos) and len(neg)
        scale = float(row[pos[0]])
        scaled = row / scale
        # After scaling: ones coeff = 1, zeros coeff = -Λ θ0  (or swapped roles)
        neg_coef = float(scaled[neg[0]])
        assert abs(neg_coef + Lam * th) < 1e-9 or abs(neg_coef + Lam / th) < 1e-9


def test_cdep_linf_binary_matches_abs_e():
    V = Variables.from_binary("X", "Y")
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    a = Assumptions.order("X", "Y")
    r0 = bound(V, RE1_P, Q, a & Assumptions.c_dependence("X", 0.0), method="lp")
    assert r0.success and abs(r0.width) < 1e-6
    r1 = bound(V, RE1_P, Q, a & Assumptions.c_dependence("X", 1.0), method="lp")
    assert r1.success
    assert abs(r1.lower - 0.35) < 1e-6
    assert abs(r1.upper - 0.85) < 1e-6


def test_ternary_treatment_msm_interpolates():
    V = Variables.from_mapping({"X": (0, 1, 2), "Y": (0, 1)})
    rng = np.random.default_rng(0)
    cells = list(V.cells())
    p = rng.dirichlet(np.ones(len(cells)))
    P = {c: float(v) for c, v in zip(cells, p)}
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    base = Assumptions.order("X", "Y")
    r_lo = bound(V, P, Q, base & Assumptions.tan_msm("X", 1.0), method="lp")
    r_mid = bound(V, P, Q, base & Assumptions.tan_msm("X", 2.0), method="lp")
    r_hi = bound(V, P, Q, base, method="lp")
    assert r_lo.success and r_mid.success and r_hi.success
    assert r_lo.width < 1e-5  # no confounding pole
    assert r_mid.width + 1e-9 >= r_lo.width
    assert r_hi.width + 1e-9 >= r_mid.width


def test_conditional_msm_on_covariate_parent():
    """C→X→Y with maximal confounding: budgets condition on C (no inert prune)."""
    V = Variables.from_binary("C", "X", "Y")
    rng = np.random.default_rng(1)
    cells = list(V.cells())
    p = rng.dirichlet(np.ones(len(cells)))
    P = {c: float(v) for c, v in zip(cells, p)}
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    a = Assumptions.diagram(
        ["C", "X", "Y"],
        directed=[("C", "X"), ("X", "Y"), ("C", "Y")],
        maximal_confounding=True,
    ) & Assumptions.tan_msm("X", 1.5)
    lp = formulate(V, P, Q, a, optimize=False)
    assert any("conditional on" in n for n in lp.notes)
    r = bound(V, P, Q, a, method="lp", optimize=False)
    assert r.success


def test_iv_conditional_msm_matches_msm_agg():
    from partialid.problems import sachs_binary_iv

    p = sachs_binary_iv(seed=2)
    a = p.assumptions & Assumptions.tan_msm("X", 3.0)
    rl = bound(p.variables, p.data, p.query, a, method="lp")
    ra = bound(p.variables, p.data, p.query, a, method="msm_agg")
    assert rl.success and ra.success
    assert abs(ra.lower - rl.lower) + abs(ra.upper - rl.upper) < 1e-6
    assert any("conditional" in n for n in rl.notes)
