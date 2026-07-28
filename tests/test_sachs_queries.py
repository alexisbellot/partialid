"""Numeric checks against Sachs et al. (2022) example formulas."""

from __future__ import annotations

from partialid import Assumptions, CausalDiagram, Query, Variables, bound


def test_sachs_61_ternary_x_risk_difference():
    """Sachs §6.1: ternary X, binary Y, RD P(Y(1)=1)-P(Y(0)=1).

    Symbolic bounds from the paper:
      lower = p00 + p11 - 1
      upper = 1 - p10 - p01
    with p_{xy} = P(X=x, Y=y).
    """
    V = Variables.from_mapping({"X": (0, 1, 2), "Y": (0, 1)})
    # Choose a distribution with positive mass on all (x,y) for x in {0,1}
    P = {
        (0, 0): 0.20,
        (0, 1): 0.10,
        (1, 0): 0.15,
        (1, 1): 0.25,
        (2, 0): 0.15,
        (2, 1): 0.15,
    }
    Q = Query.ate(outcome="Y", treatment="X", y=1, t1=1, t0=0)
    r = bound(V, P, Q, Assumptions.order("X", "Y"), method="lp")
    assert r.success
    p00, p01 = P[(0, 0)], P[(0, 1)]
    p10, p11 = P[(1, 0)], P[(1, 1)]
    lo = p00 + p11 - 1.0
    hi = 1.0 - p10 - p01
    assert abs(r.lower - lo) < 1e-6
    assert abs(r.upper - hi) < 1e-6


def test_sachs_63_measurement_error_monotone():
    """Sachs §6.3: latent Y, observed Y2, monotonicity Y2(Y=1)≥Y2(Y=0).

    Paper bounds on RD P(Y(1)=1)-P(Y(0)=1):
      max{-1, 2 p(Y2=0|X=0) - 2 p(Y2=0|X=1) - 1}
        ≤ RD ≤
      min{1, 2 p(Y2=0|X=0) - 2 p(Y2=0|X=1) + 1}
    """
    V = Variables.from_binary("X", "Y", "Y2", observed=["X", "Y2"])
    # Sachs §6.3: X unconfounded with (Y,Y2); U_R confounds Y–Y2; Y latent.
    g = CausalDiagram.from_edges(
        ["X", "Y", "Y2"],
        directed=[("X", "Y"), ("Y", "Y2")],
        bidirected=[("Y", "Y2")],
        maximal_confounding=False,
    )
    # P(X,Y2); p(Y2=0|X=0)=0.6, p(Y2=0|X=1)=0.3, P(X=1)=0.5
    P = {
        (0, 0): 0.30,  # (X, Y2)
        (0, 1): 0.20,
        (1, 0): 0.15,
        (1, 1): 0.35,
    }
    a = Assumptions.from_diagram(g) & Assumptions.mtr("Y2", "Y")
    Q = Query.ate(outcome="Y", treatment="X", y=1, t1=1, t0=0)
    r = bound(V, P, Q, a, method="lp")
    assert r.success and r.tight
    p_y2_0_x0 = 0.30 / 0.50
    p_y2_0_x1 = 0.15 / 0.50
    lo = max(-1.0, 2 * p_y2_0_x0 - 2 * p_y2_0_x1 - 1.0)
    hi = min(1.0, 2 * p_y2_0_x0 - 2 * p_y2_0_x1 + 1.0)
    assert abs(r.lower - lo) < 1e-5, (r.lower, lo, r.notes)
    assert abs(r.upper - hi) < 1e-5, (r.upper, hi, r.notes)
