"""Literature examples + instrument LP + inert pruning."""

from __future__ import annotations

import numpy as np

from partialid import Assumptions, Query, Variables, bound
from partialid.problems import (
    all_literature_problems,
    chain_order_only,
    binary_iv,
    ternary_cardinalities,
    ternary_exposure,
    two_instruments,
    shridharan_multi_treatment,
)


def test_ternary_manski_width():
    p = ternary_exposure(seed=0)
    r = bound(p.variables, p.data, p.query, p.assumptions)
    assert r.success
    assert abs(r.width - p.expected_width) < 1e-7


def test_higher_card_manski():
    p = ternary_cardinalities(4, 2, seed=1)
    r = bound(p.variables, p.data, p.query, p.assumptions, method="fiber")
    assert abs(r.width - p.expected_width) < 1e-7
    r_lp = bound(p.variables, p.data, p.query, p.assumptions, method="lp")
    assert abs(r.lower - r_lp.lower) < 1e-7


def test_inert_prune_chain():
    p = chain_order_only(4, 2, seed=7)
    r = bound(p.variables, p.data, p.query, p.assumptions)
    assert r.success
    assert any("inert prune" in n for n in r.notes)
    assert abs(r.width - p.expected_width) < 1e-6


def test_binary_iv_contains_truth():
    """P0: bounds must contain the SCM's true interventional probability."""
    from partialid.graph import CausalDiagram
    from partialid.instrument import InstrumentTypeSpace

    V = Variables.from_binary("Z", "X", "Y")
    g = CausalDiagram.from_edges(
        ["Z", "X", "Y"],
        directed=[("Z", "X"), ("X", "Y")],
        bidirected=[("X", "Y")],
    )
    rng = np.random.default_rng(0)
    ts = InstrumentTypeSpace.build(V, g, ("Z",), ("X", "Y"))
    logits = rng.normal(size=ts.n_types)
    q = np.exp(logits - logits.max())
    q /= q.sum()
    true_Q = float(sum(q[t] * ts.query_coeff(t, "Y", 1, {"X": 1}) for t in range(ts.n_types)))
    data = {}
    for z, pz in ((0, 0.45), (1, 0.55)):
        for t, qt in enumerate(q):
            r_idx = ts.eval_r_indices(t, (z,), None)
            x = ts.domain_list["X"][r_idx[0]]
            y = ts.domain_list["Y"][r_idx[1]]
            key = (z, x, y)
            data[key] = data.get(key, 0.0) + pz * qt
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    r = bound(V, data, Q, Assumptions.from_diagram(g), method="lp")
    assert r.success, r.message
    assert r.lower <= true_Q + 1e-6
    assert r.upper >= true_Q - 1e-6


def test_iv_tighter_than_naive_outer():
    p = binary_iv(seed=2)
    r_opt = bound(p.variables, p.data, p.query, p.assumptions, optimize=True, method="lp")
    r_naive = bound(p.variables, p.data, p.query, p.assumptions, optimize=False, method="lp")
    assert r_opt.success and r_naive.success
    # naive merged LP is a valid outer approximation
    assert r_naive.lower <= r_opt.lower + 1e-6
    assert r_naive.upper >= r_opt.upper - 1e-6


def test_two_instruments_feasible():
    p = two_instruments(seed=3)
    r = bound(p.variables, p.data, p.query, p.assumptions, method="lp")
    assert r.success, r.message
    assert r.width >= 0
    assert any("instrument LP" in n for n in r.notes)


def test_multi_treatment_feasible():
    p = shridharan_multi_treatment(2, 2, seed=6)
    r = bound(p.variables, p.data, p.query, p.assumptions, method="auto")
    assert r.success, r.message
    assert 0 <= r.lower <= r.upper <= 1 + 1e-8


def test_catalog_smoke():
    for p in all_literature_problems():
        r = bound(p.variables, p.data, p.query, p.assumptions, method="auto")
        assert r.success, f"{p.name}: {r.message}"
        if p.expected_width is not None:
            assert abs(r.width - p.expected_width) < 1e-5, p.name
