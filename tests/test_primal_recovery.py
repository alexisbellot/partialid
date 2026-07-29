"""Primal witness recovery after dual LP shortcut."""

from __future__ import annotations

import numpy as np

from partialid import bound
from partialid.backends import solve_bounds_lp
from partialid.formulate import formulate
from partialid.problems import binary_iv, two_instruments_card


def _assert_witness(lp, q, value, maximize: bool):
    assert q is not None
    assert np.all(np.isfinite(q))
    assert np.all(q >= -1e-8)
    q = np.maximum(q, 0.0)
    resid = lp.A_eq @ q - lp.b_eq
    assert float(np.max(np.abs(resid))) < 1e-5
    obj = float(lp.c @ q)
    assert abs(obj - value) < 1e-5


def test_dual_path_recovers_witnesses_two_iv():
    p = two_instruments_card(card_z=3, seed=15)
    lp = formulate(p.variables, p.data, p.query, p.assumptions)
    assert lp.A_eq.shape[1] > max(64, 4 * lp.A_eq.shape[0])
    r = bound(p.variables, p.data, p.query, p.assumptions, method="lp")
    assert r.success
    assert "dual_lp" in r.notes
    assert "primal_witnesses" in r.notes
    assert r.q_lower is not None and r.q_upper is not None
    _assert_witness(lp, r.q_lower, r.lower, False)
    _assert_witness(lp, r.q_upper, r.upper, True)


def test_recover_primal_false_skips_witness():
    p = two_instruments_card(card_z=3, seed=15)
    r = bound(
        p.variables,
        p.data,
        p.query,
        p.assumptions,
        method="lp",
        recover_primal=False,
    )
    assert r.success
    assert "dual_lp" in r.notes
    assert "primal_skipped" in r.notes
    assert r.q_lower is not None
    assert np.all(np.isnan(r.q_lower))


def test_binary_iv_witness_matches_objective():
    # May or may not use dual (depends on size); witnesses must be finite either way
    p = binary_iv(seed=2)
    r = bound(p.variables, p.data, p.query, p.assumptions, method="lp")
    assert r.success
    lp = formulate(p.variables, p.data, p.query, p.assumptions)
    lo, hi = solve_bounds_lp(lp, recover_primal=True)
    assert lo.success and hi.success
    _assert_witness(lp, lo.q, lo.value, False)
    _assert_witness(lp, hi.q, hi.value, True)
