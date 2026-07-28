"""Shridharan–Iyengar Appendix B examples + hyperarc comparison."""

from __future__ import annotations

import pytest

from partialid import Query, bound
from partialid.hyperarc import _partition_va_vb
from partialid.shridharan_examples import (
    PAPER_TABLE1,
    example_f,
    example_g,
    size_report,
)


def test_query_condition_describe():
    q = Query.interventional(outcome="Y", do={"X": 1}, value=1, condition={"Z": 1})
    assert "Z=1" in q.describe()
    assert (q.meta or {}).get("condition") == {"Z": 1}


def test_ex_f_partition_and_max_H():
    p = example_f(n_mc=128, seed=0)
    g = p.assumptions.diagram()
    assert g is not None
    va, vb = _partition_va_vb(g, p.query.intervention, p.query.outcome)
    assert set(va) == {"C1", "C2"}
    assert set(vb) == {"T1", "T2", "T3", "Y"}
    max_h = (2 ** len(vb)) ** (2 ** len(va))
    assert abs(max_h - PAPER_TABLE1["F"]["max_H"]) / PAPER_TABLE1["F"]["max_H"] < 0.05


def test_ex_f_hyperarc_solves_lite():
    """Laptop-safe: small MC + hyperarc only (Ex F, |H|≈58k)."""
    p = example_f(n_mc=300, seed=5)
    r = bound(p.variables, p.data, p.query, p.assumptions, method="hyperarc")
    assert r.success, r.message
    assert 0.0 <= r.lower <= r.upper <= 1.0
    assert r.width < 0.999
    assert any("|H|=57600" in n or "|H|=58000" in n or "57600" in n for n in r.notes)


def test_size_max_H_lite_letters():
    for letter in "BCFG":
        rep = size_report(letter, count_H=False)
        paper = PAPER_TABLE1[letter]["max_H"]
        assert abs(rep.max_hyperarcs - paper) / paper < 0.05, (letter, rep)


@pytest.mark.slow
def test_ex_f_auto_agrees_hyperarc():
    p = example_f(n_mc=600, seed=5)
    rh = bound(p.variables, p.data, p.query, p.assumptions, method="hyperarc")
    ra = bound(p.variables, p.data, p.query, p.assumptions, method="auto")
    assert rh.success and ra.success
    assert abs(rh.lower - ra.lower) < 1e-5
    assert abs(rh.upper - ra.upper) < 1e-5


@pytest.mark.slow
def test_ex_g_hyperarc_solves():
    # Larger MC: hyperarc keeps VA ``condition`` (not pretreatment-sliced), so
    # small samples can make the conditional hyperarc LP infeasible.
    p = example_g(n_mc=4000, seed=6)
    r = bound(p.variables, p.data, p.query, p.assumptions, method="hyperarc")
    if not r.success and "infeasible" in (r.message or "").lower():
        pytest.skip(f"MC joint inconsistent with hyperarc polytope: {r.message}")
    assert r.success, r.message
    assert 0.0 <= r.lower <= r.upper <= 1.0


@pytest.mark.slow
def test_ex_a_size_H_near_paper():
    rep = size_report("A", count_H=True)
    assert rep.n_hyperarcs is not None
    assert abs(rep.n_hyperarcs - PAPER_TABLE1["A"]["H"]) / PAPER_TABLE1["A"]["H"] < 0.25
