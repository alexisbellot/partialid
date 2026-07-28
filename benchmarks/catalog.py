"""Parametric problem catalog for benchmark suites.

Suite A uses the small correctness instances below. Scaling suites (D/E) will
extend the factories with ``n`` / ``card`` knobs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from partialid import Assumptions, CausalDiagram, Query, Variables
from partialid.problems import Problem, sachs_binary_iv

RE1_P = {(0, 0): 0.30, (0, 1): 0.20, (1, 0): 0.15, (1, 1): 0.35}

RE2_P = {
    (0, 0, 0): 0.20,
    (0, 0, 1): 0.05,
    (0, 1, 0): 0.10,
    (0, 1, 1): 0.15,
    (1, 0, 0): 0.05,
    (1, 0, 1): 0.10,
    (1, 1, 0): 0.10,
    (1, 1, 1): 0.25,
}


@dataclass(frozen=True)
class CatalogItem:
    """A named instance plus optional expected checks."""

    name: str
    family: str
    structure: str
    problem: Problem
    # expected interval (exact anchors)
    expected_lower: float | None = None
    expected_upper: float | None = None
    expected_width: float | None = None
    expected_method_prefix: str | None = None
    expect_tight: bool | None = None
    # optional SCM / analytic truth that must lie in the interval
    truth: float | None = None
    # default method for this item
    method: str = "auto"
    atol: float = 1e-6
    extra: Mapping[str, Any] | None = None


def msm_closed_form_re1(Lam: float) -> tuple[float, float, float]:
    """THEORY §8.7.3 — (lower, upper, width) for RE1 + Tan MSM."""
    if Lam <= 7 / 3:
        w = 0.15 * (Lam - 1 / Lam)
        u = 0.85 - 0.15 / Lam
        ell = 0.85 - 0.15 * Lam
    else:
        w = 0.50 * (1 - 1 / Lam)
        u = 0.85 - 0.15 / Lam
        ell = 0.35 * (1 + 1 / Lam)
    return ell, u, w


def _xy_query() -> Query:
    return Query.interventional(outcome="Y", do={"X": 1}, value=1)


def item_manski_re1() -> CatalogItem:
    V = Variables.from_binary("X", "Y")
    p = Problem(
        name="manski_re1",
        source="THEORY RE1",
        variables=V,
        data=dict(RE1_P),
        query=_xy_query(),
        assumptions=Assumptions.order("X", "Y"),
    )
    return CatalogItem(
        name="manski_re1",
        family="order_only",
        structure="order_xy",
        problem=p,
        expected_lower=0.35,
        expected_upper=0.85,
        expected_width=0.50,
        expected_method_prefix="closed",
        expect_tight=True,
        method="auto",
    )


def item_mtr_re1() -> CatalogItem:
    V = Variables.from_binary("X", "Y")
    p = Problem(
        name="mtr_re1",
        source="THEORY RE1",
        variables=V,
        data=dict(RE1_P),
        query=_xy_query(),
        assumptions=Assumptions.order("X", "Y") & Assumptions.mtr("Y", "X"),
    )
    return CatalogItem(
        name="mtr_re1",
        family="faces",
        structure="order_xy",
        problem=p,
        expected_lower=0.55,
        expected_upper=0.85,
        expected_width=0.30,
        expect_tight=True,
        method="auto",
    )


def item_msm_re1(Lam: float) -> CatalogItem:
    ell, u, w = msm_closed_form_re1(Lam)
    V = Variables.from_binary("X", "Y")
    p = Problem(
        name=f"msm_re1_L{Lam:g}",
        source="THEORY §8.7.3",
        variables=V,
        data=dict(RE1_P),
        query=_xy_query(),
        assumptions=Assumptions.order("X", "Y")
        & Assumptions.tan_msm("X", Lam),
    )
    return CatalogItem(
        name=f"msm_re1_L{Lam:g}",
        family="sensitivity",
        structure="order_xy",
        problem=p,
        expected_lower=ell,
        expected_upper=u,
        expected_width=w,
        expect_tight=True,
        method="lp",
        atol=1e-6,
        extra={"Lambda": Lam},
    )


def item_markov_point_id() -> CatalogItem:
    V = Variables.from_binary("X", "Y")
    g = CausalDiagram.from_edges(["X", "Y"], directed=[("X", "Y")])
    p = Problem(
        name="markov_xy_id",
        source="g-formula",
        variables=V,
        data=dict(RE1_P),
        query=_xy_query(),
        assumptions=Assumptions.from_diagram(g),
    )
    return CatalogItem(
        name="markov_xy_id",
        family="id",
        structure="markov_xy",
        problem=p,
        expected_lower=0.70,
        expected_upper=0.70,
        expected_width=0.0,
        expected_method_prefix="id/",
        expect_tight=True,
    )


def item_frontdoor_markov() -> CatalogItem:
    V = Variables.from_binary("X", "Z", "Y")
    g = CausalDiagram.from_edges(
        ["X", "Z", "Y"], directed=[("X", "Z"), ("Z", "Y")]
    )
    p = Problem(
        name="frontdoor_markov_chain",
        source="ID",
        variables=V,
        data=dict(RE2_P),
        query=_xy_query(),
        assumptions=Assumptions.from_diagram(g),
    )
    return CatalogItem(
        name="frontdoor_markov_chain",
        family="id",
        structure="markov_chain",
        problem=p,
        expected_lower=0.70,
        expected_upper=0.70,
        expected_width=0.0,
        expected_method_prefix="id/",
        expect_tight=True,
        atol=1e-6,
    )


def item_frontdoor_classic() -> CatalogItem:
    V = Variables.from_binary("X", "Z", "Y")
    g = CausalDiagram.from_edges(
        ["X", "Z", "Y"],
        directed=[("X", "Z"), ("Z", "Y")],
        bidirected=[("X", "Y")],
    )
    p = Problem(
        name="frontdoor_classic",
        source="ID",
        variables=V,
        data=dict(RE2_P),
        query=_xy_query(),
        assumptions=Assumptions.from_diagram(g),
    )
    return CatalogItem(
        name="frontdoor_classic",
        family="id",
        structure="frontdoor",
        problem=p,
        expected_width=0.0,
        expected_method_prefix="id/",
        expect_tight=True,
        atol=1e-8,
    )


def item_binary_iv(*, seed: int = 2) -> CatalogItem:
    prob = sachs_binary_iv(seed=seed)
    return CatalogItem(
        name="binary_iv",
        family="instrument",
        structure="iv",
        problem=prob,
        expect_tight=True,
        method="auto",
        atol=1e-8,
        extra={"seed": seed},
    )


def item_degree1_isolated_z() -> CatalogItem:
    V = Variables.from_binary("X", "Y", "Z")
    P = {
        (0, 0, 0): 0.15,
        (0, 0, 1): 0.15,
        (0, 1, 0): 0.10,
        (0, 1, 1): 0.10,
        (1, 0, 0): 0.075,
        (1, 0, 1): 0.075,
        (1, 1, 0): 0.175,
        (1, 1, 1): 0.175,
    }
    g = CausalDiagram.from_edges(
        ["X", "Y", "Z"],
        directed=[("X", "Y")],
        bidirected=[("X", "Y")],
    )
    p = Problem(
        name="degree1_isolated_z",
        source="Prop 8.4",
        variables=V,
        data=P,
        query=_xy_query(),
        assumptions=Assumptions.from_diagram(g),
    )
    return CatalogItem(
        name="degree1_isolated_z",
        family="degree1",
        structure="bow_plus_inert",
        problem=p,
        expected_lower=0.35,
        expected_upper=0.85,
        expected_width=0.50,
        expect_tight=True,
    )


def item_prop84_confounded_wl(*, seed: int = 0) -> CatalogItem:
    """Prop. 8.4: confounded WL={A,B}, WR={X,Y} — Layer-1 LP (not GPP)."""
    g = CausalDiagram.from_edges(
        ["X", "B", "A", "Y"],
        directed=[("X", "Y"), ("B", "A"), ("A", "Y")],
        bidirected=[("X", "Y"), ("A", "B")],
    )
    order = g.topological_order()
    V = Variables.from_binary(*order)
    rng = np.random.default_rng(seed)
    # Build a feasible observational law via the instrument / R-side SCM pushforward
    from partialid.degree1 import analyze_degree1
    from partialid.gpp import build_gpp

    fac = analyze_degree1(g, treatments=["X"], outcome="Y")
    cells = V.cells(order)
    P_dummy = {c: 1.0 / len(cells) for c in cells}
    Q = _xy_query()
    # Reuse GPP tensor only as a convenient rank-1 generator for P / truth
    gpp0 = build_gpp(V, P_dummy, Q, g, fac.free_districts)
    q1 = rng.dirichlet(np.ones(gpp0.A.shape[1]))
    q2 = rng.dirichlet(np.ones(gpp0.A.shape[2]))
    P_vec = np.einsum("cij,i,j->c", gpp0.A, q1, q2)
    P = {c: float(p) for c, p in zip(gpp0.cells, P_vec)}
    truth = float(np.einsum("ij,i,j->", gpp0.c_obj, q1, q2))
    p = Problem(
        name="prop84_confounded_wl",
        source="Prop. 8.4 / Sachs",
        variables=V,
        data=P,
        query=Q,
        assumptions=Assumptions.from_diagram(g),
        notes="confounded WL; R-side LP; truth known",
    )
    return CatalogItem(
        name="prop84_confounded_wl",
        family="prop84",
        structure="confounded_wl",
        problem=p,
        truth=truth,
        expected_method_prefix="lp",
        method="auto",
        atol=1e-4,
        extra={"seed": seed},
    )


def item_gpp_k2(*, seed: int = 0) -> CatalogItem:
    """True K=2 GPP: X→A breaks Prop. 8.4; rank-1 P with known truth."""
    g = CausalDiagram.from_edges(
        ["X", "B", "A", "Y"],
        directed=[("X", "Y"), ("B", "A"), ("A", "Y"), ("X", "A")],
        bidirected=[("X", "Y"), ("A", "B")],
    )
    order = g.topological_order()
    V = Variables.from_binary(*order)
    from partialid.degree1 import analyze_degree1
    from partialid.gpp import build_gpp

    fac = analyze_degree1(g, treatments=["X"], outcome="Y")
    cells = V.cells(order)
    P_dummy = {c: 1.0 / len(cells) for c in cells}
    Q = _xy_query()
    gpp0 = build_gpp(V, P_dummy, Q, g, fac.free_districts)
    rng = np.random.default_rng(seed)
    q1 = rng.dirichlet(np.ones(gpp0.A.shape[1]))
    q2 = rng.dirichlet(np.ones(gpp0.A.shape[2]))
    P_vec = np.einsum("cij,i,j->c", gpp0.A, q1, q2)
    P = {c: float(p) for c, p in zip(gpp0.cells, P_vec)}
    truth = float(np.einsum("ij,i,j->", gpp0.c_obj, q1, q2))
    p = Problem(
        name="gpp_k2_scm",
        source="Zhang GPP",
        variables=V,
        data=P,
        query=Q,
        assumptions=Assumptions.from_diagram(g),
        notes="rank-1 pushforward; truth known; no Sachs cut",
    )
    return CatalogItem(
        name="gpp_k2_scm",
        family="gpp",
        structure="two_free_districts",
        problem=p,
        truth=truth,
        expected_method_prefix="gpp",
        method="auto",
        atol=1e-4,
        extra={"seed": seed},
    )


def item_rosenbaum_anchored_re1() -> CatalogItem:
    # Γ=9 → Λ=3 MSM
    ell, u, w = msm_closed_form_re1(3.0)
    V = Variables.from_binary("X", "Y")
    p = Problem(
        name="rosenbaum_anchored_G9",
        source="THEORY Prop 8.7",
        variables=V,
        data=dict(RE1_P),
        query=_xy_query(),
        assumptions=Assumptions.order("X", "Y")
        & Assumptions.rosenbaum_gamma("X", 9.0, anchored=True),
    )
    return CatalogItem(
        name="rosenbaum_anchored_G9",
        family="sensitivity",
        structure="order_xy",
        problem=p,
        expected_lower=ell,
        expected_upper=u,
        expected_width=w,
        method="lp",
        atol=1e-6,
    )


def suite_a_items(*, quick: bool = True) -> list[CatalogItem]:
    """Correctness anchors for Suite A."""
    items = [
        item_manski_re1(),
        item_mtr_re1(),
        item_msm_re1(1.0),
        item_msm_re1(3.0),
        item_markov_point_id(),
        item_frontdoor_markov(),
        item_frontdoor_classic(),
        item_binary_iv(seed=2),
        item_degree1_isolated_z(),
        item_rosenbaum_anchored_re1(),
        item_prop84_confounded_wl(seed=0),
        item_gpp_k2(seed=0),
    ]
    if not quick:
        items.extend(
            [
                item_msm_re1(1.5),
                item_msm_re1(7 / 3),
                item_msm_re1(7.0),
            ]
        )
    return items


# ---------------------------------------------------------------------------
# Suite B — structure instances (fixed small size)
# ---------------------------------------------------------------------------


def item_bow_xy() -> CatalogItem:
    V = Variables.from_binary("X", "Y")
    g = CausalDiagram.from_edges(
        ["X", "Y"], directed=[("X", "Y")], bidirected=[("X", "Y")]
    )
    p = Problem(
        name="bow_xy",
        source="confounded pair",
        variables=V,
        data=dict(RE1_P),
        query=_xy_query(),
        assumptions=Assumptions.from_diagram(g),
    )
    return CatalogItem(
        name="bow_xy",
        family="structure",
        structure="bow",
        problem=p,
        expected_lower=0.35,
        expected_upper=0.85,
        expect_tight=True,
    )


def item_chain_order(n: int = 4, card: int = 2) -> CatalogItem:
    from partialid.problems import chain_order_only

    prob = chain_order_only(n, card, seed=7)
    return CatalogItem(
        name=prob.name,
        family="structure",
        structure="order_chain",
        problem=prob,
        expected_width=prob.expected_width,
        expect_tight=True,
        atol=1e-5,
        extra={"n": n, "card": card},
    )


def item_two_iv(*, card_z: int = 3) -> CatalogItem:
    from partialid.problems import sachs_two_instruments_card

    prob = sachs_two_instruments_card(card_z=card_z, seed=4)
    return CatalogItem(
        name=prob.name,
        family="structure",
        structure="two_iv",
        problem=prob,
        expect_tight=True,
        extra={"card_z": card_z},
    )


def item_napkin(*, seed: int = 0) -> CatalogItem:
    V = Variables.from_binary("Z1", "Z2", "X", "Y")
    rng = np.random.default_rng(seed)
    cells = V.cells(["Z1", "Z2", "X", "Y"])
    p_vec = rng.dirichlet(np.ones(len(cells)))
    P = {c: float(v) for c, v in zip(cells, p_vec)}
    g = CausalDiagram.from_edges(
        ["Z1", "Z2", "X", "Y"],
        directed=[("Z1", "X"), ("X", "Y"), ("Z1", "Z2"), ("Z2", "Y")],
        bidirected=[("X", "Z2")],
    )
    p = Problem(
        name="napkin",
        source="Shpitser napkin",
        variables=V,
        data=P,
        query=_xy_query(),
        assumptions=Assumptions.from_diagram(g),
    )
    return CatalogItem(
        name="napkin",
        family="structure",
        structure="napkin",
        problem=p,
        expected_width=0.0,
        expected_method_prefix="id/",
        expect_tight=True,
        atol=1e-10,
        extra={"seed": seed},
    )


def item_multi_treatment(n_treatments: int = 2) -> CatalogItem:
    from partialid.problems import shridharan_multi_treatment

    prob = shridharan_multi_treatment(n_treatments, 2, seed=6)
    return CatalogItem(
        name=prob.name,
        family="structure",
        structure="multi_treatment",
        problem=prob,
        expect_tight=True,
        extra={"n_treatments": n_treatments},
    )


@dataclass(frozen=True)
class StructureSpec:
    """Suite B: one structure + methods to try."""

    item: CatalogItem
    methods: tuple[str, ...]
    # soft: auto should land on a tag with this prefix (if set)
    auto_tag_prefix: str | None = None


def suite_b_specs(*, quick: bool = True) -> list[StructureSpec]:
    """Structure × method matrix (small fixed instances)."""
    specs = [
        StructureSpec(
            item_manski_re1(),
            ("auto", "fiber", "lp", "colgen", "hyperarc"),
            auto_tag_prefix="closed",
        ),
        StructureSpec(
            item_bow_xy(),
            ("auto", "fiber", "lp", "hyperarc"),
            auto_tag_prefix="fiber",
        ),
        StructureSpec(
            item_chain_order(4, 2),
            ("auto", "fiber", "lp"),
            auto_tag_prefix="fiber",
        ),
        StructureSpec(
            item_binary_iv(seed=2),
            ("auto", "lp", "hyperarc"),
            auto_tag_prefix="closed",
        ),
        StructureSpec(
            item_markov_point_id(),
            ("auto",),
            auto_tag_prefix="id/",
        ),
        StructureSpec(
            item_frontdoor_classic(),
            ("auto",),
            auto_tag_prefix="id/",
        ),
        StructureSpec(
            item_napkin(seed=0),
            ("auto",),
            auto_tag_prefix="id/",
        ),
        StructureSpec(
            item_degree1_isolated_z(),
            ("auto", "fiber", "lp"),
            auto_tag_prefix="fiber",
        ),
        StructureSpec(
            item_prop84_confounded_wl(seed=0),
            ("auto", "lp"),
            auto_tag_prefix="lp",
        ),
        StructureSpec(
            item_gpp_k2(seed=0),
            ("auto", "gpp"),
            auto_tag_prefix="gpp",
        ),
        StructureSpec(
            item_multi_treatment(2),
            ("auto", "fiber", "lp"),
            auto_tag_prefix="fiber",
        ),
    ]
    if not quick:
        specs.append(
            StructureSpec(
                item_two_iv(card_z=3),
                ("auto", "lp"),
                auto_tag_prefix="lp",
            )
        )
        specs.append(
            StructureSpec(
                item_chain_order(3, 3),
                ("auto", "fiber", "lp"),
                auto_tag_prefix="fiber",
            )
        )
    return specs


def suite_a4_items(*, quick: bool = True) -> list[CatalogItem]:
    """Track A4 Pareto catalog: fiber / lp / dual / colgen / auto head-to-head."""
    items = [
        item_manski_re1(),
        item_msm_re1(3.0),
        item_bow_xy(),
        item_binary_iv(seed=2),
        item_two_iv(card_z=2),
        item_chain_order(4, 2),
        item_multi_treatment(2),
        item_markov_point_id(),
    ]
    if not quick:
        items.extend(
            [
                item_msm_re1(7.0),
                # Z3 is the dual showcase; keep in full suite (still laptop-safe)
                item_two_iv(card_z=3),
                item_chain_order(3, 3),
            ]
        )
    return items
