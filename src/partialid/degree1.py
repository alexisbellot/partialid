"""c-degree factorization (THEORY Prop. 8.4 / Thm 8.3).

Each latent hyperedge is one exogenous source. Districts are connected
components under hyperedge incidence. A district's c-degree is the number of
hyperedges it contains.

When every free district has c-degree 1 and K≤1 (or a WL–WR cut applies), the
tight program is an LP. Higher ∑ c-degrees need bilinear / polynomial solvers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .graph import CausalDiagram


def directed_ancestors(g: CausalDiagram, seeds: Iterable[str]) -> set[str]:
    an = set(seeds)
    changed = True
    while changed:
        changed = False
        for a, b in g.directed:
            if b in an and a not in an:
                an.add(a)
                changed = True
    return an


def induced_subgraph(g: CausalDiagram, nodes: Iterable[str]) -> CausalDiagram:
    keep = set(nodes)
    return CausalDiagram(
        nodes=tuple(n for n in g.nodes if n in keep),
        directed=tuple((a, b) for a, b in g.directed if a in keep and b in keep),
        bidirected=tuple(he for he in g.bidirected if set(he) <= keep),
    )


def district_has_confounding(g: CausalDiagram, district: frozenset[str]) -> bool:
    return len(g.latent_hyperedges_in(district)) > 0


def c_degree_upper_bound(g: CausalDiagram, district: frozenset[str]) -> int:
    """c-degree = number of latent hyperedges in the district."""
    return g.c_degree(district)


def all_districts_c_degree_1(g: CausalDiagram) -> bool:
    return all(
        (not district_has_confounding(g, d)) or c_degree_upper_bound(g, d) <= 1
        for d in g.c_components()
    )


@dataclass(frozen=True)
class Degree1Factorization:
    """Result of free-block analysis for an interventional or counterfactual query."""

    ancestral: frozenset[str]
    districts: tuple[frozenset[str], ...]
    free_districts: tuple[frozenset[str], ...]
    exogenous: tuple[str, ...]
    notes: tuple[str, ...]
    lp_reducible: bool


def analyze_degree1(
    g: CausalDiagram,
    *,
    treatments: Sequence[str] | None = None,
    outcome: str | None = None,
    query: object | None = None,
    variables: object | None = None,
) -> Degree1Factorization:
    """Identify free districts for a query on ``g``.

    Free districts are confounded blocks (latent hyperedges) in the ancestral
    graph, plus unconfounded blocks that contain a *cross-world* variable
    from the query AST; see :mod:`partialid.query_footprint`.
    """
    notes: list[str] = []
    cross_world: set[str] = set()

    if query is not None:
        from .query_footprint import extract_footprint

        fp = extract_footprint(query, diagram=g, variables=variables)  # type: ignore[arg-type]
        notes.extend(list(fp.notes))
        seeds = set(fp.all_vars)
        cross_world = set(fp.cross_world_vars)
        treatments = (
            list(treatments) if treatments is not None else sorted(fp.intervention_vars)
        )
        seeds |= set(treatments)
    else:
        if treatments is None or outcome is None:
            raise TypeError("analyze_degree1 requires treatments+outcome, or query=")
        seeds = set(treatments) | {outcome}

    ancestral = directed_ancestors(g, seeds)
    ancestral |= directed_ancestors(g, treatments or ())
    notes.append(f"ancestral={sorted(ancestral)}")

    g_an = induced_subgraph(g, ancestral)
    districts = tuple(g_an.c_components())
    notes.append(f"districts={[sorted(d) for d in districts]}")
    notes.append(
        "c_degrees=" + str({tuple(sorted(d)): g_an.c_degree(d) for d in districts})
    )

    free_list: list[frozenset[str]] = []
    for d in districts:
        if district_has_confounding(g_an, d):
            free_list.append(d)
        elif cross_world and any(v in cross_world for v in d):
            free_list.append(d)
            notes.append(f"cross-world free district {sorted(d)}")
    free = tuple(free_list)
    notes.append(f"free_districts={[sorted(d) for d in free]}")

    # LP reduction only when every free block has c-degree ≤1 (cross-world
    # unconfounded singletons count as degree 1) and K≤1.
    free_cdeg_ok = all(
        (not district_has_confounding(g_an, d)) or c_degree_upper_bound(g_an, d) <= 1
        for d in free
    )
    if not free_cdeg_ok or not all_districts_c_degree_1(g_an):
        notes.append(
            f"c-degree>1 in free districts {[sorted(d) for d in free]}; "
            "LP reduction not claimed"
        )
        exo = tuple(n for n in g_an.topological_order() if not any(n in d for d in free))
        return Degree1Factorization(
            ancestral=frozenset(ancestral),
            districts=districts,
            free_districts=free,
            exogenous=exo,
            notes=tuple(notes),
            lp_reducible=False,
        )

    if len(free) > 1:
        notes.append("≥2 free districts: needs GPP or inequality completion")
        exo = tuple(n for n in g_an.topological_order() if not any(n in d for d in free))
        return Degree1Factorization(
            ancestral=frozenset(ancestral),
            districts=districts,
            free_districts=free,
            exogenous=exo,
            notes=tuple(notes),
            lp_reducible=False,
        )

    if len(free) == 1:
        D = free[0]
        exo = tuple(n for n in g_an.topological_order() if n not in D)
        notes.append(f"single free district {sorted(D)}; exo={list(exo)}")
        return Degree1Factorization(
            ancestral=frozenset(ancestral),
            districts=districts,
            free_districts=free,
            exogenous=exo,
            notes=tuple(notes),
            lp_reducible=True,
        )

    notes.append("no free district (Markovian ancestral / single-world absorbed)")
    return Degree1Factorization(
        ancestral=frozenset(ancestral),
        districts=districts,
        free_districts=(),
        exogenous=tuple(g_an.topological_order()),
        notes=tuple(notes),
        lp_reducible=True,
    )


def free_block_degree(g: CausalDiagram, district: frozenset[str]) -> int:
    """Simplex-factor degree of a free district (≥1 for cross-world unconfounded)."""
    cd = c_degree_upper_bound(g, district)
    return cd if cd >= 1 else 1
