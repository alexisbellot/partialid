"""c-degree-1 factorization (THEORY Prop. 8.4 / Thm 8.3).

For ADMGs where every district is treated as c-degree 1 (one latent per
bidirected component), the tight program is an LP on the *free* confounded
district(s), with unconfounded districts absorbed into observational kernels.

This generalizes the IV R-side LP: instruments / Markov kernels are exogenous;
only districts that contain bidirected edges keep a free type distribution.
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
        bidirected=tuple((a, b) for a, b in g.bidirected if a in keep and b in keep),
        maximal_confounding=False,
    )


def district_has_confounding(g: CausalDiagram, district: frozenset[str]) -> bool:
    for a, b in g.bidirected:
        if a in district and b in district:
            return True
    return False


def c_degree_upper_bound(g: CausalDiagram, district: frozenset[str]) -> int:
    """Upper bound on c-degree for a district in a latent-projection ADMG.

    We treat a connected bidirected component as arising from a *single* latent
    (c-degree 1). Singletons have c-degree 1 (own exogenous noise). This is the
    standard reading of pairwise ADMGs; graphs needing c-degree ≥2 require
    richer latent structure than pairwise bidirected edges.
    """
    if len(district) <= 1:
        return 1
    if district_has_confounding(g, district):
        return 1
    return 1


def all_districts_c_degree_1(g: CausalDiagram) -> bool:
    return all(c_degree_upper_bound(g, d) <= 1 for d in g.c_components())


@dataclass(frozen=True)
class Degree1Factorization:
    """Result of c-degree-1 analysis for an interventional query."""

    ancestral: frozenset[str]
    districts: tuple[frozenset[str], ...]
    free_districts: tuple[frozenset[str], ...]
    exogenous: tuple[str, ...]
    notes: tuple[str, ...]
    # True if Prop. 8.4 / Thm 8.3 LP reduction applies (≤1 free district)
    lp_reducible: bool


def analyze_degree1(
    g: CausalDiagram,
    *,
    treatments: Sequence[str],
    outcome: str,
) -> Degree1Factorization:
    """Identify free confounded districts for Q=P(outcome|do(treatments))."""
    notes: list[str] = []
    if g.maximal_confounding:
        return Degree1Factorization(
            ancestral=frozenset(g.nodes),
            districts=(frozenset(g.nodes),),
            free_districts=(frozenset(g.nodes),),
            exogenous=(),
            notes=("maximal_confounding: single block",),
            lp_reducible=True,
        )

    # Ancestral set of outcome ∪ treatments (directed)
    seeds = set(treatments) | {outcome}
    ancestral = directed_ancestors(g, seeds)
    # also include directed ancestors of treatments (instruments etc.)
    ancestral |= directed_ancestors(g, treatments)
    notes.append(f"ancestral={sorted(ancestral)}")

    g_an = induced_subgraph(g, ancestral)
    districts = tuple(g_an.c_components())
    notes.append(f"districts={[sorted(d) for d in districts]}")

    if not all_districts_c_degree_1(g_an):
        notes.append("c-degree>1 suspected; LP reduction not claimed")
        return Degree1Factorization(
            ancestral=frozenset(ancestral),
            districts=districts,
            free_districts=districts,
            exogenous=(),
            notes=tuple(notes),
            lp_reducible=False,
        )

    # Free = confounded districts that intersect the query's ancestral set in G_{\bar x}
    # Unconfounded districts are absorbed (kernels from data / ID).
    free = tuple(d for d in districts if district_has_confounding(g_an, d))
    # If outcome district is unconfounded but treatments confound something...
    # Also: if there is no confounding anywhere, free is empty → point ID territory
    notes.append(f"free_confounded={[sorted(d) for d in free]}")

    if len(free) > 1:
        notes.append("≥2 free confounded districts: needs GPP or inequality completion")
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

    # No confounded district in ancestral graph
    notes.append("no free confounded district (Markovian ancestral graph)")
    return Degree1Factorization(
        ancestral=frozenset(ancestral),
        districts=districts,
        free_districts=(),
        exogenous=tuple(g_an.topological_order()),
        notes=tuple(notes),
        lp_reducible=True,
    )
