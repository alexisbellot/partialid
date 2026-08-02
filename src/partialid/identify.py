"""Identification via Shpitser–Pearl ID (before the LP).

Only applies when an explicit diagram is given (not order-only Manski).
For interventional ``P(Y=y | do(X=x, …))`` and ATE as a difference of two
identified margins, ``bound`` skips the LP when ID succeeds.

Classical g-formula / back-door / front-door are special cases of ID — not
separate engines.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

import numpy as np

from .assumptions import AssumptionSet, DiagramAssumption, as_assumption_set
from .graph import CausalDiagram
from .id_algorithm import IDFail, evaluate_id, identify, is_identified
from .query import Query
from .variables import Variables

__all__ = [
    "IDResult",
    "has_explicit_diagram",
    "identify_query",
    "is_identified",
    "is_manski_full_latent",
    "relevant_c_components",
    "try_identify",
]


@dataclass
class IDResult:
    """Outcome of identification analysis for an interventional / ATE query."""

    identified: bool
    value: float | None = None
    criterion: str = ""  # "id" | "id/ate" | ""
    relevant_components: list[frozenset[str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    attempted: bool = False


def _ancestors(g: CausalDiagram, nodes: Iterable[str]) -> set[str]:
    an = set(nodes)
    changed = True
    while changed:
        changed = False
        for a, b in g.directed:
            if b in an and a not in an:
                an.add(a)
                changed = True
    return an


def _mutilate_do(g: CausalDiagram, intervened: Sequence[str]) -> CausalDiagram:
    inter = set(intervened)
    directed = tuple((a, b) for a, b in g.directed if b not in inter)
    return CausalDiagram(
        nodes=g.nodes,
        directed=directed,
        bidirected=g.bidirected,
    )


def has_explicit_diagram(assumptions: AssumptionSet) -> bool:
    return bool(assumptions.of_type(DiagramAssumption))


def is_manski_full_latent(g: CausalDiagram) -> bool:
    """True when one latent hyperedge covers every node (order-only Manski)."""
    return (
        len(g.nodes) > 1
        and len(g.bidirected) == 1
        and set(g.bidirected[0]) == set(g.nodes)
    )


def relevant_c_components(
    g: CausalDiagram,
    *,
    treatments: Sequence[str],
    outcome: str,
) -> list[frozenset[str]]:
    g_bar = _mutilate_do(g, treatments)
    an_y = _ancestors(g_bar, {outcome})
    comps = g.c_components()
    relevant = [c for c in comps if c & an_y]
    return relevant if relevant else comps


def _eval_interventional(
    g: CausalDiagram,
    variables: Variables,
    data: Mapping,
    *,
    treatments: Mapping[str, object],
    outcome: str,
    value: object,
    notes: list[str],
) -> float | None:
    expr = identify(g, list(treatments.keys()), [outcome])
    if isinstance(expr, IDFail) or expr.is_fail():
        notes.append("ID algorithm: not identified (hedge / FAIL)")
        return None
    try:
        val = evaluate_id(
            expr,
            variables,
            data,
            treatments=treatments,
            outcomes={outcome: value},
            order=g.topological_order(),
        )
    except Exception as ex:  # noqa: BLE001
        notes.append(f"ID evaluation failed: {ex}")
        return None
    if not np.isfinite(val):
        notes.append(f"ID expression unevaluable (got {val})")
        return None
    return float(val)


def try_identify(
    variables: Variables,
    data: Mapping,
    query: Query,
    assumptions: AssumptionSet | None,
) -> IDResult:
    """Run Shpitser–Pearl ID for interventional or ATE queries."""
    assumptions = as_assumption_set(assumptions)
    notes: list[str] = []

    if not has_explicit_diagram(assumptions):
        return IDResult(
            identified=False,
            notes=["ID skipped: no explicit DiagramAssumption (order-only / unknown G)"],
            attempted=False,
        )

    g = assumptions.diagram()
    assert g is not None
    if is_manski_full_latent(g):
        return IDResult(
            identified=False,
            notes=["ID skipped: single full-latent hyperedge (Manski / order-only)"],
            attempted=False,
            relevant_components=g.c_components(),
        )

    if variables.observed is not None and set(variables.observed) != set(variables.names):
        return IDResult(
            identified=False,
            notes=["ID skipped: latent endogenous variables (incomplete observational P)"],
            attempted=False,
            relevant_components=g.c_components(),
        )

    if query.kind == "ate":
        meta = query.meta or {}
        treatment = str(meta.get("treatment", "X"))
        t1, t0 = meta.get("t1", 1), meta.get("t0", 0)
        yv = meta.get("y", query.value)
        rel = relevant_c_components(g, treatments=[treatment], outcome=query.outcome)
        notes.append(f"C*={[sorted(c) for c in rel]}")
        if not is_identified(g, [treatment], [query.outcome]):
            notes.append("ID algorithm: not identified (hedge / FAIL)")
            return IDResult(
                identified=False,
                relevant_components=rel,
                notes=notes,
                attempted=True,
            )
        v1 = _eval_interventional(
            g,
            variables,
            data,
            treatments={treatment: t1},
            outcome=query.outcome,
            value=yv,
            notes=notes,
        )
        v0 = _eval_interventional(
            g,
            variables,
            data,
            treatments={treatment: t0},
            outcome=query.outcome,
            value=yv,
            notes=notes,
        )
        if v1 is None or v0 is None:
            return IDResult(
                identified=False,
                relevant_components=rel,
                notes=notes,
                attempted=True,
            )
        return IDResult(
            identified=True,
            value=float(v1 - v0),
            criterion="id/ate",
            relevant_components=[],
            notes=notes + ["point-identified ATE by Shpitser–Pearl ID"],
            attempted=True,
        )

    if query.kind != "interventional":
        return IDResult(
            identified=False,
            notes=[f"ID skipped: query kind {query.kind!r} (interventional / ate only)"],
            attempted=True,
            relevant_components=relevant_c_components(
                g,
                treatments=list(query.intervention.keys()),
                outcome=query.outcome,
            ),
        )

    treatments = dict(query.intervention)
    if not treatments:
        return IDResult(identified=False, notes=["ID: empty intervention"], attempted=True)

    rel = relevant_c_components(
        g, treatments=list(treatments.keys()), outcome=query.outcome
    )
    notes.append(f"C*={[sorted(c) for c in rel]}")

    val = _eval_interventional(
        g,
        variables,
        data,
        treatments=treatments,
        outcome=query.outcome,
        value=query.value,
        notes=notes,
    )
    if val is None:
        return IDResult(
            identified=False,
            value=None,
            criterion="",
            relevant_components=rel,
            notes=notes,
            attempted=True,
        )
    return IDResult(
        identified=True,
        value=val,
        criterion="id",
        relevant_components=[],
        notes=notes + ["point-identified by Shpitser–Pearl ID"],
        attempted=True,
    )


def identify_query(
    variables: Variables,
    data: Mapping,
    query: Query,
    assumptions: AssumptionSet | None = None,
) -> IDResult:
    """Public alias for :func:`try_identify`."""
    return try_identify(variables, data, query, assumptions)
