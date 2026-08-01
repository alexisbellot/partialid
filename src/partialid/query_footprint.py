"""Extract variables / cross-world structure from a query AST.

Used by program-degree analysis so interventional and counterfactual queries
share one free-block classifier.

Cross-world rule (per atom, after CUT unnesting)
-----------------------------------------------
A variable ``V`` is *cross-world relevant* if some atom either

1. contains two ``PO(V, ·)`` literals under distinct interventions (e.g. PNS), or
2. mixes ``PO(V, ·)`` with a factual ``V=·`` literal (e.g. PN / PS).

Pure single-world atoms ``PO(V, do(X=x))=v`` (interventional / ATE summands) do
*not* mark ``V`` cross-world, so Markovian outcome kernels stay absorbed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .graph import CausalDiagram
from .query import Query
from .query_ast import Atom, Cond, Expr, LinComb, PO
from .variables import Variables


def _freeze_assignment(assignment: dict) -> frozenset:
    """Freeze a flat PO assignment (constants only) for context identity."""
    items = []
    for k, v in assignment.items():
        if isinstance(v, PO):
            # nested — use placeholder; CUT should remove these before footprint
            items.append((k, ("PO", v.var, _freeze_assignment(dict(v.assignment)))))
        else:
            items.append((k, v))
    return frozenset(items)


@dataclass(frozen=True)
class QueryFootprint:
    """Structural footprint of a query for free-block analysis."""

    all_vars: frozenset[str]
    intervention_vars: frozenset[str]
    cross_world_vars: frozenset[str]
    factual_vars: frozenset[str]
    notes: tuple[str, ...] = ()


def _iter_lincomb_atoms(expr: Expr) -> Iterable[Atom]:
    if isinstance(expr, LinComb):
        yield from expr.iter_atoms()
    elif isinstance(expr, Cond):
        yield from expr.numer.iter_atoms()
        yield from expr.denom.iter_atoms()
    else:
        raise TypeError(f"unsupported expr {type(expr)!r}")


def _walk_po_vars(po: PO, out_vars: set[str], out_interv: set[str]) -> None:
    out_vars.add(po.var)
    for k, v in po.assignment.items():
        out_interv.add(k)
        out_vars.add(k)
        if isinstance(v, PO):
            _walk_po_vars(v, out_vars, out_interv)


def extract_footprint(
    query: Query,
    *,
    variables: Variables | None = None,
    diagram: CausalDiagram | None = None,
) -> QueryFootprint:
    """Build a :class:`QueryFootprint` from ``query.expr`` (CUT-unnested)."""
    if query.expr is None:
        raise ValueError("query.expr is required")

    notes: list[str] = []
    expr: Expr = query.expr
    if variables is not None and diagram is not None:
        from .query_compile import unnest_expr

        expr = unnest_expr(expr, variables, diagram, notes)

    all_vars: set[str] = set()
    intervention_vars: set[str] = set()
    factual_vars: set[str] = set()
    cross_world: set[str] = set()

    # Always include Query fields as fallback seeds
    all_vars.add(query.outcome)
    intervention_vars.update(query.intervention.keys())
    all_vars.update(query.intervention.keys())
    if query.kind in ("pns", "pn", "ps", "ate"):
        t = str((query.meta or {}).get("treatment", "X"))
        intervention_vars.add(t)
        all_vars.add(t)

    for atom in _iter_lincomb_atoms(expr):
        po_contexts: dict[str, set[frozenset]] = {}
        atom_factual: set[str] = set()
        for lit in atom.lits:
            if isinstance(lit.expr, str):
                atom_factual.add(lit.expr)
                factual_vars.add(lit.expr)
                all_vars.add(lit.expr)
            elif isinstance(lit.expr, PO):
                _walk_po_vars(lit.expr, all_vars, intervention_vars)
                if lit.expr.is_flat():
                    po_contexts.setdefault(lit.expr.var, set()).add(
                        _freeze_assignment(dict(lit.expr.assignment))
                    )
        for var, ctxs in po_contexts.items():
            if len(ctxs) >= 2:
                cross_world.add(var)
            if var in atom_factual:
                cross_world.add(var)

    if cross_world:
        notes.append(f"cross_world_vars={sorted(cross_world)}")

    return QueryFootprint(
        all_vars=frozenset(all_vars),
        intervention_vars=frozenset(intervention_vars),
        cross_world_vars=frozenset(cross_world),
        factual_vars=frozenset(factual_vars),
        notes=tuple(notes),
    )
