"""Extract variables / cross-world structure from a query AST.

Used by program-degree analysis so *any* :class:`~partialid.query_ast.Expr`
(not only named factories) shares one free-block classifier.

Free / cross-world rules (per atom, after CUT unnesting)
-------------------------------------------------------
A variable ``V`` that appears under ``PO(V, ·)`` is *cross-world relevant* if
the atom (or conditional evidence) makes its response type unidentified:

1. two ``PO(V, ·)`` literals under distinct interventions (e.g. PNS), or
2. ``PO(V, ·)`` mixed with factual ``V=·`` in the same atom (e.g. PN / PS), or
3. the atom is *multi-regime*: ≥2 distinct PO assignments appear (e.g.
   ``P(Y_z, W_x)``), in which case **every** PO variable in the atom is free, or
4. the atom mixes any PO literal with any factual literal (evidential coupling),
   in which case every PO variable in the atom is free.

For :class:`~partialid.query_ast.Cond` with a purely observational denominator,
every PO variable in the numerator is free (conditioning on evidence).

Pure single-world atoms ``PO(V, do(X=x))=v`` (and joints under one shared
assignment) do *not* mark unconfounded ``V`` free, so Markovian kernels stay
absorbed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .graph import CausalDiagram
from .query import Query
from .query_ast import Atom, Cond, Expr, LinComb, PO
from .variables import Variables

_NAMED_KINDS = frozenset({"interventional", "ate", "pns", "pn", "ps"})


def _freeze_assignment(assignment: dict) -> frozenset:
    """Freeze a flat PO assignment (constants only) for context identity."""
    items = []
    for k, v in assignment.items():
        if isinstance(v, PO):
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


def _denom_is_observational(denom: LinComb) -> bool:
    atoms = list(denom.iter_atoms())
    if not atoms:
        return False
    for atom in atoms:
        for lit in atom.lits:
            if not isinstance(lit.expr, str):
                return False
    return True


def _cross_world_from_atom(atom: Atom) -> set[str]:
    """PO-variables that are free because of this atom's regime / evidence."""
    po_contexts: dict[str, set[frozenset]] = {}
    atom_factual: set[str] = set()
    assignments: set[frozenset] = set()
    po_vars: set[str] = set()

    for lit in atom.lits:
        if isinstance(lit.expr, str):
            atom_factual.add(lit.expr)
        elif isinstance(lit.expr, PO):
            po_vars.add(lit.expr.var)
            if lit.expr.is_flat():
                frozen = _freeze_assignment(dict(lit.expr.assignment))
                assignments.add(frozen)
                po_contexts.setdefault(lit.expr.var, set()).add(frozen)

    free: set[str] = set()
    for var, ctxs in po_contexts.items():
        if len(ctxs) >= 2 or var in atom_factual:
            free.add(var)

    multi_regime = len(assignments) >= 2
    evidential = bool(atom_factual) and bool(po_vars)
    if multi_regime or evidential:
        free |= po_vars
    return free


def extract_footprint_from_expr(
    expr: Expr,
    *,
    variables: Variables | None = None,
    diagram: CausalDiagram | None = None,
) -> QueryFootprint:
    """Build a :class:`QueryFootprint` from a raw AST expression."""
    notes: list[str] = []
    if variables is not None and diagram is not None:
        from .query_compile import unnest_expr

        expr = unnest_expr(expr, variables, diagram, notes)

    all_vars: set[str] = set()
    intervention_vars: set[str] = set()
    factual_vars: set[str] = set()
    cross_world: set[str] = set()

    for atom in _iter_lincomb_atoms(expr):
        for lit in atom.lits:
            if isinstance(lit.expr, str):
                factual_vars.add(lit.expr)
                all_vars.add(lit.expr)
            elif isinstance(lit.expr, PO):
                _walk_po_vars(lit.expr, all_vars, intervention_vars)
        cross_world |= _cross_world_from_atom(atom)

    # Cond with observational evidence: all numerator PO-variables are free
    if isinstance(expr, Cond) and _denom_is_observational(expr.denom):
        for atom in expr.numer.iter_atoms():
            for lit in atom.lits:
                if isinstance(lit.expr, PO):
                    cross_world.add(lit.expr.var)
        notes.append("cond: observational denom → numerator PO-vars free")

    if cross_world:
        notes.append(f"cross_world_vars={sorted(cross_world)}")

    return QueryFootprint(
        all_vars=frozenset(all_vars),
        intervention_vars=frozenset(intervention_vars),
        cross_world_vars=frozenset(cross_world),
        factual_vars=frozenset(factual_vars),
        notes=tuple(notes),
    )


def extract_footprint(
    query: Query,
    *,
    variables: Variables | None = None,
    diagram: CausalDiagram | None = None,
) -> QueryFootprint:
    """Build a :class:`QueryFootprint` from ``query.expr`` (CUT-unnested)."""
    if query.expr is None:
        raise ValueError("query.expr is required")

    fp = extract_footprint_from_expr(
        query.expr, variables=variables, diagram=diagram
    )
    all_vars = set(fp.all_vars)
    intervention_vars = set(fp.intervention_vars)
    notes = list(fp.notes)

    # Named factories may carry seeds not yet mirrored in a sparse expr;
    # never inject default ``outcome="Y"`` for arbitrary ``from_expr`` kinds.
    if query.kind in _NAMED_KINDS:
        all_vars.add(query.outcome)
        intervention_vars.update(query.intervention.keys())
        all_vars.update(query.intervention.keys())
        if query.kind in ("pns", "pn", "ps", "ate"):
            t = str((query.meta or {}).get("treatment", "X"))
            intervention_vars.add(t)
            all_vars.add(t)

    return QueryFootprint(
        all_vars=frozenset(all_vars),
        intervention_vars=frozenset(intervention_vars),
        cross_world_vars=fp.cross_world_vars,
        factual_vars=fp.factual_vars,
        notes=tuple(notes),
    )
