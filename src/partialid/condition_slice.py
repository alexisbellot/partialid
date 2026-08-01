"""Pretreatment conditioning for interventional queries."""

from __future__ import annotations

from typing import Mapping

from .graph import CausalDiagram
from .query import Query
from .variables import Variables


def _normalize_P_local(variables: Variables, order, data: Mapping):
    from .formulate import _normalize_P

    return _normalize_P(variables, order, data)


def apply_pretreatment_condition(
    variables: Variables,
    data: Mapping,
    query: Query,
    diagram: CausalDiagram | None,
    notes: list[str],
    assumptions=None,
):
    """Slice/renormalize P on pretreatment ``condition``, drop those vars.

    For ``P(Y=y | do(X=x), W=w)`` with pretreatment ``W`` (no directed path
    from any intervened variable into ``W``), conditioning reduces to solving
    the unconditional interventional query on ``P(·|W=w)``.

    Returns ``(variables, data, query, assumptions)``.
    """
    from .assumptions import (
        AssumptionSet,
        DiagramAssumption,
        OrderAssumption,
        as_assumption_set,
    )

    cond = (query.meta or {}).get("condition")
    if not isinstance(cond, Mapping) or not cond:
        return variables, data, query, as_assumption_set(assumptions)

    cond = dict(cond)
    for name in cond:
        if name not in variables.names:
            raise ValueError(f"condition variable {name!r} not in Variables")
        if name in query.intervention:
            raise ValueError(f"cannot condition on intervened variable {name!r}")
        if name == query.outcome:
            raise ValueError("cannot condition on the query outcome")

    if diagram is not None:
        for w in cond:
            for x in query.intervention:
                if w in diagram.descendants(x) or w == x:
                    raise ValueError(f"condition on {w!r} is not pretreatment for do({x})")

    order = diagram.topological_order() if diagram is not None else tuple(variables.names)
    cells, P = _normalize_P_local(variables, order, data)
    idx = {n: order.index(n) for n in order}
    slice_mass = 0.0
    sliced: dict[tuple, float] = {}
    drop = set(cond)
    keep = tuple(n for n in order if n not in drop)
    keep_idx = [idx[n] for n in keep]
    for cell, p in zip(cells, P):
        if all(cell[idx[n]] == cond[n] for n in cond):
            slice_mass += float(p)
            key = tuple(cell[i] for i in keep_idx)
            sliced[key] = sliced.get(key, 0.0) + float(p)
    if slice_mass <= 1e-15:
        raise ValueError(f"condition {cond} has probability 0 under data")
    sliced = {k: v / slice_mass for k, v in sliced.items()}
    notes.append(f"pretreatment condition {cond} (slice mass={slice_mass:.4g}); kept {list(keep)}")
    V2 = variables.restrict(keep)
    meta = {k: v for k, v in (query.meta or {}).items() if k != "condition"}
    # Drop pretreatment condition from the query; expr stays the interventional atom
    q2 = Query(
        kind=query.kind,
        outcome=query.outcome,
        value=query.value,
        intervention=dict(query.intervention),
        meta=meta or None,
        expr=query.expr,
    )

    # Restrict order/diagram atoms to remaining variables
    a = as_assumption_set(assumptions)
    new_atoms = []
    for atom in a.atoms:
        if isinstance(atom, OrderAssumption):
            new_order = tuple(n for n in atom.order if n in keep)
            if new_order:
                new_atoms.append(OrderAssumption(order=new_order))
        elif isinstance(atom, DiagramAssumption):
            g = atom.diagram
            nodes = tuple(n for n in g.nodes if n in keep)
            directed = tuple((u, v) for u, v in g.directed if u in keep and v in keep)
            bidirected = tuple(he for he in g.bidirected if set(he) <= keep)
            new_atoms.append(
                DiagramAssumption(
                    diagram=CausalDiagram(
                        nodes=nodes,
                        directed=directed,
                        bidirected=bidirected,
                    )
                )
            )
        else:
            new_atoms.append(atom)
    return V2, sliced, q2, AssumptionSet(tuple(new_atoms))
