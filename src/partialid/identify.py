"""Identification reduction for known causal diagrams (before the LP).

Only applies when an explicit diagram is given (not order-only maximal
confounding). Implements common ID special cases for interventional
``P(Y=y | do(X=x, …))``:

- Markovian g-formula (no latent confounding)
- Back-door adjustment
- Front-door adjustment

When the query is point-identified, ``bound`` skips the LP. When it is not,
we report relevant c-components ``C*``; multi-block unidentified queries fall
back to the existing (possibly outer) LP with an explicit non-tight warning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Iterable, Mapping, Sequence

import numpy as np

from .assumptions import AssumptionSet, DiagramAssumption, as_assumption_set
from .graph import CausalDiagram
from .id_algorithm import IDFail, evaluate_id, identify
from .query import Query
from .variables import Variables


@dataclass
class IDResult:
    """Outcome of identification analysis for an interventional query."""

    identified: bool
    value: float | None = None
    criterion: str = ""  # g-formula | backdoor | frontdoor | ""
    relevant_components: list[frozenset[str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    attempted: bool = False


def _children(g: CausalDiagram, node: str) -> set[str]:
    return {b for a, b in g.directed if a == node}


def _parents_set(g: CausalDiagram, node: str) -> set[str]:
    return set(g.parents(node))


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


def _descendants(g: CausalDiagram, nodes: Iterable[str]) -> set[str]:
    de = set(nodes)
    changed = True
    while changed:
        changed = False
        for a, b in g.directed:
            if a in de and b not in de:
                de.add(b)
                changed = True
    return de


def _has_directed_path(g: CausalDiagram, src: str, dst: str) -> bool:
    if src == dst:
        return True
    return dst in _descendants(g, {src})


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


class _Prob:
    def __init__(self, variables: Variables, order: Sequence[str], data: Mapping):
        self.order = tuple(order)
        self.idx = {v: i for i, v in enumerate(self.order)}
        self.domains = {v: tuple(variables.domains[v]) for v in self.order}
        cells = variables.cells(self.order)
        self.P = np.zeros(len(cells), dtype=float)
        for key, val in data.items():
            if isinstance(key, dict):
                cell = tuple(key[v] for v in self.order)
            else:
                cell = tuple(key)
            self.P[cells.index(cell)] += float(val)
        s = float(self.P.sum())
        if s <= 0:
            raise ValueError("data probabilities must sum to a positive value")
        self.P /= s
        self.cells = cells

    def _match(self, cell: tuple, cond: Mapping[str, object]) -> bool:
        for v, val in cond.items():
            if cell[self.idx[v]] != val:
                return False
        return True

    def prob(self, **assignment: object) -> float:
        return float(
            sum(self.P[i] for i, cell in enumerate(self.cells) if self._match(cell, assignment))
        )

    def cond(self, target: Mapping[str, object], given: Mapping[str, object]) -> float:
        num = self.prob(**{**given, **target})
        den = self.prob(**given)
        if den <= 1e-15:
            return float("nan")
        return num / den


def _expand_latents(g: CausalDiagram) -> CausalDiagram:
    """Expand each latent hyperedge to one exogenous node pointing to all members."""
    nodes = list(g.nodes)
    directed = list(g.directed)
    for i, he in enumerate(g.bidirected):
        u = f"__U{i}"
        nodes.append(u)
        for v in he:
            directed.append((u, v))
    return CausalDiagram(
        nodes=tuple(nodes),
        directed=tuple(directed),
        bidirected=(),
    )


def _d_connected(g: CausalDiagram, x: str, y: str, given: set[str]) -> bool:
    if x == y:
        return True
    an = _ancestors(g, {x, y} | given)
    nodes = [n for n in g.nodes if n in an]
    undirected: set[tuple[str, str]] = set()
    parents: dict[str, set[str]] = {n: set() for n in nodes}
    for a, b in g.directed:
        if a in an and b in an:
            undirected.add(tuple(sorted((a, b))))
            parents[b].add(a)
    for n in nodes:
        pa = list(parents[n])
        for i in range(len(pa)):
            for j in range(i + 1, len(pa)):
                undirected.add(tuple(sorted((pa[i], pa[j]))))
    adj: dict[str, set[str]] = {n: set() for n in nodes if n not in given}
    for a, b in undirected:
        if a in given or b in given:
            continue
        if a in adj and b in adj:
            adj[a].add(b)
            adj[b].add(a)
    if x not in adj or y not in adj:
        return False
    stack = [x]
    seen = {x}
    while stack:
        u = stack.pop()
        if u == y:
            return True
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return False


def _valid_backdoor_set(g: CausalDiagram, treatment: str, outcome: str, z: set[str]) -> bool:
    if treatment in z or outcome in z:
        return False
    if z & _descendants(g, {treatment}):
        return False
    g_exp = _expand_latents(g)
    g_under = CausalDiagram(
        nodes=g_exp.nodes,
        directed=tuple((a, b) for a, b in g_exp.directed if a != treatment),
        bidirected=(),
    )
    return not _d_connected(g_under, treatment, outcome, set(z))


def _find_backdoor_set(g: CausalDiagram, treatment: str, outcome: str) -> set[str] | None:
    candidates: list[set[str]] = [set()]
    pa = _parents_set(g, treatment)
    if pa:
        candidates.append(set(pa))
    pool = [
        n
        for n in g.nodes
        if n not in (treatment, outcome) and n not in _descendants(g, {treatment})
    ]
    for n in pool:
        candidates.append({n})
    for i, a in enumerate(pool):
        for b in pool[i + 1 :]:
            candidates.append({a, b})
    for z in candidates:
        if _valid_backdoor_set(g, treatment, outcome, z):
            return z
    return None


def _front_door_mediators(g: CausalDiagram, treatment: str, outcome: str) -> set[str] | None:
    if treatment == outcome:
        return None
    cand = (_children(g, treatment) & _ancestors(g, {outcome})) - {treatment, outcome}
    if not cand:
        return None
    z = set(cand)
    directed_wo = tuple((a, b) for a, b in g.directed if a not in z and b not in z)
    g_wo = CausalDiagram(
        nodes=tuple(n for n in g.nodes if n not in z),
        directed=directed_wo,
        bidirected=tuple(he for he in g.bidirected if not any(n in z for n in he)),
    )
    if treatment in g_wo.nodes and outcome in g_wo.nodes:
        if _has_directed_path(g_wo, treatment, outcome):
            return None
    for m in z:
        if not _valid_backdoor_set(g, treatment, m, set()):
            return None
    for m in z:
        if not _valid_backdoor_set(g, m, outcome, {treatment}):
            return None
    return z


def _eval_g_formula(
    prob: _Prob,
    g: CausalDiagram,
    treatments: Mapping[str, object],
    outcome: str,
    value: object,
) -> float:
    order = g.topological_order()
    intervened = set(treatments)
    free = [v for v in order if v not in intervened and v != outcome]
    domains = [prob.domains[v] for v in free]
    total = 0.0
    for assign in product(*domains) if free else [()]:
        world = dict(zip(free, assign))
        world[outcome] = value
        world.update(treatments)
        term = 1.0
        ok = True
        for v in order:
            if v in intervened:
                continue
            given = {}
            for p in g.parents(v):
                if p in treatments:
                    given[p] = treatments[p]
                else:
                    given[p] = world[p]
            cp = prob.cond({v: world[v]}, given) if given else prob.prob(**{v: world[v]})
            if not np.isfinite(cp):
                ok = False
                break
            term *= cp
        if ok:
            total += term
    return float(total)


def _eval_backdoor(
    prob: _Prob,
    treatments: Mapping[str, object],
    outcome: str,
    value: object,
    z: set[str],
) -> float:
    z_vars = sorted(z)
    if not z_vars:
        return float(prob.cond({outcome: value}, dict(treatments)))
    domains = [prob.domains[v] for v in z_vars]
    total = 0.0
    for assign in product(*domains):
        z_asg = dict(zip(z_vars, assign))
        pz = prob.prob(**z_asg)
        if pz <= 0:
            continue
        py = prob.cond({outcome: value}, {**dict(treatments), **z_asg})
        if not np.isfinite(py):
            continue
        total += py * pz
    return float(total)


def _eval_frontdoor(
    prob: _Prob,
    treatment: str,
    x_val: object,
    outcome: str,
    y_val: object,
    z: set[str],
) -> float:
    z_vars = sorted(z)
    x_dom = prob.domains[treatment]
    total = 0.0
    for z_asg_t in product(*[prob.domains[v] for v in z_vars]):
        z_asg = dict(zip(z_vars, z_asg_t))
        pz_given_x = prob.cond(z_asg, {treatment: x_val})
        if not np.isfinite(pz_given_x) or pz_given_x <= 0:
            continue
        inner = 0.0
        for xp in x_dom:
            pxp = prob.prob(**{treatment: xp})
            py = prob.cond({outcome: y_val}, {treatment: xp, **z_asg})
            if np.isfinite(py):
                inner += py * pxp
        total += pz_given_x * inner
    return float(total)


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


def try_identify(
    variables: Variables,
    data: Mapping,
    query: Query,
    assumptions: AssumptionSet | None,
) -> IDResult:
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
    # Manski: one latent covering every node — never point-identified for do()
    if (
        len(g.nodes) > 1
        and len(g.bidirected) == 1
        and set(g.bidirected[0]) == set(g.nodes)
    ):
        return IDResult(
            identified=False,
            notes=["ID skipped: single full-latent hyperedge (Manski / order-only)"],
            attempted=False,
            relevant_components=g.c_components(),
        )

    if query.kind != "interventional":
        return IDResult(
            identified=False,
            notes=[f"ID skipped: query kind {query.kind!r} (interventional only)"],
            attempted=True,
            relevant_components=relevant_c_components(
                g,
                treatments=list(query.intervention.keys()),
                outcome=query.outcome,
            ),
        )

    if variables.observed is not None and set(variables.observed) != set(variables.names):
        return IDResult(
            identified=False,
            notes=["ID skipped: latent endogenous variables (incomplete observational P)"],
            attempted=False,
            relevant_components=g.c_components(),
        )

    treatments = dict(query.intervention)
    if not treatments:
        return IDResult(identified=False, notes=["ID: empty intervention"], attempted=True)

    order = g.topological_order()
    prob = _Prob(variables, order, data)
    rel = relevant_c_components(g, treatments=list(treatments.keys()), outcome=query.outcome)
    notes.append(f"C*={[sorted(c) for c in rel]}")

    # --- Full Shpitser–Pearl ID (primary) ---
    expr = identify(g, list(treatments.keys()), [query.outcome])
    if not isinstance(expr, IDFail) and not expr.is_fail():
        try:
            val = evaluate_id(
                expr,
                variables,
                data,
                treatments=treatments,
                outcomes={query.outcome: query.value},
                order=order,
            )
            if np.isfinite(val):
                return IDResult(
                    identified=True,
                    value=float(val),
                    criterion="id",
                    relevant_components=[],
                    notes=notes + ["point-identified by Shpitser–Pearl ID"],
                    attempted=True,
                )
            notes.append(f"ID expression unevaluable (got {val})")
        except Exception as ex:  # noqa: BLE001
            notes.append(f"ID evaluation failed: {ex}")
    else:
        notes.append("ID algorithm: not identified (hedge / FAIL)")

    # --- Fast special cases (also used if ID eval failed) ---
    if not g.bidirected:
        try:
            val = _eval_g_formula(prob, g, treatments, query.outcome, query.value)
            if np.isfinite(val):
                return IDResult(
                    identified=True,
                    value=float(val),
                    criterion="g-formula",
                    relevant_components=[],
                    notes=notes + ["point-identified by truncated factorization"],
                    attempted=True,
                )
        except Exception as ex:  # noqa: BLE001
            notes.append(f"g-formula failed: {ex}")

    if len(treatments) == 1:
        t0 = next(iter(treatments))
        z = _find_backdoor_set(g, t0, query.outcome)
        # Only accept nonempty backdoor (empty set is unsafe with mediators /
        # buggy d-sep); nonempty sets still helpful as a fast path.
        if z:
            val = _eval_backdoor(prob, treatments, query.outcome, query.value, z)
            if np.isfinite(val):
                return IDResult(
                    identified=True,
                    value=float(val),
                    criterion="backdoor",
                    relevant_components=[],
                    notes=notes + [f"point-identified by back-door Z={sorted(z)}"],
                    attempted=True,
                )

    if len(treatments) == 1:
        t0 = next(iter(treatments))
        z = _front_door_mediators(g, t0, query.outcome)
        if z is not None:
            val = _eval_frontdoor(prob, t0, treatments[t0], query.outcome, query.value, z)
            if np.isfinite(val):
                return IDResult(
                    identified=True,
                    value=float(val),
                    criterion="frontdoor",
                    relevant_components=[],
                    notes=notes + [f"point-identified by front-door Z={sorted(z)}"],
                    attempted=True,
                )

    return IDResult(
        identified=False,
        value=None,
        criterion="",
        relevant_components=rel,
        notes=notes,
        attempted=True,
    )
