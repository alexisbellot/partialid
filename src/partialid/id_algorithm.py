"""Shpitser–Pearl ID algorithm on ADMGs (bidirected latent confounding).

Implements the recursive ID procedure (Shpitser & Pearl 2006; cf. DoWhy
``id_identifier``) with c-components = bidirected districts. Returns an
expression tree that can be evaluated on a discrete observational table.

Graph-theoretic steps are cheap (poly in |V|). Evaluating the expression
sums over intermediate variables and can be costly for large domains — but
typically far cheaper than a full response-type LP, and necessary for
correctness when our LP uses a merged outer approximation.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Mapping, Sequence

import numpy as np

from .graph import CausalDiagram
from .variables import Variables

# ---------------------------------------------------------------------------
# Expression AST
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CondFactor:
    """Observational factor P(outcome_vars | condition_vars)."""

    outcome_vars: tuple[str, ...]
    condition_vars: tuple[str, ...]


@dataclass
class IDExpr:
    """Product of factors / sub-expressions, marginalized over ``sum_vars``."""

    factors: list  # CondFactor | IDExpr
    sum_vars: tuple[str, ...] = ()

    def is_fail(self) -> bool:
        return False


@dataclass
class IDFail:
    def is_fail(self) -> bool:
        return True


# ---------------------------------------------------------------------------
# ADMG helpers
# ---------------------------------------------------------------------------


def _districts(g: CausalDiagram, nodes: Iterable[str] | None = None) -> list[frozenset[str]]:
    """Bidirected connected components (c-components / districts)."""
    node_set = set(nodes) if nodes is not None else set(g.nodes)
    parent = {n: n for n in node_set}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in g.bidirected:
        if a in node_set and b in node_set:
            union(a, b)
    comps: dict[str, set[str]] = {}
    for n in node_set:
        comps.setdefault(find(n), set()).add(n)
    # stable
    out: list[frozenset[str]] = []
    seen: set[str] = set()
    for n in g.nodes:
        if n not in node_set:
            continue
        r = find(n)
        if r not in seen:
            seen.add(r)
            out.append(frozenset(comps[r]))
    return out


def _ancestors(g: CausalDiagram, seeds: Iterable[str], nodes: set[str] | None = None) -> set[str]:
    """Directed ancestors, inclusive, optionally restricted to ``nodes``."""
    allow = set(g.nodes) if nodes is None else set(nodes)
    an = {s for s in seeds if s in allow}
    changed = True
    while changed:
        changed = False
        for a, b in g.directed:
            if b in an and a in allow and a not in an:
                an.add(a)
                changed = True
    return an


def _subgraph(g: CausalDiagram, nodes: Iterable[str]) -> CausalDiagram:
    keep = set(nodes)
    return CausalDiagram(
        nodes=tuple(n for n in g.nodes if n in keep),
        directed=tuple((a, b) for a, b in g.directed if a in keep and b in keep),
        bidirected=tuple((a, b) for a, b in g.bidirected if a in keep and b in keep),
        maximal_confounding=False,
    )


def _topo(g: CausalDiagram, nodes: Iterable[str] | None = None) -> list[str]:
    allow = set(g.nodes) if nodes is None else set(nodes)
    indeg = {n: 0 for n in allow}
    children: dict[str, list[str]] = {n: [] for n in allow}
    for a, b in g.directed:
        if a in allow and b in allow:
            indeg[b] += 1
            children[a].append(b)
    queue = [n for n in g.nodes if n in allow and indeg[n] == 0]
    order: list[str] = []
    while queue:
        n = queue.pop(0)
        order.append(n)
        for c in children[n]:
            indeg[c] -= 1
            if indeg[c] == 0:
                queue.append(c)
    if len(order) != len(allow):
        # fall back to declaration order among allow
        return [n for n in g.nodes if n in allow]
    return order


# ---------------------------------------------------------------------------
# ID recursion
# ---------------------------------------------------------------------------


def identify(
    g: CausalDiagram,
    treatments: Sequence[str],
    outcomes: Sequence[str],
) -> IDExpr | IDFail:
    """Return an identifying expression for P(outcomes | do(treatments)), or fail."""
    if g.maximal_confounding and len(g.nodes) > 1 and not g.bidirected:
        # treat as one district
        pass
    return _id(
        g,
        frozenset(treatments),
        frozenset(outcomes),
        frozenset(g.nodes),
    )


def _id(
    g: CausalDiagram,
    X: frozenset[str],
    Y: frozenset[str],
    V: frozenset[str],
) -> IDExpr | IDFail:
    gV = _subgraph(g, V)
    X = X & V
    Y = Y & V

    # Line 1: no intervention
    if not X:
        return IDExpr(
            factors=[CondFactor(outcome_vars=tuple(sorted(V)), condition_vars=())],
            sum_vars=tuple(sorted(V - Y)),
        )

    # Line 2: restrict to ancestors of Y
    an = _ancestors(gV, Y)
    if V - an:
        return _id(g, X & an, Y, frozenset(an))

    # Line 3: force inactive interventions
    # G_{\bar X}: remove edges *into* X
    directed_bar = tuple((a, b) for a, b in gV.directed if b not in X)
    g_bar = CausalDiagram(
        nodes=gV.nodes,
        directed=directed_bar,
        bidirected=gV.bidirected,
        maximal_confounding=False,
    )
    an_bar = _ancestors(g_bar, Y)
    W = (V - X) - an_bar
    if W:
        return _id(g, X | frozenset(W), Y, V)

    # Line 4: c-component factorization of G_{V\\X}
    VminusX = V - X
    comps = _districts(gV, VminusX)
    if len(comps) > 1:
        factors: list = []
        for S in comps:
            sub = _id(g, V - S, frozenset(S), V)
            if isinstance(sub, IDFail) or sub.is_fail():
                return IDFail()
            factors.append(sub)
        return IDExpr(factors=factors, sum_vars=tuple(sorted(V - Y - X)))

    # Line 5: single district S in G_{V\\X}
    S = comps[0]
    comps_G = _districts(gV, V)

    # Hedge: whole graph is one district
    if len(comps_G) == 1 and comps_G[0] == V:
        return IDFail()

    # Line 6: S itself is a district of G
    if S in comps_G:
        # ∏_{v in S topo} P(v | past) , sum over S\Y
        topo = _topo(gV, S)
        factors = []
        prev: list[str] = []
        # conditioning set is all predecessors in V that precede v in a topo of V
        topo_V = _topo(gV, V)
        for v in topo_V:
            if v not in S:
                continue
            # predecessors among V in topo order
            cond = tuple(u for u in topo_V if u == v and False)  # placeholder
        factors = []
        seen_before: list[str] = []
        for v in topo_V:
            if v in S:
                factors.append(CondFactor(outcome_vars=(v,), condition_vars=tuple(seen_before)))
            seen_before.append(v)
        return IDExpr(factors=factors, sum_vars=tuple(sorted(S - Y)))

    # Line 7: S proper subset of some district S' of G
    for Sprime in comps_G:
        if S < Sprime:  # proper subset
            return _id(g, X & Sprime, Y, frozenset(Sprime))

    return IDFail()


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


class _Table:
    def __init__(self, variables: Variables, order: Sequence[str], data: Mapping):
        self.order = tuple(order)
        self.idx = {v: i for i, v in enumerate(self.order)}
        self.domains = {v: tuple(variables.domains[v]) for v in self.order}
        cells = variables.cells(self.order)
        self.P = np.zeros(len(cells), dtype=float)
        for key, val in data.items():
            cell = tuple(key[v] for v in self.order) if isinstance(key, dict) else tuple(key)
            self.P[cells.index(cell)] += float(val)
        s = float(self.P.sum())
        if s <= 0:
            raise ValueError("empty distribution")
        self.P /= s
        self.cells = cells

    def _match(self, cell: tuple, asg: Mapping[str, object]) -> bool:
        for v, val in asg.items():
            if v in self.idx and cell[self.idx[v]] != val:
                return False
        return True

    def joint(self, asg: Mapping[str, object]) -> float:
        return float(sum(self.P[i] for i, c in enumerate(self.cells) if self._match(c, asg)))

    def cond(
        self,
        outcome: Mapping[str, object],
        given: Mapping[str, object],
    ) -> float:
        den = self.joint(given)
        if den <= 1e-15:
            return float("nan")
        return self.joint({**given, **outcome}) / den


def _eval_expr(
    expr: IDExpr | CondFactor,
    table: _Table,
    fixed: Mapping[str, object],
) -> float:
    if isinstance(expr, CondFactor):
        # With current assignment in ``fixed`` for all vars, return
        # P(outcome=fixed[outcome] | condition=fixed[condition])
        out = {v: fixed[v] for v in expr.outcome_vars}
        given = {v: fixed[v] for v in expr.condition_vars}
        return table.cond(out, given)

    assert isinstance(expr, IDExpr)
    sum_vars = [v for v in expr.sum_vars if v not in fixed]
    if not sum_vars:
        # product of factors
        val = 1.0
        for f in expr.factors:
            piece = _eval_expr(f, table, fixed)
            if not np.isfinite(piece):
                return float("nan")
            val *= piece
        return val

    domains = [table.domains[v] for v in sum_vars]
    total = 0.0
    for assign in product(*domains):
        world = dict(fixed)
        world.update(dict(zip(sum_vars, assign)))
        val = 1.0
        ok = True
        for f in expr.factors:
            piece = _eval_expr(f, table, world)
            if not np.isfinite(piece):
                ok = False
                break
            val *= piece
        if ok:
            total += val
    return float(total)


def evaluate_id(
    expr: IDExpr,
    variables: Variables,
    data: Mapping,
    *,
    treatments: Mapping[str, object],
    outcomes: Mapping[str, object],
    order: Sequence[str],
) -> float:
    """Evaluate identifying expression at concrete treatment/outcome values."""
    table = _Table(variables, order, data)
    fixed = {**dict(treatments), **dict(outcomes)}
    return _eval_expr(expr, table, fixed)


def is_identified(g: CausalDiagram, treatments: Sequence[str], outcomes: Sequence[str]) -> bool:
    expr = identify(g, treatments, outcomes)
    return not isinstance(expr, IDFail) and not expr.is_fail()
