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
class BaseMarg:
    """Observational marginal P(keep) of the input table."""

    keep: frozenset[str]


@dataclass
class IDExpr:
    """Product of factors / sub-expressions, marginalized over ``sum_vars``."""

    factors: list  # BaseMarg | Ratio | IDExpr
    sum_vars: tuple[str, ...] = ()

    def is_fail(self) -> bool:
        return False


@dataclass
class Ratio:
    """Quotient ``num / den`` of two expressions (a conditional)."""

    num: object
    den: object


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

    for he in g.bidirected:
        members = [n for n in he if n in node_set]
        if len(members) < 2:
            continue
        root = members[0]
        for n in members[1:]:
            union(root, n)
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
        bidirected=tuple(he for he in g.bidirected if set(he) <= keep),
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
    nodes = frozenset(g.nodes)
    expr = _id(g, frozenset(treatments), frozenset(outcomes), nodes, BaseMarg(nodes))
    if isinstance(expr, IDFail):
        return expr
    return IDExpr(factors=[expr])


def _marg(P, scope: frozenset[str], keep: frozenset[str]):
    """Marginal onto ``keep`` of distribution ``P`` over ``scope``.

    Variables of ``P`` outside ``scope`` are fixed by the enclosing context.
    """
    if isinstance(P, BaseMarg):
        return BaseMarg(P.keep - (scope - keep))
    drop = scope - keep
    if not drop:
        return P
    return IDExpr(factors=[P], sum_vars=tuple(sorted(drop)))


def _kernel_product(P, scope: frozenset[str], targets: frozenset[str], topo: list[str]) -> list:
    """Factors ∏_{v in targets} P(v | predecessors of v in ``topo``) under ``P``."""
    factors: list = []
    before: list[str] = []
    for v in topo:
        if v in targets:
            given = frozenset(before)
            num = _marg(P, scope, (given | {v}) & scope)
            factors.append(Ratio(num, _marg(P, scope, given & scope)) if given & scope else num)
        before.append(v)
    return factors


def _id(
    g: CausalDiagram,
    X: frozenset[str],
    Y: frozenset[str],
    V: frozenset[str],
    P,
):
    """ID(Y, X, P, G[V]) where ``P`` is a distribution over ``V``.

    Variables referenced by ``P`` outside ``V`` are fixed by the caller.
    """
    gV = _subgraph(g, V)
    X = X & V
    Y = Y & V

    # Line 1: no intervention
    if not X:
        return _marg(P, V, Y)

    # Line 2: restrict to ancestors of Y
    an = _ancestors(gV, Y)
    if V - an:
        keep = frozenset(an)
        return _id(g, X & keep, Y, keep, _marg(P, V, keep))

    # Line 3: force inactive interventions
    # G_{\bar X}: remove edges *into* X
    directed_bar = tuple((a, b) for a, b in gV.directed if b not in X)
    g_bar = CausalDiagram(
        nodes=gV.nodes,
        directed=directed_bar,
        bidirected=gV.bidirected,
    )
    an_bar = _ancestors(g_bar, Y)
    W = (V - X) - an_bar
    if W:
        return _id(g, X | frozenset(W), Y, V, P)

    # Line 4: c-component factorization of G_{V\\X}
    VminusX = V - X
    comps = _districts(gV, VminusX)
    if len(comps) > 1:
        factors: list = []
        for S in comps:
            sub = _id(g, V - S, frozenset(S), V, P)
            if isinstance(sub, IDFail):
                return IDFail()
            factors.append(sub)
        return IDExpr(factors=factors, sum_vars=tuple(sorted(V - Y - X)))

    # Line 5: single district S in G_{V\\X}
    S = comps[0]
    comps_G = _districts(gV, V)

    # Hedge: whole graph is one district
    if len(comps_G) == 1 and comps_G[0] == V:
        return IDFail()

    # The order must be topological in G, not only in G[V].
    topo_V = [v for v in _topo(g) if v in V]

    # Line 6: S is a district of G[V]: ∑_{S\Y} ∏_{v∈S} P(v | v_π^{(i-1)})
    if S in comps_G:
        return IDExpr(
            factors=_kernel_product(P, V, S, topo_V),
            sum_vars=tuple(sorted(S - Y)),
        )

    # Line 7: S ⊂ S' (a district of G[V]); recurse with P' = ∏_{v∈S'} P(v | v_π^{(i-1)})
    for Sprime in comps_G:
        if S < Sprime:
            P_new = IDExpr(factors=_kernel_product(P, V, Sprime, topo_V))
            return _id(g, X & Sprime, Y, frozenset(Sprime), P_new)

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
    expr: IDExpr | BaseMarg | Ratio,
    table: _Table,
    fixed: Mapping[str, object],
) -> float:
    if isinstance(expr, BaseMarg):
        # Line 3 of ID adds interventions whose value the expression does not
        # depend on; when the context leaves one unassigned, any value is valid.
        asg = {v: fixed.get(v, table.domains[v][0]) for v in expr.keep}
        return table.joint(asg)

    if isinstance(expr, Ratio):
        den = _eval_expr(expr.den, table, fixed)
        if not np.isfinite(den) or den <= 1e-15:
            return float("nan")
        return _eval_expr(expr.num, table, fixed) / den

    assert isinstance(expr, IDExpr)
    sum_vars = list(expr.sum_vars)
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
