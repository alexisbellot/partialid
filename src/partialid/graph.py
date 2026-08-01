"""Causal diagrams and topological orders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


def normalize_hyperedge(edge: Sequence[str]) -> tuple[str, ...]:
    """Sort and dedupe a latent hyperedge; require at least two nodes."""
    nodes = tuple(sorted(set(edge)))
    if len(nodes) < 2:
        raise ValueError(f"latent hyperedge must cover ≥2 nodes; got {tuple(edge)}")
    return nodes


def normalize_bidirected(
    bidirected: Iterable[Sequence[str]],
) -> tuple[tuple[str, ...], ...]:
    """Normalize latent hyperedges (pairs or larger) to sorted unique tuples."""
    seen: set[tuple[str, ...]] = set()
    out: list[tuple[str, ...]] = []
    for e in bidirected:
        he = normalize_hyperedge(e)
        if he not in seen:
            seen.add(he)
            out.append(he)
    return tuple(out)


def full_latent_hyperedge(nodes: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    """One latent covering all listed nodes (Manski / maximal confounding)."""
    he = tuple(sorted(set(nodes)))
    if len(he) < 2:
        return ()
    return (he,)


@dataclass(frozen=True)
class CausalDiagram:
    """DAG over endogenous variables, with optional latent confounding.

    Parameters
    ----------
    nodes :
        Endogenous variable names.
    directed :
        Directed edges ``(parent, child)``.
    bidirected :
        Latent confounding **hyperedges** (each a tuple of ≥2 endogenous nodes).
        Each hyperedge is one independent exogenous variable pointing to all
        listed nodes. Structure is entirely determined by these hyperedges:

        - ``[("X", "Y")]`` — one latent on ``{X,Y}`` (c-degree 1)
        - ``[("X", "M"), ("M", "Y")]`` — two latents (c-degree 2)
        - ``[("X", "M", "Y")]`` — one latent on all three (c-degree 1)
        - ``[]`` — no latent confounding (Markovian)
    """

    nodes: tuple[str, ...]
    directed: tuple[tuple[str, str], ...] = ()
    bidirected: tuple[tuple[str, ...], ...] = ()

    def __post_init__(self) -> None:
        node_set = set(self.nodes)
        for a, b in self.directed:
            if a not in node_set or b not in node_set:
                raise ValueError(f"directed edge {(a, b)} uses unknown node")
        bi = normalize_bidirected(self.bidirected) if self.bidirected else ()
        for he in bi:
            for n in he:
                if n not in node_set:
                    raise ValueError(f"bidirected hyperedge {he} uses unknown node")
        if bi != self.bidirected:
            object.__setattr__(self, "bidirected", bi)

    @classmethod
    def from_order(cls, order: Sequence[str]) -> CausalDiagram:
        """Order-only Manski diagram: complete directed order + one full latent.

        Equivalent to ``bidirected=[tuple(sorted(order))]`` (single exogenous
        on every node). For a Markovian order, build ``from_edges`` with
        ``bidirected=[]`` instead.
        """
        nodes = tuple(order)
        directed = []
        for i, child in enumerate(nodes):
            for parent in nodes[:i]:
                directed.append((parent, child))
        return cls(
            nodes=nodes,
            directed=tuple(directed),
            bidirected=full_latent_hyperedge(nodes),
        )

    @classmethod
    def from_edges(
        cls,
        nodes: Sequence[str],
        directed: Iterable[tuple[str, str]] = (),
        bidirected: Iterable[Sequence[str]] = (),
    ) -> CausalDiagram:
        return cls(
            nodes=tuple(nodes),
            directed=tuple(directed),
            bidirected=normalize_bidirected(bidirected),
        )

    def parents(self, node: str) -> tuple[str, ...]:
        return tuple(a for a, b in self.directed if b == node)

    def children(self, node: str) -> tuple[str, ...]:
        return tuple(b for a, b in self.directed if a == node)

    def descendants(self, node: str) -> frozenset[str]:
        """Strict directed descendants of ``node`` (not including itself)."""
        out: set[str] = set()
        stack = list(self.children(node))
        while stack:
            n = stack.pop()
            if n in out:
                continue
            out.add(n)
            stack.extend(self.children(n))
        return frozenset(out)

    def topological_order(self) -> tuple[str, ...]:
        """Kahn topological sort; raises if cyclic."""
        indeg = {n: 0 for n in self.nodes}
        children: dict[str, list[str]] = {n: [] for n in self.nodes}
        for a, b in self.directed:
            indeg[b] += 1
            children[a].append(b)
        queue = [n for n in self.nodes if indeg[n] == 0]
        order: list[str] = []
        while queue:
            n = queue.pop(0)
            order.append(n)
            for c in children[n]:
                indeg[c] -= 1
                if indeg[c] == 0:
                    queue.append(c)
        if len(order) != len(self.nodes):
            raise ValueError("causal diagram has a directed cycle")
        return tuple(order)

    def latent_hyperedges_in(self, district: Iterable[str]) -> tuple[tuple[str, ...], ...]:
        """Hyperedges wholly contained in ``district``."""
        keep = set(district)
        return tuple(he for he in self.bidirected if set(he) <= keep)

    def c_degree(self, district: Iterable[str]) -> int:
        """Number of latent hyperedges in ``district``.

        Unconfounded singletons (no hyperedge) have c-degree 1 for their
        private exogenous noise, but they are not *free confounded* blocks.
        """
        d = frozenset(district)
        n_lat = len(self.latent_hyperedges_in(d))
        if n_lat == 0:
            return 1
        return n_lat

    def c_components(self) -> list[frozenset[str]]:
        """Connected components under latent hyperedge incidence."""
        parent = {n: n for n in self.nodes}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for he in self.bidirected:
            if not he:
                continue
            root = he[0]
            for n in he[1:]:
                union(root, n)
        comps: dict[str, set[str]] = {}
        for n in self.nodes:
            comps.setdefault(find(n), set()).add(n)
        out: list[frozenset[str]] = []
        seen: set[str] = set()
        for n in self.nodes:
            r = find(n)
            if r not in seen:
                seen.add(r)
                out.append(frozenset(comps[r]))
        return out
