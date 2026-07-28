"""Causal diagrams and topological orders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


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
        Latent confounding edges ``(i, j)`` (unordered in spirit; stored sorted).
        If ``None`` and ``maximal_confounding`` is True, every pair shares a latent.
    maximal_confounding :
        If True (default when no bidirected edges given under order-only),
        put all nodes in one c-component.
    """

    nodes: tuple[str, ...]
    directed: tuple[tuple[str, str], ...] = ()
    bidirected: tuple[tuple[str, str], ...] = ()
    maximal_confounding: bool = False

    def __post_init__(self) -> None:
        node_set = set(self.nodes)
        for a, b in self.directed:
            if a not in node_set or b not in node_set:
                raise ValueError(f"directed edge {(a, b)} uses unknown node")
        for a, b in self.bidirected:
            if a not in node_set or b not in node_set:
                raise ValueError(f"bidirected edge {(a, b)} uses unknown node")

    @classmethod
    def from_order(
        cls,
        order: Sequence[str],
        *,
        maximal_confounding: bool = True,
    ) -> CausalDiagram:
        """Order-only diagram: edges from each predecessor to each successor.

        With ``maximal_confounding=True`` (default), one c-component — the
        Prop. 0.2 maximal-confounding / Manski regime.
        """
        nodes = tuple(order)
        directed = []
        for i, child in enumerate(nodes):
            for parent in nodes[:i]:
                directed.append((parent, child))
        return cls(
            nodes=nodes,
            directed=tuple(directed),
            bidirected=(),
            maximal_confounding=maximal_confounding,
        )

    @classmethod
    def from_edges(
        cls,
        nodes: Sequence[str],
        directed: Iterable[tuple[str, str]] = (),
        bidirected: Iterable[tuple[str, str]] = (),
        *,
        maximal_confounding: bool = False,
    ) -> CausalDiagram:
        bi = tuple((a, b) if a <= b else (b, a) for a, b in ((e[0], e[1]) for e in bidirected))
        return cls(
            nodes=tuple(nodes),
            directed=tuple(directed),
            bidirected=bi,
            maximal_confounding=maximal_confounding,
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

    def c_components(self) -> list[frozenset[str]]:
        """Connected components under bidirected edges (plus maximal option)."""
        if self.maximal_confounding:
            return [frozenset(self.nodes)]
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

        for a, b in self.bidirected:
            union(a, b)
        comps: dict[str, set[str]] = {}
        for n in self.nodes:
            comps.setdefault(find(n), set()).add(n)
        # stable order by first appearance in nodes
        out: list[frozenset[str]] = []
        seen: set[str] = set()
        for n in self.nodes:
            r = find(n)
            if r not in seen:
                seen.add(r)
                out.append(frozenset(comps[r]))
        return out
