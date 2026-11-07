"""Graphical checks for a single linear program on every positive table.

The checks follow the checklist in ``WHEN_THE_BOUNDS_ARE_AN_LP.md``:
delete a hyperedge contained in another, then test saturation of a
component that the program does not enforce, and covering of one that
it does. Covering of a component with three or more hyperedges is the
sufficient fiber lemma (one hyperedge contains every variable that has
a parent; every other variable lies on at most one hyperedge). The
mixing step is not a theorem and is not implemented.
"""

from __future__ import annotations

from .graph import CausalDiagram


def without_contained_hyperedges(diagram: CausalDiagram) -> CausalDiagram:
    """Drop a hyperedge that is a subset of another, and drop exact duplicates.

    Contained hyperedges do not change the realizable laws. ``c_degree`` on
    the original diagram is left alone; callers that count λ opt in.
    """
    hes = list(diagram.bidirected)
    drop: set[int] = set()
    for i, he in enumerate(hes):
        s = set(he)
        for j, other in enumerate(hes):
            if i == j:
                continue
            o = set(other)
            if s < o or (s == o and i > j):
                drop.add(i)
                break
    if not drop:
        return diagram
    kept = tuple(he for i, he in enumerate(hes) if i not in drop)
    return CausalDiagram(nodes=diagram.nodes, directed=diagram.directed, bidirected=kept)


def _hyperedges(diagram: CausalDiagram, district: frozenset[str]) -> tuple[tuple[str, ...], ...]:
    """Hyperedges of ``district`` after deleting contained ones."""
    return without_contained_hyperedges(
        CausalDiagram(
            nodes=tuple(diagram.topological_order()),
            directed=diagram.directed,
            bidirected=diagram.latent_hyperedges_in(district),
        )
    ).bidirected


def outside_parents(diagram: CausalDiagram, district: frozenset[str]) -> tuple[str, ...]:
    """Parents of ``district`` that lie outside it, in topological order."""
    inside = set(district)
    found: set[str] = set()
    for v in district:
        for p in diagram.parents(v):
            if p not in inside:
                found.add(p)
    return tuple(n for n in diagram.topological_order() if n in found)


def _reaches(diagram: CausalDiagram, src: str, dst: str) -> bool:
    """True when a directed path of positive length runs from ``src`` to ``dst``."""
    if src == dst:
        return False
    seen: set[str] = set()
    stack = list(diagram.children(src))
    while stack:
        n = stack.pop()
        if n == dst:
            return True
        if n in seen:
            continue
        seen.add(n)
        stack.extend(diagram.children(n))
    return False


def _reachability(diagram: CausalDiagram, district: frozenset[str]) -> bool:
    """Every outside parent reaches, or is a parent of, every variable in ``district``."""
    for z in outside_parents(diagram, district):
        for y in district:
            if z in diagram.parents(y) or _reaches(diagram, y, z):
                continue
            return False
    return True


def _cross_edge(
    diagram: CausalDiagram, hes: tuple[tuple[str, ...], ...]
) -> bool:
    h, k = set(hes[0]), set(hes[1])
    left, right = h - k, k - h
    if not left or not right:
        return False
    for a in left:
        for b in right:
            if (a, b) not in diagram.directed and (b, a) not in diagram.directed:
                return False
    return True


def _two_sided_blind(
    diagram: CausalDiagram,
    district: frozenset[str],
    hes: tuple[tuple[str, ...], ...],
) -> bool:
    """True when some intersection node is blind, from both private sides, to an outside parent."""
    h, k = set(hes[0]), set(hes[1])
    left, right, inter = h - k, k - h, h & k
    outside = outside_parents(diagram, district)

    def blind(w: str, rho: str) -> bool:
        if rho in diagram.parents(w):
            return False
        return any(
            z in diagram.parents(w) and z not in diagram.parents(rho) for z in outside
        )

    for rho in inter:
        if any(blind(w, rho) for w in left) and any(blind(w, rho) for w in right):
            return True
    return False


def saturation_status(diagram: CausalDiagram, district: frozenset[str]) -> str:
    """``saturated``, ``unsaturated``, or ``unknown``.

    ``unknown`` means three or more hyperedges: no graphical saturation test
    is claimed. One hyperedge is saturated exactly when reachability holds.
    Two hyperedges also need the cross edge and no two-sided blindness.
    """
    hes = _hyperedges(diagram, district)
    if len(hes) >= 3:
        return "unknown"
    if len(hes) <= 1:
        return "saturated" if _reachability(diagram, district) else "unsaturated"
    if not _cross_edge(diagram, hes) or _two_sided_blind(diagram, district, hes):
        return "unsaturated"
    if not _reachability(diagram, district):
        return "unsaturated"
    return "saturated"


def sufficient_hyperedge(
    diagram: CausalDiagram, district: frozenset[str]
) -> tuple[str, ...] | None:
    """A hyperedge witnessing the sufficient fiber lemma, if one exists.

    The hyperedge contains every variable that has a parent. Every variable
    off it lies on at most one hyperedge, so the fiber is a polytope of
    local assignments. Returns None for an unconfounded district (no
    hyperedge) and for the λ=2 root-private shape only when that shape is
    the lemma; callers that already have a root-private solver need not
    use this.
    """
    hes = _hyperedges(diagram, district)
    if not hes:
        return None
    nodes = set(district)
    nonroots = {v for v in district if diagram.parents(v)}
    for he in hes:
        H = set(he)
        if any(v not in H for v in nonroots):
            continue
        off = nodes - H
        if all(sum(v in set(e) for e in hes) <= 1 for v in off):
            return he
    return None


def is_covered(diagram: CausalDiagram, district: frozenset[str]) -> bool:
    """True when the fiber is known to be a polytope.

    Unconfounded districts, single hyperedges, the λ=2 root-private shape,
    and the sufficient fiber lemma.
    """
    hes = _hyperedges(diagram, district)
    if len(hes) <= 1:
        return True
    return sufficient_hyperedge(diagram, district) is not None


def side_root_blocks(
    diagram: CausalDiagram, district: frozenset[str], hyperedge: tuple[str, ...]
) -> tuple[tuple[str, ...], ...] | None:
    """Partition of the variables off ``hyperedge``, one block per side hyperedge.

    Roots that lie on no hyperedge are singleton blocks. Returns None when
    the partition is not the one the sufficient lemma allows.
    """
    hes = _hyperedges(diagram, district)
    H = set(hyperedge)
    off = set(district) - H
    assigned: set[str] = set()
    blocks: list[tuple[str, ...]] = []
    for he in hes:
        if set(he) == H:
            continue
        priv = tuple(v for v in he if v in off)
        if not priv:
            continue
        block = frozenset(priv)
        if block & assigned:
            return None
        assigned |= block
        blocks.append(tuple(v for v in diagram.topological_order() if v in block))
    for v in diagram.topological_order():
        if v in off and v not in assigned:
            blocks.append((v,))
    return tuple(blocks)
