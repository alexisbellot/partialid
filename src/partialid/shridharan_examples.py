"""Shridharan–Iyengar (JMLR 2023 / arXiv:2308.02709) Appendix B examples.

Implements Examples A–G: Sachs-partition ADMGs, logit–Bernoulli DGPs with
shared Gaussian latents (U_A, U_B), and the paper's conditional interventional
queries. Joint tables are obtained by Monte Carlo integration over (U_A, U_B).
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product
from typing import Callable, Mapping, Sequence

import numpy as np

from .assumptions import Assumptions
from .graph import CausalDiagram
from .problems import Problem
from .query import Query
from .variables import Variables

# Paper Table 1 (JMLR 24:22-1081) — reference sizes for comparison.
PAPER_TABLE1: dict[str, dict[str, float]] = {
    "A": {"R": 1.3e8, "max_H": 1.0e6, "H": 2.3e3},
    "B": {"R": 4.2e6, "max_H": 1.0e6, "H": 7.1e4},
    "C": {"R": 4.2e6, "max_H": 1.0e6, "H": 4.4e4},
    "D": {"R": 6.3e57, "max_H": 1.7e7, "H": 9.4e6},
    "E": {"R": 3.2e32, "max_H": 1.7e7, "H": 9.4e6},
    "F": {"R": 1.8e13, "max_H": 6.5e4, "H": 5.8e4},
    "G": {"R": 3.0e23, "max_H": 1.0e6, "H": 9.2e3},
}

# Paper Table 2 — pruned-LP construction times (seconds).
PAPER_TABLE2_tH: dict[str, float] = {
    "A": 2.0,
    "B": 3.9,
    "C": 3.7,
    "F": 0.6,
    "G": 2.4,
}


def _sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


def _bern(val: int, logit: float) -> float:
    p = float(_sigmoid(logit))
    return p if val == 1 else 1.0 - p


def _clique(nodes: Sequence[str]) -> list[tuple[str, str]]:
    return [(a, b) for a, b in combinations(nodes, 2)]


def _mc_joint(
    order: Sequence[str],
    logit_fn: Callable[[float, float, Mapping[str, int]], Mapping[str, float]],
    *,
    n_mc: int = 4000,
    seed: int = 0,
) -> dict[tuple, float]:
    """Average P(V | U_A, U_B) over U_A, U_B ~ N(0,1)."""
    rng = np.random.default_rng(seed)
    n = len(order)
    mass = {cell: 0.0 for cell in product((0, 1), repeat=n)}
    for _ in range(n_mc):
        ua, ub = float(rng.normal()), float(rng.normal())
        for cell in mass:
            world = dict(zip(order, cell))
            logits = logit_fn(ua, ub, world)
            p = 1.0
            for v in order:
                p *= _bern(int(world[v]), float(logits[v]))
            mass[cell] += p
    s = sum(mass.values())
    if s <= 0:
        raise RuntimeError("DGP integration produced zero mass")
    return {k: v / s for k, v in mass.items()}


def _problem(
    letter: str,
    *,
    nodes: Sequence[str],
    directed: Sequence[tuple[str, str]],
    va: Sequence[str],
    vb: Sequence[str],
    do: Mapping[str, int],
    condition: Mapping[str, int],
    logit_fn: Callable[[float, float, Mapping[str, int]], Mapping[str, float]],
    n_mc: int,
    seed: int,
    notes: str = "",
) -> Problem:
    order = list(va) + list(vb)
    assert set(order) == set(nodes)
    V = Variables.from_binary(*order)
    bi = _clique(va) + _clique(vb)
    g = CausalDiagram.from_edges(
        nodes=list(nodes),
        directed=list(directed),
        bidirected=bi,
        maximal_confounding=False,
    )
    data = _mc_joint(order, logit_fn, n_mc=n_mc, seed=seed)
    Q = Query.interventional(
        outcome="Y",
        do=dict(do),
        value=1,
        condition=dict(condition),
    )
    return Problem(
        name=f"shridharan_ex{letter}",
        source="Shridharan–Iyengar JMLR 2023 Appendix B",
        variables=V,
        data=data,
        query=Q,
        assumptions=Assumptions.from_diagram(g),
        notes=notes or f"Example {letter}; VA={list(va)} VB={list(vb)}",
    )


# ---------------------------------------------------------------------------
# Examples A–G
# ---------------------------------------------------------------------------


def example_a(*, n_mc: int = 4000, seed: int = 0) -> Problem:
    """P(Y(X2=1)=1 | Z1=1, Z2=1)."""
    va = ("Z1", "Z2")
    vb = ("S1", "X1", "S2", "X2", "Y")
    nodes = va + vb
    directed = [
        ("S1", "X1"),
        ("S1", "S2"),
        ("X1", "S2"),
        ("Z2", "S2"),
        ("S2", "X2"),
        ("Z1", "X2"),
        ("Z2", "X2"),
        ("S2", "Y"),
        ("X2", "Y"),
        ("Z2", "Y"),
    ]

    def logits(ua: float, ub: float, w: Mapping[str, int]) -> dict[str, float]:
        return {
            "Z1": ua,
            "Z2": ua,
            "S1": ub,
            "X1": ub + w["S1"],
            "S2": w["S1"] + ub + w["X1"] + w["Z2"],
            "X2": w["S2"] + ub + w["Z1"] + w["Z2"],
            "Y": ub + w["S2"] + w["X2"] + w["Z2"],
        }

    return _problem(
        "A",
        nodes=nodes,
        directed=directed,
        va=va,
        vb=vb,
        do={"X2": 1},
        condition={"Z1": 1, "Z2": 1},
        logit_fn=logits,
        n_mc=n_mc,
        seed=seed,
    )


def example_b(*, n_mc: int = 4000, seed: int = 1) -> Problem:
    """P(Y(A=1,B=1)=1 | C=1, F=1)."""
    va = ("C", "F")
    vb = ("A", "B", "D", "E", "Y")
    nodes = va + vb
    directed = [
        ("C", "A"),
        ("F", "A"),
        ("C", "B"),
        ("F", "B"),
        ("A", "D"),
        ("A", "E"),
        ("B", "E"),
        ("D", "Y"),
        ("C", "Y"),
        ("E", "Y"),
    ]

    def logits(ua: float, ub: float, w: Mapping[str, int]) -> dict[str, float]:
        return {
            "C": ua,
            "F": ua,
            "A": w["C"] + w["F"] + ub,
            "B": w["C"] + w["F"] + ub,
            "D": w["A"] + ub,
            "E": w["A"] + w["B"] + ub,
            "Y": ub + w["D"] + w["C"] + w["E"],
        }

    return _problem(
        "B",
        nodes=nodes,
        directed=directed,
        va=va,
        vb=vb,
        do={"A": 1, "B": 1},
        condition={"C": 1, "F": 1},
        logit_fn=logits,
        n_mc=n_mc,
        seed=seed,
    )


def example_c(*, n_mc: int = 4000, seed: int = 2) -> Problem:
    """P(Y(M1=1)=1 | W1=1, W3=1)."""
    va = ("W1", "W3")
    vb = ("T", "M1", "M2", "Y", "X3")
    nodes = va + vb
    directed = [
        ("W1", "T"),
        ("W3", "T"),
        ("T", "M1"),
        ("W3", "M1"),
        ("M1", "M2"),
        ("W1", "M2"),
        ("M2", "Y"),
        ("Y", "X3"),
        ("M1", "X3"),
        ("T", "X3"),
    ]

    def logits(ua: float, ub: float, w: Mapping[str, int]) -> dict[str, float]:
        return {
            "W1": ua,
            "W3": ua,
            "T": w["W1"] + w["W3"] + ub,
            "M1": w["T"] + w["W3"] + ub,
            "M2": w["M1"] + w["W1"] + ub,
            "Y": w["M2"] + ub,
            "X3": ub + w["Y"] + w["M1"] + w["T"],
        }

    return _problem(
        "C",
        nodes=nodes,
        directed=directed,
        va=va,
        vb=vb,
        do={"M1": 1},
        condition={"W1": 1, "W3": 1},
        logit_fn=logits,
        n_mc=n_mc,
        seed=seed,
    )


def example_d(*, n_mc: int = 2000, seed: int = 3) -> Problem:
    """P(Y(D=1)=1 | E=1, F=1). LP too large after pruning (paper §3)."""
    va = ("E", "F")
    vb = ("A", "B", "C", "D", "Y", "G")
    nodes = va + vb
    directed = [
        ("F", "A"),
        ("E", "A"),
        ("E", "B"),
        ("F", "B"),
        ("A", "B"),
        ("B", "C"),
        ("E", "C"),
        ("F", "C"),
        ("A", "C"),
        ("B", "D"),
        ("E", "D"),
        ("F", "D"),
        ("A", "D"),
        ("C", "D"),
        ("E", "Y"),
        ("D", "Y"),
        ("A", "G"),
        ("B", "G"),
        ("C", "G"),
        ("D", "G"),
        ("Y", "G"),
        ("E", "G"),
        ("F", "G"),
    ]

    def logits(ua: float, ub: float, w: Mapping[str, int]) -> dict[str, float]:
        return {
            "F": ua,
            "E": ua,
            "A": w["F"] + w["E"] + ub,
            "B": w["E"] + w["F"] + w["A"] + ub,
            "C": w["B"] + w["E"] + w["F"] + w["A"] + ub,
            "D": w["B"] + w["E"] + w["F"] + w["A"] + w["C"] + ub,
            "Y": w["E"] + w["D"] + ub,
            "G": (w["A"] + w["B"] + w["C"] + w["D"] + w["Y"] + w["E"] + w["F"] + ub),
        }

    return _problem(
        "D",
        nodes=nodes,
        directed=directed,
        va=va,
        vb=vb,
        do={"D": 1},
        condition={"E": 1, "F": 1},
        logit_fn=logits,
        n_mc=n_mc,
        seed=seed,
        notes="Example D; pruned LP still too large (paper §3)",
    )


def example_e(*, n_mc: int = 2000, seed: int = 4) -> Problem:
    """P(Y(C=1,F=1)=1 | A=1, B=1). LP too large after pruning."""
    va = ("A", "B")
    vb = ("C", "D", "E", "G", "F", "Y")
    nodes = va + vb
    # Topo: A,B | C,D,E,G,F,Y  (F depends on G)
    directed = [
        ("A", "C"),
        ("B", "C"),
        ("A", "D"),
        ("C", "D"),
        ("B", "D"),
        ("A", "E"),
        ("B", "E"),
        ("A", "G"),
        ("B", "G"),
        ("C", "G"),
        ("D", "G"),
        ("A", "F"),
        ("C", "F"),
        ("B", "F"),
        ("D", "F"),
        ("E", "F"),
        ("G", "F"),
        ("A", "Y"),
        ("E", "Y"),
        ("B", "Y"),
        ("F", "Y"),
    ]

    def logits(ua: float, ub: float, w: Mapping[str, int]) -> dict[str, float]:
        return {
            "A": ua,
            "B": ua,
            "C": w["A"] + w["B"] + ub,
            "D": w["A"] + w["C"] + w["B"] + ub,
            "E": w["A"] + w["B"] + ub,
            "G": ub + w["A"] + w["B"] + w["C"] + w["D"],
            "F": w["A"] + w["C"] + w["B"] + w["D"] + w["E"] + w["G"] + ub,
            "Y": ub + w["A"] + w["E"] + w["B"] + w["F"],
        }

    return _problem(
        "E",
        nodes=nodes,
        directed=directed,
        va=va,
        vb=vb,
        do={"C": 1, "F": 1},
        condition={"A": 1, "B": 1},
        logit_fn=logits,
        n_mc=n_mc,
        seed=seed,
        notes="Example E; pruned LP still too large (paper §3)",
    )


def example_f(*, n_mc: int = 4000, seed: int = 5) -> Problem:
    """P(Y(T1=T2=T3=1)=1 | C1=1, C2=1) — multi-cause style."""
    va = ("C1", "C2")
    vb = ("T1", "T2", "T3", "Y")
    nodes = va + vb
    directed = [
        ("C1", "T1"),
        ("C2", "T1"),
        ("C1", "T2"),
        ("C2", "T2"),
        ("C1", "T3"),
        ("C2", "T3"),
        ("T1", "Y"),
        ("T2", "Y"),
        ("T3", "Y"),
        ("C2", "Y"),
    ]

    def logits(ua: float, ub: float, w: Mapping[str, int]) -> dict[str, float]:
        return {
            "C1": ua,
            "C2": ua,
            "T1": w["C1"] + w["C2"] + ub,
            "T2": w["C1"] + w["C2"] + ub,
            "T3": w["C1"] + w["C2"] + ub,
            "Y": ub + w["T1"] + w["T2"] + w["T3"] + w["C2"],
        }

    return _problem(
        "F",
        nodes=nodes,
        directed=directed,
        va=va,
        vb=vb,
        do={"T1": 1, "T2": 1, "T3": 1},
        condition={"C1": 1, "C2": 1},
        logit_fn=logits,
        n_mc=n_mc,
        seed=seed,
    )


def example_g(*, n_mc: int = 4000, seed: int = 6) -> Problem:
    """P(Y(B=1)=1 | W1=1, W2=1)."""
    va = ("W1", "W2")
    vb = ("D", "A", "C", "B", "Y")
    nodes = va + vb
    directed = [
        ("W1", "D"),
        ("W1", "A"),
        ("D", "A"),
        ("A", "C"),
        ("D", "C"),
        ("W1", "C"),
        ("D", "B"),
        ("C", "B"),
        ("A", "B"),
        ("W1", "B"),
        ("W2", "B"),
        ("D", "Y"),
        ("C", "Y"),
        ("A", "Y"),
        ("B", "Y"),
        ("W1", "Y"),
    ]

    def logits(ua: float, ub: float, w: Mapping[str, int]) -> dict[str, float]:
        return {
            "W1": ua,
            "W2": ua,
            "D": w["W1"] + ub,
            "A": w["W1"] + w["D"] + ub,
            "C": w["A"] + w["D"] + w["W1"] + ub,
            "B": w["D"] + w["C"] + w["A"] + w["W1"] + w["W2"] + ub,
            "Y": ub + w["D"] + w["C"] + w["A"] + w["B"] + w["W1"],
        }

    return _problem(
        "G",
        nodes=nodes,
        directed=directed,
        va=va,
        vb=vb,
        do={"B": 1},
        condition={"W1": 1, "W2": 1},
        logit_fn=logits,
        n_mc=n_mc,
        seed=seed,
    )


EXAMPLES: dict[str, Callable[..., Problem]] = {
    "A": example_a,
    "B": example_b,
    "C": example_c,
    "D": example_d,
    "E": example_e,
    "F": example_f,
    "G": example_g,
}


def all_examples(*, n_mc: int = 4000, seed: int = 0) -> list[Problem]:
    out = []
    for i, letter in enumerate("ABCDEFG"):
        kwargs: dict = {"seed": seed + i}
        if letter in "DE":
            kwargs["n_mc"] = min(n_mc, 2000)
        else:
            kwargs["n_mc"] = n_mc
        out.append(EXAMPLES[letter](**kwargs))
    return out


@dataclass(frozen=True)
class SizeReport:
    letter: str
    va: tuple[str, ...]
    vb: tuple[str, ...]
    raw_types_est: int
    max_hyperarcs: int
    n_hyperarcs: int | None
    paper_R: float
    paper_max_H: float
    paper_H: float


def size_report(
    letter: str,
    problem: Problem | None = None,
    *,
    count_H: bool = True,
) -> SizeReport:
    """Compare local |R| / max|H| / |H| estimates to paper Table 1."""
    from .hyperarc import (
        _estimate_raw_types,
        _partition_va_vb,
        enumerate_valid_hyperarcs,
    )

    p = problem or EXAMPLES[letter](n_mc=64, seed=0)  # tiny MC; sizes ignore P
    diagram = p.assumptions.diagram()
    assert diagram is not None
    va, vb = _partition_va_vb(diagram, p.query.intervention, p.query.outcome)
    raw = _estimate_raw_types(p.variables, diagram, vb)
    max_h = (2 ** len(vb)) ** (2 ** len(va)) if va else 2 ** len(vb)
    n_h: int | None = None
    if count_H and max_h <= 2_000_000:
        n_h = len(enumerate_valid_hyperarcs(p.variables, diagram, va, vb))
    ref = PAPER_TABLE1[letter]
    return SizeReport(
        letter=letter,
        va=va,
        vb=vb,
        raw_types_est=raw,
        max_hyperarcs=int(max_h),
        n_hyperarcs=n_h,
        paper_R=ref["R"],
        paper_max_H=ref["max_H"],
        paper_H=ref["H"],
    )
