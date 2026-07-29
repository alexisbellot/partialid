"""Literature problem catalog (IV, confounded exposure, multi-cause benchmarks).

Pure-Python instances for wall-clock and correctness benchmarks.
No R / causaloptim dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .assumptions import Assumptions, AssumptionSet
from .graph import CausalDiagram
from .query import Query
from .variables import Variables


@dataclass(frozen=True)
class Problem:
    """A named partial-identification instance."""

    name: str
    source: str
    variables: Variables
    data: dict
    query: Query
    assumptions: AssumptionSet
    notes: str = ""
    # optional analytic checks
    expected_width: float | None = None


def _softmax_table(keys: list[tuple], seed: int = 0) -> dict[tuple, float]:
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=len(keys))
    w = np.exp(logits - logits.max())
    w = w / w.sum()
    return {k: float(v) for k, v in zip(keys, w)}


def _iv_data_from_scm(
    variables: Variables,
    diagram: CausalDiagram,
    instruments: tuple[str, ...],
    r_side: tuple[str, ...],
    *,
    seed: int = 0,
) -> dict[tuple, float]:
    """Sample observational P from a valid instrument SCM (guarantees feasibility)."""
    from .instrument import InstrumentTypeSpace

    rng = np.random.default_rng(seed)
    ts = InstrumentTypeSpace.build(variables, diagram, instruments, r_side)
    logits = rng.normal(size=ts.n_types)
    q = np.exp(logits - logits.max())
    q = q / q.sum()
    i_cells = ts.i_cells
    pi_logits = rng.normal(size=len(i_cells))
    pi = np.exp(pi_logits - pi_logits.max())
    pi = pi / pi.sum()
    P_i = {ic: float(p) for ic, p in zip(i_cells, pi)}

    order = list(instruments) + list(r_side)
    data: dict[tuple, float] = {cell: 0.0 for cell in variables.cells(order)}
    for ic, p_i in P_i.items():
        for t, qt in enumerate(q):
            r_idx = ts.eval_r_indices(t, ic, None)
            r_vals = [ts.domain_list[v][r_idx[k]] for k, v in enumerate(r_side)]
            cell = tuple(list(ic) + r_vals)
            data[cell] = data.get(cell, 0.0) + p_i * qt
    s = sum(data.values())
    return {k: v / s for k, v in data.items()}


# ---------------------------------------------------------------------------
# Confounded exposure & instruments
# ---------------------------------------------------------------------------


def ternary_exposure(*, seed: int = 0) -> Problem:
    """Ternary confounded exposure X∈{0,1,2}, binary Y.

    Query P(Y=1 | do(X=1)). Under maximal confounding, Manski:
    width = 1 - P(X=1).
    """
    V = Variables.from_mapping({"X": (0, 1, 2), "Y": (0, 1)})
    keys = V.cells(["X", "Y"])
    raw = _softmax_table(keys, seed=seed)
    # ensure full support
    data = {k: max(v, 1e-3) for k, v in raw.items()}
    s = sum(data.values())
    data = {k: v / s for k, v in data.items()}
    px1 = sum(v for (x, y), v in data.items() if x == 1)
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    a = Assumptions.order("X", "Y")
    return Problem(
        name="ternary_exposure",
        source="ternary confounded exposure",
        variables=V,
        data=data,
        query=Q,
        assumptions=a,
        notes="natural bounds; width=1-P(X=1)",
        expected_width=1.0 - px1,
    )


def ternary_cardinalities(card_x: int = 4, card_y: int = 2, *, seed: int = 1) -> Problem:
    """Higher-cardinality variant of the confounded exposure example."""
    V = Variables.from_mapping({"X": tuple(range(card_x)), "Y": tuple(range(card_y))})
    keys = V.cells(["X", "Y"])
    data = _softmax_table(keys, seed=seed)
    do_x = min(1, card_x - 1)
    do_y = min(1, card_y - 1)
    px = sum(v for (x, y), v in data.items() if x == do_x)
    Q = Query.interventional(outcome="Y", do={"X": do_x}, value=do_y)
    return Problem(
        name=f"card_X{card_x}_Y{card_y}",
        source="Higher-cardinality confounded exposure",
        variables=V,
        data=data,
        query=Q,
        assumptions=Assumptions.order("X", "Y"),
        expected_width=1.0 - px,
    )


def binary_iv(*, seed: int = 2) -> Problem:
    """Classic binary IV: Z→X→Y with X↔Y confounding (Balke–Pearl)."""
    V = Variables.from_binary("Z", "X", "Y")
    g = CausalDiagram.from_edges(
        nodes=["Z", "X", "Y"],
        directed=[("Z", "X"), ("X", "Y")],
        bidirected=[("X", "Y")],
        maximal_confounding=False,
    )
    data = _iv_data_from_scm(V, g, ("Z",), ("X", "Y"), seed=seed)
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    return Problem(
        name="binary_iv",
        source="Balke–Pearl binary IV",
        variables=V,
        data=data,
        query=Q,
        assumptions=Assumptions.from_diagram(g),
        notes="tight instrument LP on R-side types",
    )


def two_instruments(*, seed: int = 3) -> Problem:
    """Two binary instruments Z1,Z2 → X → Y, X↔Y confounded."""
    V = Variables.from_binary("Z1", "Z2", "X", "Y")
    g = CausalDiagram.from_edges(
        nodes=["Z1", "Z2", "X", "Y"],
        directed=[("Z1", "X"), ("Z2", "X"), ("X", "Y")],
        bidirected=[("X", "Y")],
        maximal_confounding=False,
    )
    data = _iv_data_from_scm(V, g, ("Z1", "Z2"), ("X", "Y"), seed=seed)
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    return Problem(
        name="two_instruments",
        source="two-instrument IV",
        variables=V,
        data=data,
        query=Q,
        assumptions=Assumptions.from_diagram(g),
        notes="64 R-side response parameters for binary instruments",
    )


def two_instruments_card(
    card_z: int = 3,
    card_x: int = 2,
    card_y: int = 2,
    *,
    seed: int = 4,
) -> Problem:
    """Two instruments with raised cardinality on Z."""
    V = Variables.from_mapping(
        {
            "Z1": tuple(range(card_z)),
            "Z2": tuple(range(card_z)),
            "X": tuple(range(card_x)),
            "Y": tuple(range(card_y)),
        }
    )
    g = CausalDiagram.from_edges(
        nodes=["Z1", "Z2", "X", "Y"],
        directed=[("Z1", "X"), ("Z2", "X"), ("X", "Y")],
        bidirected=[("X", "Y")],
    )
    data = _iv_data_from_scm(V, g, ("Z1", "Z2"), ("X", "Y"), seed=seed)
    Q = Query.interventional(
        outcome="Y",
        do={"X": 1 if card_x > 1 else 0},
        value=1 if card_y > 1 else 0,
    )
    return Problem(
        name=f"two_iv_Z{card_z}",
        source="Higher-card two-instrument IV",
        variables=V,
        data=data,
        query=Q,
        assumptions=Assumptions.from_diagram(g),
    )


# ---------------------------------------------------------------------------
# Shridharan–Iyengar (JMLR 2023 / arXiv:2308.02709)
# ---------------------------------------------------------------------------


def shridharan_running_iv(*, seed: int = 5) -> Problem:
    """Shridharan Fig. 2: Z→X→Y, query P(Y(X=1)=1 | Z=1).

    We report the unconditional interventional P(Y=1|do(X=1)) as the main
    numeric target (same LP family); the conditional-on-Z query is a linear
    functional of the same type distribution.
    """
    return binary_iv(seed=seed)._replace_name(
        "shridharan_fig2_iv", "Shridharan–Iyengar 2023 Fig. 2"
    )


def shridharan_multi_treatment(
    n_treatments: int = 3,
    card: int = 2,
    *,
    seed: int = 6,
) -> Problem:
    """Multi-confounded treatments T1..Tk → Y (Shridharan multi-cause style).

    Context variables omitted (VA empty); one shared latent over all T and Y.
    """
    t_names = [f"T{i}" for i in range(n_treatments)]
    domains = {n: tuple(range(card)) for n in t_names}
    domains["Y"] = (0, 1)
    V = Variables.from_mapping(domains)
    order = t_names + ["Y"]
    keys = V.cells(order)
    data = _softmax_table(keys, seed=seed)
    directed = [(t, "Y") for t in t_names]
    # chain among treatments optional; add T_i → T_{i+1}
    for i in range(len(t_names) - 1):
        directed.append((t_names[i], t_names[i + 1]))
    bidirected = [(t_names[i], "Y") for i in range(len(t_names))]
    bidirected += [
        (t_names[i], t_names[j]) for i in range(len(t_names)) for j in range(i + 1, len(t_names))
    ]
    g = CausalDiagram.from_edges(
        nodes=order,
        directed=directed,
        bidirected=bidirected,
        maximal_confounding=True,  # single U_B block
    )
    do = {t_names[0]: 1 if card > 1 else 0}
    Q = Query.interventional(outcome="Y", do=do, value=1)
    return Problem(
        name=f"shridharan_multi_T{n_treatments}_c{card}",
        source="Shridharan–Iyengar multi-cause style",
        variables=V,
        data=data,
        query=Q,
        assumptions=Assumptions.from_diagram(g),
        notes="maximal confounding on treatments+Y",
    )


def Problem_replace_name(self: Problem, name: str, source: str) -> Problem:
    return Problem(
        name=name,
        source=source,
        variables=self.variables,
        data=self.data,
        query=self.query,
        assumptions=self.assumptions,
        notes=self.notes,
        expected_width=self.expected_width,
    )


# monkey-patch helper
Problem._replace_name = Problem_replace_name  # type: ignore[attr-defined]


def chain_order_only(n: int, card: int = 2, *, seed: int = 7) -> Problem:
    """Length-n chain under order-only (tests inert-variable pruning)."""
    names = ["X"] + [f"Z{i}" for i in range(n - 2)] + ["Y"] if n > 2 else ["X", "Y"]
    if n == 2:
        names = ["X", "Y"]
    V = Variables.from_mapping({nm: tuple(range(card)) for nm in names})
    keys = V.cells(names)
    data = _softmax_table(keys, seed=seed)
    Q = Query.interventional(
        outcome="Y", do={"X": 1 if card > 1 else 0}, value=1 if card > 1 else 0
    )
    # expected Manski width on (X,Y) margin
    # P(X=do_x)
    do_x = 1 if card > 1 else 0
    xi = names.index("X")
    px = sum(v for k, v in data.items() if k[xi] == do_x)
    return Problem(
        name=f"chain_n{n}_c{card}",
        source="order-only chain (inert Z pruning)",
        variables=V,
        data=data,
        query=Q,
        assumptions=Assumptions.order(*names),
        expected_width=1.0 - px,
    )


def all_literature_problems() -> list[Problem]:
    """Default catalog for benchmarks (keeps mid-size instances)."""
    return [
        ternary_exposure(),
        ternary_cardinalities(4, 2),
        ternary_cardinalities(3, 3),
        binary_iv(),
        two_instruments(),
        shridharan_multi_treatment(2, 2),  # T0,T1,Y — manageable type count
        chain_order_only(3, 2),
        chain_order_only(4, 2),
        chain_order_only(3, 3),
    ]
