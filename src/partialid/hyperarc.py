"""Shridharan–Iyengar hyperarc LP (JMLR 2023 Algorithm 1).

Aggregates response-function variables into valid hyperarcs VA → VB for
VA/VB-partition graphs, without enumerating the full type space R.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Mapping, Sequence

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from .backends import SolveResult
from .formulate import _normalize_P
from .graph import CausalDiagram
from .query import Query
from .variables import Variables


@dataclass
class HyperarcResult:
    lower: float
    upper: float
    q_lower: np.ndarray | None
    q_upper: np.ndarray | None
    success: bool
    message: str
    n_hyperarcs: int
    n_raw_types_est: int
    closed_form: bool
    notes: list[str]


def _estimate_raw_types(variables: Variables, diagram: CausalDiagram, vb: Sequence[str]) -> int:
    n = 1
    for v in vb:
        parents = diagram.parents(v)
        n_in = 1
        for p in parents:
            n_in *= variables.size(p)
        n *= variables.size(v) ** max(n_in, 1) if parents else variables.size(v)
    return int(n)


def _bidirected_component(diagram: CausalDiagram, seed: str) -> set[str]:
    """Undirected connected component of ``seed`` in the bidirected graph."""
    adj: dict[str, set[str]] = {n: set() for n in diagram.nodes}
    for he in diagram.bidirected:
        for i, a in enumerate(he):
            for b in he[i + 1 :]:
                adj[a].add(b)
                adj[b].add(a)
    seen = {seed}
    stack = [seed]
    while stack:
        u = stack.pop()
        for v in adj[u]:
            if v not in seen:
                seen.add(v)
                stack.append(v)
    return seen


def _partition_va_vb(
    diagram: CausalDiagram,
    intervention: Mapping[str, object],
    outcome: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """VA/VB / Shridharan partition.

    VA = observed roots that do **not** share a latent confounder with the
    outcome (Assumption 1: no (A,B) pair shares a confounder; VB may include
    roots that only have latent parents, e.g. S1 in Example A).
    VB = remaining nodes (treatments / mediators / outcomes).
    Intervened nodes lie in VB (Assumption: I ⊆ B).
    """
    order = diagram.topological_order()
    roots = {v for v in order if not diagram.parents(v)}
    out_comp = _bidirected_component(diagram, outcome)
    va_list = [
        v
        for v in order
        if v in roots and v not in intervention and v != outcome and v not in out_comp
    ]
    # Keep only VA nodes that parent something outside VA (Assumption 1.2)
    va_set = set(va_list)
    va_list = [v for v in va_list if any(c not in va_set for c in _children(diagram, v))]
    va = tuple(va_list)
    vb = tuple(v for v in order if v not in va)
    return va, vb


def _children(diagram: CausalDiagram, v: str) -> list[str]:
    return [c for p, c in diagram.directed if p == v]


def _critical_vars(
    diagram: CausalDiagram,
    intervention: Mapping[str, object],
    outcome: str,
) -> set[str]:
    """Variables with a path to outcome in G_do(VI)."""
    # remove incoming to intervened nodes
    children: dict[str, list[str]] = {n: [] for n in diagram.nodes}
    for p, c in diagram.directed:
        if c in intervention:
            continue
        children[p].append(c)
    # BFS ancestors of outcome in reverse = nodes that reach outcome
    rev: dict[str, list[str]] = {n: [] for n in diagram.nodes}
    for p, cs in children.items():
        for c in cs:
            rev[c].append(p)
    reach = {outcome}
    stack = [outcome]
    while stack:
        u = stack.pop()
        for p in rev[u]:
            if p not in reach:
                reach.add(p)
                stack.append(p)
    reach |= set(intervention)
    return reach


def _is_valid_hyperarc(
    diagram: CausalDiagram,
    va: Sequence[str],
    vb: Sequence[str],
    variables: Variables,
    h: Mapping[tuple, tuple],
) -> bool:
    """Theorem 5: same parent config ⇒ same child value across VA inputs."""
    # For each v in VB, collect implied (parent_vals → child_val) and check consistency
    implied: dict[str, dict[tuple, object]] = {v: {} for v in vb}
    for a_key, b_vals in h.items():
        world = {**dict(zip(va, a_key)), **dict(zip(vb, b_vals))}
        for v in vb:
            parents = diagram.parents(v)
            pkey = tuple(world[p] for p in parents) if parents else ()
            val = world[v]
            if pkey in implied[v] and implied[v][pkey] != val:
                return False
            implied[v][pkey] = val
    return True


def _cL_cU(
    h: Mapping[tuple, tuple],
    va: Sequence[str],
    vb: Sequence[str],
    intervention: Mapping[str, object],
    outcome: str,
    out_val: object,
    qA: Mapping[str, object] | None,
    critical: set[str],
) -> tuple[int, int]:
    """Theorems 6–7: coefficients from hyperarc outputs alone."""
    cL = 0
    cU = 1
    for a_key, b_vals in h.items():
        world = {**dict(zip(va, a_key)), **dict(zip(vb, b_vals))}
        # check context match on critical VA
        if qA is not None:
            ok_ctx = all(world.get(k) == qA[k] for k in qA if k in critical)
        else:
            ok_ctx = True
        if not ok_ctx:
            continue
        if any(world.get(k) != intervention[k] for k in intervention):
            # Case: does not realize intervention on this row — skip for both tests
            # Theorem 6 needs some row with vI=qI, vO=qO
            continue
        if world[outcome] == out_val:
            cL = 1
        else:
            cU = 0
    return cL, cU


def _closed_form_thm8(
    variables: Variables,
    data: Mapping,
    diagram: CausalDiagram,
    query: Query,
    va: Sequence[str],
    vb: Sequence[str],
    critical: set[str],
    notes: list[str],
) -> HyperarcResult | None:
    """Theorem 8 when A ⊆ C(Q): bounds are sums of conditional probs."""
    if not set(va).issubset(critical):
        return None
    if query.kind != "interventional":
        return None

    order = list(va) + list(vb)
    cells, P = _normalize_P(variables, order, data)
    # Unconditional: if VA empty, use joint. If VA nonempty without qA, cannot use Thm8
    # unless we interpret query as unconditional — then A ⊆ C(Q) with nonempty A
    # needs conditioning; skip.
    meta = query.meta or {}
    qA = meta.get("condition")  # optional Mapping

    if va and qA is None:
        return None

    intervention = dict(query.intervention)
    out = query.outcome
    out_val = query.value

    if not va:
        # αL = Σ_{vB: vI=qI, vO=qO} P(vB)
        # αU = 1 - Σ_{vB: vI=qI, vO≠qO} P(vB)
        lo = 0.0
        hi_comp = 0.0
        for cell, p in zip(cells, P):
            world = dict(zip(order, cell))
            if any(world.get(k) != intervention[k] for k in intervention):
                continue
            if world[out] == out_val:
                lo += float(p)
            else:
                hi_comp += float(p)
        hi = 1.0 - hi_comp
        notes.append("Shridharan Thm 8 closed form (A=∅)")
        return HyperarcResult(
            lower=lo,
            upper=hi,
            q_lower=None,
            q_upper=None,
            success=True,
            message="closed form",
            n_hyperarcs=0,
            n_raw_types_est=_estimate_raw_types(variables, diagram, vb),
            closed_form=True,
            notes=notes,
        )

    # Conditional on VA=qA
    assert isinstance(qA, Mapping)
    # renormalize P(VB | VA=qA)
    num_lo = 0.0
    num_hi_comp = 0.0
    den = 0.0
    for cell, p in zip(cells, P):
        world = dict(zip(order, cell))
        if any(world[k] != qA[k] for k in va):
            continue
        den += float(p)
        if any(world.get(k) != intervention[k] for k in intervention):
            continue
        if world[out] == out_val:
            num_lo += float(p)
        else:
            num_hi_comp += float(p)
    if den <= 0:
        return HyperarcResult(
            lower=float("nan"),
            upper=float("nan"),
            q_lower=None,
            q_upper=None,
            success=False,
            message="P(VA=qA)=0",
            n_hyperarcs=0,
            n_raw_types_est=0,
            closed_form=True,
            notes=notes,
        )
    lo = num_lo / den
    hi = 1.0 - num_hi_comp / den
    notes.append("Shridharan Thm 8 closed form (A ⊆ C(Q))")
    return HyperarcResult(
        lower=lo,
        upper=hi,
        q_lower=None,
        q_upper=None,
        success=True,
        message="closed form",
        n_hyperarcs=0,
        n_raw_types_est=_estimate_raw_types(variables, diagram, vb),
        closed_form=True,
        notes=notes,
    )


def enumerate_valid_hyperarcs(
    variables: Variables,
    diagram: CausalDiagram,
    va: Sequence[str],
    vb: Sequence[str],
) -> list[dict[tuple, tuple]]:
    """Enumerate valid h : VA → VB (Theorem 5)."""
    a_cells = variables.cells(va) if va else [()]
    b_cells = variables.cells(vb)
    # Each hyperarc assigns every a_cell → some b_cell
    # |B|^|A| can be large; for binary small VA/VB this is fine
    valid: list[dict[tuple, tuple]] = []
    for assignment in product(b_cells, repeat=len(a_cells)):
        h = {a: b for a, b in zip(a_cells, assignment)}
        if _is_valid_hyperarc(diagram, va, vb, variables, h):
            valid.append(h)
    return valid


def solve_hyperarc(
    variables: Variables,
    data: Mapping,
    query: Query,
    diagram: CausalDiagram,
) -> HyperarcResult:
    """Build and solve the Shridharan pruned LP (or closed form)."""
    notes: list[str] = []
    if query.kind != "interventional":
        return HyperarcResult(
            lower=float("nan"),
            upper=float("nan"),
            q_lower=None,
            q_upper=None,
            success=False,
            message="hyperarc currently supports interventional queries",
            n_hyperarcs=0,
            n_raw_types_est=0,
            closed_form=False,
            notes=notes,
        )

    va, vb = _partition_va_vb(diagram, query.intervention, query.outcome)
    notes.append(f"partition VA={list(va)} VB={list(vb)}")
    critical = _critical_vars(diagram, query.intervention, query.outcome)
    notes.append(f"critical={sorted(critical)}")

    cf = _closed_form_thm8(variables, data, diagram, query, va, vb, critical, notes)
    if cf is not None:
        return cf

    # Full hyperarc LP
    order = list(va) + list(vb)
    cells, P = _normalize_P(variables, order, data)
    # Conditional probabilities P(VB|VA)
    if va:
        # group by VA
        from collections import defaultdict

        joint: dict[tuple, dict[tuple, float]] = defaultdict(lambda: defaultdict(float))
        for cell, p in zip(cells, P):
            a = tuple(cell[i] for i in range(len(va)))
            b = tuple(cell[len(va) :])
            joint[a][b] += float(p)
        pcond: dict[tuple, dict[tuple, float]] = {}
        for a, dist in joint.items():
            s = sum(dist.values())
            pcond[a] = {b: v / s for b, v in dist.items()} if s > 0 else {}
    else:
        pcond = {(): {tuple(cell): float(p) for cell, p in zip(cells, P)}}

    H = enumerate_valid_hyperarcs(variables, diagram, va, vb)
    raw_est = _estimate_raw_types(variables, diagram, vb)
    notes.append(f"|H|={len(H)} raw_types≈{raw_est}")

    meta = query.meta or {}
    qA = meta.get("condition")
    qA_map = dict(qA) if isinstance(qA, Mapping) else None

    cL = np.zeros(len(H))
    cU = np.zeros(len(H))
    for i, h in enumerate(H):
        cl, cu = _cL_cU(
            h,
            va,
            vb,
            dict(query.intervention),
            query.outcome,
            query.value,
            qA_map,
            critical,
        )
        cL[i] = cl
        cU[i] = cu

    # Constraints: for every (vA, vB), Σ_{h: h(vA)=vB} q_h = P(vB | vA)
    # (Shridharan Eq. 5). When qA is set, only that slice is required for
    # identifying the conditional query; we still impose all slices so q is a
    # valid global hyperarc distribution (needed for unconditional queries).
    a_cells = variables.cells(va) if va else [()]
    b_cells = variables.cells(vb)
    eq_rows = []
    eq_rhs = []
    for a in a_cells:
        dist = pcond.get(a, {})
        for b in b_cells:
            row = np.zeros(len(H))
            for j, h in enumerate(H):
                if h.get(a) == b:
                    row[j] = 1.0
            eq_rows.append(row)
            eq_rhs.append(float(dist.get(b, 0.0)))

    A = np.vstack(eq_rows) if eq_rows else np.zeros((0, len(H)))
    n = len(H)
    m = len(eq_rhs)
    if n == 0 or m == 0:
        return HyperarcResult(
            lower=float("nan"),
            upper=float("nan"),
            q_lower=None,
            q_upper=None,
            success=False,
            message="empty hyperarc LP",
            n_hyperarcs=n,
            n_raw_types_est=raw_est,
            closed_form=False,
            notes=notes,
        )

    def _solve(c_obj: np.ndarray, maximize: bool) -> SolveResult:
        c = -c_obj if maximize else c_obj
        res = linprog(
            c,
            A_eq=sparse.csr_matrix(A),
            b_eq=np.asarray(eq_rhs),
            bounds=[(0, None)] * n,
            method="highs",
        )
        if not res.success:
            return SolveResult(
                value=float("nan"),
                q=np.zeros(n),
                success=False,
                message=res.message,
            )
        val = float(-res.fun if maximize else res.fun)
        return SolveResult(value=val, q=res.x, success=True, message=res.message)

    lo = _solve(cL, False)
    hi = _solve(cU, True)
    success = lo.success and hi.success
    notes.append("Shridharan hyperarc LP")
    return HyperarcResult(
        lower=lo.value if lo.success else float("nan"),
        upper=hi.value if hi.success else float("nan"),
        q_lower=lo.q if lo.success else None,
        q_upper=hi.q if hi.success else None,
        success=success,
        message=f"lo:{lo.message}; hi:{hi.message}",
        n_hyperarcs=n,
        n_raw_types_est=raw_est,
        closed_form=False,
        notes=notes,
    )
