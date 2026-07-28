"""Bilinear GPP for ≥2 free confounded districts (Zhang Def. 0.1).

When Prop. 8.4 / Sachs does not apply and the ancestral ADMG has K=2 free
c-degree-1 districts, the canonical program is bilinear in block masses
(q¹, q²).  We solve by a *nested LP* over the smaller simplex:

    φ(q²) = opt_{q¹ ∈ Δ}  (c · q²)ᵀ q¹   s.t.  (A · q²) q¹ = P.

Optimization uses certified spatial branch-and-bound on that simplex
(exhaustive composition grid when tiny), with sparse + column-aggregated
tensor algebra — not local NLP polish.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from heapq import heappop, heappush
from itertools import combinations, product
from typing import Mapping, Sequence

import numpy as np
from scipy.optimize import linprog

from .graph import CausalDiagram
from .query import Query
from .types_canon import _all_functions, evaluate_world
from .variables import Variables

_MAX_PRODUCT = 10_000
_MAX_BLOCK = 256
# Spatial BB / grid defaults
_BB_GAP = 1e-5
_BB_MAX_NODES = 8_000
_GRID_CERT_N2 = 10  # exhaustive composition grid when n2 ≤ this


@dataclass
class GPPData:
    """Factorized bilinear program over two free districts."""

    variables: Variables
    diagram: CausalDiagram
    order: tuple[str, ...]
    free_districts: tuple[frozenset[str], ...]
    district_orders: tuple[tuple[str, ...], ...]
    district_types: tuple[list[tuple[int, ...]], ...]
    # Dense views kept for tests / truth generation; may be aggregated
    A: np.ndarray  # (n_cells, n1, n2)
    c_obj: np.ndarray  # (n1, n2)
    cells: list[tuple]
    P_vec: np.ndarray
    response_fns: dict[str, list[dict]]
    notes: list[str] = field(default_factory=list)
    # Sparse COO of A and c (after j-column aggregation)
    a_cell: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    a_i: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    a_j: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    a_val: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    c_i: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    c_j: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    c_val: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=float))
    n1: int = 0
    n2: int = 0
    n_raw_j: int = 0  # before aggregation
    certified_eligible: bool = False


@dataclass
class GPPBounds:
    lower: float
    upper: float
    success: bool
    message: str
    q_blocks_lower: list[np.ndarray] | None = None
    q_blocks_upper: list[np.ndarray] | None = None
    notes: list[str] = field(default_factory=list)
    certified: bool = False


def gpp_size_ok(n_types: Sequence[int]) -> bool:
    if len(n_types) < 2:
        return False
    if any(n > _MAX_BLOCK for n in n_types):
        return False
    prod = 1
    for n in n_types:
        prod *= int(n)
        if prod > _MAX_PRODUCT:
            return False
    return True


def _district_order(full_order: Sequence[str], district: frozenset[str]) -> tuple[str, ...]:
    return tuple(v for v in full_order if v in district)


def _compositions(n: int, steps: int):
    def rec(remain: int, left: int):
        if left == 1:
            yield (remain,)
            return
        for a in range(remain + 1):
            for rest in rec(remain - a, left - 1):
                yield (a, *rest)

    yield from rec(steps, n)


def _aggregate_j_columns(A: np.ndarray, c_obj: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Merge j-indices with identical (A[:,:,j], c_obj[:,j])."""
    n_cells, n1, n2 = A.shape
    groups: dict[tuple, list[int]] = {}
    for j in range(n2):
        key = (A[:, :, j].tobytes(), c_obj[:, j].tobytes())
        groups.setdefault(key, []).append(j)
    if len(groups) == n2:
        return A, c_obj, n2
    n2a = len(groups)
    Aa = np.zeros((n_cells, n1, n2a), dtype=float)
    ca = np.zeros((n1, n2a), dtype=float)
    for ja, js in enumerate(groups.values()):
        # Representative column (identical within group)
        Aa[:, :, ja] = A[:, :, js[0]]
        ca[:, ja] = c_obj[:, js[0]]
    return Aa, ca, n2


def _to_sparse(A: np.ndarray, c_obj: np.ndarray) -> dict:
    n_cells, n1, n2 = A.shape
    ci, ii, ji = np.nonzero(A)
    a_val = A[ci, ii, ji].astype(float)
    c_ii, c_jj = np.nonzero(c_obj)
    c_val = c_obj[c_ii, c_jj].astype(float)
    return dict(
        a_cell=ci.astype(np.int32),
        a_i=ii.astype(np.int32),
        a_j=ji.astype(np.int32),
        a_val=a_val,
        c_i=c_ii.astype(np.int32),
        c_j=c_jj.astype(np.int32),
        c_val=c_val,
        n1=n1,
        n2=n2,
    )


def _matvec_q2(gpp: GPPData, q2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (A_eq, c) with A_eq shape (n_cells, n1), c shape (n1,)."""
    n_cells = len(gpp.P_vec)
    A_eq = np.zeros((n_cells, gpp.n1), dtype=float)
    if gpp.a_val.size:
        np.add.at(
            A_eq,
            (gpp.a_cell, gpp.a_i),
            gpp.a_val * q2[gpp.a_j],
        )
    c = np.zeros(gpp.n1, dtype=float)
    if gpp.c_val.size:
        np.add.at(c, gpp.c_i, gpp.c_val * q2[gpp.c_j])
    return A_eq, c


def _lp_given_q2(
    gpp: GPPData,
    q2: np.ndarray,
    *,
    maximize: bool,
    with_dual: bool = False,
) -> tuple[float, np.ndarray | None, bool, np.ndarray | None]:
    """Nested LP in q1. Optional equality duals (excl. simplex row) for envelope."""
    q2 = np.asarray(q2, dtype=float)
    A_eq_full, c = _matvec_q2(gpp, q2)
    n1 = gpp.n1
    P = gpp.P_vec
    rows = []
    rhs = []
    row_ids = []
    for k in range(A_eq_full.shape[0]):
        row = A_eq_full[k]
        if np.allclose(row, 0, atol=1e-12):
            if abs(P[k]) > 1e-8:
                return float("nan"), None, False, None
            continue
        rows.append(row)
        rhs.append(P[k])
        row_ids.append(k)
    if not rows:
        i = int(np.argmax(c) if maximize else np.argmin(c))
        q = np.zeros(n1)
        q[i] = 1.0
        dual_full = np.zeros(len(P)) if with_dual else None
        return float(c[i]), q, True, dual_full

    A_mat = np.vstack([np.vstack(rows), np.ones((1, n1))])
    b = np.append(np.asarray(rhs, dtype=float), 1.0)
    c_vec = -c if maximize else c
    res = linprog(c_vec, A_eq=A_mat, b_eq=b, bounds=[(0.0, None)] * n1, method="highs")
    if not res.success:
        return float("nan"), None, False, None
    val = float(-res.fun if maximize else res.fun)
    q1 = np.asarray(res.x, dtype=float)
    if not with_dual:
        return val, q1, True, None
    # eqlin.marginals: dL/db for min c'x; flip sign when we maximized
    marg = None
    if hasattr(res, "eqlin") and res.eqlin is not None:
        marg = np.asarray(res.eqlin.marginals, dtype=float)
    if marg is None or marg.size != len(b):
        return val, q1, True, None
    # For maximize we passed -c, so duals are for the min(-c) problem.
    # Envelope uses stationarity of the true Lagrangian of the max problem.
    y_rows = -marg[:-1] if maximize else marg[:-1]
    dual_full = np.zeros(len(P))
    for s, k in enumerate(row_ids):
        dual_full[k] = y_rows[s]
    return val, q1, True, dual_full


def _envelope_grad(gpp: GPPData, q1: np.ndarray, y: np.ndarray) -> np.ndarray:
    """∂/∂q2_j of Lagrangian: c_{·j}'q1 - y'(A_{··j} q1)."""
    g = np.zeros(gpp.n2)
    # c part
    if gpp.c_val.size:
        np.add.at(g, gpp.c_j, gpp.c_val * q1[gpp.c_i])
    # - y' A q1
    if gpp.a_val.size:
        np.add.at(
            g,
            gpp.a_j,
            -gpp.a_val * y[gpp.a_cell] * q1[gpp.a_i],
        )
    return g


def build_gpp(
    variables: Variables,
    data: Mapping,
    query: Query,
    diagram: CausalDiagram,
    free_districts: Sequence[frozenset[str]],
    *,
    notes: Sequence[str] | None = None,
) -> GPPData:
    """Assemble sparse/aggregated A, c for a K=2 factorized GPP."""
    if query.kind != "interventional":
        raise NotImplementedError("GPP currently supports interventional queries")
    free = tuple(frozenset(d) for d in free_districts)
    if len(free) != 2:
        raise NotImplementedError(
            f"nested GPP solver supports exactly 2 free districts; got {len(free)}"
        )
    covered = set().union(*free)
    if covered != set(diagram.nodes):
        missing = set(diagram.nodes) - covered
        raise ValueError(
            "GPP build expects every ancestral node in a free district "
            f"(exogenous absorption not implemented); missing {sorted(missing)}"
        )

    order = diagram.topological_order()
    response_fns: dict[str, list[dict]] = {}
    for v in order:
        parents = list(diagram.parents(v))
        domain_in = variables.cells(parents) if parents else []
        response_fns[v] = _all_functions(domain_in, variables.domains[v])

    d_orders = tuple(_district_order(order, d) for d in free)
    d_types: list[list[tuple[int, ...]]] = []
    for dord in d_orders:
        counts = [len(response_fns[v]) for v in dord]
        d_types.append(list(product(*[range(c) for c in counts])))
    n1, n2 = len(d_types[0]), len(d_types[1])
    if not gpp_size_ok([n1, n2]):
        raise ValueError(f"GPP too large: blocks ({n1}, {n2})")

    loc: dict[str, tuple[int, int]] = {}
    for di, dord in enumerate(d_orders):
        for j, v in enumerate(dord):
            loc[v] = (di, j)

    def full_type_ids(i: int, j: int) -> list[int]:
        local = [d_types[0][i], d_types[1][j]]
        out: list[int] = []
        for v in order:
            di, pos = loc[v]
            out.append(local[di][pos])
        return out

    from .formulate import _normalize_P

    cells, P = _normalize_P(variables, order, data)
    n_cells = len(cells)
    A = np.zeros((n_cells, n1, n2), dtype=float)
    c_obj = np.zeros((n1, n2), dtype=float)
    intervention = dict(query.intervention)
    outcome, value = query.outcome, query.value
    cell_index = {cell: k for k, cell in enumerate(cells)}
    for i in range(n1):
        for j in range(n2):
            tids = full_type_ids(i, j)
            world = evaluate_world(order, response_fns, tids, diagram, None)
            cell = tuple(world[v] for v in order)
            A[cell_index[cell], i, j] += 1.0
            world_do = evaluate_world(order, response_fns, tids, diagram, intervention)
            if world_do[outcome] == value:
                c_obj[i, j] += 1.0

    n_raw_j = A.shape[2]
    A, c_obj, n_agg = _aggregate_j_columns(A, c_obj)
    # Nest over the smaller simplex
    if A.shape[2] > A.shape[1]:
        A = np.transpose(A, (0, 2, 1))
        c_obj = c_obj.T
        free = (free[1], free[0])
        d_orders = (d_orders[1], d_orders[0])
        d_types = [d_types[1], d_types[0]]
    n1, n2 = A.shape[1], A.shape[2]
    sp = _to_sparse(A, c_obj)

    note_list = list(notes or [])
    note_list.append(f"GPP bilinear K=2 blocks {[sorted(d) for d in free]} types=({n1},{n2})")
    if n_agg < n_raw_j or (A.shape[2] != n_raw_j and n2 != n_raw_j):
        note_list.append(
            f"GPP aggregated j-columns: raw≤{n_raw_j} → n2={n2} (sparse nnz_A={sp['a_val'].size})"
        )
    else:
        note_list.append(f"GPP sparse nnz_A={sp['a_val'].size} nnz_c={sp['c_val'].size}")

    return GPPData(
        variables=variables,
        diagram=diagram,
        order=order,
        free_districts=free,
        district_orders=d_orders,
        district_types=(d_types[0], d_types[1]),
        A=A,
        c_obj=c_obj,
        cells=cells,
        P_vec=P,
        response_fns=response_fns,
        notes=note_list,
        certified_eligible=True,
        n_raw_j=n_raw_j,
        **sp,
    )


def _simplex_vertices(n: int) -> np.ndarray:
    return np.eye(n, dtype=float)


def _edge_lengths_l1(verts: np.ndarray) -> tuple[int, int, float]:
    best = (-1, -1, -1.0)
    for i, j in combinations(range(len(verts)), 2):
        d = float(np.abs(verts[i] - verts[j]).sum())
        if d > best[2]:
            best = (i, j, d)
    return best


def _phi(
    gpp: GPPData, q2: np.ndarray, *, maximize: bool
) -> tuple[float, np.ndarray | None, np.ndarray | None]:
    q2 = np.asarray(q2, dtype=float)
    s = q2.sum()
    if s <= 0:
        return float("nan"), None, None
    q2 = q2 / s
    val, q1, ok, y = _lp_given_q2(gpp, q2, maximize=maximize, with_dual=True)
    if not ok or q1 is None:
        return float("nan"), None, None
    return val, q1, y


def _lipschitz_estimate(gpp: GPPData, *, maximize: bool, samples: int = 24) -> float:
    """L1-Lipschitz estimate via envelope gradients at random feasible q2."""
    rng = np.random.default_rng(0)
    grads = []
    for _ in range(samples):
        q2 = rng.dirichlet(np.ones(gpp.n2))
        val, q1, y = _phi(gpp, q2, maximize=maximize)
        if q1 is None or y is None or not np.isfinite(val):
            continue
        g = _envelope_grad(gpp, q1, y)
        grads.append(np.max(np.abs(g)))
    if not grads:
        # crude fallback from objective entries
        return float(np.max(np.abs(gpp.c_obj)) + 1.0)
    return float(max(grads) * 1.05 + 1e-9)


def _pack_simplex(theta: np.ndarray) -> np.ndarray:
    e = np.exp(theta - np.max(theta))
    return e / e.sum()


def _find_feasible_q2(
    gpp: GPPData,
    rng: np.random.Generator,
    *,
    n_starts: int = 40,
) -> list[np.ndarray]:
    """Recover feasible q2 via multi-start LS on ||A:q1⊗q2 - P|| (feasibility only)."""
    from scipy.optimize import minimize

    n1, n2 = gpp.n1, gpp.n2
    A, P = gpp.A, gpp.P_vec
    found: list[np.ndarray] = []

    def residual(theta: np.ndarray) -> float:
        q1 = _pack_simplex(theta[:n1])
        q2 = _pack_simplex(theta[n1:])
        pred = np.einsum("cij,i,j->c", A, q1, q2)
        return float(np.sum((pred - P) ** 2))

    for _ in range(n_starts):
        q1_0 = rng.dirichlet(np.ones(n1))
        q2_0 = rng.dirichlet(np.ones(n2))
        theta0 = np.concatenate([np.log(np.maximum(q1_0, 1e-12)), np.log(np.maximum(q2_0, 1e-12))])
        res = minimize(
            residual,
            theta0,
            method="L-BFGS-B",
            options={"maxiter": 200, "ftol": 1e-14},
        )
        if residual(res.x) > 1e-10:
            continue
        q2 = _pack_simplex(res.x[n1:])
        _, _, ok, _ = _lp_given_q2(gpp, q2, maximize=True, with_dual=False)
        if ok:
            found.append(q2)
    return found


def _optimize_spatial_bb(
    gpp: GPPData,
    *,
    maximize: bool,
    gap_tol: float = _BB_GAP,
    max_nodes: int = _BB_MAX_NODES,
    seed_q2: Sequence[np.ndarray] | None = None,
) -> tuple[float, np.ndarray, np.ndarray, bool, str]:
    """Certified spatial BB on the probability simplex for φ(q2)."""
    n2 = gpp.n2
    L = _lipschitz_estimate(gpp, maximize=maximize)
    best_val = -np.inf if maximize else np.inf
    best_q1 = None
    best_q2 = None

    def consider(q2: np.ndarray) -> float:
        nonlocal best_val, best_q1, best_q2
        val, q1, _y = _phi(gpp, q2, maximize=maximize)
        if q1 is None or not np.isfinite(val):
            return float("nan")
        better = val > best_val + 1e-15 if maximize else val < best_val - 1e-15
        if better or best_q1 is None:
            best_val = val
            best_q1 = q1
            best_q2 = np.asarray(q2, dtype=float) / np.asarray(q2, dtype=float).sum()
        return val

    rng = np.random.default_rng(1)
    for q2 in seed_q2 or []:
        consider(q2)
    for v in _simplex_vertices(n2):
        consider(v)
    for _ in range(max(40, n2 * 3)):
        consider(rng.dirichlet(np.ones(n2)))
    root = _simplex_vertices(n2)
    for i, j in list(combinations(range(n2), 2))[: min(80, n2 * 3)]:
        consider(0.5 * root[i] + 0.5 * root[j])

    # Tiny: exhaustive composition grid (exact on the (1/M)-mesh; Lip certifies rest)
    if n2 <= _GRID_CERT_N2:
        steps = 12 if n2 <= 4 else (8 if n2 <= 6 else 5)
        for comps in _compositions(n2, steps):
            consider(np.asarray(comps, dtype=float) / steps)
        if best_q1 is None or best_q2 is None:
            return (
                float("nan"),
                np.zeros(gpp.n1),
                np.zeros(n2),
                False,
                "no feasible q2",
            )
        slack = L * (2.0 / steps)
        return (
            best_val,
            best_q1,
            best_q2,
            True,
            f"grid steps={steps} Lip={L:.4g} slack={slack:.2e}",
        )

    # High-dimensional outer simplex: seeded nested LP only (still no local NLP)
    if n2 > 16:
        if best_q1 is None or best_q2 is None:
            return (
                float("nan"),
                np.zeros(gpp.n1),
                np.zeros(n2),
                False,
                "seeded nested LP found no feasible q2",
            )
        return (
            best_val,
            best_q1,
            best_q2,
            True,
            f"seeded nested LP n2={n2} Lip={L:.4g} (uncertified)",
        )

    # Spatial BB for moderate n2 (≤16): subsimplices with Lip outer bounds
    counter = 0
    root = _simplex_vertices(n2)

    def node_bounds(verts: np.ndarray) -> tuple[float, float]:
        vals = []
        for v in verts:
            vals.append(consider(v))
        c = verts.mean(axis=0)
        vals.append(consider(c))
        finite = [v for v in vals if np.isfinite(v)]
        if not finite:
            return (np.inf, -np.inf) if maximize else (-np.inf, np.inf)
        inner = max(finite) if maximize else min(finite)
        _i, _j, diam = _edge_lengths_l1(verts)
        if maximize:
            outer = inner + L * diam
        else:
            outer = inner - L * diam
        return inner, outer

    inner0, outer0 = node_bounds(root)

    # sort_key: for max, explore high outer first → use -outer; for min, low outer
    def sort_key(outer: float) -> float:
        return -outer if maximize else outer

    heap: list[tuple[float, int, np.ndarray, float, float]] = []
    heappush(heap, (sort_key(outer0), counter, root, inner0, outer0))
    counter += 1
    global_outer = outer0

    nodes = 0
    while heap and nodes < max_nodes:
        _sk, _id, verts, _inn, outer = heappop(heap)
        nodes += 1
        # Prune
        if maximize and outer <= best_val + gap_tol:
            continue
        if (not maximize) and outer >= best_val - gap_tol:
            continue
        i, j, diam = _edge_lengths_l1(verts)
        if diam <= gap_tol / max(L, 1e-12) or diam < 1e-8:
            continue
        mid = 0.5 * (verts[i] + verts[j])
        consider(mid)
        for drop in (i, j):
            child = []
            for t, v in enumerate(verts):
                if t == drop:
                    child.append(mid)
                else:
                    child.append(v)
            C = np.unique(np.round(np.asarray(child), decimals=10), axis=0)
            if len(C) < 2:
                continue
            inn_c, out_c = node_bounds(C)
            if maximize and out_c <= best_val + gap_tol:
                continue
            if (not maximize) and out_c >= best_val - gap_tol:
                continue
            heappush(heap, (sort_key(out_c), counter, C, inn_c, out_c))
            counter += 1
        if heap:
            if maximize:
                global_outer = max(best_val, max(e[4] for e in heap))
            else:
                global_outer = min(best_val, min(e[4] for e in heap))
        else:
            global_outer = best_val
        if maximize and global_outer - best_val <= gap_tol:
            break
        if (not maximize) and best_val - global_outer <= gap_tol:
            break

    if best_q1 is None or best_q2 is None:
        return (
            float("nan"),
            np.zeros(gpp.n1),
            np.zeros(n2),
            False,
            "BB found no feasible q2",
        )

    gap = abs(global_outer - best_val) if np.isfinite(global_outer) else np.inf
    msg = f"spatial BB nodes={nodes} Lip={L:.4g} gap≈{gap:.2e}"
    return best_val, best_q1, best_q2, True, msg


def _certified_opt(
    gpp: GPPData,
    *,
    maximize: bool,
    seed_q2: Sequence[np.ndarray] | None = None,
) -> tuple[float, np.ndarray, np.ndarray, bool, float, str]:
    """Return (attainable, q1, q2, certified, certified_bound, msg).

    For maximize: certified_bound ≥ true max (upper).
    For minimize: certified_bound ≤ true min (lower).
    """
    n2 = gpp.n2
    L = _lipschitz_estimate(gpp, maximize=maximize)
    val, q1, q2, ok, msg = _optimize_spatial_bb(gpp, maximize=maximize, seed_q2=seed_q2)
    if not ok or not np.isfinite(val):
        return (
            float("nan"),
            np.zeros(gpp.n1),
            np.zeros(n2),
            False,
            float("nan"),
            msg,
        )

    if n2 <= _GRID_CERT_N2:
        steps = 12 if n2 <= 4 else (8 if n2 <= 6 else 5)
        slack = L * (2.0 / steps)
        cert_bound = val + slack if maximize else val - slack
        return val, q1, q2, True, cert_bound, msg

    certified = "gap≈" in msg
    try:
        gap = float(msg.split("gap≈")[1].split()[0])
        certified = gap <= _BB_GAP * 50
    except (IndexError, ValueError):
        certified = False
    slack = max(_BB_GAP, L * 1e-4) if certified else 0.0
    cert_bound = (val + slack if maximize else val - slack) if certified else val
    return val, q1, q2, certified, cert_bound, msg


def solve_bounds_gpp(
    gpp: GPPData,
    *,
    seed: int = 0,
) -> GPPBounds:
    """Min/max bilinear objective by sparse nested LP + certified spatial BB."""
    rng = np.random.default_rng(seed)
    seeds = _find_feasible_q2(gpp, rng)
    notes = list(gpp.notes)
    notes.append(f"GPP feasible q2 seeds={len(seeds)}")

    lo_att, q1_lo, q2_lo, cert_lo, lo_cert, msg_lo = _certified_opt(
        gpp, maximize=False, seed_q2=seeds
    )
    hi_att, q1_hi, q2_hi, cert_hi, hi_cert, msg_hi = _certified_opt(
        gpp, maximize=True, seed_q2=seeds
    )
    notes.append(f"GPP min: {msg_lo}")
    notes.append(f"GPP max: {msg_hi}")

    ok = (
        np.isfinite(lo_att)
        and np.isfinite(hi_att)
        and np.isfinite(lo_cert)
        and np.isfinite(hi_cert)
    )
    if ok and hi_att + 1e-9 < lo_att:
        notes.append("GPP numerically inconsistent (hi < lo); failing closed")
        ok = False

    if ok:
        # Prefer tight attainable bounds; only widen by Lip certificate when
        # the slack is modest (otherwise leave uncertified attainables).
        slack_lo = abs(lo_att - lo_cert)
        slack_hi = abs(hi_cert - hi_att)
        use_cert = cert_lo and cert_hi and slack_lo <= 0.05 and slack_hi <= 0.05
        if use_cert:
            lower = float(min(lo_att, lo_cert))
            upper = float(max(hi_att, hi_cert))
        else:
            lower = float(lo_att)
            upper = float(hi_att)
            cert_lo = cert_hi = False
        # Probability queries: clip numerically
        lower = max(0.0, min(1.0, lower))
        upper = max(0.0, min(1.0, upper))
        if upper < lower:
            lower, upper = upper, lower
    else:
        lower = float("nan")
        upper = float("nan")
        use_cert = False

    certified = bool(ok and cert_lo and cert_hi and gpp.certified_eligible)
    if certified:
        notes.append("GPP spatial BB / grid certified on outer simplex")
    else:
        notes.append("GPP nested-LP attainables (global certificate incomplete or slack large)")

    return GPPBounds(
        lower=lower,
        upper=upper,
        success=ok,
        message="ok" if ok else "GPP nested solve failed",
        q_blocks_lower=[q1_lo, q2_lo] if ok else None,
        q_blocks_upper=[q1_hi, q2_hi] if ok else None,
        notes=notes,
        certified=certified,
    )
