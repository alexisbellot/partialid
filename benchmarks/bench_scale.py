"""Scaling: matrix-free / lazy colgen + c_canonical wall-clock."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from partialid import Assumptions, CausalDiagram, Query, Variables, bound
from partialid.lazy_types import LazyTypeSpace
from partialid.column_gen import solve_bounds_colgen_lazy
from partialid.formulate import _normalize_P

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

RE1_P = {(0, 0): 0.30, (0, 1): 0.20, (1, 0): 0.15, (1, 1): 0.35}


def _med(fn, repeats: int = 5) -> float:
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(sorted(ts)[len(ts) // 2])


def _sparse_chain(n: int, seed: int = 0):
    """Binary chain X0→…→X{n-1} (sparse parents; exponential only in length via |cells|)."""
    names = [f"X{i}" for i in range(n)]
    V = Variables.from_binary(*names)
    directed = [(names[i], names[i + 1]) for i in range(n - 1)]
    g = CausalDiagram.from_edges(names, directed, maximal_confounding=True)
    a = Assumptions.from_diagram(g)
    rng = np.random.default_rng(seed)
    cells = V.cells(names)
    p = rng.dirichlet(np.ones(len(cells)))
    P = {c: float(v) for c, v in zip(cells, p)}
    Q = Query.interventional(outcome=names[-1], do={names[0]: 1}, value=1)
    return V, P, Q, a


def main():
    rows = []

    V = Variables.from_binary("X", "Y")
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    a = Assumptions.order("X", "Y") & Assumptions.tan_msm("X", 3.0)
    for m in ("lp", "colgen"):
        t = _med(lambda mm=m: bound(V, RE1_P, Q, a, method=mm))
        r = bound(V, RE1_P, Q, a, method=m)
        rows.append(
            {
                "name": "re1_msm",
                "method": m,
                "ms": t * 1000,
                "interval": [r.lower, r.upper],
                "notes": r.notes[-3:],
            }
        )
        print(f"re1_msm {m}: {t*1000:.2f}ms [{r.lower:.4f},{r.upper:.4f}]")

    a0 = Assumptions.order("X", "Y")
    for m in ("auto", "c_canonical", "colgen"):
        t = _med(lambda mm=m: bound(V, RE1_P, Q, a0, method=mm))
        r = bound(V, RE1_P, Q, a0, method=m)
        rows.append(
            {
                "name": "re1_manski",
                "method": m,
                "ms": t * 1000,
                "interval": [r.lower, r.upper],
                "method_tag": r.method,
            }
        )
        print(f"re1_manski {m}: {t*1000:.2f}ms method={r.method}")

    # Sparse chains: lazy colgen vs fiber / c_canonical
    for n in (3, 5):  # n=5 ≈512 types; larger chains are slow with naive fiber walk
        Vn, Pn, Qn, an = _sparse_chain(n)
        diagram = an.diagram()
        assert diagram is not None
        lazy = LazyTypeSpace.from_diagram(Vn, diagram)
        cells, Pvec = _normalize_P(Vn, lazy.order, Pn)

        t_fiber = _med(lambda: bound(Vn, Pn, Qn, an))
        t_cc = _med(lambda: bound(Vn, Pn, Qn, an, method="c_canonical"))
        # Force lazy API (bound() only auto-routes lazy when N>4096)
        t_lazy = _med(lambda: solve_bounds_colgen_lazy(lazy, cells, Pvec, Qn))
        rf = bound(Vn, Pn, Qn, an)
        rc = bound(Vn, Pn, Qn, an, method="c_canonical")
        rl = solve_bounds_colgen_lazy(lazy, cells, Pvec, Qn)
        row = {
            "name": f"sparse_chain_n{n}",
            "n_types_est": lazy.n_types_est,
            "fiber_ms": t_fiber * 1000,
            "c_canonical_ms": t_cc * 1000,
            "lazy_colgen_ms": t_lazy * 1000,
            "fiber": [rf.lower, rf.upper],
            "c_canonical": [rc.lower, rc.upper],
            "lazy": [rl.lower, rl.upper],
            "lazy_ok": bool(
                rl.success
                and abs(rl.lower - rf.lower) < 1e-5
                and abs(rl.upper - rf.upper) < 1e-5
            ),
        }
        rows.append(row)
        print(
            f"chain n={n} N≈{lazy.n_types_est}: fiber={row['fiber_ms']:.1f}ms "
            f"c_can={row['c_canonical_ms']:.1f}ms lazy={row['lazy_colgen_ms']:.1f}ms "
            f"ok={row['lazy_ok']}"
        )

    path = OUT / "scale_colgen.json"
    path.write_text(json.dumps(rows, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
