#!/usr/bin/env python3
"""Benchmark P1–P3 universe features (timing smoke).

  PYTHONPATH=src:benchmarks python benchmarks/bench_universe_p1_p3.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np

from partialid import (
    Assumptions,
    Query,
    Variables,
    audit_bound,
    bound,
    bound_ci,
    bound_path,
)
from partialid.problems import sachs_binary_iv


def _time(fn, repeats: int = 5) -> float:
    # warmup
    fn()
    ts = []
    for _ in range(repeats):
        t0 = perf_counter()
        fn()
        ts.append(perf_counter() - t0)
    return float(np.median(ts))


def main() -> int:
    V = Variables.from_binary("X", "Y")
    P = {(0, 0): 0.30, (0, 1): 0.20, (1, 0): 0.15, (1, 1): 0.35}
    a = Assumptions.order("X", "Y")
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)

    rows = []

    def add(name, fn):
        ms = 1000.0 * _time(fn)
        rows.append((name, ms))
        print(f"{name:40s} {ms:8.2f} ms")

    add("manski auto", lambda: bound(V, P, Q, a))
    add("mts lp", lambda: bound(V, P, Q, a & Assumptions.mts("X", "Y"), method="lp"))
    add("pns lp", lambda: bound(V, P, Query.pns(), a, method="lp"))
    add("pn lp", lambda: bound(V, P, Query.pn(), a, method="lp"))
    add(
        "path Lambda (closed)",
        lambda: bound_path(V, P, Q, a, param="Lambda", n_grid=17),
    )
    add(
        "path c (grid)",
        lambda: bound_path(V, P, Q, a, param="c", n_grid=17, prefer_closed=False),
    )
    add(
        "path kappa",
        lambda: bound_path(V, P, Q, a, param="kappa", n_grid=17),
    )

    p = sachs_binary_iv(seed=2)
    add(
        "IV+MSM lp",
        lambda: bound(
            p.variables,
            p.data,
            p.query,
            p.assumptions & Assumptions.tan_msm("X", 2.0),
            method="lp",
        ),
    )
    add(
        "IV+MSM path Λ",
        lambda: bound_path(
            p.variables,
            p.data,
            p.query,
            p.assumptions,
            param="Lambda",
            n_grid=9,
            prefer_closed=False,
        ),
    )
    add(
        "bound_ci n_boot=30",
        lambda: bound_ci(V, P, Q, a, n_boot=30, n=400, seed=0),
    )
    r = bound(V, P, Q, a & Assumptions.tan_msm("X", 2.0))
    add(
        "audit_bound",
        lambda: audit_bound(
            r, query=Q, assumptions=a & Assumptions.tan_msm("X", 2.0)
        ),
    )

    out = ROOT / "benchmarks" / "results" / "bench_universe_p1_p3.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write("# Universe P1–P3 timing smoke\n\n")
        f.write("| Task | median ms |\n|---|---:|\n")
        for name, ms in rows:
            f.write(f"| {name} | {ms:.2f} |\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
