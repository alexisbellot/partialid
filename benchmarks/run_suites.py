#!/usr/bin/env python3
"""Wall-clock benchmarks: fiber vs full LP vs column generation.

Suite S0: correctness timings on RE1 / MSM
Suite S1: scale binary chains (order-only + MTR)
Suite S2: MSM Lambda grid cold starts
"""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import numpy as np

from partialid import Assumptions, Query, Variables, bound, bound_grid

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)


def _median_time(fn, repeats: int = 5, warmup: int = 1) -> float:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def suite_s0():
    V = Variables.from_binary("X", "Y")
    P = {(0, 0): 0.30, (0, 1): 0.20, (1, 0): 0.15, (1, 1): 0.35}
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    a = Assumptions.order("X", "Y")
    rows = []
    for name, assumptions, method in [
        ("manski_fiber", a, "fiber"),
        ("manski_lp", a, "lp"),
        ("manski_colgen", a, "colgen"),
        ("msm3_lp", a & Assumptions.tan_msm("X", 3.0), "lp"),
        ("msm3_colgen", a & Assumptions.tan_msm("X", 3.0), "colgen"),
        ("mtr_fiber", a & Assumptions.mtr("Y", "X"), "fiber"),
    ]:
        med = _median_time(lambda: bound(V, P, Q, assumptions, method=method))
        r = bound(V, P, Q, assumptions, method=method)
        rows.append(
            {
                "suite": "S0",
                "name": name,
                "method": method,
                "median_s": med,
                "width": r.width,
                "n_types": 8,
            }
        )
    return rows


def _chain_data(n: int) -> tuple[Variables, dict, Query]:
    names = ["X"] + [f"Z{i}" for i in range(n - 2)] + ["Y"] if n > 2 else ["X", "Y"]
    if n == 2:
        names = ["X", "Y"]
    V = Variables.from_binary(*names)
    # Put mass only varying X,Y like RE1; intermediates uniform
    order = list(names)
    cells = V.cells(order)
    P = {c: 0.0 for c in cells}
    base = {(0, 0): 0.30, (0, 1): 0.20, (1, 0): 0.15, (1, 1): 0.35}
    mid = names[1:-1]
    if not mid:
        for (x, y), p in base.items():
            P[(x, y)] = p
    else:
        mid_cells = V.cells(mid)
        share = 1.0 / len(mid_cells)
        for (x, y), p in base.items():
            for mc in mid_cells:
                key = (x, *mc, y)
                P[key] = p * share
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    return V, P, Q


def suite_s1(max_n: int = 4):
    rows = []
    for n in range(2, max_n + 1):
        V, P, Q = _chain_data(n)
        a = Assumptions.order(*V.names)
        # count types via one formulate
        from partialid.formulate import formulate

        lp = formulate(V, P, Q, a)
        n_types = lp.type_space.n_types
        for method in ("fiber", "lp", "colgen"):
            try:
                med = _median_time(
                    lambda m=method: bound(V, P, Q, a, method=m), repeats=3
                )
                r = bound(V, P, Q, a, method=method)
                width = r.width
                ok = r.success
            except Exception as e:  # noqa: BLE001
                med, width, ok = float("nan"), float("nan"), False
                err = str(e)
            else:
                err = ""
            rows.append(
                {
                    "suite": "S1",
                    "n": n,
                    "n_types": n_types,
                    "method": method,
                    "median_s": med,
                    "width": width,
                    "success": ok,
                    "error": err,
                }
            )
        # MTR variant
        a_mtr = a & Assumptions.mtr("Y", "X")
        for method in ("fiber", "lp"):
            med = _median_time(lambda m=method: bound(V, P, Q, a_mtr, method=m), repeats=3)
            r = bound(V, P, Q, a_mtr, method=method)
            rows.append(
                {
                    "suite": "S1_mtr",
                    "n": n,
                    "n_types": n_types,
                    "method": method,
                    "median_s": med,
                    "width": r.width,
                    "success": r.success,
                    "error": "",
                }
            )
    return rows


def suite_s2():
    V = Variables.from_binary("X", "Y")
    P = {(0, 0): 0.30, (0, 1): 0.20, (1, 0): 0.15, (1, 1): 0.35}
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    a = Assumptions.order("X", "Y")
    values = list(np.linspace(1.0, 20.0, 50))

    from partialid.api import bound_grid_cold

    def cold():
        bound_grid_cold(V, P, Q, a, param="Lambda", values=values, method="lp")

    def warm():
        bound_grid(V, P, Q, a, param="Lambda", values=values, method="lp", warm_start=True)

    med_cold = _median_time(cold, repeats=3)
    med_warm = _median_time(warm, repeats=3)
    return [
        {
            "suite": "S2",
            "name": "msm_grid_50_cold_lp",
            "median_s": med_cold,
            "n_points": len(values),
        },
        {
            "suite": "S2",
            "name": "msm_grid_50_warm_lp",
            "median_s": med_warm,
            "n_points": len(values),
            "speedup_vs_cold": med_cold / med_warm if med_warm > 0 else None,
        },
    ]


def main():
    all_rows = []
    print("Running S0...")
    all_rows.extend(suite_s0())
    print("Running S1 (n<=3; n=4 explodes type count under full enum)...")
    all_rows.extend(suite_s1(max_n=3))
    print("Running S2...")
    all_rows.extend(suite_s2())

    out_path = OUT / "wallclock.json"
    out_path.write_text(json.dumps(all_rows, indent=2))
    print(f"Wrote {out_path}")
    for row in all_rows:
        if "median_s" in row:
            print(
                f"  {row.get('suite')} {row.get('name', row.get('method'))} "
                f"n={row.get('n', '-')} types={row.get('n_types', '-')}: "
                f"{row['median_s']*1000:.2f} ms"
            )


if __name__ == "__main__":
    main()
