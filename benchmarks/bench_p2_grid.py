"""P2: 50-point MSM grid cold vs warm (persistent HiGHS)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from partialid import Assumptions, Query, Variables, bound_grid
from partialid.api import bound_grid_cold
from partialid.highs_grid import highspy_available

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)


def _median(fn, repeats: int = 3) -> float:
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(sorted(ts)[len(ts) // 2])


def _instance(k: int):
    V = Variables.from_mapping({"X": [0, 1], "Y": list(range(k))})
    rng = np.random.default_rng(0)
    cells = [(x, y) for x in (0, 1) for y in range(k)]
    p = rng.dirichlet(np.ones(len(cells)))
    P = {c: float(v) for c, v in zip(cells, p)}
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=0)
    a = Assumptions.order("X", "Y")
    return V, P, Q, a


def main():
    values = list(np.linspace(1.0, 20.0, 50))
    rows = []
    print(f"highspy_available={highspy_available()}")
    for k in (2, 10, 20):
        V, P, Q, a = _instance(k)
        # correctness check once
        warm = bound_grid(V, P, Q, a, param="Lambda", values=values, warm_start=True)
        cold = bound_grid_cold(V, P, Q, a, param="Lambda", values=values)
        maxdiff = max(
            abs(w[1].lower - c[1].lower) + abs(w[1].upper - c[1].upper)
            for w, c in zip(warm, cold)
        )
        assert maxdiff < 1e-6, maxdiff

        t_cold = _median(
            lambda: bound_grid_cold(V, P, Q, a, param="Lambda", values=values)
        )
        t_warm = _median(
            lambda: bound_grid(
                V, P, Q, a, param="Lambda", values=values, warm_start=True
            )
        )
        t_tmpl = _median(
            lambda: bound_grid(
                V, P, Q, a, param="Lambda", values=values, warm_start=False
            )
        )
        row = {
            "suite": "P2",
            "k_Y": k,
            "n_points": len(values),
            "highspy": highspy_available(),
            "cold_s": t_cold,
            "template_scipy_s": t_tmpl,
            "warm_s": t_warm,
            "speedup_vs_cold": t_cold / t_warm if t_warm > 0 else float("inf"),
            "max_abs_diff": maxdiff,
            "notes": warm[0][1].notes[-1] if warm else "",
        }
        rows.append(row)
        print(
            f"k={k:2d} cold={t_cold*1000:7.1f}ms tmpl={t_tmpl*1000:7.1f}ms "
            f"warm={t_warm*1000:7.1f}ms speedup={row['speedup_vs_cold']:.1f}x "
            f"diff={maxdiff:.1e}"
        )

    out = OUT / "p2_msm_grid.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
