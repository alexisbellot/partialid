"""Track B success metrics: analytic RE1 path + 50-pt ≤ 5 full-solve cost."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import numpy as np

from partialid import Assumptions, Query, Variables, bound, bound_path
from partialid.api import bound_grid_cold
from partialid.closed_forms import msm_closed_form_re1
from partialid.highs_grid import highspy_available

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)

RE1_P = {(0, 0): 0.30, (0, 1): 0.20, (1, 0): 0.15, (1, 1): 0.35}


def _med(fn, repeats: int = 5, warmup: int = 1) -> float:
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(statistics.median(ts))


def main() -> None:
    V = Variables.from_binary("X", "Y")
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    a = Assumptions.order("X", "Y")
    values = list(np.linspace(1.0, 20.0, 50))

    # --- Metric 1: RE1 matches analytic ---
    path = bound_path(V, RE1_P, Q, a, param="Lambda", theta_min=1.0, theta_max=20.0)
    max_err = 0.0
    for lam in values:
        ell, u, w = path.at(float(lam))
        e2, u2, w2 = msm_closed_form_re1(float(lam))
        max_err = max(max_err, abs(ell - e2), abs(u - u2), abs(w - w2))
    m1 = max_err < 1e-12 and path.closed_form

    # --- Metric 2: 50-pt path ≤ 5 full solves (RE1 LP template) ---
    t_path = _med(
        lambda: bound_path(
            V,
            RE1_P,
            Q,
            a,
            param="Lambda",
            values=values,
            prefer_closed=False,
            breakpoints=False,
            warm_start=True,
        )
    )
    t_1 = _med(
        lambda: bound(
            V, RE1_P, Q, a & Assumptions.tan_msm("X", 3.0), method="lp"
        ),
        repeats=7,
    )
    t_cold = _med(
        lambda: bound_grid_cold(V, RE1_P, Q, a, param="Lambda", values=values),
        repeats=3,
    )
    m2 = t_path <= 5.0 * t_1

    row = {
        "highspy": highspy_available(),
        "analytic_max_err": max_err,
        "analytic_pass": m1,
        "breakpoints": path.breakpoints.tolist(),
        "path_50_s": t_path,
        "single_s": t_1,
        "cold_50_s": t_cold,
        "path_over_single": t_path / t_1 if t_1 > 0 else None,
        "leq_5_solves_pass": m2,
        "speedup_vs_cold": t_cold / t_path if t_path > 0 else None,
    }
    OUT.joinpath("track_b_metrics.json").write_text(json.dumps(row, indent=2))
    print(json.dumps(row, indent=2))
    print(
        f"B metrics: analytic={'PASS' if m1 else 'FAIL'}, "
        f"50pt≤5solves={'PASS' if m2 else 'FAIL'} "
        f"({row['path_over_single']:.2f}× single)"
    )


if __name__ == "__main__":
    main()
