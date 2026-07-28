#!/usr/bin/env python3
"""Naive vs optimized wall-clock comparison.

Naive = full type expansion + HiGHS LP (optimize=False).
Optimized = inert prune + instrument R-side LP + fiber when separable.
"""

from __future__ import annotations

import json
import statistics
import time
import traceback
from pathlib import Path

from partialid import bound
from partialid.formulate import formulate
from partialid.problems import (
    chain_order_only,
    sachs_binary_iv,
    sachs_ternary_cardinalities,
    sachs_ternary_exposure,
    sachs_two_instruments,
    shridharan_multi_treatment,
)

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)

TIMEOUT_S = 15.0
MAX_NAIVE_TYPES = 5000  # skip naive expand beyond this


def _median(fn, repeats=3, warmup=1):
    for _ in range(warmup):
        fn()
    xs = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        xs.append(time.perf_counter() - t0)
    return statistics.median(xs)


def _estimate_joint_types(p) -> int:
    """|R| under the diagram's directed parent sets (maximal block)."""
    from partialid.assumptions import as_assumption_set
    from partialid.graph import CausalDiagram

    a = as_assumption_set(p.assumptions)
    g = a.diagram()
    if g is None:
        g = CausalDiagram.from_order(p.variables.names, maximal_confounding=True)
    n = 1
    for v in g.topological_order():
        parents = list(g.parents(v))
        if not parents:
            n *= p.variables.size(v)
        else:
            in_size = 1
            for pa in parents:
                in_size *= p.variables.size(pa)
            n *= p.variables.size(v) ** in_size
    return n


def _safe_run(p, *, optimize: bool, method: str = "auto"):
    if not optimize:
        est = _estimate_joint_types(p)
        if est > MAX_NAIVE_TYPES:
            return {
                "optimize": False,
                "success": False,
                "error": f"skipped: estimated {est} types > {MAX_NAIVE_TYPES}",
                "n_types": est,
                "skipped": True,
            }

    t0 = time.perf_counter()
    try:
        lp = formulate(
            p.variables, p.data, p.query, p.assumptions, optimize=optimize
        )
        n_types = int(lp.type_space.n_types)
        if time.perf_counter() - t0 > TIMEOUT_S:
            return {
                "optimize": optimize,
                "success": False,
                "error": "timeout during formulate",
                "n_types": n_types,
            }

        def run():
            return bound(
                p.variables,
                p.data,
                p.query,
                p.assumptions,
                method=method if optimize else "lp",
                optimize=optimize,
            )

        repeats = 3 if n_types < 200 else 1
        med = _median(run, repeats=repeats, warmup=1 if n_types < 500 else 0)
        r = run()
        return {
            "optimize": optimize,
            "method_used": r.method,
            "n_types": n_types,
            "median_s": med,
            "formulate_s": r.formulate_time,
            "solve_s": r.solve_time,
            "width": r.width,
            "lower": r.lower,
            "upper": r.upper,
            "success": r.success,
            "notes": r.notes,
            "error": "" if r.success else r.message,
        }
    except Exception as e:  # noqa: BLE001
        return {
            "optimize": optimize,
            "success": False,
            "error": f"{e}",
            "traceback": traceback.format_exc(limit=3),
            "n_types": None,
        }


def main():
    problems = [
        sachs_ternary_exposure(),
        sachs_ternary_cardinalities(4, 2),
        sachs_ternary_cardinalities(5, 2, seed=10),
        chain_order_only(3, 2),
        chain_order_only(4, 2),  # naive may be huge
        chain_order_only(3, 3),
        sachs_binary_iv(),
        sachs_two_instruments(),
        shridharan_multi_treatment(2, 2),
    ]

    rows = []
    print(f"{'problem':40s} {'naive types':>12s} {'opt types':>10s} "
          f"{'naive ms':>10s} {'opt ms':>10s} {'speedup':>8s} {'width ok':>8s}")
    for p in problems:
        print(f"\n→ {p.name}")
        naive = _safe_run(p, optimize=False)
        opt = _safe_run(p, optimize=True, method="auto")
        row = {"name": p.name, "source": p.source, "naive": naive, "optimized": opt}

        # speedup / agreement
        if naive.get("success") and opt.get("success"):
            speedup = naive["median_s"] / max(opt["median_s"], 1e-12)
            width_ok = bool(abs(naive["width"] - opt["width"]) < 1e-4)
            row["speedup"] = float(speedup)
            row["width_agree"] = width_ok
            print(
                f"{p.name:40s} {naive.get('n_types'):>12} {opt.get('n_types'):>10} "
                f"{naive['median_s']*1000:>10.2f} {opt['median_s']*1000:>10.2f} "
                f"{speedup:>8.1f}x {str(width_ok):>8s}"
            )
        else:
            row["speedup"] = None
            row["width_agree"] = None
            print(
                f"{p.name:40s} naive={naive.get('n_types')} "
                f"ok={naive.get('success')} err={str(naive.get('error',''))[:40]} | "
                f"opt={opt.get('n_types')} ok={opt.get('success')}"
            )
        rows.append(row)

    out = OUT / "naive_vs_optimized.json"
    out.write_text(json.dumps(rows, indent=2))
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
