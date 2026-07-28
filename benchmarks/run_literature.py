#!/usr/bin/env python3
"""Literature + scaling wall-clock suite (pure Python; no R).

Compares fiber / lp / colgen on Sachs- and Shridharan-style instances.
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
    all_literature_problems,
    chain_order_only,
    sachs_ternary_cardinalities,
    sachs_two_instruments_card,
    shridharan_multi_treatment,
)

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(exist_ok=True)


def _median_time(fn, repeats: int = 3, warmup: int = 1) -> float:
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


def _n_types(p) -> int:
    lp = formulate(p.variables, p.data, p.query, p.assumptions)
    return int(lp.type_space.n_types)


def run_problem(p, methods=("auto", "lp", "fiber", "colgen")) -> list[dict]:
    rows = []
    try:
        n_types = _n_types(p)
    except Exception as e:  # noqa: BLE001
        return [
            {
                "name": p.name,
                "source": p.source,
                "error": f"formulate: {e}",
                "success": False,
            }
        ]

    for method in methods:
        if method == "fiber":
            # skip when not separable
            from partialid.fiber_solve import is_fiber_separable
            from partialid.formulate import formulate as form

            lp = form(p.variables, p.data, p.query, p.assumptions)
            if not is_fiber_separable(lp):
                continue
        if method == "colgen" and n_types > 256:
            # avoid pathological colgen on large instrument LPs for now
            continue
        try:

            def _run(m=method):
                return bound(p.variables, p.data, p.query, p.assumptions, method=m)

            med = _median_time(_run, repeats=3 if n_types < 128 else 1, warmup=1)
            r = _run()
            rows.append(
                {
                    "name": p.name,
                    "source": p.source,
                    "method": method,
                    "n_types": n_types,
                    "median_s": med,
                    "formulate_s": r.formulate_time,
                    "solve_s": r.solve_time,
                    "width": r.width,
                    "lower": r.lower,
                    "upper": r.upper,
                    "success": r.success,
                    "notes": r.notes,
                    "error": r.message if not r.success else "",
                }
            )
        except Exception as e:  # noqa: BLE001
            rows.append(
                {
                    "name": p.name,
                    "source": p.source,
                    "method": method,
                    "n_types": n_types,
                    "success": False,
                    "error": f"{e}\n{traceback.format_exc(limit=2)}",
                }
            )
    return rows


def main():
    problems = list(all_literature_problems())
    # extra scaling / cardinality (keep type counts modest)
    problems += [
        sachs_ternary_cardinalities(5, 2, seed=10),
        sachs_ternary_cardinalities(3, 4, seed=11),
        sachs_two_instruments_card(card_z=2, seed=12),
        chain_order_only(5, 2, seed=13),  # inert prune → binary XY
        # card_z=3 → 2^(3*3)=512 X-types × 4 Y-types = 2048 (LP ok, skip colgen)
        sachs_two_instruments_card(card_z=3, seed=15),
    ]

    all_rows = []
    print(f"Running {len(problems)} literature/scaling problems...")
    for p in problems:
        print(f"  · {p.name} ({p.source})")
        rows = run_problem(p)
        all_rows.extend(rows)
        for row in rows:
            if row.get("success"):
                print(
                    f"      {row['method']}: {row['median_s']*1000:.2f} ms  "
                    f"types={row['n_types']}  w={row['width']:.4f}"
                )
            else:
                print(f"      {row.get('method', '?')}: FAIL {row.get('error', '')[:80]}")

    out = OUT / "literature_wallclock.json"
    out.write_text(json.dumps(all_rows, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
