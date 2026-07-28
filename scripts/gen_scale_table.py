#!/usr/bin/env python3
"""Generate a small user-facing scale table (Phase 3).

Writes ``docs/scale_table.md``. Keep instances tiny so CI/nightly stays fast.
"""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

from partialid import Assumptions, Query, Variables, bound

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "scale_table.md"


def _time_ms(fn, repeats: int = 5) -> float:
    # warmup
    fn()
    xs = []
    for _ in range(repeats):
        t0 = perf_counter()
        fn()
        xs.append((perf_counter() - t0) * 1000.0)
    xs.sort()
    return xs[len(xs) // 2]


def main() -> None:
    rows: list[tuple[str, str, str, float]] = []

    V = Variables.from_binary("X", "Y")
    P = {(0, 0): 0.30, (0, 1): 0.20, (1, 0): 0.15, (1, 1): 0.35}
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)

    cases = [
        ("RE1 Manski", Assumptions.manski("X", "Y"), "auto"),
        ("RE1 MSM Λ=3", Assumptions.sensitivity_msm("X", "Y", 3.0), "auto"),
        ("RE1 MSM Λ=3", Assumptions.sensitivity_msm("X", "Y", 3.0), "lp"),
    ]

    # short binary chain (inert prune → XY-sized)
    V3 = Variables.from_binary("X", "Z", "Y")
    cells = V3.cells()
    P3 = {c: 1.0 / len(cells) for c in cells}
    Q3 = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    cases.append(("chain n=3 order", Assumptions.order("X", "Z", "Y"), "auto"))

    for name, a, method in cases:
        if "chain" in name:
            vv, pp, qq = V3, P3, Q3
        else:
            vv, pp, qq = V, P, Q

        def run(vv=vv, pp=pp, qq=qq, a=a, method=method):
            return bound(vv, pp, qq, a, method=method)

        ms = _time_ms(run)
        r = run()
        rows.append((name, method, r.method, ms))

    lines = [
        "# Scale snapshot",
        "",
        "Median wall-clock over 5 timed calls after one warmup (this machine).",
        "Regenerate with `python scripts/gen_scale_table.py`.",
        "",
        "| Instance | requested | routed (`BoundResult.method`) | median ms |",
        "|---|---|---|---:|",
    ]
    for name, req, routed, ms in rows:
        lines.append(f"| {name} | `{req}` | `{routed}` | {ms:.2f} |")
    lines.extend(
        [
            "",
            "These are **smoke** timings for docs — not a claim against Autobounds /",
            "`causaloptim`. Larger frontiers live under `benchmarks/`.",
            "",
        ]
    )
    OUT.write_text("\n".join(lines))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
