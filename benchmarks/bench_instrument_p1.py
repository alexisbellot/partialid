#!/usr/bin/env python3
"""P1 microbench: instrument LP scaling (esp. two instruments, |Z|=3)."""

from __future__ import annotations

import statistics
import time

from partialid import bound
from partialid.formulate import formulate
from partialid.problems import binary_iv, two_instruments, two_instruments_card


def _med(fn, n=5, warmup=1):
    for _ in range(warmup):
        fn()
    xs = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        xs.append(time.perf_counter() - t0)
    return statistics.median(xs)


def report(name, p, repeats=5):
    lp = formulate(p.variables, p.data, p.query, p.assumptions)
    notes = " | ".join(lp.notes)
    n_types = lp.type_space.n_types
    raw = getattr(lp.type_space, "raw_types", n_types)

    def run():
        return bound(p.variables, p.data, p.query, p.assumptions, method="lp")

    med = _med(run, n=repeats)
    r = run()
    print(
        f"{name:28s} raw={raw:<6} cols={n_types:<6} "
        f"{med*1000:8.2f} ms  w={r.width:.4f}  [{notes}]"
    )
    return {"name": name, "raw": raw, "cols": n_types, "median_s": med, "width": r.width}


def main():
    print("Instrument LP P1 benchmark (array + sparse + aggregate)")
    rows = [
        report("binary_iv", binary_iv(seed=2)),
        report("two_iv_Z2", two_instruments(seed=3)),
        report("two_iv_Z3", two_instruments_card(card_z=3, seed=15), repeats=3),
    ]
    z3 = rows[-1]
    # Prior baseline ~5s on Z3; target ≥10× → ≤500ms
    print(f"\nZ3 median {z3['median_s']*1000:.1f} ms "
          f"(prior ~5000 ms → speedup ~{5000/(z3['median_s']*1000):.1f}×)")


if __name__ == "__main__":
    main()
