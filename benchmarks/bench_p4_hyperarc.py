"""P4: hyperarc vs instrument/fiber wall-clock on literature instances."""

from __future__ import annotations

import json
import time
from pathlib import Path

from partialid import bound
from partialid.problems import (
    binary_iv,
    ternary_exposure,
    shridharan_multi_treatment,
)

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)


def _med(fn, repeats: int = 5) -> float:
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(sorted(ts)[len(ts) // 2])


def main():
    rows = []
    cases = [
        ("re1_style_ternary", ternary_exposure()),
        ("iv_binary", binary_iv(seed=2)),
        ("multi_T2", shridharan_multi_treatment(2, 2)),
        ("multi_T3", shridharan_multi_treatment(3, 2)),
    ]
    for name, p in cases:
        methods = ["auto", "hyperarc"]
        row = {"name": name, "source": p.source}
        ref = bound(p.variables, p.data, p.query, p.assumptions)
        for m in methods:
            def run(mm=m):
                return bound(
                    p.variables, p.data, p.query, p.assumptions, method=mm
                )

            # correctness
            r = run()
            ok = bool(
                r.success
                and abs(r.lower - ref.lower) < 1e-6
                and abs(r.upper - ref.upper) < 1e-6
            )
            t = _med(run)
            row[f"{m}_ms"] = float(t * 1000)
            row[f"{m}_ok"] = ok
            if m == "hyperarc":
                row["hyperarc_notes"] = list(r.notes[:4])
        rows.append(row)
        print(
            f"{name}: auto={row['auto_ms']:.2f}ms hyperarc={row['hyperarc_ms']:.2f}ms "
            f"ok={row['hyperarc_ok']} notes={row.get('hyperarc_notes')}"
        )

    path = OUT / "p4_hyperarc.json"
    path.write_text(json.dumps(rows, indent=2))
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
