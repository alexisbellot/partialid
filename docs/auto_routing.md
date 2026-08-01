# How ``bound(..., method="auto")`` chooses an algorithm

Prefer leaving ``method="auto"`` (the default). Explicit solver names are
power-user escape hatches.

## Routing (simplified)

```text
assumptions + query + data
        │
        ▼
   closed form?  ──yes──► analytic interval
        │ no
        ▼
   ID on explicit diagram? ──yes──► point interval (identified)
        │ no
        ▼
   MSM aggregate applicable? ──yes──► msm_agg (sensitivity-native)
        │ no
        ▼
   fiber-separable faces? ──yes──► fiber scan (no LP)
        │ no
        ▼
   equality-only & n ≫ m? ──yes──► dual LP (tagged lp/dual)
        │ no
        ▼
   huge type space? ──yes──► column generation (colgen)
        │ no
        ▼
   full primal LP (HiGHS via SciPy)
```

Special assumption classes force other paths earlier:

| Hint | When |
|---|---|
| `mip` | `support_bound(k)` |
| `convex` | `f_sensitivity` (CVXPY) |
| `gpp` | \(K=2\) free c-degree-1 districts (factor bilinear) |
| `gpp/latent` | one district with c-degree 2 (two latent hyperedges) |
| `fractional` | conditional ratio programs |

Diagram program class (before numeric solve) is summarized by
`classify_program` — see [Graphs](graphs.md#program-degree-which-solver).

## What you should do

1. Call `bound(...)` or `bound_path(...)` with recipes / composed assumptions.
2. Read `BoundResult.method` and `notes` if you care which engine ran.
3. Check `tight` — outer approximations set `tight=False` and emit `UserWarning`.

## Scale snapshot

A small reproducible wall-clock table lives in
[`docs/scale_table.md`](scale_table.md). Regenerate with:

```bash
pip install -e ".[dev]"
PYTHONPATH=src:benchmarks python scripts/gen_scale_table.py
```
