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
   ID (Shpitser–Pearl) on explicit diagram?
        │ yes ──► point interval (method id / id/ate)
        │ no
        ▼
   MSM aggregate applicable? ──yes──► msm_agg
        │ no
        ▼
   classify_program (C2 + ID → POINT, else post-cut degree)
        │
        ├─ POINT / LINEAR ──► instrument LP (WL–WR / C2) or fiber / primal LP
        ├─ BILINEAR_*     ──► gpp / gpp/latent   (unless prefer_gpp=False → outer)
        └─ POLYNOMIAL     ──► outer LP by default
                              └─ prefer_poly=True → gpp/poly (factor K=3, size-gated)
        │
        ▼
   within LINEAR / fiber path, size heuristics may still pick:
        fiber → dual LP → colgen → primal HiGHS
```

`classify_program` follows program-degree theory (`THEORY_PROGRAM_DEGREE` §6): ancestral trim, footprint C1, directed-cut
C2, then ID (→ `POINT`) or \(\sum\lambda\). See
[Graphs](graphs.md#program-degree-which-solver). Use `is_identified` /
`identify_query` to test identification without an LP.

## Formulation knobs

| Flag | Default | Effect |
|---|---|---|
| `prefer_gpp` | `True` | Bilinear factor/latent → nested GPP; `False` forces outer LP |
| `prefer_poly` | `False` | Deg≥3 → outer LP; `True` tries `gpp/poly` when the size gate passes |

Pass them on `bound(...)` or `formulate(...)`.

## Method hints (special paths)

| Hint | When |
|---|---|
| `mip` | `support_bound(k)` |
| `convex` | `f_sensitivity` (CVXPY) |
| `gpp` | \(K=2\) free c-degree-1 districts (factor bilinear) |
| `gpp/latent` | one district with c-degree 2 (two latent hyperedges) |
| `gpp/poly` | deg≥3 with `prefer_poly=True` and size gate (factor \(K=3\)) |
| `fractional` | conditional ratio programs |

## What you should do

1. Call `bound(...)` or `bound_path(...)` with recipes / composed assumptions.
2. Read `BoundResult.method` and `notes` if you care which engine ran.
3. Check `tight` — outer approximations and uncertified GPP attainables set
   `tight=False` and emit `UserWarning`. Inspect `notes` for
   `bounds_may_be_outer` vs `bounds_may_be_inner`.

## Scale snapshot

A small reproducible wall-clock table lives in
[`docs/scale_table.md`](scale_table.md). Regenerate with:

```bash
pip install -e ".[dev]"
PYTHONPATH=src:benchmarks python scripts/gen_scale_table.py
```
