# Scale snapshot

Median wall-clock over 5 timed calls after one warmup (this machine).
Regenerate with `python scripts/gen_scale_table.py`.

| Instance | requested | routed (`BoundResult.method`) | median ms |
|---|---|---|---:|
| RE1 Manski | `auto` | `closed/manski_binary_xy` | 0.02 |
| RE1 MSM Λ=3 | `auto` | `closed/tan_msm_re1_L3` | 0.01 |
| RE1 MSM Λ=3 | `lp` | `lp` | 3.90 |
| chain n=3 order | `auto` | `fiber` | 0.31 |

These are **smoke** timings for docs — not a claim against Autobounds /
`causaloptim`. Larger frontiers live under `benchmarks/`.
