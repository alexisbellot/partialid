# Solvers (advanced)

Numeric partial identification for discrete SCMs in the **linear reduction**:
face assumptions and **anchored** sensitivity budgets. Prefer
`method="auto"` — see [Auto routing](../auto_routing.md).

## In scope (LP / convex / MIP / bilinear)

- Order-only / maximal confounding (Manski)
- Directed exclusions, MTR, no-defiers
- Tan MSM, \(c\)-dependence, anchored Rosenbaum \(\Gamma\)
- Margin budgets; \(f\)-sensitivity via CVXPY
- Instrument R-side LPs; support bounds via MIP
- **Bilinear GPP**: \(K=2\) free c-degree-1 districts (`gpp`) or one
  c-degree-2 district with two latent hyperedges (`gpp/latent`)
- **Polynomial (opt-in):** factor \(K=3\) nested multilinear via
  `bound(..., prefer_poly=True)` → `gpp/poly` when the size gate passes

## Out of core (workarounds)

| Assumption | Why | Workaround |
|---|---|---|
| Unanchored Rosenbaum \(\Gamma\) | Bilinear | `anchored=True` |
| Raw MI / KL budgets | Transcendental | margin budget proxy |
| Hard independence c-split | Factorization incomplete | merged outer / ID |
| Large multi-district GPP | Size gate | outer LP (`tight=False`) |
| Degree ≥3 (default) | Polynomial | outer LP (`tight=False`); opt in with `prefer_poly=True` |
| Latent c-degree ≥3 | Beyond size-gated poly | outer LP even with `prefer_poly=True` |

Latent hyperedges: each `bidirected` tuple is one exogenous source. Pairwise
`[("X","M"),("M","Y")]` is c-degree 2 (bilinear); `[("X","M","Y")]` is one
latent (LP). No `maximal_confounding` flag — Manski is a full hyperedge
`[tuple(nodes)]`. See [Graphs](../graphs.md#program-degree-which-solver).

## Algorithm sketch

1. **Closed form** when a known pattern matches (Manski / Tan binary XY, …).
2. **Shpitser–Pearl ID** on explicit diagrams (interventional / ATE → point interval).
3. **MSM aggregate** for common sensitivity programs.
4. **`classify_program`** (C2 + ID → POINT, else post-cut degree) → formulate:
   - LINEAR / POINT → instrument LP (WL–WR / C2) or TypeSpace LP / fiber
   - BILINEAR_* → `gpp` / `gpp/latent` (`prefer_gpp=True`; else outer)
   - POLYNOMIAL → outer by default; `prefer_poly=True` → `gpp/poly` when gated
5. Within LP-shaped programs, size heuristics may still pick **fiber**,
   **dual LP**, **column generation**, or **primal HiGHS**.

Speed tables in this repo’s benchmarks compare against a *naive* full-type LP —
not against Autobounds or R/`causaloptim`. See also
[Auto routing](../auto_routing.md) and the [scale table](../scale_table.md).
