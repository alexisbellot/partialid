# Solvers (advanced)

Numeric partial identification for discrete SCMs in the **linear reduction**:
face assumptions and **anchored** sensitivity budgets. Prefer
`method="auto"` — see [Auto routing](../auto_routing.md).

## In scope (LP / convex / MIP)

- Order-only / maximal confounding (Manski)
- Directed exclusions, MTR, no-defiers
- Tan MSM, \(c\)-dependence, anchored Rosenbaum \(\Gamma\)
- Margin budgets; \(f\)-sensitivity via CVXPY (`partialid[convex]`)
- Instrument R-side LPs; support bounds via MIP

## Out of core (workarounds)

| Assumption | Why | Workaround |
|---|---|---|
| Unanchored Rosenbaum \(\Gamma\) | Bilinear | `anchored=True` |
| Raw MI / KL budgets | Transcendental | margin budget proxy |
| Hard independence c-split | Factorization incomplete | merged outer / ID |
| Large multi-district GPP | Polynomial | outer LP (`tight=False`) |

## Algorithm sketch

1. **Closed form** when a known pattern matches (Manski / Tan binary XY, …).
2. **ID** on explicit diagrams (point interval when identified).
3. **MSM aggregate** for common sensitivity programs.
4. **Fiber** scan when faces separate by observational cell.
5. **Dual LP** when equality-only and \(n \gg m\).
6. **Column generation** for huge type spaces.
7. Else **primal HiGHS** (SciPy); optional persistent HiGHS via `highspy`.

Speed tables in this repo’s benchmarks compare against a *naive* full-type LP —
not against Autobounds or R/`causaloptim`. See also
[Auto routing](../auto_routing.md) and the [scale table](../scale_table.md).
