# Stability contract (0.x)

Applies until **1.0.0**. See also [`product.md`](product.md).

## Versioning

- **SemVer** on PyPI tags (`major.minor.patch`, with optional pre-releases `aN` / `bN` / `rcN`).
- **0.x** means the API may evolve. Breaking changes are allowed in minor releases
  (`0.2` → `0.3`) but should be called out in the changelog.
- Prefer **deprecation for one minor** (`DeprecationWarning`) before removing a
  public name, once that name has appeared in a non-alpha PyPI release.
- Patch releases (`0.1.1`) are for bugfixes and docs only — no intentional
  breaking API changes.

## Public API

**Only names in `partialid.__all__` are public.**

- Anything else under `partialid.*` (including modules not re-exported) is
  **private** and may change without notice.
- Prefer `from partialid import bound, Assumptions, …` over deep imports
  (`from partialid.formulate import …`).

### Stability tiers within `__all__` (0.x)

**Core (treat as stable-intent; changes need deprecation):**

- `Variables`, `CausalDiagram`
- `Query` (named factories: `interventional`, `ate`, `pns`, `pn`, `ps`, …)
- `Assumptions`, `AssumptionSet` (including recipes `manski`, `sensitivity_msm`, `iv`)
- `ObservationalData`, `from_counts`, `from_records`, `from_dataframe`
- `bound`, `bound_grid`, `bound_path`
- `BoundResult`, `BoundPathResult`

**Experimental (exported, but may change shape in 0.x without long deprecation):**

- Query AST: `PO`, `Lit`, `Atom`, `LinComb`, `Cond`
- `bound_ci`, `BoundCIResult`
- `audit_assumptions`, `audit_bound`, `AuditReport`
- `try_closed_form`, `ClosedFormResult`
- `Problem`, `all_literature_problems`

Experimental symbols stay importable for power users and tests; do not build
long-lived downstream APIs on them until 1.0 promotes them to core.

## `BoundResult.tight` and warnings

| Signal | Meaning |
|---|---|
| `tight=True` (default) | Solver claims sharp bounds under the stated discrete LP/MIP/convex program for this assumption class. |
| `tight=False` | Bounds are a **valid outer approximation** (contain the true identified set) but may be **loose**. Typical cause: multi c-component / merged-latent fallback when factors are not separated by ID. |
| `warnings` / `UserWarning` | Human-readable copy of the same issue (and related notes). Emitted when the result is returned. |
| `notes` | Machine-oriented tags (e.g. `bounds_may_be_outer`). |

**Contract:** we do not silently return outer bounds as if they were sharp.
Callers should check `result.tight` (or catch `UserWarning`) in automated pipelines.

Numeric values are floating-point; compare with a tolerance appropriate to the
LP solver (on the order of `1e-6`–`1e-8` for well-scaled binary problems).

## Method argument

`bound(..., method="auto")` is the supported default. Explicit methods
(`fiber`, `lp`, `dual`, `colgen`, `mip`, `convex`, `hyperarc`, `c_canonical`,
`gpp`, `closed`) are power-user escape hatches; routing heuristics for `"auto"`
may improve across 0.x without a major bump.
