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
- Program degree: `classify_program`, `ProgramDegree`, `ProgramKind`
- Identification: `is_identified`, `identify_query`, `IDResult`
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
| `tight=False` + `bounds_may_be_outer` | Bounds are a **valid outer approximation** (contain the true identified set) but may be **loose**. Typical causes: degree ≥3 (default outer), GPP size-gate failure, hard `independence`, or other merged-latent fallback. |
| `tight=False` + `bounds_may_be_inner` | Uncertified GPP / poly nested-LP **attainables**: may be slightly **inner** (not a certified outer). Emitted when the global Lip certificate is incomplete. |
| `warnings` / `UserWarning` | Human-readable copy of the same issue (and related notes). Emitted when the result is returned. |
| `notes` | Machine-oriented tags (`bounds_may_be_outer`, `bounds_may_be_inner`, …). |

**Contract:** we do not silently return outer (or uncertified) bounds as if they
were sharp. Callers should check `result.tight` and `result.notes` (or catch
`UserWarning`) in automated pipelines. Do **not** treat every `tight=False` as
a conservative outer — inspect the note tags.

Numeric values are floating-point; compare with a tolerance appropriate to the
LP solver (on the order of `1e-6`–`1e-8` for well-scaled binary problems).

## Method argument

`bound(..., method="auto")` is the supported default. Explicit methods
(`fiber`, `lp`, `dual`, `colgen`, `mip`, `convex`, `hyperarc`, `c_canonical`,
`gpp`, `msm_agg`, `fractional`, `closed`) are power-user escape hatches; routing
heuristics for `"auto"` may improve across 0.x without a major bump. Fractional
programs tag results as `lp/fractional` when solved via the LP backend.
