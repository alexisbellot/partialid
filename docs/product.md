# Product boundaries (Phase 0)

Locked decisions for **partialid**. This page is the product contract.

## Name

| | |
|---|---|
| PyPI | `partialid` |
| Import | `import partialid` |
| Former names | `pi-solver` / `pi_solver` (do not use) |

## Positioning

> Bound causal effects from discrete data by composing explicit assumptions
> from the literature — without requiring a fully identified DAG.

**Assumption composition is first-class.** Orders, diagrams, monotonicity,
instruments, and sensitivity budgets combine with `&`. Graphs and LP solvers
are infrastructure. Users should rarely need method flags (`dual`, `colgen`, …).

Sensitivity paths (`bound_path`) are a supported workflow when a budget varies;
they are not the only (or primary) product story.

## Monorepo layout

| Path | Role | In **wheel**? | In **sdist**? |
|---|---|---|---|
| `src/partialid/` | Installable library | yes | yes |
| `tests/` | Pytest suite | no | yes |
| `notebooks/` | Tutorial vignettes + CSV data (canonical examples) | no | yes |
| `docs/` | User / product docs | no | yes |
| `benchmarks/` | Scale / method experiments | no | **no** |
| `latex/`, `papers/`, `Figures/` | Research collateral (local only) | no | **no** |
| `README.md`, `LICENSE`, `CHANGELOG.md`, `CITATION.cff`, `SECURITY.md` | Package metadata | metadata | yes |
| Research notes (`THEORY*.md`, roadmaps, …) | Local working notes | no | **no** |
| `scripts/` | Maintainer utilities | no | **no** |

Enforced for sdists via the repository `MANIFEST.in`. The wheel contains only
the `partialid` package (and metadata).

## What we optimize for

1. Practitioner workflows: observational table → composed assumptions → bounds
   (optional sensitivity path)
2. Trust: never silently return outer or uncertified bounds as if sharp
   (`tight` + note tags)
3. Scale on discrete systems where structure (fiber, dual, prune) matters
4. A small, documented public API on a SemVer 0.x train

## Non-goals (near term)

- Symbolic vertex enumeration (`causaloptim` / symbolic vertex enumeration)
- Shipping the paper tree inside the installable artifact
- Requiring users to choose advanced solvers for common cases
