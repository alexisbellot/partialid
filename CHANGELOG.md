# Changelog

All notable changes to **partialid** are documented here.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
versioning follows [SemVer](https://semver.org/) (0.x API may evolve; see
[`docs/stability.md`](docs/stability.md)).

## [Unreleased]

### Fixed
- Measurement-error / latent-endogenous graphs: keep observed proxies in the
  ancestral set, skip spurious ID→POINT when the outcome is latent, and
  remargin observational tables on observed coordinates only.

## [0.1.0a5] — 2026-08-04

### Added
- Numbered vignette tutorials under `notebooks/` (`01`–`07`) with synthetic
  study CSVs in `notebooks/data/` (job training, encouragement IV, 8-variable
  panel).
- Latent **hyperedges** in `CausalDiagram.bidirected` (pairs or larger tuples):
  each hyperedge is one exogenous source.
- `classify_program` (`THEORY_PROGRAM_DEGREE` §6): ancestral trim + footprint C1
  + directed-cut C2, then \(\sum\lambda\) → POINT / LINEAR / BILINEAR_* /
  POLYNOMIAL. Records `wl_wr_cut`, `in_hypothesis_H`, post-cut `free_districts`.
  Formulate/bound dispatch from that label (no classify↔solver split-brain).
- Sharp nested-LP for bilinear: factor \(K=2\) (`gpp`) and latent c-degree 2
  (`gpp/latent`).
- Opt-in nested multilinear for factor \(K=3\): `bound(..., prefer_poly=True)`
  → `gpp/poly` when the size gate passes. **Default for deg≥3 remains outer**
  (`tight=False`).
- `classify_program` / bilinear–GPP routing accept **counterfactual** LinComb
  queries (PNS, ATE, …): cross-world variables become free blocks (e.g. mediated
  `X↔M→Y` is linear for `P(Y|do(X))` and bilinear/factor for PNS).
- PN/PS (`Cond` with fixed observational denominator) route to GPP via
  Tian–Pearl linearization (no merged-outer fallback).
- Program-degree / GPP routing is AST-first: any `Query.from_expr` /
  `Query.conditional` (e.g. `P(Y_z, W_x | R)`), not only named factories.
  Multi-regime joints and observational evidence free the relevant blocks.
- `CITATION.cff`, `CONTRIBUTING.md`, docs Tutorials page, optional
  `[notebooks]` extra.

### Changed
- Identification is **Shpitser–Pearl ID only** (g-formula / back-door /
  front-door fallbacks removed as separate engines). `classify_program` sets
  `POINT` when ID succeeds; public `is_identified` / `identify_query`; ATE
  point-ID when both margins are identified (`method="id/ate"`).
- Docs overview refreshed for classify routing: `prefer_poly=False` by default
  (deg≥3 → outer; opt in with `prefer_poly=True`), `prefer_gpp` on `bound`,
  auto-routing flowchart, graphs / concepts / README / API index.
- Single install includes pandas, HiGHS (`highspy`), and CVXPY. Removed the
  `data` / `fast` / `convex` / `all` extras.
- Removed `maximal_confounding`. Manski / order-only is
  `bidirected=[tuple(nodes)]` (also what `from_order` / `Assumptions.order` emit).
- Pairwise bidirected chains (e.g. `X↔M`, `M↔Y`) are c-degree 2 (bilinear).
  Use `bidirected=[("X","M","Y")]` for one shared latent.
- `Assumptions.independence`: always forces merged-latent outer (`tight=False`);
  sharp product factorization remains unimplemented.
- Unanchored `tan_msm(..., anchored=False)` raises at the factory (mirrors
  Rosenbaum).
- `BoundResult.tight=False` distinguishes `bounds_may_be_outer` vs
  `bounds_may_be_inner` (uncertified GPP attainables).
- Fractional LP results tag `method` as `lp/fractional` (not bare `lp`).
- README install story prefers git/editable until PyPI publish; version `0.1.0a5`.
- Sdist excludes `THEORY_PROGRAM_DEGREE.md` / `CLAUDE.md`; includes notebooks +
  citation/contributing files.

### Fixed
- MkDocs `--strict` links (notebooks / source paths no longer leave the docs tree).
- Docs cite `THEORY_PROGRAM_DEGREE` §6 (not `THEORY.md` Move 6) for
  `classify_program`.
- `graphs.md` distinguishes Manski full hyperedge from deg≥3 outer fallback.
- GPP build failures no longer swallow arbitrary `Exception` into outer LP
  (only `ValueError` / `NotImplementedError` / `RuntimeError`).

## [0.1.0a4] — 2026-07-27

### Changed
- `Assumptions.iv(..., monotone=False)` by default (opt in to no-defiers); avoids
  infeasible LPs on many valid IV laws.

## [0.1.0a3] — 2026-07-27

### Added
- Phase 3 quality bar: ruff format+lint, mypy on public modules, `py.typed`,
  pre-commit config, coverage gate (≥85% on public API modules).
- CI jobs: lint/types, pytest matrix (3.10–3.13 + macOS 3.12), extras smoke,
  examples, nightly benchmark workflow.
- Docs: `docs/auto_routing.md`, `docs/scale_table.md` (+ `scripts/gen_scale_table.py`).

## [0.1.0a2] — 2026-07-27

### Added
- Data ingress: `from_counts`, `from_records`, `from_dataframe` (`ObservationalData`).
- Recipes: `Assumptions.manski`, `sensitivity_msm`, `iv`.
- `BoundResult.summary` / `to_dict`, `BoundPathResult.summary` / `to_dict`,
  `Query.describe`.
- Examples: sensitivity-path quickstart + CSV workflow (`examples/from_csv.py`).

## [0.1.0a1] — 2026-07-26

### Added
- First installable alpha of `partialid` (renamed from `pi-solver` / `pi_solver`).
- Core API: `Variables`, `Query`, `Assumptions`, `bound`, `bound_grid`, `bound_path`.
- BSD-3-Clause license, packaging metadata, CI, and sdist inclusion policy.
- Optional extras: `fast` (highspy), `convex` (cvxpy), `data` (pandas; helpers in Phase 2).
  *(Superseded: these are core dependencies as of 0.1.0a5.)*

### Notes
- 0.x pre-release: public API may change before 0.1.0.
- Research roadmaps and paper sources remain in the git repo, not the wheel.
