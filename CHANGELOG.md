# Changelog

All notable changes to **partialid** are documented here.

Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
versioning follows [SemVer](https://semver.org/) (0.x API may evolve; see
[`docs/stability.md`](docs/stability.md)).

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

### Notes
- 0.x pre-release: public API may change before 0.1.0.
- Research roadmaps and paper sources remain in the git repo, not the wheel.
