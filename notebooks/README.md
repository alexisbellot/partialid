# partialid tutorials

Self-contained Jupyter notebooks that teach **partialid** through short case
studies. You do **not** need to have read the package docs first.

Each notebook follows the same rhythm: inspect data → state a causal question →
state assumptions → call `bound` → interpret the interval in plain language.

## Quick start

From the repository root:

```bash
pip install -e ".[notebooks]"
jupyter notebook notebooks/01_first_bounds.ipynb
```

Or open any `0*.ipynb` from the `notebooks/` folder in JupyterLab / VS Code.
CI executes every code cell under this folder via pytest.

## Curriculum

### Core path (start here)

| # | Notebook | Study | What you learn |
|---|---|---|---|
| 1 | [`01_first_bounds.ipynb`](01_first_bounds.ipynb) | A — job training | Load a table, Manski bounds, compose MTR/MTS with `&`, read `BoundResult` (`width`, `tight`, `method`) |
| 2 | [`02_assumption_composition.ipynb`](02_assumption_composition.ipynb) | A | Structural faces vs sensitivity budgets; recipes vs atoms; infeasible assumptions |
| 3 | [`03_sensitivity_paths.ipynb`](03_sensitivity_paths.ipynb) | A | `bound_path`, breakdown points, `bound_grid`; how to report a sensitivity curve |
| 4 | [`04_instruments_and_graphs.ipynb`](04_instruments_and_graphs.ipynb) | B — encouragement | `Assumptions.iv`, explicit `CausalDiagram`, point ID vs partial ID, IV+MSM, measurement error |
| 5 | [`05_queries_and_results.ipynb`](05_queries_and_results.ipynb) | A | Interventional / ATE / PN–PNS, data ingress, experimental audit/CI, reporting checklist |

### Optional

| # | Notebook | Study | What you learn |
|---|---|---|---|
| 6 | [`06_panel_and_scale.ipynb`](06_panel_and_scale.ipynb) | C — 8-variable panel | Wide microdata, inert-variable pruning, strata, when cardinality / multi-cause bite |
| 7 | [`07_advanced_identification.ipynb`](07_advanced_identification.ipynb) | (synthetic graphs) | `classify_program`, `tight`, `prefer_poly` — advanced / **experimental** APIs |

```text
01 → 02 → 03 → 04 → 05
                 └─→ 07 (optional)
05 ─→ 06 (optional)
```

## Studies and data

Synthetic CSVs (no network download) live in [`data/`](data/). Details and seeds:
[`data/SOURCES.md`](data/SOURCES.md).

| File | Used in | Columns |
|---|---|---|
| [`data/job_training.csv`](data/job_training.csv) | 01–03, 05 | `treated`, `employed` |
| [`data/encouragement_trial.csv`](data/encouragement_trial.csv) | 04 | `offer`, `enrolled`, `employed` |
| [`data/job_training_panel.csv`](data/job_training_panel.csv) | 06 | `site`, `age_bin`, `educ`, `prior_emp`, `urban`, `spouse`, `treated`, `employed` |

Regenerate:

```bash
PYTHONPATH=src python scripts/generate_notebook_data.py
```

**Study A** — observational job-training evaluation (binary treatment / outcome).  
**Study B** — randomized offer, selective take-up, employment (classic IV).  
**Study C** — multi-site panel with covariates; same causal question as A on a wider file.

## Running tips

- Prefer `pip install -e .` so notebooks import this repo’s `partialid` (no `sys.path` hacks).
- Each notebook resolves `data/*.csv` via `__file__` when set (CI does this) and
  falls back to `notebooks/data/` from the repo root.
- Leave `method="auto"`. Always check `BoundResult.tight` before treating an
  interval as sharp.
- Full assumption catalog and API reference: [`docs/assumptions.md`](../docs/assumptions.md),
  [`docs/api/index.md`](../docs/api/index.md).
