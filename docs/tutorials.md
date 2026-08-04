# Tutorials

Numbered vignette notebooks live in the repository under
[`notebooks/`](https://github.com/alexisbellot/partialid/tree/main/notebooks)
(also included in the source distribution). They are not installed into the
Python package wheel — clone the repo or download the sdist to run them.

| Notebook | Focus |
|---|---|
| [`01_first_bounds`](https://github.com/alexisbellot/partialid/blob/main/notebooks/01_first_bounds.ipynb) | First Manski / MTR / MSM bounds |
| [`02_assumption_composition`](https://github.com/alexisbellot/partialid/blob/main/notebooks/02_assumption_composition.ipynb) | Composing atoms with `&` |
| [`03_sensitivity_paths`](https://github.com/alexisbellot/partialid/blob/main/notebooks/03_sensitivity_paths.ipynb) | `bound_path` curves |
| [`04_instruments_and_graphs`](https://github.com/alexisbellot/partialid/blob/main/notebooks/04_instruments_and_graphs.ipynb) | IV and diagrams |
| [`05_queries_and_results`](https://github.com/alexisbellot/partialid/blob/main/notebooks/05_queries_and_results.ipynb) | Queries, `BoundResult`, audit |
| [`06_panel_and_scale`](https://github.com/alexisbellot/partialid/blob/main/notebooks/06_panel_and_scale.ipynb) | Larger / panel systems |
| [`07_advanced_identification`](https://github.com/alexisbellot/partialid/blob/main/notebooks/07_advanced_identification.ipynb) | ID, `classify_program`, GPP |

Curriculum notes and data sources:
[`notebooks/README.md`](https://github.com/alexisbellot/partialid/blob/main/notebooks/README.md).

CI executes every notebook cell via `tests/test_docs_recipes.py`.
