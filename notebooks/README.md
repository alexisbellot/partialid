# partialid notebooks

Runnable Jupyter tutorials. From the repo root:

```bash
pip install -e .
jupyter notebook notebooks/quickstart.ipynb
```

| Notebook | Contents |
|---|---|
| [`quickstart.ipynb`](quickstart.ipynb) | First bounds, `&` composition, counts, ATE |
| [`assumption_catalog.ipynb`](assumption_catalog.ipynb) | Structural faces + sensitivity budgets |
| [`msm_sensitivity.ipynb`](msm_sensitivity.ipynb) | Tan MSM paths and breakdown |
| [`sensitivity_budgets.ipynb`](sensitivity_budgets.ipynb) | Λ, Γ, $c$, κ, grids, support sparsity |
| [`instrumental_variables.ipynb`](instrumental_variables.ipynb) | Binary IV, two instruments, IV+MSM |
| [`queries_and_audit.ipynb`](queries_and_audit.ipynb) | ATE / PN–PS, query AST, audit, CI |
| [`graphs_and_identification.ipynb`](graphs_and_identification.ipynb) | Diagrams, ID, front-door, `tight` |
| [`measurement_error.ipynb`](measurement_error.ipynb) | Latent outcomes / misclassification |
| [`data_io.ipynb`](data_io.ipynb) | Counts, records, CSV / DataFrame |
| [`larger_systems.ipynb`](larger_systems.ipynb) | Higher cardinality, chains, multi-cause |

CI executes every `*.ipynb` code cell under this folder.
