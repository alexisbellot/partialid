# Tutorial data sources

All files here are **synthetic**. They are designed to look like study extracts
so notebooks can teach a full analysis workflow (inspect rows → crosstab →
bound) without network downloads. Regenerate with:

```bash
PYTHONPATH=src python scripts/generate_notebook_data.py
```

| File | N | Columns | Generative process |
|---|---|---|---|
| `job_training.csv` | 2000 | `treated`, `employed` | Multinomial sample from the classic RE1 law \(P=(0.30,0.20,0.15,0.35)\) used in package docs/examples. Seed `42`. |
| `encouragement_trial.csv` | 3000 | `offer`, `enrolled`, `employed` | Unit rows sampled from a feasible binary-IV observational law (`partialid.problems.binary_iv`, seed `2`), remapped `Z,X,Y → offer,enrolled,employed`. Row sample seed `7`. |
| `job_training_panel.csv` | 5000 | `site`, `age_bin`, `educ`, `prior_emp`, `urban`, `spouse`, `treated`, `employed` | Logistic SCM with shared latent `U` driving selection into treatment and the outcome (confounding), plus covariate imbalance across sites. Seed `11`. |

## Column meanings (vignettes)

**Study A / C — job training**

- `treated` — program participation (0/1)
- `employed` — employment at follow-up (0/1)
- `site` — study site / region (0,1,2)
- `age_bin` — age group (0,1,2)
- `educ` — high-school or above (0/1)
- `prior_emp` — employed in prior year (0/1)
- `urban` — urban residence (0/1)
- `spouse` — partner employed (0/1)

**Study B — encouragement trial**

- `offer` — randomized offer of training (instrument)
- `enrolled` — take-up / enrollment (treatment)
- `employed` — employment at follow-up (outcome)
