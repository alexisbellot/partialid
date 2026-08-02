#!/usr/bin/env python3
"""Regenerate synthetic CSVs under notebooks/data/ for the tutorial curriculum.

Run from the repo root:

    PYTHONPATH=src python scripts/generate_notebook_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks" / "data"

# Classic RE1 margins used throughout docs / examples.
RE1 = {
    (0, 0): 0.30,
    (0, 1): 0.20,
    (1, 0): 0.15,
    (1, 1): 0.35,
}


def _sample_from_table(
    columns: list[str],
    probs: dict[tuple, float],
    n: int,
    *,
    seed: int,
) -> pd.DataFrame:
    keys = list(probs.keys())
    p = np.array([probs[k] for k in keys], dtype=float)
    p = p / p.sum()
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(keys), size=n, p=p)
    rows = [keys[i] for i in idx]
    return pd.DataFrame(rows, columns=columns)


def write_job_training() -> None:
    df = _sample_from_table(["treated", "employed"], RE1, n=2000, seed=42)
    df.to_csv(OUT / "job_training.csv", index=False)
    print(f"wrote job_training.csv  n={len(df)}")


def write_encouragement_trial() -> None:
    import sys

    sys.path.insert(0, str(ROOT / "src"))
    from partialid.problems import binary_iv

    p = binary_iv(seed=2)
    # Remap Z,X,Y → offer, enrolled, employed
    remapped = {
        (z, x, y): float(v)
        for (z, x, y), v in p.data.items()
    }
    df = _sample_from_table(
        ["offer", "enrolled", "employed"],
        remapped,
        n=3000,
        seed=7,
    )
    df.to_csv(OUT / "encouragement_trial.csv", index=False)
    print(f"wrote encouragement_trial.csv  n={len(df)}")


def write_job_training_panel() -> None:
    """Eight-column panel with confounding and covariate imbalance."""
    rng = np.random.default_rng(11)
    n = 5000

    site = rng.integers(0, 3, size=n)
    age_bin = rng.integers(0, 3, size=n)
    educ = (rng.random(n) < (0.35 + 0.1 * (site == 2))).astype(int)
    prior_emp = (rng.random(n) < (0.40 + 0.15 * educ + 0.05 * (age_bin == 2))).astype(int)
    urban = (rng.random(n) < (0.55 + 0.1 * (site == 0))).astype(int)
    spouse = (rng.random(n) < (0.45 + 0.1 * prior_emp)).astype(int)

    # Latent propensity / motivation correlates with covariates and drives treatment + outcome.
    u = rng.normal(size=n)
    logit_t = (
        -0.8
        + 0.6 * educ
        + 0.5 * prior_emp
        + 0.3 * urban
        + 0.2 * (age_bin >= 1)
        + 0.4 * u
        + 0.15 * (site == 1)
    )
    p_t = 1.0 / (1.0 + np.exp(-logit_t))
    treated = (rng.random(n) < p_t).astype(int)

    logit_y = (
        -0.5
        + 0.7 * treated
        + 0.5 * educ
        + 0.6 * prior_emp
        + 0.2 * spouse
        + 0.15 * urban
        + 0.5 * u
        + 0.1 * (age_bin == 2)
    )
    p_y = 1.0 / (1.0 + np.exp(-logit_y))
    employed = (rng.random(n) < p_y).astype(int)

    df = pd.DataFrame(
        {
            "site": site,
            "age_bin": age_bin,
            "educ": educ,
            "prior_emp": prior_emp,
            "urban": urban,
            "spouse": spouse,
            "treated": treated,
            "employed": employed,
        }
    )
    df.to_csv(OUT / "job_training_panel.csv", index=False)
    print(f"wrote job_training_panel.csv  n={len(df)}")
    print("  treated rate:", float(df["treated"].mean()))
    print("  employed rate:", float(df["employed"].mean()))
    print(
        "  employed | treated=1:",
        float(df.loc[df["treated"] == 1, "employed"].mean()),
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    write_job_training()
    write_encouragement_trial()
    write_job_training_panel()


if __name__ == "__main__":
    main()
