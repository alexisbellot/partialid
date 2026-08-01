"""Practitioner data ingress: counts, records, and optional DataFrames."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from .variables import Variables


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "from_dataframe requires pandas. Install with: pip install partialid"
        ) from e
    return pd


def _coerce_cell(values: Sequence[Any]) -> tuple:
    return tuple(values)


def _infer_domains(
    columns: Sequence[str],
    cells: Iterable[tuple],
    domains: Mapping[str, Sequence] | None,
) -> dict[str, tuple]:
    if domains is not None:
        missing = [c for c in columns if c not in domains]
        if missing:
            raise ValueError(f"domains missing columns: {missing}")
        out = {c: tuple(domains[c]) for c in columns}
        return out
    buckets: dict[str, set] = {c: set() for c in columns}
    for cell in cells:
        if len(cell) != len(columns):
            raise ValueError(f"cell {cell!r} has length {len(cell)}, expected {len(columns)}")
        for name, val in zip(columns, cell, strict=True):
            buckets[name].add(val)
    out = {}
    for c in columns:
        vals = buckets[c]
        if not vals:
            raise ValueError(f"no observed values for column {c!r}")
        try:
            out[c] = tuple(sorted(vals))
        except TypeError:
            out[c] = tuple(sorted(vals, key=lambda x: (str(type(x)), str(x))))
    return out


@dataclass(frozen=True)
class ObservationalData:
    """Discrete observational law ready for ``bound`` / ``bound_path``.

    Attributes
    ----------
    variables :
        Domains for each column (declaration order = ``columns``).
    P :
        Probability mass keyed by tuples in ``columns`` order (sums to 1).
    columns :
        Variable names matching keys of ``P``.
    n_obs :
        Number of raw rows if built from records / a DataFrame; ``None`` for
        an already-aggregated probability table.
    notes :
        Normalization / coercion messages for the user.
    """

    variables: Variables
    P: dict[tuple, float]
    columns: tuple[str, ...]
    n_obs: int | None = None
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if tuple(self.variables.names) != self.columns:
            raise ValueError("variables.names must match columns order")
        s = float(sum(self.P.values()))
        if abs(s - 1.0) > 1e-6:
            raise ValueError(
                f"ObservationalData.P must be a probability table (sum={s}). "
                "Use from_counts(..., normalize=True) or from_records."
            )

    @property
    def mass(self) -> float:
        return float(sum(self.P.values()))


def from_counts(
    columns: Sequence[str],
    counts: Mapping[tuple, float],
    *,
    domains: Mapping[str, Sequence] | None = None,
    normalize: bool = True,
    tol: float = 1e-8,
) -> ObservationalData:
    """Build ``(Variables, P)`` from an aggregated contingency table.

    Parameters
    ----------
    columns :
        Names for each position in the count keys (e.g. ``(\"X\", \"Y\")``).
    counts :
        Mapping from value tuples to non-negative weights (counts or masses).
    domains :
        Optional explicit domains per column; otherwise inferred from keys.
    normalize :
        If True (default), renormalize so ``P`` sums to 1. If False, require
        the weights to already sum to approximately 1.
    tol :
        Absolute tolerance when checking whether the table already sums to 1.
    """
    cols = tuple(columns)
    if not cols:
        raise ValueError("columns must be non-empty")
    if not counts:
        raise ValueError("counts is empty")

    clean: dict[tuple, float] = {}
    for key, val in counts.items():
        cell = _coerce_cell(key if not isinstance(key, dict) else tuple(key[c] for c in cols))
        if len(cell) != len(cols):
            raise ValueError(
                f"count key {key!r} has length {len(cell)}, expected {len(cols)} for columns {cols}"
            )
        v = float(val)
        if v < 0:
            raise ValueError(f"negative weight {v} for cell {cell}")
        clean[cell] = clean.get(cell, 0.0) + v

    total = float(sum(clean.values()))
    if total <= 0:
        raise ValueError("counts must sum to a positive value")

    notes: list[str] = []
    if abs(total - 1.0) <= tol:
        if abs(total - 1.0) > 0:
            P = {k: v / total for k, v in clean.items()}
            notes.append(f"renormalized mass {total:.6g} → 1")
        else:
            P = {k: float(v) for k, v in clean.items()}
    elif normalize:
        P = {k: v / total for k, v in clean.items()}
        kind = "counts" if total > 1.0 + tol else "weights"
        notes.append(f"normalized {kind} (sum={total:.6g}) to probabilities")
    else:
        raise ValueError(
            f"probabilities sum to {total:.6g}, not 1 "
            f"(tolerance {tol}). Pass normalize=True to renormalize, "
            "or fix the table."
        )

    dom = _infer_domains(cols, clean.keys(), domains)
    # Ensure every domain cell exists in P with zero if needed? Not required —
    # formulate fills missing with 0. Keep only positive mass keys.
    V = Variables.from_mapping(dom)
    return ObservationalData(variables=V, P=P, columns=cols, n_obs=None, notes=tuple(notes))


def from_records(
    columns: Sequence[str],
    records: Sequence[Sequence[Any]],
    *,
    domains: Mapping[str, Sequence] | None = None,
) -> ObservationalData:
    """Build an empirical law from unit-level discrete rows (no pandas)."""
    cols = tuple(columns)
    if not cols:
        raise ValueError("columns must be non-empty")
    if len(records) == 0:
        raise ValueError("records is empty")

    cells: list[tuple] = []
    for i, row in enumerate(records):
        if len(row) != len(cols):
            raise ValueError(f"row {i} has length {len(row)}, expected {len(cols)} for {cols}")
        cells.append(_coerce_cell(row))

    ctr = Counter(cells)
    return from_counts(cols, ctr, domains=domains, normalize=True)


def from_dataframe(
    df,
    columns: Sequence[str] | None = None,
    *,
    domains: Mapping[str, Sequence] | None = None,
) -> ObservationalData:
    """Build an empirical law from a pandas DataFrame of discrete columns.

    Requires pandas (installed with ``partialid``).
    """
    pd = _require_pandas()
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"df must be a pandas DataFrame, got {type(df)!r}")
    cols = tuple(columns) if columns is not None else tuple(df.columns)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"dataframe missing columns: {missing}")
    if df.empty:
        raise ValueError("dataframe has no rows")

    sub = df.loc[:, list(cols)]
    if sub.isna().any().any():
        bad = [c for c in cols if sub[c].isna().any()]
        raise ValueError(
            f"dataframe has missing values in {bad}; drop or impute before calling from_dataframe"
        )

    records = [tuple(row) for row in sub.itertuples(index=False, name=None)]
    data = from_records(cols, records, domains=domains)
    return ObservationalData(
        variables=data.variables,
        P=data.P,
        columns=data.columns,
        n_obs=len(records),
        notes=data.notes + (f"from_dataframe n_obs={len(records)}",),
    )
