"""Discrete endogenous variables and their domains."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class Variables:
    """Ordered collection of discrete endogenous variables.

    The declaration order is *not* a causal order; pass a causal order
    (or diagram) via assumptions.

    ``observed`` lists variables that appear in the observational law ``P``.
    Endogenous nodes missing from ``observed`` are *latent* (e.g. true
    outcome under measurement error; Sachs et al. 2022 §6.3): they still
    have response types and may appear in queries, but do not index ``P``.
    """

    names: tuple[str, ...]
    domains: dict[str, tuple]
    observed: frozenset[str] | None = None  # None → all names observed

    def __post_init__(self) -> None:
        if len(self.names) != len(set(self.names)):
            raise ValueError("variable names must be unique")
        for n in self.names:
            if n not in self.domains:
                raise ValueError(f"missing domain for {n!r}")
            if len(self.domains[n]) == 0:
                raise ValueError(f"empty domain for {n!r}")
        if self.observed is not None:
            bad = set(self.observed) - set(self.names)
            if bad:
                raise ValueError(f"observed unknowns: {sorted(bad)}")
            object.__setattr__(self, "observed", frozenset(self.observed))

    @classmethod
    def from_binary(cls, *names: str, observed: Sequence[str] | None = None) -> Variables:
        obs = frozenset(observed) if observed is not None else None
        return cls(
            names=tuple(names),
            domains={n: (0, 1) for n in names},
            observed=obs,
        )

    @classmethod
    def from_mapping(
        cls,
        domains: Mapping[str, Sequence],
        *,
        observed: Sequence[str] | None = None,
    ) -> Variables:
        names = tuple(domains.keys())
        obs = frozenset(observed) if observed is not None else None
        return cls(
            names=names,
            domains={k: tuple(v) for k, v in domains.items()},
            observed=obs,
        )

    def is_observed(self, name: str) -> bool:
        if self.observed is None:
            return True
        return name in self.observed

    def observed_names(self, order: Sequence[str] | None = None) -> tuple[str, ...]:
        """Observed variables, optionally filtered/ordered by ``order``."""
        seq = list(order) if order is not None else list(self.names)
        return tuple(n for n in seq if self.is_observed(n))

    def __len__(self) -> int:
        return len(self.names)

    def __contains__(self, name: str) -> bool:
        return name in self.domains

    def size(self, name: str) -> int:
        return len(self.domains[name])

    def index(self, name: str) -> int:
        return self.names.index(name)

    def cells(self, names: Sequence[str] | None = None) -> list[tuple]:
        """Cartesian product of domains for ``names`` (default: all)."""
        keys = list(names) if names is not None else list(self.names)
        return list(product(*(self.domains[k] for k in keys)))

    def cell_index(self, cell: tuple, names: Sequence[str] | None = None) -> int:
        keys = list(names) if names is not None else list(self.names)
        cells = self.cells(keys)
        try:
            return cells.index(tuple(cell))
        except ValueError as e:
            raise KeyError(f"unknown cell {cell} for variables {keys}") from e

    def restrict(self, names: Iterable[str]) -> Variables:
        names_t = tuple(names)
        obs = None
        if self.observed is not None:
            obs = frozenset(n for n in names_t if n in self.observed)
        return Variables(
            names=names_t,
            domains={n: self.domains[n] for n in names_t},
            observed=obs,
        )
