"""Transcribed symbolic / oracle bound formulas (causaloptim-style, no R).

- Manski / natural bounds: closed form
- Binary IV: Balke–Pearl polytope evaluated as a tiny explicit LP (same
  numeric object ``causaloptim`` emits symbolically for Z→X→Y with X↔Y)
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
from scipy.optimize import linprog


def manski_binary_xy(
    P: Mapping[tuple, float],
    *,
    do_x: int = 1,
    y: int = 1,
) -> tuple[float, float]:
    """Natural / Manski bounds for P(Y=y | do(X=do_x)) under max confounding."""
    lo = sum(v for (x, yy), v in P.items() if x == do_x and yy == y)
    hi = lo + sum(v for (x, yy), v in P.items() if x != do_x)
    return float(lo), float(hi)


def manski_exposure(
    P: Mapping[tuple, float],
    *,
    do_x: object,
    y: object = 1,
    x_index: int = 0,
    y_index: int = 1,
) -> tuple[float, float]:
    """Manski for categorical exposure (categorical exposure)."""
    lo = sum(v for cell, v in P.items() if cell[x_index] == do_x and cell[y_index] == y)
    hi = lo + sum(v for cell, v in P.items() if cell[x_index] != do_x)
    return float(lo), float(hi)


def balke_pearl_iv(P: Mapping[tuple, float]) -> tuple[float, float]:
    """Balke–Pearl bounds for P(Y=1 | do(X=1)) on binary IV Z→X→Y, X↔Y.

    Implements the canonical 16-type response LP (4×4) with conditional
    moment constraints — the numeric evaluation of the symbolic program
    that ``causaloptim`` derives for this graph (no R runtime).
    Cells keyed ``(z, x, y)``.
    """

    # Response types: rX ∈ {0,1,2,3} maps Z→X; rY ∈ {0,1,2,3} maps X→Y.
    # Index: bit0 = f(0), bit1 = f(1) for binary domain.
    def fn_val(code: int, inp: int) -> int:
        return (code >> inp) & 1

    # Conditional P(X=x,Y=y | Z=z)
    def g(z: int, x: int, y: int) -> float:
        pz = sum(float(P.get((z, xx, yy), 0.0)) for xx in (0, 1) for yy in (0, 1))
        if pz <= 0:
            return 0.0
        return float(P.get((z, x, y), 0.0)) / pz

    # Variables q[rx + 4*ry], rx,ry in 0..3
    n = 16
    # Equalities: for each z,x,y: sum_{types consistent} q = g(z,x,y)
    rows = []
    rhs = []
    for z in (0, 1):
        for x in (0, 1):
            for y in (0, 1):
                row = np.zeros(n)
                for rx in range(4):
                    if fn_val(rx, z) != x:
                        continue
                    for ry in range(4):
                        if fn_val(ry, x) != y:
                            continue
                        row[rx + 4 * ry] = 1.0
                rows.append(row)
                rhs.append(g(z, x, y))
    A_eq = np.vstack(rows)
    b_eq = np.asarray(rhs)
    # Objective: P(Y=1 | do(X=1)) = sum q over types with g_Y(1)=1
    c = np.zeros(n)
    for rx in range(4):
        for ry in range(4):
            if fn_val(ry, 1) == 1:
                c[rx + 4 * ry] = 1.0

    def _opt(maximize: bool) -> float:
        cc = -c if maximize else c
        res = linprog(cc, A_eq=A_eq, b_eq=b_eq, bounds=[(0, None)] * n, method="highs")
        if not res.success:
            return float("nan")
        return float(-res.fun if maximize else res.fun)

    return _opt(False), _opt(True)


RE1_P = {(0, 0): 0.30, (0, 1): 0.20, (1, 0): 0.15, (1, 1): 0.35}
