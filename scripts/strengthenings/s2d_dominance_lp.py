"""Mediated PNS as an LP with a data-dependent objective (row-dominance lemma).

Q = sum_{i,j} q1_i C_ij q2_j. If some v* in F2 maximizes every row functional
e_i^T C v over F2 simultaneously, then max Q = max_{q1 in F1} q1^T C v*.
For mediated PNS the row functionals are t = P(r_Y = id) (rows with
W_1=1, W_0=0) and s = P(r_Y = flip) (rows with W_1=0, W_0=1), with
t - s = a1 - a0 fixed, so both are maximized at t = min(a1, 1-a0) and minimized
at t = max(0, a1-a0):
  upper = max_{q1} alpha(q1) min(a1, 1-a0) + beta(q1) min(a0, 1-a1)
  lower = min_{q1} alpha(q1) (a1-a0)_+     + beta(q1) (a0-a1)_+
with alpha = P(W_1=1, W_0=0), beta = P(W_1=0, W_0=1), a_w = P(Y=1|W=w).
"""

from __future__ import annotations

import numpy as np
from common import lp
from s2_separable import BOW, bow_constraints, mediated, rand_joint


def dominance_lp(Pxw, py1):
    a0, a1 = py1
    A, b = bow_constraints(Pxw)
    alpha = np.array([float(rw[1] == 1 and rw[0] == 0) for rx, rw in BOW])
    beta = np.array([float(rw[1] == 0 and rw[0] == 1) for rx, rw in BOW])
    up = lp(alpha * min(a1, 1 - a0) + beta * min(a0, 1 - a1), A, b, sense="max")[0]
    lo = lp(alpha * max(0, a1 - a0) + beta * max(0, a0 - a1), A, b, sense="min")[0]
    return lo, up


if __name__ == "__main__":
    rng = np.random.default_rng(3)
    err = 0.0
    for _ in range(300):
        Pxw, py1 = rand_joint(rng), rng.uniform(0.02, 0.98, 2)
        err = max(err, np.abs(np.subtract(dominance_lp(Pxw, py1), mediated(Pxw, py1))).max())
    print(f"dominance LP vs vertex enumeration, 300 random laws: max |diff| = {err:.2e}")
    Pxw = {(0, 0): 0.25, (0, 1): 0.25, (1, 0): 0.15, (1, 1): 0.35}
    print("re3: dominance LP PNS = [%.4f, %.4f]" % dominance_lp(Pxw, (0.4, 0.6)))
