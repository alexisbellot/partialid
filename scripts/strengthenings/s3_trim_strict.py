"""Is the ancestral trim lossless? The D-descendant example, certified.

Graph: X -> Y -> D, hyperedges {X,Y} and {X,D}. Query P(Y_1 = 1).
SCM: X = f_X(U1, U2), Y = f_Y(X, U1), D = f_D(Y, U2), U1 _||_ U2.
An(Y) = {X, Y}; dropping D leaves the bow, whose sharp bounds are Manski.

By Lemma S1 (S1 = {Y}, S12 = {X}, S2 = {D}) the model is exactly
    { mu(r_Y, x, r_D) >= 0 : sum_x mu(r_Y, x, r_D) = a(r_Y) nu(r_D) },
with a, nu the type laws of Y and D. For fixed nu this is an LP in (mu, a).
  inner value : the fixed-nu LP at the best nu found (a witness model);
  outer bound : branch and bound over boxes of nu with McCormick envelopes for
                w = a nu (valid LP relaxation on each box).
"""

from __future__ import annotations

import itertools

import numpy as np
from common import MAPS, lp

TYPES = [(ry, x, rd) for ry in MAPS for x in (0, 1) for rd in MAPS]
NT, NA, NN = len(TYPES), 4, 4
IDX = {t: i for i, t in enumerate(TYPES)}


def obs_rows(P):
    A, b = [], []
    for x, y, d in itertools.product((0, 1), repeat=3):
        row = np.zeros(NT)
        for (ry, tx, rd), i in IDX.items():
            if tx == x and ry[x] == y and rd[y] == d:
                row[i] = 1
        A.append(row)
        b.append(P[(x, y, d)])
    return np.array(A), np.array(b)


def qvec():
    return np.array([float(ry[1] == 1) for ry, x, rd in TYPES])


def fixed_nu(P, nu, sense):
    """Variables (mu, a). sum_x mu(ry,x,rd) - nu(rd) a(ry) = 0."""
    A_obs, b_obs = obs_rows(P)
    A = [np.concatenate([r, np.zeros(NA)]) for r in A_obs]
    b = list(b_obs)
    for i, ry in enumerate(MAPS):
        for j, rd in enumerate(MAPS):
            row = np.zeros(NT + NA)
            for x in (0, 1):
                row[IDX[(ry, x, rd)]] = 1
            row[NT + i] = -nu[j]
            A.append(row)
            b.append(0.0)
    c = np.concatenate([qvec(), np.zeros(NA)])
    return lp(c, np.array(A), np.array(b), sense=sense)[0]


def mccormick(P, lo, hi, sense):
    """Relaxation on the box lo <= nu <= hi. Variables (mu, a, nu, w)."""
    nv = NT + NA + NN + NA * NN
    ia, inu, iw = NT, NT + NA, NT + NA + NN
    A_obs, b_obs = obs_rows(P)
    A_eq = [np.concatenate([r, np.zeros(nv - NT)]) for r in A_obs]
    b_eq = list(b_obs)
    A_ub, b_ub = [], []

    def W(i, j):
        return iw + i * NN + j

    for i, ry in enumerate(MAPS):
        for j, rd in enumerate(MAPS):
            row = np.zeros(nv)
            for x in (0, 1):
                row[IDX[(ry, x, rd)]] = 1
            row[W(i, j)] = -1
            A_eq.append(row)
            b_eq.append(0.0)
            # w >= lo a ; w >= hi a + nu - hi ; w <= hi a ; w <= lo a + nu - lo
            for coef_a, coef_nu, const, sgn in ((lo[j], 0, 0, 1), (hi[j], 1, -hi[j], 1),
                                               (hi[j], 0, 0, -1), (lo[j], 1, -lo[j], -1)):
                r = np.zeros(nv)
                r[W(i, j)] = -sgn
                r[ia + i] = sgn * coef_a
                r[inu + j] = sgn * coef_nu
                A_ub.append(r)
                b_ub.append(-sgn * const)
    for i in range(NA):  # sum_j w_ij = a_i
        r = np.zeros(nv); r[[W(i, j) for j in range(NN)]] = 1; r[ia + i] = -1
        A_eq.append(r); b_eq.append(0.0)
    for j in range(NN):  # sum_i w_ij = nu_j, box on nu
        r = np.zeros(nv); r[[W(i, j) for i in range(NA)]] = 1; r[inu + j] = -1
        A_eq.append(r); b_eq.append(0.0)
        r = np.zeros(nv); r[inu + j] = 1; A_ub.append(r); b_ub.append(hi[j])
        r = np.zeros(nv); r[inu + j] = -1; A_ub.append(r); b_ub.append(-lo[j])
    r = np.zeros(nv); r[inu:inu + NN] = 1; A_eq.append(r); b_eq.append(1.0)
    c = np.zeros(nv); c[:NT] = qvec()
    val, z = lp(c, np.array(A_eq), np.array(b_eq), np.array(A_ub), np.array(b_ub), sense)
    return val, (None if z is None else z[inu:inu + NN])


def certified(P, sense, tol=1e-3, max_nodes=4000):
    """Returns (inner, outer): inner attained by a witness, outer a valid bound."""
    sgn = 1 if sense == "max" else -1
    best_val, best_nu = -np.inf, None
    for nu in itertools.product(np.linspace(0, 1, 11), repeat=3):  # coarse incumbent
        if sum(nu) <= 1 + 1e-12:
            nu4 = np.array([*nu, 1 - sum(nu)])
            v = fixed_nu(P, nu4, sense)
            if v is not None and sgn * v > best_val:
                best_val, best_nu = sgn * v, nu4
    stack, nodes, outer = [(np.zeros(NN), np.ones(NN))], 0, -np.inf
    while stack and nodes < max_nodes:
        lo, hi = stack.pop()
        nodes += 1
        ub, nu = mccormick(P, lo, hi, sense)
        if ub is None:
            continue
        ub *= sgn
        if nu is not None:
            nu = np.clip(nu, 0, None); nu /= nu.sum()
            v = fixed_nu(P, nu, sense)
            if v is not None and sgn * v > best_val:
                best_val, best_nu = sgn * v, nu
        if ub <= best_val + tol:
            outer = max(outer, ub)
            continue
        k = int(np.argmax(hi - lo))
        mid = (lo[k] + hi[k]) / 2
        h1, l2 = hi.copy(), lo.copy()
        h1[k], l2[k] = mid, mid
        stack += [(lo, h1), (l2, hi)]
    if stack:  # unfinished boxes: bound them by their relaxations
        for lo, hi in stack:
            ub, _ = mccormick(P, lo, hi, sense)
            if ub is not None:
                outer = max(outer, sgn * ub)
    return sgn * best_val, sgn * max(outer, best_val), nodes


def law_from_scm(seed):
    """Random product SCM with U1 = (r_Y, e1), U2 = (r_D, e2) and X depending on both."""
    rng = np.random.default_rng(seed)
    a, nu = rng.dirichlet(np.ones(4)), rng.dirichlet(np.ones(4))
    kx = rng.beta(0.3, 0.3, size=(4, 4))  # P(X=1 | r_Y, r_D)
    P = {k: 0.0 for k in itertools.product((0, 1), repeat=3)}
    for i, ry in enumerate(MAPS):
        for j, rd in enumerate(MAPS):
            for x in (0, 1):
                px = kx[i, j] if x else 1 - kx[i, j]
                y = ry[x]
                P[(x, y, rd[y])] += a[i] * nu[j] * px
    return P


def manski(P):
    p1 = sum(v for k, v in P.items() if k[0] == 1)
    p11 = P[(1, 1, 0)] + P[(1, 1, 1)]
    return p11, p11 + 1 - p1


if __name__ == "__main__":
    for seed in (0, 1, 2, 3, 4, 5):
        P = law_from_scm(seed)
        ml, mu_ = manski(P)
        li, lo_out, n1 = certified(P, "min")
        ui, up_out, n2 = certified(P, "max")
        print(f"seed {seed}: trimmed (Manski) [{ml:.4f}, {mu_:.4f}]   full model: "
              f"lower in [{lo_out:.4f}, {li:.4f}], upper in [{ui:.4f}, {up_out:.4f}]  "
              f"(nodes {n1}+{n2})")
