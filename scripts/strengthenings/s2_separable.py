"""Separable coupled programs: exact bounds by vertex enumeration, and a
concavity test showing that no fixed-matrix LP represents them.

Separable = the leftover factors q1, q2 live in separate c-factors, so the
feasible set is a product F1 x F2 of polytopes, but an atom of Q multiplies
them: Q = q1^T C q2. A bilinear function on a product of polytopes attains its
max/min at a pair of vertices, so
    max Q = max_{v vertex of F2} max_{q1 in F1} q1^T C v      (one LP per vertex).

Instances (binary):
  mediated PNS   X -> W -> Y, X <-> W, U_Y alone. F1: q over (r_X, r_W),
                 F2: q over r_Y with P(Y=1|W=w) fixed.
  two bows       X -> Y, X <-> Y  and  Z -> W, Z <-> W (disconnected),
                 Q = P(Y_1=1, W_1=1) = P(Y_1=1) P(W_1=1).
"""

from __future__ import annotations

import itertools

import numpy as np
from common import MAPS, lp

BOW = [(rx, ry) for rx in (0, 1) for ry in MAPS]  # bow types: X = rx, Y = ry(X)


def bow_constraints(Pxy):
    """sum q(rx, ry) 1[rx=x] 1[ry(x)=y] = P(x, y)."""
    A = [[1.0 if rx == x and ry[x] == y else 0.0 for rx, ry in BOW]
         for x, y in itertools.product((0, 1), repeat=2)]
    return np.array(A), np.array([Pxy[x, y] for x, y in itertools.product((0, 1), repeat=2)])


def kernel_constraints(py1):
    """q over r_Y in MAPS with sum q 1[r(w)=1] = P(Y=1|W=w), sum q = 1."""
    A = [[1.0 if r[w] == 1 else 0.0 for r in MAPS] for w in (0, 1)] + [[1.0] * 4]
    return np.array(A), np.array([py1[0], py1[1], 1.0])


def vertices(A, b, tol=1e-12):
    """All basic feasible solutions of {z >= 0 : A z = b} (small instances)."""
    m, n = A.shape
    rank = np.linalg.matrix_rank(A)
    out = []
    for cols in itertools.combinations(range(n), rank):
        sub = A[:, cols]
        if np.linalg.matrix_rank(sub) < rank:
            continue
        sol, *_ = np.linalg.lstsq(sub, b, rcond=None)
        if np.abs(sub @ sol - b).max() > 1e-9 or sol.min() < -tol:
            continue
        z = np.zeros(n)
        z[list(cols)] = np.clip(sol, 0, None)
        if not any(np.allclose(z, v) for v in out):
            out.append(z)
    return out


def separable_bounds(A1, b1, A2, b2, C):
    lo, hi = np.inf, -np.inf
    for v in vertices(A2, b2):
        c = C @ v
        mx, _ = lp(c, A1, b1, sense="max")
        mn, _ = lp(c, A1, b1, sense="min")
        if mx is None:
            return None
        lo, hi = min(lo, mn), max(hi, mx)
    return lo, hi


# ---------------------------------------------------------------- mediated PNS
C_MED = np.array([[float(ry[rw[1]] == 1 and ry[rw[0]] == 0) for ry in MAPS] for rx, rw in BOW])
C_MED_YX = np.array([[float(ry[rw[1]] == 1) for ry in MAPS] for rx, rw in BOW])


def mediated(Pxw, py1, query="PNS"):
    A1, b1 = bow_constraints(Pxw)  # (X, W) bow; types (r_X, r_W)
    A2, b2 = kernel_constraints(py1)
    return separable_bounds(A1, b1, A2, b2, C_MED if query == "PNS" else C_MED_YX)


def free_joint_mediated(Pxw, py1):
    """Free joint over (r_X, r_W, r_Y): drops U_Y _||_ U (outer bound)."""
    types = [(rx, rw, ry) for rx, rw in BOW for ry in MAPS]
    A, b = [], []
    for x, w, y in itertools.product((0, 1), repeat=3):
        A.append([1.0 if rx == x and rw[x] == w and ry[w] == y else 0.0 for rx, rw, ry in types])
        b.append(Pxw[x, w] * (py1[w] if y == 1 else 1 - py1[w]))
    c = [float(ry[rw[1]] == 1 and ry[rw[0]] == 0) for rx, rw, ry in types]
    return lp(c, np.array(A), np.array(b), sense="min")[0], lp(c, np.array(A), np.array(b))[0]


# -------------------------------------------------------------- two bows
C_TWO = np.array([[float(ry[1] == 1 and rw[1] == 1) for rz, rw in BOW] for rx, ry in BOW])


def two_bows(Pxy, Pzw):
    A1, b1 = bow_constraints(Pxy)
    A2, b2 = bow_constraints(Pzw)
    return separable_bounds(A1, b1, A2, b2, C_TWO)


# ----------------------------------------------------- concavity test helpers
def rand_joint(rng):
    p = rng.dirichlet(np.ones(4))
    return {k: p[i] for i, k in enumerate(itertools.product((0, 1), repeat=2))}


def mix(a, b, t):
    if isinstance(a, dict):
        return {k: (1 - t) * a[k] + t * b[k] for k in a}
    return (1 - t) * np.asarray(a) + t * np.asarray(b)


def concavity_scan(f, draw, n, rng):
    """max over segments of chord - midpoint (upper; >0 violates concavity) and
    midpoint - chord (lower; >0 violates convexity)."""
    worst_u, worst_l, arg_u, arg_l = 0.0, 0.0, None, None
    for _ in range(n):
        th0, th1 = draw(rng), draw(rng)
        (l0, u0), (l1, u1), (lm, um) = f(th0), f(th1), f(mix_theta(th0, th1, 0.5))
        vu, vl = (u0 + u1) / 2 - um, lm - (l0 + l1) / 2
        if vu > worst_u:
            worst_u, arg_u = vu, (th0, th1, u0, um, u1)
        if vl > worst_l:
            worst_l, arg_l = vl, (th0, th1, l0, lm, l1)
    return worst_u, worst_l, arg_u, arg_l


def mix_theta(a, b, t):
    return tuple(mix(x, y, t) for x, y in zip(a, b))


def fixed_margin_direction(rng, eps):
    """Perturbation of a 2x2 joint keeping both margins fixed."""
    d = eps * rng.uniform(-1, 1)
    return {(0, 0): d, (0, 1): -d, (1, 0): -d, (1, 1): d}


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    # Compatible mediated table (re3): P(x,w) = margin of (eq:re2), P(Y=1|W)=(0.4, 0.6).
    Pxw = {(0, 0): 0.25, (0, 1): 0.25, (1, 0): 0.15, (1, 1): 0.35}
    py1 = (0.4, 0.6)
    print("mediated graph on re3 = P(x,w) * P(y|w), P(Y=1|W) = (0.4, 0.6):")
    print("  table:", {(x, w, y): round(Pxw[x, w] * (py1[w] if y else 1 - py1[w]), 4)
                       for x, w, y in itertools.product((0, 1), repeat=3)})
    print("  P(Y_1=1) sharp (LP, degree 1):   [%.4f, %.4f]" % mediated(Pxw, py1, "Yx"))
    print("  PNS sharp (vertex enumeration):  [%.4f, %.4f]" % mediated(Pxw, py1))
    print("  PNS free joint (outer):          [%.4f, %.4f]" % free_joint_mediated(Pxw, py1))

    # Concavity on unrestricted segments of the c-factors (P(x,w), P(Y=1|w)).
    f = lambda th: mediated(th[0], th[1])
    draw = lambda r: (rand_joint(r), r.uniform(0.05, 0.95, 2))
    wu, wl, au, al = concavity_scan(f, draw, 400, rng)
    print(f"\nmediated PNS, random c-factor segments: max concavity violation of upper "
          f"{wu:.4f}, of convexity of lower {wl:.4f}")
    if au:
        th0, th1, u0, um, u1 = au
        print("  upper example: u(0)=%.4f u(1/2)=%.4f u(1)=%.4f" % (u0, um, u1))

    # Same, but along segments that keep P(X), P(W), P(Y) fixed.
    def draw_fixed(r):
        base = (Pxw, np.array(py1))
        d = fixed_margin_direction(r, 0.1)
        e = r.uniform(-0.15, 0.15)
        pw = (Pxw[0, 0] + Pxw[1, 0], Pxw[0, 1] + Pxw[1, 1])
        dy = np.array([e, -e * pw[0] / pw[1]])  # keeps P(Y=1) = sum_w P(w) P(Y=1|w)
        return base, (d, dy)

    worst = 0.0
    best = None
    for _ in range(400):
        (P0, y0), (d, dy) = draw_fixed(rng)
        Pa = {k: P0[k] - d[k] for k in P0}
        Pb = {k: P0[k] + d[k] for k in P0}
        if min(Pa.values()) <= 0 or min(Pb.values()) <= 0:
            continue
        ua, ub, um = mediated(Pa, y0 - dy)[1], mediated(Pb, y0 + dy)[1], mediated(P0, y0)[1]
        v = (ua + ub) / 2 - um
        if v > worst:
            worst, best = v, (ua, um, ub, d[(0, 0)], dy)
    print(f"mediated PNS, segments with P(X), P(W), P(Y) fixed: max concavity violation {worst:.4f}")
    if best:
        print("  u(-) = %.4f, u(0) = %.4f, u(+) = %.4f (d00=%.4f, dP(Y=1|W)=%s)"
              % (best[0], best[1], best[2], best[3], np.round(best[4], 4)))

    # Two bows: analytic u = u1 * u2; numeric check and concavity violation.
    print("\ntwo bows, Q = P(Y_1=1, W_1=1):")
    Pxy = {(0, 0): 0.30, (0, 1): 0.20, (1, 0): 0.15, (1, 1): 0.35}
    print("  on (eq:re1) x (eq:re1): [%.4f, %.4f]   (Manski product: [%.4f, %.4f])"
          % (*two_bows(Pxy, Pxy), 0.35 ** 2, 0.85 ** 2))
    for s in (+1, -1):
        d = {(0, 0): 0.0, (0, 1): 0.0, (1, 0): 0.1, (1, 1): -0.1}
        A = {k: Pxy[k] - d[k] for k in Pxy}
        B = {k: Pxy[k] + d[k] for k in Pxy}
        A2 = {k: Pxy[k] - s * d[k] for k in Pxy}
        B2 = {k: Pxy[k] + s * d[k] for k in Pxy}
        lo_a, up_a = two_bows(A, A2)
        lo_b, up_b = two_bows(B, B2)
        lo_m, up_m = two_bows(Pxy, Pxy)
        print(f"  segment sign {s:+d}: upper (end, mid, end) = ({up_a:.4f}, {up_m:.4f}, {up_b:.4f}) "
              f"chord-mid = {(up_a + up_b) / 2 - up_m:+.4f};  lower = ({lo_a:.4f}, {lo_m:.4f}, "
              f"{lo_b:.4f}) mid-chord = {lo_m - (lo_a + lo_b) / 2:+.4f}")
