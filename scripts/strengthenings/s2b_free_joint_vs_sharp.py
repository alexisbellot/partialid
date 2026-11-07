"""Is the free-joint LP (one simplex over all response types, RHS = joint P)
sharp on the separable coupled instances? Compared with the exact vertex
enumeration of s2_separable on random compatible laws.
"""

from __future__ import annotations

import itertools

import numpy as np
from common import MAPS, lp
from s2_separable import BOW, free_joint_mediated, mediated, rand_joint, two_bows


def free_joint_two_bows(Pxy, Pzw):
    types = [(a, b) for a in BOW for b in BOW]
    A, bvec = [], []
    for x, y, z, w in itertools.product((0, 1), repeat=4):
        A.append([1.0 if rx == x and ry[x] == y and rz == z and rw[z] == w else 0.0
                  for (rx, ry), (rz, rw) in types])
        bvec.append(Pxy[x, y] * Pzw[z, w])
    c = [float(ry[1] == 1 and rw[1] == 1) for (rx, ry), (rz, rw) in types]
    A, bvec = np.array(A), np.array(bvec)
    return lp(c, A, bvec, sense="min")[0], lp(c, A, bvec)[0]


if __name__ == "__main__":
    rng = np.random.default_rng(1)
    n = 200
    gm, gb = [], []
    for _ in range(n):
        Pxw, py1 = rand_joint(rng), rng.uniform(0.02, 0.98, 2)
        s, f = mediated(Pxw, py1), free_joint_mediated(Pxw, py1)
        gm.append((s[0] - f[0]) + (f[1] - s[1]))
        P1, P2 = rand_joint(rng), rand_joint(rng)
        s, f = two_bows(P1, P2), free_joint_two_bows(P1, P2)
        gb.append((s[0] - f[0]) + (f[1] - s[1]))
    for name, g in (("mediated PNS", gm), ("two-bow conjunction", gb)):
        g = np.array(g)
        print(f"{name:20s} free joint - sharp: max {g.max():.2e}, "
              f"#laws with gap > 1e-7: {(g > 1e-7).sum()}/{n}")
