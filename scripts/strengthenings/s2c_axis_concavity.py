"""Mediated PNS: concavity of the sharp upper bound along segments that move one
c-factor only. With P(x,w) fixed the joint P(x,w,y) = P(x,w) P(y|w) is affine
in P(Y=1|W); with P(Y=1|W) fixed it is affine in P(x,w). Such a segment is a
segment both in c-factor and in joint-law coordinates, so a violation rules out
an LP whose right-hand side is affine in either.
"""

from __future__ import annotations

import numpy as np
from s2_separable import mediated, rand_joint

if __name__ == "__main__":
    rng = np.random.default_rng(2)
    best = {"move P(y|w)": (0, None), "move P(x,w)": (0, None)}
    for _ in range(600):
        Pxw, Pxw2 = rand_joint(rng), rand_joint(rng)
        y0, y1 = rng.uniform(0.02, 0.98, 2), rng.uniform(0.02, 0.98, 2)
        u = lambda P, y: mediated(P, y)[1]
        # move P(y|w) only
        v = (u(Pxw, y0) + u(Pxw, y1)) / 2 - u(Pxw, (y0 + y1) / 2)
        if v > best["move P(y|w)"][0]:
            best["move P(y|w)"] = (v, (Pxw, y0, y1))
        # move P(x,w) only
        Pm = {k: (Pxw[k] + Pxw2[k]) / 2 for k in Pxw}
        v = (u(Pxw, y0) + u(Pxw2, y0)) / 2 - u(Pm, y0)
        if v > best["move P(x,w)"][0]:
            best["move P(x,w)"] = (v, (Pxw, Pxw2, y0))
    for k, (v, arg) in best.items():
        print(f"{k}: max (chord - midpoint) of upper bound = {v:.4f}")

    # A clean rounded witness for the write-up (move P(y|w) only).
    Pxw = {(0, 0): 0.25, (0, 1): 0.25, (1, 0): 0.15, (1, 1): 0.35}
    print("\nexplicit witness, P(x,w) = margin of (eq:re2):")
    grid = [(a, b) for a in np.linspace(0.1, 0.9, 9) for b in np.linspace(0.1, 0.9, 9)]
    top = (0, None)
    for ya in grid:
        for yb in grid:
            ya_, yb_ = np.array(ya), np.array(yb)
            v = (mediated(Pxw, ya_)[1] + mediated(Pxw, yb_)[1]) / 2 - mediated(Pxw, (ya_ + yb_) / 2)[1]
            if v > top[0] + 1e-12:
                top = (v, (ya_, yb_))
    v, (ya, yb) = top
    print(f"  P(Y=1|W) endpoints {np.round(ya, 2)} and {np.round(yb, 2)}, midpoint {np.round((ya + yb) / 2, 2)}")
    print("  upper bounds: %.4f, %.4f (mid), %.4f; chord - mid = %.4f"
          % (mediated(Pxw, ya)[1], mediated(Pxw, (ya + yb) / 2)[1], mediated(Pxw, yb)[1], v))
