"""How often does U1 _||_ U2 tighten the pair chain / IV-with-Z<->X beyond the free joint?

Laws are drawn from random product SCMs, so they are compatible with the model.
Reports the fraction of laws with a gap > 1e-6 and the largest gap found, and
prints one law (rounded to 3 decimals, renormalized) exhibiting a gap.
"""

from __future__ import annotations

import itertools

import numpy as np
from common import interval
from s1_lambda2_lift import MAPS, indep_constraints, obs_constraints, query_vec

Gs = list(itertools.product(MAPS, repeat=len(MAPS)))


def random_law(rng, alpha):
    p1 = rng.dirichlet(alpha * np.ones(2 * len(Gs))).reshape(2, len(Gs))
    p2 = rng.dirichlet(np.ones(len(MAPS)))
    P = {k: 0.0 for k in itertools.product((0, 1), repeat=3)}
    for a in (0, 1):
        for g, G in enumerate(Gs):
            for j, rb in enumerate(MAPS):
                m = G[j][a]
                P[(a, m, rb[m])] += p1[a, g] * p2[j]
    return P


def gaps(P, kind):
    A_obs, b_obs = obs_constraints(P)
    A_ind, b_ind = indep_constraints(P)
    c = query_vec(kind)
    flo, fhi, _, _ = interval(c, A_obs, b_obs)
    lo, hi, _, _ = interval(c, np.vstack([A_obs, A_ind]), np.concatenate([b_obs, b_ind]))
    return (lo - flo) + (fhi - hi), (flo, fhi, lo, hi)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    for kind in ("pair_Yx", "pair_PNS", "iv_Yx", "iv_PNS"):
        n, hit, best = 300, 0, (0.0, None, None)
        for _ in range(n):
            P = random_law(rng, alpha=rng.choice([0.02, 0.1, 1.0]))
            g, iv = gaps(P, kind)
            hit += g > 1e-6
            if g > best[0]:
                best = (g, P, iv)
        print(f"{kind:9s} gap>0 on {hit}/{n} laws; largest total gap {best[0]:.4f}")
        if best[1] is not None and kind == "pair_Yx":
            P = {k: round(v, 3) for k, v in best[1].items()}
            s = sum(P.values())
            P = {k: v / s for k, v in P.items()}
            g, (flo, fhi, lo, hi) = gaps(P, kind)
            print("   example law P(x,w,y):", {k: round(v, 4) for k, v in P.items()})
            print(f"   free joint [{flo:.4f}, {fhi:.4f}]  sharp [{lo:.4f}, {hi:.4f}]")
