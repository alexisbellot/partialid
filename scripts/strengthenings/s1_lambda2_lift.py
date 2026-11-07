"""Two-hyperedge lifting LP (Proposition S1 in THEORY_STRENGTHENINGS.md).

District with hyperedges H1, H2. S1 = variables only in H1, S12 = in both,
S2 = only in H2. Lemma S1: the product model equals the set of joint type laws
mu(r_S1, r_S12, r_S2) with the single constraint r_S1 _||_ r_S2. When S1 is a
set of roots its type law is P(S1), so the constraint is linear:
    sum_{r12} mu(r1, r12, r2) = P(S1 = r1) * nu(r2),   nu(r2) = sum mu(., ., r2).

Instances (all binary):
  pair chain   X -> W -> Y,  {X,W}, {W,Y}:  S1={X}, S12={W}, S2={Y}
  IV with Z<->X  Z -> X -> Y,  {Z,X}, {X,Y}: S1={Z}, S12={X}, S2={Y}
Both have the same shape: a root A, a middle M with parent A, an end B with
parent M. Types: a in {0,1}, r_M, r_B in MAPS.

For each instance and query we report
  free joint (no independence)  >=  lifted LP  (sharp by Lemma S1),
and build an explicit product SCM (U1 _||_ U2) from the LP optimum whose
observational law and query value are recomputed from scratch.
"""

from __future__ import annotations

import itertools

import numpy as np
from common import MAPS, interval

TYPES = [(a, rm, rb) for a in (0, 1) for rm in MAPS for rb in MAPS]
IDX = {t: i for i, t in enumerate(TYPES)}


def obs_constraints(P):
    """P(a, m, b) = sum mu(a, rM, rB) 1[rM(a)=m] 1[rB(m)=b]."""
    A, b = [], []
    for a, m, bb in itertools.product((0, 1), repeat=3):
        row = np.zeros(len(TYPES))
        for (ta, rm, rb), i in IDX.items():
            if ta == a and rm[a] == m and rb[m] == bb:
                row[i] = 1
        A.append(row)
        b.append(P[(a, m, bb)])
    return np.array(A), np.array(b)


def indep_constraints(P):
    """sum_{rM} mu(a, rM, rB) - P(A=a) * sum_{a', rM} mu(a', rM, rB) = 0."""
    pa = {a: sum(v for k, v in P.items() if k[0] == a) for a in (0, 1)}
    A = []
    for a in (0, 1):
        for rb in MAPS:
            row = np.zeros(len(TYPES))
            for (ta, rm, trb), i in IDX.items():
                if trb == rb:
                    row[i] -= pa[a]
                    if ta == a:
                        row[i] += 1
            A.append(row)
    return np.array(A), np.zeros(len(A))


def query_vec(kind, x=1):
    """Queries on the end variable B under do(A=x) (pair chain: do(X)); here
    'A' is the intervened root for the pair chain and 'M' for the IV."""
    c = np.zeros(len(TYPES))
    for (a, rm, rb), i in IDX.items():
        if kind == "pair_Yx":       # P(Y_x = 1), Y_x = rB(rM(x))
            c[i] = rb[rm[x]] == 1
        elif kind == "pair_PNS":    # P(Y_1=1, Y_0=0)
            c[i] = rb[rm[1]] == 1 and rb[rm[0]] == 0
        elif kind == "iv_Yx":       # P(Y_x = 1), Y_x = rB(x)
            c[i] = rb[x] == 1
        elif kind == "iv_PNS":
            c[i] = rb[1] == 1 and rb[0] == 0
    return c


def witness(mu, P):
    """Explicit product SCM from a feasible mu (constructive proof of Lemma S1).

    U2 = r_B ~ nu.  U1 = (a, G), G : MAPS -> MAPS a random map whose coordinates
    G(r_B) are drawn independently from kappa(. | a, r_B) = mu(a,.,r_B)/(P(a) nu(r_B)).
    Then r_M := G(U2). Returns the joint law of (a, r_M, r_B) induced by
    p1 (x) p2, computed by summing over all 4^4 maps G."""
    pa = {a: sum(v for k, v in P.items() if k[0] == a) for a in (0, 1)}
    nu = {rb: sum(mu[IDX[(a, rm, rb)]] for a in (0, 1) for rm in MAPS) for rb in MAPS}
    kappa = {}
    for a in (0, 1):
        for rb in MAPS:
            den = pa[a] * nu[rb]
            for rm in MAPS:
                kappa[a, rm, rb] = mu[IDX[(a, rm, rb)]] / den if den > 1e-15 else 0.25
    joint = np.zeros(len(TYPES))
    for a in (0, 1):
        for G in itertools.product(MAPS, repeat=len(MAPS)):  # G[j] = image of MAPS[j]
            p1 = pa[a] * np.prod([kappa[a, G[j], rb] for j, rb in enumerate(MAPS)])
            if p1 == 0:
                continue
            for j, rb in enumerate(MAPS):
                joint[IDX[(a, G[j], rb)]] += p1 * nu[rb]
    return joint


def run(name, P, kinds):
    A_obs, b_obs = obs_constraints(P)
    A_ind, b_ind = indep_constraints(P)
    A_lift = np.vstack([A_obs, A_ind])
    b_lift = np.concatenate([b_obs, b_ind])
    print(f"== {name}")
    for kind in kinds:
        c = query_vec(kind)
        flo, fhi, _, _ = interval(c, A_obs, b_obs)
        lo, hi, xlo, xhi = interval(c, A_lift, b_lift)
        errs = []
        for z in (xlo, xhi):
            j = witness(z, P)
            errs.append(max(np.abs(A_obs @ j - b_obs).max(), abs(c @ j - c @ z)))
        print(f"  {kind:9s} free joint [{flo:.4f}, {fhi:.4f}]   "
              f"lifted (sharp) [{lo:.4f}, {hi:.4f}]   witness max err {max(errs):.1e}")


def balke_pearl(P, kind):
    """Classical IV (Z _||_ U): q(rX, rY), P(x,y|z) = sum q 1[rX(z)=x] 1[rY(x)=y]."""
    types = [(rx, ry) for rx in MAPS for ry in MAPS]
    pz = {z: sum(v for k, v in P.items() if k[0] == z) for z in (0, 1)}
    A, b = [], []
    for z, x, y in itertools.product((0, 1), repeat=3):
        A.append([1.0 if rx[z] == x and ry[x] == y else 0.0 for rx, ry in types])
        b.append(P[(z, x, y)] / pz[z])
    c = [float(ry[1] == 1) if kind == "iv_Yx" else float(ry[1] == 1 and ry[0] == 0)
         for rx, ry in types]
    lo, hi, _, _ = interval(c, np.array(A), np.array(b))
    return lo, hi


def random_pair_chain_law(seed):
    """Observational law of a random product SCM on the pair chain."""
    rng = np.random.default_rng(seed)
    Gs = list(itertools.product(MAPS, repeat=len(MAPS)))  # U1 = (x, G), G: r_Y -> r_W
    p1 = rng.dirichlet(0.2 * np.ones(2 * len(Gs))).reshape(2, len(Gs))
    p2 = rng.dirichlet(np.ones(len(MAPS)))                # U2 = r_Y
    P = {k: 0.0 for k in itertools.product((0, 1), repeat=3)}
    for x in (0, 1):
        for g, G in enumerate(Gs):
            for j, rb in enumerate(MAPS):
                w = G[j][x]
                P[(x, w, rb[w])] += p1[x, g] * p2[j]
    return P


if __name__ == "__main__":
    RE2 = {
        (0, 0, 0): 0.20, (0, 0, 1): 0.05, (0, 1, 0): 0.10, (0, 1, 1): 0.15,
        (1, 0, 0): 0.05, (1, 0, 1): 0.10, (1, 1, 0): 0.10, (1, 1, 1): 0.25,
    }
    run("pair chain on (eq:re2)", RE2, ["pair_Yx", "pair_PNS"])
    for s in (1, 2):
        run(f"pair chain, random product-SCM law (seed {s})",
            random_pair_chain_law(s), ["pair_Yx", "pair_PNS"])

    run("IV with Z<->X on (eq:re2) read as P(z,x,y)", RE2, ["iv_Yx", "iv_PNS"])
    for kind in ("iv_Yx", "iv_PNS"):
        lo, hi = balke_pearl(RE2, kind)
        print(f"  Balke-Pearl {kind:9s} [{lo:.4f}, {hi:.4f}]")
