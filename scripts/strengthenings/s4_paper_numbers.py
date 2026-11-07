"""Numbers quoted in the simplified Section 3 / Appendix A.

1. Descendant example (ex:desc): a 3-decimal table summing to one, Manski
   (trimmed) interval and certified full-model bounds.
2. Mediated graph on the compatible table (eq:re3): sharp P(Y_1=1) and the
   free-joint interval over (R_X, R_W, R_Y) (drops U _||_ U_Y).
3. Pair chain on (eq:re2): sharp P(Y_1=1) and PNS.
"""

import itertools

import numpy as np
from common import MAPS, interval
from s1_lambda2_lift import indep_constraints, obs_constraints, query_vec
from s2_separable import mediated
from s3_trim_strict import certified, law_from_scm, manski

RE2 = {
    (0, 0, 0): 0.20, (0, 0, 1): 0.05, (0, 1, 0): 0.10, (0, 1, 1): 0.15,
    (1, 0, 0): 0.05, (1, 0, 1): 0.10, (1, 1, 0): 0.10, (1, 1, 1): 0.25,
}

# 1. descendant table
P = {k: round(v, 3) for k, v in law_from_scm(1).items()}
kmax = max(P, key=P.get)
P[kmax] = round(P[kmax] + 1 - sum(P.values()), 3)
assert abs(sum(P.values()) - 1) < 1e-12
print("descendant table:", {"".join(map(str, k)): v for k, v in P.items()})
print("  trimmed (Manski): [%.3f, %.3f]" % manski(P))
li, lo, _ = certified(P, "min", tol=5e-4)
ui, uo, _ = certified(P, "max", tol=5e-4)
print(f"  full model: lower in [{lo:.4f}, {li:.4f}], upper in [{ui:.4f}, {uo:.4f}]")

# 2. mediated graph on re3
Pxw = {(0, 0): 0.25, (0, 1): 0.25, (1, 0): 0.15, (1, 1): 0.35}
py1 = (0.4, 0.6)
print("re3 sharp P(Y_1=1): [%.3f, %.3f]" % mediated(Pxw, py1, "Yx"))
types = [(rx, rw, ry) for rx in (0, 1) for rw in MAPS for ry in MAPS]
A, b = [], []
for x, w, y in itertools.product((0, 1), repeat=3):
    A.append([1.0 if rx == x and rw[x] == w and ry[w] == y else 0.0 for rx, rw, ry in types])
    b.append(Pxw[x, w] * (py1[w] if y else 1 - py1[w]))
c = [float(ry[rw[1]] == 1) for rx, rw, ry in types]
lo_, hi_, _, _ = interval(c, np.array(A), np.array(b))
print("re3 free joint P(Y_1=1): [%.3f, %.3f]" % (lo_, hi_))

# 3. pair chain on re2
A_obs, b_obs = obs_constraints(RE2)
A_ind, b_ind = indep_constraints(RE2)
for kind in ("pair_Yx", "pair_PNS"):
    lo_, hi_, _, _ = interval(query_vec(kind), np.vstack([A_obs, A_ind]),
                              np.concatenate([b_obs, b_ind]))
    print(f"pair chain {kind}: [{lo_:.3f}, {hi_:.3f}]")
