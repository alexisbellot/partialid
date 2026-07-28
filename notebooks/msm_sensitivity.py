# %% [markdown]
# MSM sensitivity path (partialid)
#
# Runnable percent-format notebook. Execute with:
#
# ```bash
# pip install partialid jupytext
# jupytext --to notebook notebooks/msm_sensitivity.py
# # or: python notebooks/msm_sensitivity.py
# ```

# %%
from partialid import Assumptions, Query, Variables, bound, bound_path

V = Variables.from_binary("X", "Y")
P = {(0, 0): 0.30, (0, 1): 0.20, (1, 0): 0.15, (1, 1): 0.35}
Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)

# %%
r = bound(V, P, Q, Assumptions.sensitivity_msm("X", "Y", Lambda=3.0))
print(r.summary(Q))
assert abs(r.width - 1 / 3) < 1e-3

# %%
path = bound_path(
    V,
    P,
    Q,
    Assumptions.manski("X", "Y"),
    param="Lambda",
    theta_min=1.0,
    theta_max=5.0,
)
print(path.summary(at=3.0))
for lam in (1.0, 2.0, 3.0, 5.0):
    lo, hi, w = path.at(lam)
    print(f"Λ={lam:g}: [{lo:.4f}, {hi:.4f}]  width={w:.4f}")
