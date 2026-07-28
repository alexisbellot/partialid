"""Suite C — sensitivity / budget sweeps on fixed XY (and optional IV)."""

from __future__ import annotations

import time
import warnings

import numpy as np

from catalog import RE1_P, msm_closed_form_re1
from harness import (
    RESULTS_DIR,
    BenchRow,
    machine_meta,
    write_jsonl,
    write_summary_md,
)
from partialid import Assumptions, Query, Variables, bound, bound_grid
from partialid.api import bound_grid_cold


def _xy():
    V = Variables.from_binary("X", "Y")
    Q = Query.interventional(outcome="Y", do={"X": 1}, value=1)
    a = Assumptions.order("X", "Y")
    return V, RE1_P, Q, a


def _row(
    *,
    name: str,
    family: str,
    param: str,
    value: float | None,
    result,
    total_ms: float,
    check: str,
    check_message: str = "",
    extra: dict | None = None,
) -> BenchRow:
    notes = list(result.notes) if result is not None else []
    return BenchRow(
        suite="C",
        name=name,
        family=family,
        structure="order_xy",
        assumptions=param,
        method=getattr(result, "method", "lp") if result else "lp",
        median_ms=total_ms,
        total_ms=total_ms,
        formulate_ms=(result.formulate_time or 0) * 1000 if result else None,
        solve_ms=(result.solve_time or 0) * 1000 if result else None,
        lower=float(result.lower) if result and result.success else None,
        upper=float(result.upper) if result and result.success else None,
        width=float(result.width) if result and result.success else None,
        tight=bool(result.tight) if result else None,
        success=bool(result.success) if result else False,
        method_tag=result.method if result else "",
        notes=notes,
        check=check,
        check_message=check_message,
        extra={"param": param, "value": value, **(extra or {})},
    )


def _sweep_msm(*, quick: bool) -> list[BenchRow]:
    V, P, Q, a = _xy()
    values = (
        list(np.linspace(1.0, 7.0, 8))
        if quick
        else list(np.linspace(1.0, 20.0, 50))
    )
    rows: list[BenchRow] = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t0 = time.perf_counter()
        warm = bound_grid(
            V, P, Q, a, param="Lambda", values=values, method="lp", warm_start=True
        )
        t_warm = (time.perf_counter() - t0) * 1000

        t0 = time.perf_counter()
        cold = bound_grid_cold(V, P, Q, a, param="Lambda", values=values, method="lp")
        t_cold = (time.perf_counter() - t0) * 1000

    # Per-point correctness vs closed form
    for (lam, rw), (_, rc) in zip(warm, cold):
        ell, u, w = msm_closed_form_re1(float(lam))
        ok = (
            rw.success
            and abs(rw.lower - ell) < 1e-5
            and abs(rw.upper - u) < 1e-5
            and abs(rw.width - w) < 1e-5
            and abs(rw.lower - rc.lower) < 1e-6
            and abs(rw.upper - rc.upper) < 1e-6
        )
        msg = "" if ok else f"warm/cold/closed mismatch at Λ={lam}"
        rows.append(
            _row(
                name=f"msm_L{lam:g}",
                family="msm_grid",
                param="Lambda",
                value=float(lam),
                result=rw,
                total_ms=t_warm / len(values),
                check="pass" if ok else "fail",
                check_message=msg,
                extra={"grid": "warm"},
            )
        )

    speedup = t_cold / t_warm if t_warm > 0 else None
    # Aggregate warm vs cold timing row
    rows.append(
        BenchRow(
            suite="C",
            name="msm_grid_warm_vs_cold",
            family="msm_grid",
            structure="order_xy",
            assumptions="Lambda",
            method="lp",
            median_ms=t_warm,
            total_ms=t_warm,
            success=True,
            check="pass",
            check_message="",
            notes=[f"highspy_or_template warm={t_warm:.1f}ms cold={t_cold:.1f}ms"],
            extra={
                "warm_ms": t_warm,
                "cold_ms": t_cold,
                "speedup": speedup,
                "n_points": len(values),
            },
        )
    )
    print(
        f"  [OK] msm_grid n={len(values)} warm={t_warm:.1f}ms cold={t_cold:.1f}ms "
        f"speedup={speedup:.2f}x" if speedup else f"  msm_grid warm={t_warm} cold={t_cold}"
    )
    return rows


def _sweep_cdep(*, quick: bool) -> list[BenchRow]:
    V, P, Q, a = _xy()
    values = [0.0, 0.1, 0.25, 0.5] if quick else list(np.linspace(0.0, 0.5, 11))
    rows = []
    prev_w = -1.0
    for c in values:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t0 = time.perf_counter()
            r = bound(
                V, P, Q, a & Assumptions.c_dependence("X", c), method="lp"
            )
            ms = (time.perf_counter() - t0) * 1000
        ok = r.success and r.width + 1e-9 >= prev_w  # monotone nondecreasing gap
        if c == 0.0:
            ok = ok and abs(r.width) < 1e-6  # no confounding slack → point?
            # c=0 means |e(r)-e0|<=0 → e(r)=e0 → like no unmeasured confounding on propensity
            # Actually c=0 forces e(r)=e0 for all r → point ID at P(Y=1|X=1)=0.70
            ok = r.success and abs(r.width) < 1e-5 and abs(r.lower - 0.70) < 1e-4
        msg = "" if ok else f"c-dep failed/monotone at c={c} width={r.width}"
        rows.append(
            _row(
                name=f"cdep_c{c:g}",
                family="c_dependence",
                param="c",
                value=float(c),
                result=r,
                total_ms=ms,
                check="pass" if ok else "fail",
                check_message=msg,
            )
        )
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] cdep c={c:g} width={r.width:.4f} {msg}")
        if r.success:
            prev_w = float(r.width)
    return rows


def _sweep_rosenbaum(*, quick: bool) -> list[BenchRow]:
    V, P, Q, a = _xy()
    gammas = [1.0, 4.0, 9.0] if quick else [1.0, 2.25, 4.0, 9.0, 16.0, 25.0]
    rows = []
    for g in gammas:
        lam = float(np.sqrt(g))
        ell, u, w = msm_closed_form_re1(lam)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t0 = time.perf_counter()
            r = bound(
                V,
                P,
                Q,
                a & Assumptions.rosenbaum_gamma("X", g, anchored=True),
                method="lp",
            )
            ms = (time.perf_counter() - t0) * 1000
        ok = (
            r.success
            and abs(r.lower - ell) < 1e-5
            and abs(r.upper - u) < 1e-5
            and abs(r.width - w) < 1e-5
        )
        msg = "" if ok else f"Rosenbaum Γ={g} ≠ MSM Λ={lam}"
        rows.append(
            _row(
                name=f"rosenbaum_G{g:g}",
                family="rosenbaum",
                param="Gamma",
                value=float(g),
                result=r,
                total_ms=ms,
                check="pass" if ok else "fail",
                check_message=msg,
                extra={"Lambda_equiv": lam},
            )
        )
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] Rosenbaum Γ={g:g} width={r.width:.4f} {msg}")
    return rows


def _sweep_margin(*, quick: bool) -> list[BenchRow]:
    V, P, Q, a = _xy()
    # margin around point reference (observational P(Y=1|X=1)=0.70)
    kappas = [0.0, 0.05, 0.15] if quick else list(np.linspace(0.0, 0.3, 7))
    rows = []
    prev_w = -1.0
    for kap in kappas:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t0 = time.perf_counter()
            r = bound(
                V,
                P,
                Q,
                a & Assumptions.margin_budget(kap, reference="point"),
                method="lp",
            )
            ms = (time.perf_counter() - t0) * 1000
        ok = r.success and r.width + 1e-9 >= prev_w
        if kap == 0.0:
            ok = r.success and abs(r.width) < 1e-5
        msg = "" if ok else f"margin κ={kap} failed"
        rows.append(
            _row(
                name=f"margin_k{kap:g}",
                family="margin",
                param="kappa",
                value=float(kap),
                result=r,
                total_ms=ms,
                check="pass" if ok else "fail",
                check_message=msg,
            )
        )
        status = "OK" if ok else "FAIL"
        print(f"  [{status}] margin κ={kap:g} width={r.width:.4f} {msg}")
        if r.success:
            prev_w = float(r.width)
    return rows


def _sweep_f_sensitivity(*, quick: bool) -> list[BenchRow]:
    try:
        import cvxpy  # noqa: F401
    except ImportError:
        return [
            BenchRow(
                suite="C",
                name="f_sensitivity_skip",
                family="f_sensitivity",
                check="skip",
                check_message="cvxpy not installed",
                success=False,
            )
        ]
    V, P, Q, a = _xy()
    rhos = [0.0, 0.05] if quick else [0.0, 0.01, 0.05, 0.1, 0.2]
    rows = []
    for f in ("kl", "chi2", "tv"):
        for rho in rhos:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                t0 = time.perf_counter()
                try:
                    r = bound(
                        V,
                        P,
                        Q,
                        a & Assumptions.f_sensitivity("X", rho, f=f),
                        method="convex",
                    )
                    ms = (time.perf_counter() - t0) * 1000
                    err = ""
                except Exception as exc:  # noqa: BLE001
                    r = None
                    ms = float("nan")
                    err = str(exc)
            if r is None:
                rows.append(
                    BenchRow(
                        suite="C",
                        name=f"fsens_{f}_r{rho:g}",
                        family="f_sensitivity",
                        check="fail",
                        check_message=err,
                        success=False,
                        error=err,
                        extra={"f": f, "rho": rho},
                    )
                )
                print(f"  [FAIL] f-sens {f} ρ={rho}: {err}")
                continue
            ok = r.success
            if rho == 0.0:
                ok = ok and abs(r.width) < 1e-3  # near point-ID
            msg = "" if ok else (r.message or "f-sens failed")
            rows.append(
                _row(
                    name=f"fsens_{f}_r{rho:g}",
                    family="f_sensitivity",
                    param="rho",
                    value=float(rho),
                    result=r,
                    total_ms=ms,
                    check="pass" if ok else "fail",
                    check_message=msg,
                    extra={"f": f},
                )
            )
            status = "OK" if ok else "FAIL"
            print(f"  [{status}] f-sens {f} ρ={rho:g} width={r.width:.4f} {msg}")
    return rows


def _sweep_support(*, quick: bool) -> list[BenchRow]:
    V, P, Q, a = _xy()
    # Cor. 7.7 on RE1: k<4 refutes fully-supported data; k≥4 non-binding Manski
    ks = [3, 4] if quick else [1, 2, 3, 4, 8]
    rows = []
    for k in ks:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t0 = time.perf_counter()
            r = bound(
                V, P, Q, a & Assumptions.support_bound(k), method="mip"
            )
            ms = (time.perf_counter() - t0) * 1000
        if k < 4:
            ok = not r.success  # infeasible / refuted
            msg = "" if ok else "expected refute for k<4 on RE1"
        else:
            ok = (
                r.success
                and abs(r.lower - 0.35) < 1e-4
                and abs(r.upper - 0.85) < 1e-4
            )
            msg = "" if ok else "k>=4 should recover Manski"
        rows.append(
            _row(
                name=f"support_k{k}",
                family="support",
                param="k",
                value=float(k),
                result=r,
                total_ms=ms,
                check="pass" if ok else "fail",
                check_message=msg,
            )
        )
        status = "OK" if ok else "FAIL"
        w = f"{r.width:.4f}" if r.success else "infeas"
        print(f"  [{status}] support k={k} width={w} {msg}")
    return rows


def run_suite_c(*, quick: bool = True) -> list[BenchRow]:
    rows: list[BenchRow] = []
    print("  -- MSM grid --")
    rows.extend(_sweep_msm(quick=quick))
    print("  -- c-dependence --")
    rows.extend(_sweep_cdep(quick=quick))
    print("  -- Rosenbaum --")
    rows.extend(_sweep_rosenbaum(quick=quick))
    print("  -- margin κ --")
    rows.extend(_sweep_margin(quick=quick))
    print("  -- f-sensitivity --")
    rows.extend(_sweep_f_sensitivity(quick=quick))
    print("  -- support k --")
    rows.extend(_sweep_support(quick=quick))
    return rows


def main(quick: bool = True) -> int:
    print(f"Suite C (sensitivity sweeps) quick={quick}")
    rows = run_suite_c(quick=quick)
    meta = machine_meta()
    meta["suite"] = "C"
    meta["quick"] = quick
    out = RESULTS_DIR / "suite_C.jsonl"
    write_jsonl(out, rows, meta=meta)
    md = RESULTS_DIR / "suite_C_summary.md"
    write_summary_md(md, rows, title="Suite C — sensitivity")
    n_fail = sum(1 for r in rows if r.check == "fail")
    n_skip = sum(1 for r in rows if r.check == "skip")
    print(f"Wrote {out} and {md} ({n_fail} failures, {n_skip} skipped)")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
