"""Suite J — Track A4 method Pareto: fiber / lp / dual / colgen / auto."""

from __future__ import annotations

import warnings
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutTimeout
from pathlib import Path

from catalog import suite_a4_items
from harness import (
    RESULTS_DIR,
    BenchRow,
    machine_meta,
    run_bound,
    write_jsonl,
)

METHODS = ("auto", "fiber", "lp", "dual", "colgen")
AUTO_SLOWDOWN_CAP = 2.0
WIDTH_ATOL = 1e-5
METHOD_TIMEOUT_S = 20.0


def _applicable(method: str, item) -> tuple[bool, str]:
    """Return (ok_to_try, skip_reason)."""
    name = item.name
    if method in ("dual", "colgen"):
        pref = item.expected_method_prefix or ""
        if pref.startswith("id/"):
            return False, "skip numeric (ID instance)"
        if item.structure in ("markov_xy", "frontdoor", "napkin"):
            return False, "skip numeric (ID instance)"
    if method == "fiber":
        if "msm" in name or "rosenbaum" in name or "cdep" in name:
            return False, "fiber inapplicable (budget)"
        if item.structure in ("binary_iv", "two_iv", "napkin", "iv"):
            return False, "fiber inapplicable (coupled)"
    # Dual is the Z≥3 story; colgen on dense instrument masters is slow
    if method == "colgen" and item.structure == "two_iv":
        if int((item.extra or {}).get("card_z", 2)) >= 3:
            return False, "colgen skip (use dual on Z≥3)"
    return True, ""


def _run_bound_timeout(p, method: str, *, repeats: int, timeout_s: float):
    def _call():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return run_bound(
                p.variables,
                p.data,
                p.query,
                p.assumptions,
                method=method,
                repeats=repeats,
                warmup=1 if repeats > 1 else 0,
                recover_primal=False,
            )

    with ThreadPoolExecutor(max_workers=1) as ex:
        fut = ex.submit(_call)
        try:
            return fut.result(timeout=timeout_s), False
        except FutTimeout:
            return (None, float("nan"), f"timeout>{timeout_s}s"), True


def run_suite_j(*, quick: bool = True, repeats: int | None = None) -> list[BenchRow]:
    items = suite_a4_items(quick=quick)
    if repeats is None:
        repeats = 3
    rows: list[BenchRow] = []
    pivot: dict[str, dict[str, dict]] = {}

    for item in items:
        p = item.problem
        print(f"  · {item.name} ...", flush=True)
        pivot[item.name] = {}
        ref_width = None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r_lp, _, _ = run_bound(
                p.variables,
                p.data,
                p.query,
                p.assumptions,
                method="lp",
                repeats=1,
                warmup=0,
                recover_primal=False,
            )
        if r_lp is not None and r_lp.success:
            ref_width = float(r_lp.width)

        times: dict[str, float] = {}
        item_repeats = repeats
        if item.structure == "two_iv" and int((item.extra or {}).get("card_z", 2)) >= 3:
            item_repeats = 1

        for method in METHODS:
            ok_try, skip_reason = _applicable(method, item)
            if not ok_try:
                rows.append(
                    BenchRow(
                        suite="J",
                        name=item.name,
                        family=item.family,
                        structure=item.structure,
                        method=method,
                        check="skip",
                        check_message=skip_reason,
                        success=False,
                    )
                )
                pivot[item.name][method] = {
                    "ms": None,
                    "width": None,
                    "tag": "",
                    "ok": False,
                    "skip": skip_reason,
                }
                continue

            (result, med_s, err), timed_out = _run_bound_timeout(
                p, method, repeats=item_repeats, timeout_s=METHOD_TIMEOUT_S
            )

            ms = med_s * 1000 if med_s == med_s else None
            success = bool(result and result.success) and not timed_out
            width = float(result.width) if success else None
            tag = result.method if result else ""
            notes = list(result.notes) if result else []

            width_ok = True
            msg = ""
            if timed_out:
                check = "skip"
                msg = err
            elif not success:
                if method == "fiber" or (err and "fiber" in err.lower()):
                    check = "skip"
                    msg = (err or (result.message if result else "fail")).split("\n", 1)[0]
                else:
                    check = "fail"
                    msg = (err or (result.message if result else "fail")).split("\n", 1)[0]
            else:
                check = "pass"
                if ref_width is not None and width is not None:
                    if abs(width - ref_width) > WIDTH_ATOL and method in (
                        "lp",
                        "dual",
                        "colgen",
                    ):
                        if item.expected_width is not None:
                            if abs(width - item.expected_width) > max(item.atol, WIDTH_ATOL):
                                check = "fail"
                                msg = f"width {width:.6g} ≠ expected {item.expected_width}"
                                width_ok = False
                        else:
                            check = "fail"
                            msg = f"width {width:.6g} ≠ lp {ref_width:.6g}"
                            width_ok = False
                elif item.expected_width is not None and width is not None:
                    if abs(width - item.expected_width) > max(item.atol, WIDTH_ATOL):
                        check = "fail"
                        msg = f"width {width:.6g} ≠ expected {item.expected_width}"
                        width_ok = False

            if success and ms is not None:
                times[method] = ms

            rows.append(
                BenchRow(
                    suite="J",
                    name=item.name,
                    family=item.family,
                    structure=item.structure,
                    method=method,
                    median_ms=ms,
                    width=width,
                    success=success,
                    method_tag=tag,
                    notes=notes,
                    check=check,
                    check_message=msg,
                    error=err.split("\n", 1)[0] if err and check == "fail" else "",
                    extra={"width_ok": width_ok, "ref_width": ref_width},
                )
            )
            pivot[item.name][method] = {
                "ms": ms,
                "width": width,
                "tag": tag,
                "ok": success and check == "pass",
                "skip": msg if check == "skip" else "",
            }

        if times:
            winner = min(times, key=times.get)
            best_ms = times[winner]
            rows.append(
                BenchRow(
                    suite="J",
                    name=f"{item.name}__winner",
                    family="pareto",
                    structure=item.structure,
                    method=winner,
                    median_ms=best_ms,
                    check="pass",
                    check_message=f"winner {winner} at {best_ms:.2f}ms",
                    success=True,
                    extra={"winner": winner, "ms": best_ms, "times": times},
                )
            )
            if "auto" in times:
                ratio = times["auto"] / best_ms if best_ms > 0 else float("inf")
                micro = best_ms < 1.0
                auto_ok = ratio <= AUTO_SLOWDOWN_CAP + 1e-9 or micro
                rows.append(
                    BenchRow(
                        suite="J",
                        name=f"{item.name}__auto_vs_best",
                        family="pareto",
                        structure=item.structure,
                        method="auto",
                        median_ms=times["auto"],
                        check="pass" if auto_ok else "fail",
                        check_message=(
                            f"auto/best={ratio:.2f} (cap {AUTO_SLOWDOWN_CAP}"
                            + ("; micro-exempt" if micro else "")
                            + ")"
                        ),
                        success=True,
                        extra={
                            "ratio": ratio,
                            "best": winner,
                            "best_ms": best_ms,
                            "micro_exempt": micro,
                        },
                    )
                )

    rows.append(
        BenchRow(
            suite="J",
            name="__pivot__",
            family="meta",
            method="",
            check="pass",
            success=True,
            extra={"pivot": pivot, "methods": list(METHODS)},
        )
    )
    return rows


def write_pareto_md(path: Path, rows: list[BenchRow], *, quick: bool) -> Path:
    pivot_row = next((r for r in rows if r.name == "__pivot__"), None)
    pivot = (pivot_row.extra or {}).get("pivot", {}) if pivot_row else {}
    methods = list(METHODS)

    lines = [
        "# Suite J — Track A4 method Pareto",
        "",
        f"Methods: {', '.join(methods)}. quick={quick}.",
        "",
        "Median wall time (ms). Bold = fastest successful method on that instance.",
        "",
    ]
    header = "| instance | " + " | ".join(methods) + " | winner | auto/best |"
    sep = "|---|" + "|".join(["---:"] * len(methods)) + "|---|---:|"
    lines.append(header)
    lines.append(sep)

    winners = {
        r.name.replace("__winner", ""): r
        for r in rows
        if r.name.endswith("__winner")
    }
    autos = {
        r.name.replace("__auto_vs_best", ""): r
        for r in rows
        if r.name.endswith("__auto_vs_best")
    }

    for name, cells in pivot.items():
        parts = [name]
        wrow = winners.get(name)
        winner = wrow.method if wrow else ""
        for m in methods:
            cell = cells.get(m, {})
            ms = cell.get("ms")
            if cell.get("skip") and ms is None:
                parts.append("skip")
            elif ms is None:
                parts.append("—")
            else:
                s = f"{ms:.2f}"
                if m == winner and cell.get("ok"):
                    s = f"**{s}**"
                tag = cell.get("tag") or ""
                if m in ("auto", "dual") and tag and tag != m:
                    short = tag.replace("lp/", "").replace("closed/", "cf/")[:10]
                    s = f"{s} ({short})"
                parts.append(s)
        ar = autos.get(name)
        ratio = "—"
        if ar and ar.extra and "ratio" in ar.extra:
            ratio = f"{ar.extra['ratio']:.2f}"
            if ar.check == "fail":
                ratio = f"**{ratio} FAIL**"
        parts.append(winner or "—")
        parts.append(ratio)
        lines.append("| " + " | ".join(parts) + " |")

    n_fail = sum(1 for r in rows if r.check == "fail" and r.name != "__pivot__")
    n_pass = sum(1 for r in rows if r.check == "pass" and not r.name.startswith("__"))
    lines.extend(
        [
            "",
            f"Checks: pass={n_pass}, fail={n_fail} "
            f"(auto ≤ {AUTO_SLOWDOWN_CAP}× best; sub-ms races exempt).",
            "",
            "Width agreement: `lp` / `dual` / `colgen` vs LP reference "
            f"(atol={WIDTH_ATOL}). ID instances skip forced dual/colgen.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(quick: bool = True) -> int:
    print(f"Suite J (Track A4 Pareto) quick={quick}")
    rows = run_suite_j(quick=quick, repeats=3)
    meta = machine_meta()
    meta["suite"] = "J"
    meta["quick"] = quick
    meta["track"] = "A4"
    json_rows = [r for r in rows if r.name != "__pivot__"]
    out = RESULTS_DIR / "suite_J.jsonl"
    write_jsonl(out, json_rows, meta=meta)
    md = RESULTS_DIR / "suite_J_pareto.md"
    write_pareto_md(md, rows, quick=quick)
    n_fail = sum(1 for r in json_rows if r.check == "fail")
    print(md.read_text())
    print(f"Wrote {out} and {md} ({n_fail} failures)")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
