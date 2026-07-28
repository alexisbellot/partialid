"""Suite B — structure × method matrix (fixed small instances)."""

from __future__ import annotations

import warnings

from catalog import StructureSpec, suite_b_specs
from harness import (
    RESULTS_DIR,
    assert_close,
    machine_meta,
    row_from_bound,
    run_bound,
    write_jsonl,
    write_summary_md,
    BenchRow,
)


def _check_structure_run(
    spec: StructureSpec,
    method: str,
    result,
    error: str,
    *,
    ref_interval: tuple[float, float] | None,
) -> tuple[str, str]:
    if error:
        return "fail", error
    if result is None or not result.success:
        return "fail", getattr(result, "message", "no result")

    msgs: list[str] = []
    ok = True
    item = spec.item
    atol = item.atol

    if method == "auto" and spec.auto_tag_prefix:
        tag = result.method or ""
        if not tag.startswith(spec.auto_tag_prefix):
            ok = False
            msgs.append(
                f"auto tag {tag!r} expected prefix {spec.auto_tag_prefix!r}"
            )

    if item.expected_lower is not None and method in ("auto", "fiber", "lp", "hyperarc"):
        # colgen/gpp may differ slightly; still check auto/fiber/lp
        if abs(result.lower - item.expected_lower) > atol:
            ok = False
            msgs.append(f"lower {result.lower} != {item.expected_lower}")
        if item.expected_upper is not None and abs(result.upper - item.expected_upper) > atol:
            ok = False
            msgs.append(f"upper {result.upper} != {item.expected_upper}")

    if item.expected_width is not None and method in ("auto", "fiber", "lp"):
        if abs(result.width - item.expected_width) > atol:
            ok = False
            msgs.append(f"width {result.width} != {item.expected_width}")

    if item.expect_tight is True and method == "auto" and not result.tight:
        ok = False
        msgs.append("expected tight on auto")

    if item.truth is not None:
        if result.lower - atol > item.truth or result.upper + atol < item.truth:
            ok = False
            msgs.append(f"truth {item.truth} not in [{result.lower},{result.upper}]")

    # Cross-method agreement vs reference (auto interval)
    if ref_interval is not None and method not in ("auto",):
        rlo, rhi = ref_interval
        if abs(result.lower - rlo) > max(atol, 1e-4) or abs(result.upper - rhi) > max(
            atol, 1e-4
        ):
            # hyperarc / colgen should match on Manski; gpp may differ from outer
            if method in ("fiber", "lp", "colgen", "hyperarc") and item.family != "gpp":
                ok = False
                msgs.append(
                    f"interval [{result.lower},{result.upper}] disagrees with auto "
                    f"[{rlo},{rhi}]"
                )

    return ("pass" if ok else "fail"), "; ".join(msgs)


def run_suite_b(*, quick: bool = True, repeats: int = 1) -> list[BenchRow]:
    specs = suite_b_specs(quick=quick)
    rows: list[BenchRow] = []
    for spec in specs:
        p = spec.item.problem
        ref: tuple[float, float] | None = None
        for method in spec.methods:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result, med, err = run_bound(
                    p.variables,
                    p.data,
                    p.query,
                    p.assumptions,
                    method=method,
                    repeats=repeats,
                )
            if method == "auto" and result is not None and result.success:
                ref = (float(result.lower), float(result.upper))
            check, msg = _check_structure_run(
                spec, method, result, err, ref_interval=ref
            )
            name = f"{spec.item.name}__{method}"
            assum = ", ".join(p.assumptions.describe())
            rows.append(
                row_from_bound(
                    suite="B",
                    name=name,
                    result=result,
                    median_s=med,
                    method=method,
                    error=err,
                    family=spec.item.family,
                    structure=spec.item.structure,
                    assumptions=assum,
                    check=check,
                    check_message=msg,
                    extra={"base": spec.item.name, **dict(spec.item.extra or {})},
                )
            )
            status = "OK" if check == "pass" else "FAIL"
            tag = result.method if result is not None else "—"
            w = (
                f"{result.width:.4f}"
                if result is not None and result.success
                else "—"
            )
            print(
                f"  [{status}] {name:36s} tag={tag:16s} width={w:8s} {msg}"
            )
    return rows


def main(quick: bool = True) -> int:
    print(f"Suite B (structure × method) quick={quick}")
    rows = run_suite_b(quick=quick)
    meta = machine_meta()
    meta["suite"] = "B"
    meta["quick"] = quick
    out = RESULTS_DIR / "suite_B.jsonl"
    write_jsonl(out, rows, meta=meta)
    md = RESULTS_DIR / "suite_B_summary.md"
    write_summary_md(md, rows, title="Suite B — structure × method")
    n_fail = sum(1 for r in rows if r.check == "fail")
    print(f"Wrote {out} and {md} ({n_fail} failures)")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
