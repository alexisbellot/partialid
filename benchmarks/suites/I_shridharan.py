"""Suite I — Shridharan–Iyengar Appendix B (arXiv:2308.02709) head-to-head."""

from __future__ import annotations

import time
import warnings

from harness import (
    RESULTS_DIR,
    BenchRow,
    machine_meta,
    write_jsonl,
    write_summary_md,
)
from partialid import bound
from partialid.shridharan_examples import (
    EXAMPLES,
    PAPER_TABLE1,
    PAPER_TABLE2_tH,
    size_report,
)


SKIP_SOLVE = ("D", "E")


def _parse_H(notes: list[str]) -> int | None:
    for n in notes:
        if n.startswith("|H|="):
            try:
                return int(n.split("=", 1)[1])
            except ValueError:
                return None
    return None


def run_suite_i(*, quick: bool = True, lite: bool = False) -> list[BenchRow]:
    """Compare Appendix B examples to paper Table 1 / solve times.

    Modes
    -----
    lite :
        Ex F only (smallest hyperarc LP). No |H| enumeration for A–E/G.
        Safe for laptops; ~10–30s.
    quick :
        Size checks for all letters (no heavy |H| except F); solve F+G hyperarc only.
    full (``quick=False, lite=False``) :
        Also solve A (+ optional B/C); count |H| for A/G (can take minutes / heavy RAM).
    """
    rows: list[BenchRow] = []
    if lite:
        n_mc = 400
        letters = ("F",)
        size_letters = ("F", "B", "C", "G")  # max_H only; skip A (same as B/C)
        count_h_letters = frozenset({"F"})
        methods = ("hyperarc",)
    elif quick:
        n_mc = 800
        letters = ("F", "G")
        size_letters = "ABCDEFG"
        count_h_letters = frozenset({"F"})
        methods = ("hyperarc",)
    else:
        n_mc = 2000
        letters = ("F", "G", "A")
        size_letters = "ABCDEFG"
        count_h_letters = frozenset({"F", "A", "G"})
        methods = ("hyperarc", "auto")

    for letter in size_letters:
        count_h = letter in count_h_letters
        if letter in "DE":
            count_h = False
        t0 = time.perf_counter()
        rep = size_report(letter, count_H=count_h)
        dt = time.perf_counter() - t0
        paper = PAPER_TABLE1[letter]
        max_ok = abs(rep.max_hyperarcs - paper["max_H"]) / paper["max_H"] < 0.05
        h_ok = (
            rep.n_hyperarcs is not None
            and abs(rep.n_hyperarcs - paper["H"]) / paper["H"] < 0.25
        )
        msg = (
            f"VA={list(rep.va)} VB={list(rep.vb)}; "
            f"|R|≈{rep.raw_types_est:.2e} (paper {rep.paper_R:.2e}); "
            f"maxH={rep.max_hyperarcs:.2e} (paper {rep.paper_max_H:.2e}); "
            f"|H|={rep.n_hyperarcs} (paper {rep.paper_H:.2e})"
        )
        if h_ok:
            msg += " |H|~match"
        rows.append(
            BenchRow(
                suite="I",
                name=f"ex{letter}__sizes",
                family="shridharan",
                method="size",
                check="pass" if max_ok else "fail",
                check_message=msg,
                success=True,
                median_ms=dt * 1000,
                extra={
                    "letter": letter,
                    "raw_types_est": rep.raw_types_est,
                    "max_hyperarcs": rep.max_hyperarcs,
                    "n_hyperarcs": rep.n_hyperarcs,
                    "paper": paper,
                    "va": list(rep.va),
                    "vb": list(rep.vb),
                    "h_match": h_ok,
                },
            )
        )

    for letter in letters:
        p = EXAMPLES[letter](n_mc=n_mc, seed=0)
        for method in methods:
            t0 = time.perf_counter()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    r = bound(
                        p.variables,
                        p.data,
                        p.query,
                        p.assumptions,
                        method=method,
                    )
                    ok = bool(r.success)
                    msg = r.message or ""
                except Exception as e:  # noqa: BLE001
                    r = None
                    ok = False
                    msg = str(e)
            dt = time.perf_counter() - t0
            paper_t = PAPER_TABLE2_tH.get(letter)
            n_h = _parse_H(list(getattr(r, "notes", []) or [])) if r else None
            rows.append(
                BenchRow(
                    suite="I",
                    name=f"ex{letter}__{method}",
                    family="shridharan",
                    method=method,
                    method_tag=getattr(r, "method", method) if r else method,
                    check="pass" if ok else "fail",
                    check_message=msg[:200],
                    success=ok,
                    lower=float(r.lower) if r and r.success else None,
                    upper=float(r.upper) if r and r.success else None,
                    width=float(r.width) if r and r.success else None,
                    median_ms=dt * 1000,
                    notes=list(getattr(r, "notes", []) or [])[:8] if r else [],
                    extra={
                        "letter": letter,
                        "query": p.query.describe(),
                        "paper_tH_construct_s": paper_t,
                        "our_wall_s": dt,
                        "n_hyperarcs": n_h,
                        "n_mc": n_mc,
                    },
                )
            )

    if not lite:
        for letter in SKIP_SOLVE:
            rows.append(
                BenchRow(
                    suite="I",
                    name=f"ex{letter}__solve_skip",
                    family="shridharan",
                    method="skip",
                    check="skip",
                    check_message=(
                        "paper: pruned LP still intractable; greedy Alg 5 not implemented"
                    ),
                    success=True,
                    extra={"letter": letter, "paper_H": PAPER_TABLE1[letter]["H"]},
                )
            )

    return rows


def main(quick: bool = True, lite: bool = False) -> int:
    mode = "lite" if lite else ("quick" if quick else "full")
    print(f"Suite I (Shridharan Appendix B) mode={mode}")
    rows = run_suite_i(quick=quick, lite=lite)
    meta = machine_meta()
    meta["suite"] = "I"
    meta["quick"] = quick
    out = RESULTS_DIR / "suite_I.jsonl"
    write_jsonl(out, rows, meta=meta)
    md = RESULTS_DIR / "suite_I_summary.md"
    write_summary_md(md, rows, title="Suite I — Shridharan–Iyengar Appendix B")
    n_fail = sum(1 for r in rows if r.check == "fail")
    print(f"Wrote {out} and {md} ({n_fail} failures)")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--lite",
        action="store_true",
        help="Ex F only; skip heavy |H| enumeration (laptop-safe)",
    )
    args = ap.parse_args()
    raise SystemExit(main(quick=args.quick, lite=args.lite))
