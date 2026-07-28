"""Doctests and example smoke for Phase 3."""

from __future__ import annotations

import doctest
import runpy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_init_doctest():
    import partialid

    result = doctest.testmod(partialid, verbose=False)
    assert result.failed == 0


@pytest.mark.parametrize(
    "script",
    ["examples/quickstart.py", "examples/from_csv.py"],
)
def test_examples_run(script: str, monkeypatch):
    path = ROOT / script
    assert path.is_file()
    monkeypatch.setattr(sys, "argv", [str(path)])
    runpy.run_path(str(path), run_name="__main__")
