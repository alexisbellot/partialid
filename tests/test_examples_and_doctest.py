"""Doctests for Phase 3 (tutorials live under notebooks/, executed in CI)."""

from __future__ import annotations

import doctest


def test_init_doctest():
    import partialid

    result = doctest.testmod(partialid, verbose=False)
    assert result.failed == 0
