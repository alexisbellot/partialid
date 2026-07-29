# Publishing partialid

## Local build check

```bash
pip install -e ".[dev]"
python -m build
twine check dist/*
```

Install the wheel in a clean venv (no `PYTHONPATH`):

```bash
python -m venv /tmp/partialid-smoke
/tmp/partialid-smoke/bin/pip install dist/partialid-*.whl
/tmp/partialid-smoke/bin/pip install pytest
/tmp/partialid-smoke/bin/pytest -m "not slow"
```

## TestPyPI (Trusted Publishing)

Dry-run index. Private GitHub repos are fine; free Pages is unrelated.

1. Account on [TestPyPI](https://test.pypi.org/) (separate from real PyPI).
2. Create project **`partialid`** if needed.
3. **Publishing** → **Add a new pending publisher**:
   - PyPI project: `partialid`
   - Owner: `alexisbellot`
   - Repository: `partialid`
   - Workflow: `publish-testpypi.yml`
   - Environment: `testpypi`
4. GitHub → Settings → Environments → create **`testpypi`**.
5. Publish the commit you want (prefer a new tag on current `main`):

```bash
# version in pyproject.toml should match the tag
git tag v0.1.0a4
git push origin v0.1.0a4
```

Or Actions → **Publish to TestPyPI** → **Run workflow**.

6. Install from a clean venv (not inside the repo tree):

```bash
cd /tmp
python -m venv /tmp/pid-test && source /tmp/pid-test/bin/activate
pip install -U pip
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ partialid==0.1.0a4
python -c "import partialid; print(partialid.__version__)"
```

(`--extra-index-url` pulls NumPy/SciPy from real PyPI.)

Project page: https://test.pypi.org/project/partialid/

## PyPI (production)

This is what makes `pip install partialid` work for everyone.

Workflow: `.github/workflows/publish-pypi.yml`
(**manual `workflow_dispatch` only** — tags do not auto-publish to production).

1. Account on [pypi.org](https://pypi.org/).
2. Create project **`partialid`** if needed (or let the first Trusted Publishing
   upload create it via a pending publisher).
3. **Publishing** → **Add a new pending publisher**:
   - PyPI project: `partialid`
   - Owner: `alexisbellot`
   - Repository: `partialid`
   - Workflow: `publish-pypi.yml`
   - Environment: `pypi`
4. GitHub → Settings → Environments → create **`pypi`**
   (optional: required reviewers for an extra safety gate).
5. Push the workflow file to `main`, then Actions → **Publish to PyPI** →
   **Run workflow** on the branch/commit whose `pyproject.toml` version you
   want (e.g. `0.1.0a4`).
6. Install:

```bash
pip install partialid
# or pin:
pip install 'partialid==0.1.0a4'
```

Project page: https://pypi.org/project/partialid/

### Release hygiene

- Bump `version` in `pyproject.toml` before each upload; PyPI rejects reuse.
- Prefer tagging the same commit you publish (`git tag v0.1.0a5 && git push origin v0.1.0a5`) even though production is manual — keeps GitHub releases aligned.
- Smoke-test on TestPyPI first, then run **Publish to PyPI**.
- Hold a stable `0.1.0` until external alpha feedback is in; alphas (`0.1.0aN`) are fine on PyPI.
