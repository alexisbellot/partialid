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

1. Create the project on [TestPyPI](https://test.pypi.org/) if needed (`partialid`).
2. Under **Publishing** → **Add a new pending publisher**:
   - PyPI project: `partialid`
   - Owner: `alexisbellot`
   - Repository: `partialid`
   - Workflow: `publish-testpypi.yml`
   - Environment: `testpypi`
3. In GitHub → Settings → Environments, create `testpypi` (optional protection rules).
4. Tag and push:

```bash
git tag v0.1.0a1
git push origin v0.1.0a1
```

Or run **Publish to TestPyPI** via `workflow_dispatch`.

5. Install from TestPyPI:

```bash
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ partialid==0.1.0a1
```

(`--extra-index-url` pulls NumPy/SciPy from real PyPI.)

## PyPI (later)

Mirror the TestPyPI publisher config on [pypi.org](https://pypi.org/) and add a
separate workflow (or gate) for production. Do not publish `0.1.0` until
external alpha feedback is in.
