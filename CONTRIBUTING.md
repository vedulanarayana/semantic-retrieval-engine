# Contributing

## Setup

```bash
pip install -r requirements.txt
pip install ruff black pytest
./scripts/build_cpp.sh   # needed for the brute-force index and its tests
```

## Before opening a PR

```bash
ruff check .
black .
pytest tests/
```

CI runs the same three checks, so a clean run locally means a clean run in CI.

## Style

- Comments explain *why*, not *what* — skip comments that just restate the code.
- Keep changes scoped to what the PR is about; unrelated cleanup belongs in its own PR.
- Add or update a test alongside any behavior change.
