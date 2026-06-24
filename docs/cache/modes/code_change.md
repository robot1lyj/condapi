# Mode: Code Change

## Continue Loading
- Check `git status --short` and `git log --oneline -5` for current state.
- Read relevant source files before editing.

## Change Rules
- Surgical edits: change only what the task requires.
- Match surrounding code style (comments, naming, formatting).
- Python: ruff format (line length 120), ruff lint. Import order via ruff isort.
- New configs go in `src/openpi/training/config.py` (frozen dataclass, register in `_CONFIGS`).
- New robot policies go in `src/openpi/policies/` (subclass `Policy`).
- Data transforms go in `src/openpi/transforms.py`.
- Don't add dependencies without checking `requirements-pi-pip.txt`.

## Build & Verify
- Lint: `ruff check .` and `ruff format .`
- Tests: `conda run -n pi-conda python -m pytest --strict-markers -m "not manual"`
- Smoke test: `conda run -n pi-conda python scripts/train_test.py` (if applicable)
- For new training configs: verify `create_trained_policy()` works before training.

## Commit
- Chinese commit messages (per AGENTS.md).
- Focused, single-topic commits.
- Auto stage all: `git add -A` then `git commit -m "<message>"`.
- Never push; user handles push.

## Docs Writeback
- Every meaningful change → `docs/CHANGELOG.md`
- Default behavior/parameters/baseline changes → also `README.md`
- New configs/policies → architecture doc if needed
- Follow `kernel.md` writeback classes at end of task.
