# Repository Guidelines

## Project Structure & Module Organization
- `src/openpi/`: core library code (models, policies, training, shared utilities).
- `packages/openpi-client/`: client package for robot-side inference and IO helpers.
- `scripts/`: runnable entry points (training, serving, data prep).
- `examples/`: end-to-end workflows and robot-specific READMEs.
- `docs/`: longer-form docs (Docker, remote inference, troubleshooting).
- `third_party/` and submodules: vendored deps; keep changes scoped and justified.

## Build, Test, and Development Commands
Use `uv` for dependency management (Python 3.11). Typical setup:
```bash
git submodule update --init --recursive
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```
Common workflows:
```bash
uv run scripts/compute_norm_stats.py --config-name pi05_libero
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_libero --exp-name=my_experiment
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi05_libero --policy.dir=checkpoints/pi05_libero/my_experiment/20000
```
Optional Docker path:
```bash
docker compose -f scripts/docker/compose.yml up --build
```

## Coding Style & Naming Conventions
- Python: 4-space indentation; target version `py311`.
- Formatting: `ruff format .` (line length 120).
- Linting: `ruff check .` (imports are sorted via Ruff isort settings).
- Prefer explicit, descriptive names for configs and experiment runs (e.g., `pi05_libero`, `my_experiment`).

## Testing Guidelines
- Framework: `pytest` (test discovery in `src/`, `scripts/`, `packages/`).
- Naming: tests are `*_test.py` (see `src/openpi/*_test.py`).
- Run all tests:
```bash
uv run pytest
```
- Manual-only tests are marked `manual`:
```bash
uv run pytest -m manual
```

## Commit & Pull Request Guidelines
- Commit messages are short and imperative; history sometimes uses prefixes like `feat:` or `chore:`—use if it helps clarity.
- PRs should have a clear title/description, pass tests, and include formatting/linting (`pre-commit install`, then `pre-commit run --all-files`, `ruff check .`, `ruff format .`).
- For larger features (new robots/environments), open an issue/discussion first to align on scope.

## Agent Automation
- After completing requested changes, automatically run `git add -A` and `git commit -m "<message>"`.
- Never `git push`; the user will push.
- Commit messages are written by the agent when not explicitly provided.
- Commit messages should be in Chinese.

## Environment & Configuration Tips
- Repo is tested on Ubuntu 22.04; GPU is required for most training/inference.
- Checkpoints download to `~/.cache/openpi`; override with `OPENPI_DATA_HOME` if needed.
