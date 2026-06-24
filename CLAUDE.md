# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Context System

This repo uses a layered "Context OS". On session start, read files in this order:

1. `AGENTS.md` — agent rules, safety, code style, docs writeback.
2. `docs/cache/kernel.md` — hot facts, security kernel, budget rules.
3. `docs/cache/context_index.md` — routes to the right `docs/cache/modes/*.md` mode pack.
4. Read at most one mode pack; only cross modes when the task spans them.

`README.md` is the product baseline (cold reference), not hot context.
`docs/CHANGELOG.md` is the chronological change log.
`docs/ARCHITECTURE.md` is the human-readable architecture.

## Archive Points

- `conda-pi` — current branch, Piper 双臂离线 Conda 训练路径.

## Project Overview

openpi is Physical Intelligence's open-source robotics VLA (Vision-Language-Action) model repository. It provides pre-trained checkpoints (pi0, pi0-FAST, pi0.5), fine-tuning scripts, and a policy server for robot inference. Supports both JAX (all models) and PyTorch (pi0/pi0.5 only) backends.

## Environment Setup

This repo uses a **conda-based offline bundle** workflow (Python 3.11, CUDA 12):

```bash
# Build offline bundle (on internet-connected machine)
bash scripts/conda/build_offline_bundle.sh artifacts/pi-conda-offline-bundle

# Install from bundle (on training server)
bash scripts/conda/install_offline_bundle.sh --bundle-dir artifacts/pi-conda-offline-bundle --env-name pi-conda

# Patch HuggingFace transformers (required for PyTorch path)
conda run -n pi-conda python scripts/conda/patch_transformers.py --openpi-dir .

# Install packages in editable mode
conda run -n pi-conda pip install --no-build-isolation --no-deps -e .
conda run -n pi-conda pip install --no-build-isolation --no-deps -e packages/openpi-client
```

Key environment variables:
- `OPENPI_DATA_HOME` — override checkpoint download location (default: `~/.cache/openpi`)
- `XLA_PYTHON_CLIENT_MEM_FRACTION` — control JAX GPU memory (e.g., `0.9`)
- `JAX_PLATFORMS` — set to `cpu` when no GPU is available (handled automatically by `conftest.py`)

## Commands

```bash
# Lint and format
ruff check .                        # Lint
ruff format .                       # Format
pre-commit install && pre-commit run --all-files  # Pre-commit (ruff + ruff-format)

# Tests (discover in src/, scripts/, packages/)
conda run -n pi-conda python -m pytest                              # All non-manual tests
conda run -n pi-conda python -m pytest --strict-markers -m "not manual"  # CI equivalent
conda run -n pi-conda python -m pytest -m manual                    # Manual tests only

# Training
conda run -n pi-conda python scripts/compute_norm_stats.py --config-name <CONFIG>
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 conda run -n pi-conda python scripts/train.py <CONFIG> --exp-name=<NAME>
conda run -n pi-conda python scripts/train_pytorch.py <CONFIG> --exp-name=<NAME>

# PyTorch DDP training
torchrun --standalone --nnodes=1 --nproc_per_node=<N> scripts/train_pytorch.py <CONFIG> --exp_name=<NAME>

# Serving (policy inference server on port 8000)
conda run -n pi-conda python scripts/serve_policy.py policy:checkpoint --policy.config=<CONFIG> --policy.dir=<PATH>

# Data repair
conda run -n pi-conda python scripts/repair_lerobot_subset.py
```

Line length: **120** (ruff configured). Target Python: **3.11**.

## Architecture

### Package layout

```
src/openpi/           # Core library (the `openpi` package)
  models/             # JAX model implementations (pi0, pi0_fast, gemma, siglip, vit)
  models_pytorch/     # PyTorch model implementations (pi0/pi0.5, gemma)
  policies/           # Policy wrappers — map model outputs → robot actions
  training/           # Training infrastructure (configs, data loader, checkpoints, sharding)
  shared/             # Shared utilities (downloading, normalization, image tools)
  serving/            # WebSocket policy server
packages/openpi-client/  # Client package (`openpi_client`) for robot-side inference
scripts/              # Runnable entry points (train.py, serve_policy.py, etc.)
examples/             # Platform-specific examples (ALOHA, DROID, LIBERO, UR5)
```

### Core concepts

**Model** (`models/model.py`): The `BaseModel` abstract class defines the interface all models must implement — `compute_loss()` and `sample_actions()`. Three model types exist via `ModelType` enum: `PI0` (flow-based), `PI0_FAST` (autoregressive with FAST tokenizer), `PI0.5` (pi0 with knowledge insulation). Models are created via `BaseModelConfig.create(rng)`.

**Config** (`training/config.py`): The central configuration registry. All training configs are defined as frozen dataclasses and registered in the `_CONFIGS` dict (mapping name → `TrainConfig`). Configs are accessed via `_config.cli()` using `tyro` for CLI parsing. A `TrainConfig` bundles: model config, data config (LeRobot repo, transforms, norm stats), optimizer config, weight loader, FSDP settings, and training hyperparameters.

The `DataConfig` specifies data transforms as a pipeline:
1. `repack_transforms` — dataset-specific → common format
2. `data_transforms` — robot-specific transformations
3. `model_transforms` — model-specific (tokenization, resizing, prompt injection)

**Policy** (`policies/policy.py`): Wraps a model with pre/post-processing transforms to produce robot actions. Policies are created via `create_trained_policy(config_name, checkpoint_dir)` in `policies/policy_config.py`. Each robot platform has its own policy subclass (ALOHA, DROID, LIBERO, Piper).

**Training**: Two parallel training paths exist:
- JAX (`scripts/train.py`): Uses Flax NNX, FSDP sharding, Optax optimizer. Data loaders produce `(Observation, Actions)` tuples. Training loop is `train_step` → JIT compiled with sharding specs.
- PyTorch (`scripts/train_pytorch.py`): Uses `PI0Pytorch` model, PyTorch DDP for multi-GPU. Mirrors the JAX config/data pipeline.

**Serving** (`serving/websocket_policy_server.py`): WebSocket-based server that loads a policy and streams action chunks to clients. Clients use `openpi_client.websocket_client_policy.WebsocketClientPolicy`.

### Data format

The model expects a fixed set of camera views: `base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb` at 224×224 resolution. Data transforms produce a nested dict with `image`, `image_mask`, `state`, `tokenized_prompt`, and `actions` keys, which is converted to `Observation` and `Actions` objects.

### Checkpoints

Training checkpoints are managed by `training/checkpoints.py` using Orbax. Checkpoints contain model params, optimizer state, EMA params, step counter, data loader state, and norm stats. JAX checkpoints can be converted to PyTorch via `examples/convert_jax_model_to_pytorch.py`.

### Weight loading

`training/weight_loaders.py` defines strategies for loading pretrained weights. `WeightLoader.load()` returns a subset of the full model params, allowing partial initialization (e.g., loading base model weights for fine-tuning). The training script validates shapes and dtypes match.

## Important constraints

- **WandB is forced offline** in training (`WANDB_MODE=offline`). The `_configure_offline_wandb()` function is called at the start of `train.py`.
- **Training is JAX-first**; PyTorch support exists for pi0/pi0.5 but pi0-FAST is JAX-only.
- **Batch size must be divisible by device count** in JAX training (checked at startup).
- **GPU is required** for most operations; `conftest.py` auto-detects GPU absence and sets `JAX_PLATFORMS=cpu` for tests.
- **RTC compatibility**: Any changes to RTC (Real-Time Control) must keep the old inference path usable, coexisting via `rtc_mode` switch or automatic fallback.
- **Git submodules** (`third_party/aloha`, `third_party/libero`) must be initialized: `git submodule update --init --recursive`
- **Pre-commit excludes** `third_party/`, `docker/`, `transformers_replace/` from linting.
