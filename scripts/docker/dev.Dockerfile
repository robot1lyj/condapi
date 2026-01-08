# Dev/offline image for openpi. Builds a self-contained environment.
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04@sha256:2d913b09e6be8387e1a10976933642c73c840c0b735f0bf3c28d97fc9bc422e0
COPY --from=ghcr.io/astral-sh/uv:0.5.1 /uv /uvx /bin/

WORKDIR /app

RUN apt-get update \
    && apt-get install -y git git-lfs linux-headers-generic build-essential clang \
    && rm -rf /var/lib/apt/lists/*

ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT=/.venv
ENV PATH="/.venv/bin:$PATH"

RUN uv venv --python 3.11.9 $UV_PROJECT_ENVIRONMENT

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=packages/openpi-client/pyproject.toml,target=packages/openpi-client/pyproject.toml \
    --mount=type=bind,source=packages/openpi-client/src,target=packages/openpi-client/src \
    GIT_LFS_SKIP_SMUDGE=1 uv sync --frozen --dev --no-install-project

COPY . /app

RUN GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

RUN python -c "import transformers; print(transformers.__file__)" \
    | xargs dirname \
    | xargs -I{} cp -r /app/src/openpi/models_pytorch/transformers_replace/* {}

CMD ["/bin/bash"]
