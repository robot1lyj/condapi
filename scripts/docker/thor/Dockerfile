# Native JAX candidate, NOT a validated YAM deployment until replay tests pass.
# Build on Thor; the server's CUDA 12 training environment is intentionally unchanged.
ARG BASE_IMAGE=nvcr.io/nvidia/jax:26.05-py3
FROM ${BASE_IMAGE}
WORKDIR /app
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false \
    JAX_PLATFORMS=cuda \
    JAX_DEFAULT_MATMUL_PRECISION=highest \
    OPENPI_DATA_HOME=/cache \
    HF_HUB_OFFLINE=1 \
    WANDB_MODE=disabled \
    PIP_DISABLE_PIP_VERSION_CHECK=1
COPY scripts/docker/thor/jax_smoke.py /opt/thor/jax_smoke.py
# The first target is deliberately a GPU test harness with the vendor stack intact.
# Application dependencies are added only after inspecting the base image versions.
CMD ["python3", "/opt/thor/jax_smoke.py"]
