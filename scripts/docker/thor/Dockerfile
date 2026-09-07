# Native JAX: base-model recorded replay validated; NOT finetuned robot acceptance.
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
COPY scripts/docker/thor/preserve_vendor_stack.py /opt/thor/preserve_vendor_stack.py
COPY scripts/docker/thor/requirements-inference.txt /opt/thor/requirements-inference.txt
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
RUN python3 /opt/thor/preserve_vendor_stack.py /opt/thor/vendor-constraints.txt \
    && python3 -m pip install --no-cache-dir --index-url ${PIP_INDEX_URL} \
       -c /opt/thor/vendor-constraints.txt -r /opt/thor/requirements-inference.txt
COPY pyproject.toml README.md LICENSE /app/
COPY src /app/src
COPY packages/openpi-client /app/packages/openpi-client
COPY scripts/thor /app/scripts/thor
RUN python3 -m pip install --no-build-isolation --no-deps /app /app/packages/openpi-client \
    && python3 -m pip freeze > /opt/thor/resolved-packages.txt
# The default remains a GPU smoke; select a recorded replay command explicitly.
CMD ["python3", "/opt/thor/jax_smoke.py"]
