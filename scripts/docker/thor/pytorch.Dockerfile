# Candidate Pi-family backend. Not an accepted numerical/performance baseline yet.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.05-py3
FROM ${BASE_IMAGE}
WORKDIR /app
ENV JAX_PLATFORMS=cpu \
    OPENPI_DATA_HOME=/cache \
    HF_HUB_OFFLINE=1 \
    WANDB_MODE=disabled \
    TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0 \
    PIP_DISABLE_PIP_VERSION_CHECK=1
# Keep NVIDIA's torch/CUDA/TensorRT versions. JAX is CPU-only transform/IO
# infrastructure here; checkpoint restoration uses the separate converter image.
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
RUN python3 -m pip install --no-cache-dir --index-url ${PIP_INDEX_URL} \
    'jax==0.6.2' 'jaxlib==0.6.2' 'flax==0.10.2' 'orbax-checkpoint==0.11.13' \
    'transformers==4.53.2' 'augmax==0.3.4' 'beartype==0.19.0' \
    'jaxtyping==0.2.36' 'dm-tree==0.1.9' 'ml_collections==1.0.0' \
    'sentencepiece==0.2.0' 'numpydantic==1.10.0' 'tyro==0.9.35' \
    'tqdm-loggable==0.4.1' 'chex==0.1.89' 'websockets==17.1' \
    'av==18.1.0' 'hatchling==1.32.0' 'pytest==8.4.2'
COPY pyproject.toml README.md LICENSE /app/
COPY src /app/src
COPY packages/openpi-client /app/packages/openpi-client
COPY scripts/thor /app/scripts/thor
COPY scripts/conda/patch_transformers.py /app/scripts/conda/patch_transformers.py
RUN python3 /app/scripts/conda/patch_transformers.py --openpi-dir /app \
    && python3 -m pip install --no-build-isolation --no-deps /app /app/packages/openpi-client \
    && python3 -m pip freeze > /opt/pytorch-candidate-packages.txt
CMD ["python3", "-c", "import torch; print(torch.__version__, torch.cuda.is_available())"]
