# CPU-only conversion tool derived from the already validated Pi JAX image.
# No GPU access, clock changes or model weights are baked into this image.
FROM openpi-pi:thor-jax-20260907
ENV JAX_PLATFORMS=cpu TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=0
ARG PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
RUN python3 -m pip install --no-cache-dir --index-url ${PIP_INDEX_URL} \
    'transformers==4.53.2' 'safetensors==0.5.3' 'pytest==8.4.2'
COPY src /app/src
COPY examples/convert_jax_model_to_pytorch.py /app/examples/convert_jax_model_to_pytorch.py
COPY scripts/conda/patch_transformers.py /app/scripts/conda/patch_transformers.py
RUN python3 /app/scripts/conda/patch_transformers.py --openpi-dir /app \
    && python3 -m pip install --no-build-isolation --no-deps /app \
    && python3 -m pip freeze > /opt/thor/conversion-packages.txt
CMD ["python3", "/app/examples/convert_jax_model_to_pytorch.py", "--help"]
