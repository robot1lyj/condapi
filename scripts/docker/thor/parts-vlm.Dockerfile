# Offline label review only; no robot devices or production service entry point.
ARG BASE_IMAGE=parts-vision:thor-sam3-0570b3a-20261010-r3
FROM ${BASE_IMAGE}
COPY wheels /opt/parts-vlm-wheels
RUN python -m pip install --no-index --no-deps /opt/parts-vlm-wheels/*.whl \
    && python -c "from transformers import Qwen3VLForConditionalGeneration, AutoProcessor" \
    && python -m pip freeze --all > /opt/parts-vlm-wheels/installed-freeze.txt
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
