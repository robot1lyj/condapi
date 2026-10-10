# Build context contains the pinned upstream sam3/ directory and these files.
# Thor inference only. No robot devices, model credentials or weights in image.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch@sha256:9024018b27e9ad043d1b88984b4fb7b705df9778f3688ca7cbaf399031421cde
FROM ${BASE_IMAGE}

ARG SAM3_REVISION=0570b3a5be9c4e694f23d85232fb55f4a6f1f7fc
LABEL org.opencontainers.image.title="PARTS grasp vision on Thor"
LABEL ai.wuyan.sam3.revision=${SAM3_REVISION}

# SAM3 declares NumPy <2. Use a private venv to preserve the vendor runtime.
RUN python -m venv --system-site-packages /opt/parts-vision
ENV PATH=/opt/parts-vision/bin:${PATH} \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    OMP_NUM_THREADS=4
COPY preserve_vendor_stack.py /opt/parts-build/preserve_vendor_stack.py
RUN python /opt/parts-build/preserve_vendor_stack.py /opt/parts-build/vendor-all.txt --backend pytorch \
    && sed '/^numpy==/d' /opt/parts-build/vendor-all.txt > /opt/parts-build/vendor-gpu.txt
COPY parts-vision-requirements.txt /opt/parts-build/requirements.txt
RUN python -m pip install --no-cache-dir -c /opt/parts-build/vendor-gpu.txt -r /opt/parts-build/requirements.txt
COPY sam3 /opt/sam3
RUN python -m pip install --no-cache-dir --no-deps /opt/sam3 \
    && python -m pip freeze --all > /opt/parts-build/installed-freeze.txt
COPY parts_vision_audit.py /opt/parts-vision-audit.py
ENTRYPOINT ["/opt/parts-vision/bin/python"]
CMD ["/opt/parts-vision-audit.py", "--output", "/tmp/environment-audit.json"]
