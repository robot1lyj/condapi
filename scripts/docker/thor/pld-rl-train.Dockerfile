# Dedicated offline RL environment; the vendor CUDA/PyTorch stack stays intact.
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch@sha256:9024018b27e9ad043d1b88984b4fb7b705df9778f3688ca7cbaf399031421cde
FROM ${BASE_IMAGE}
LABEL org.opencontainers.image.title="YAM offline RL training on Thor"
RUN python -m venv --system-site-packages /opt/pld-rl
ENV PATH=/opt/pld-rl/bin:${PATH} \
    YAM_RL_TRAINING_TARGET=thor-offline \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_OFFLINE=1 \
    OPENBLAS_NUM_THREADS=1 \
    OMP_NUM_THREADS=2
WORKDIR /workspace
ENTRYPOINT ["/opt/pld-rl/bin/python"]
# The idle worker holds no model or CUDA context; training is an explicit exec.
CMD ["-c", "import threading; threading.Event().wait()"]
