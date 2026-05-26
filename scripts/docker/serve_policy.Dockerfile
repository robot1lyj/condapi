# Dockerfile for serving a PI policy with the conda environment.

# Build the container:
# docker build . -t openpi_server -f scripts/docker/serve_policy.Dockerfile

# Run the container:
# docker run --rm -it --network=host -v .:/app --gpus=all openpi_server /bin/bash

FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04@sha256:2d913b09e6be8387e1a10976933642c73c840c0b735f0bf3c28d97fc9bc422e0

WORKDIR /app

RUN apt-get update \
    && apt-get install -y git git-lfs curl ca-certificates linux-headers-generic build-essential clang ffmpeg libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Silence NGC license banner in interactive shells.
RUN for f in /etc/profile /etc/bash.bashrc /etc/profile.d/*; do \
    if [ -f "$f" ] && grep -q "NGC-DL-CONTAINER-LICENSE" "$f"; then \
        sed -i '/NGC-DL-CONTAINER-LICENSE/d' "$f"; \
    fi; \
  done

ENV CONDA_DIR=/opt/conda
ENV PATH="${CONDA_DIR}/bin:${PATH}"
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

RUN curl -fsSL https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh -o /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p "${CONDA_DIR}" \
    && rm /tmp/miniforge.sh \
    && conda clean -afy

COPY environment.pi-conda.yml /tmp/environment.pi-conda.yml
COPY scripts/conda/requirements-pi-pip.txt /tmp/requirements-pi-pip.txt
RUN conda env create -f /tmp/environment.pi-conda.yml \
    && conda run -n pi-conda python -m pip install -r /tmp/requirements-pi-pip.txt \
    && conda clean -afy

COPY . /app

RUN conda run -n pi-conda python -m pip install --no-build-isolation --no-deps -e packages/openpi-client \
    && conda run -n pi-conda python -m pip install --no-build-isolation --no-deps -e . \
    && conda run -n pi-conda python scripts/conda/patch_transformers.py --openpi-dir /app

CMD /bin/bash -c "conda run -n pi-conda python scripts/serve_policy.py $SERVER_ARGS"
