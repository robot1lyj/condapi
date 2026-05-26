# Dev/offline image for openpi. Builds a conda-based environment.
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04@sha256:2d913b09e6be8387e1a10976933642c73c840c0b735f0bf3c28d97fc9bc422e0

ARG USER_NAME=linyongjia
ARG USER_UID=1110
ARG USER_GID=1011

WORKDIR /app

RUN apt-get update \
    && apt-get install -y git git-lfs wget curl rsync ca-certificates linux-headers-generic build-essential clang sudo libgl1 libglib2.0-0 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Silence NGC license banner in interactive shells.
RUN for f in /etc/profile /etc/bash.bashrc /etc/profile.d/*; do \
    if [ -f "$f" ] && grep -q "NGC-DL-CONTAINER-LICENSE" "$f"; then \
        sed -i '/NGC-DL-CONTAINER-LICENSE/d' "$f"; \
    fi; \
  done

RUN if ! getent group "${USER_GID}" >/dev/null; then groupadd -g "${USER_GID}" "${USER_NAME}"; fi \
    && if ! id -u "${USER_NAME}" >/dev/null 2>&1; then useradd -m -u "${USER_UID}" -g "${USER_GID}" -s /bin/bash "${USER_NAME}"; fi \
    && mkdir -p /openpi_cache "/home/${USER_NAME}/.cache" \
    && chown -R "${USER_UID}:${USER_GID}" /openpi_cache "/home/${USER_NAME}/.cache" \
    && echo "${USER_NAME} ALL=(ALL) NOPASSWD:ALL" > "/etc/sudoers.d/${USER_NAME}" \
    && chmod 440 "/etc/sudoers.d/${USER_NAME}"

ENV USER="${USER_NAME}"
ENV HOME="/home/${USER_NAME}"
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

SHELL ["conda", "run", "-n", "pi-conda", "/bin/bash", "-c"]

CMD ["/bin/bash"]
