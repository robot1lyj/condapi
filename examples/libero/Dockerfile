# Dockerfile for the LIBERO benchmark.

# Build the container:
# docker build . -t libero -f examples/libero/Dockerfile

# Run the container:
# docker run --rm -it --network=host -v .:/app -v /tmp/.X11-unix:/tmp/.X11-unix:ro -e DISPLAY=$DISPLAY --gpus all libero /bin/bash

FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04@sha256:2d913b09e6be8387e1a10976933642c73c840c0b735f0bf3c28d97fc9bc422e0

RUN apt-get update && \
    apt-get install -y \
    make \
    g++ \
    clang \
    libosmesa6-dev \
    libgl1-mesa-glx \
    libegl1 \
    libglew-dev \
    libglfw3-dev \
    libgles2-mesa-dev \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    python3.8 \
    python3-pip

WORKDIR /app

# Copy the requirements files so we can install dependencies.
# The rest of the project is mounted as a volume, so we don't need to rebuild on changes.
# This strategy is best for development-style usage.
COPY ./examples/libero/requirements.txt /tmp/requirements.txt
COPY ./third_party/libero/requirements.txt /tmp/requirements-libero.txt
COPY ./packages/openpi-client/pyproject.toml /tmp/openpi-client/pyproject.toml

# Install python dependencies.
RUN python3.8 -m pip install --upgrade pip \
    && python3.8 -m pip install -r /tmp/requirements.txt -r /tmp/requirements-libero.txt /tmp/openpi-client/pyproject.toml --extra-index-url https://download.pytorch.org/whl/cu113
ENV PYTHONPATH=/app:/app/packages/openpi-client/src:/app/third_party/libero

# Create a default config file to avoid an input prompt from LIBERO's init script.
# https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/libero/libero/__init__.py
ENV LIBERO_CONFIG_PATH=/tmp/libero
RUN mkdir -p /tmp/libero && cat <<'EOF' > /tmp/libero/config.yaml
benchmark_root: /app/third_party/libero/libero/libero
bddl_files: /app/third_party/libero/libero/libero/bddl_files
init_states: /app/third_party/libero/libero/libero/init_files
datasets: /app/third_party/libero/libero/datasets
assets: /app/third_party/libero/libero/libero/assets
EOF

RUN mkdir -p /usr/share/glvnd/egl_vendor.d && echo '{"file_format_version" : "1.0.0", "ICD" : { "library_path" : "libEGL_nvidia.so.0" }}' > /usr/share/glvnd/egl_vendor.d/10_nvidia.json

CMD ["/bin/bash", "-c", "python3.8 examples/libero/main.py $CLIENT_ARGS"]
