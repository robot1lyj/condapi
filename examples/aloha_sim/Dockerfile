# Dockerfile for the Aloha simulation environment.

# Build the container:
# docker build . -t aloha_sim -f examples/aloha_sim/Dockerfile

# Run the container:
# docker run --rm -it --network=host -v .:/app aloha_sim /bin/bash

FROM python:3.11-slim@sha256:370c586a6ffc8c619e6d652f81c094b34b14b8f2fb9251f092de23f16e299b78

RUN apt-get update && \
    apt-get install -y \
    libosmesa6-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libglfw3-dev \
    libgles2-mesa-dev
ENV MUJOCO_GL=egl

WORKDIR /app

# Copy the requirements files so we can install dependencies.
# The rest of the project is mounted as a volume, so we don't need to rebuild on changes.
# This strategy is best for development-style usage.
COPY ./examples/aloha_sim/requirements.txt /tmp/requirements.txt
COPY ./packages/openpi-client/pyproject.toml /tmp/openpi-client/pyproject.toml

# Install python dependencies.
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /tmp/requirements.txt /tmp/openpi-client/pyproject.toml
ENV PYTHONPATH=/app:/app/src:/app/packages/openpi-client/src

CMD ["/bin/bash", "-c", "python examples/aloha_sim/main.py"]
