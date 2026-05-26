# Dockerfile for the simple client.

# Build the container:
# docker build . -t simple_client -f examples/simple_client/Dockerfile

# Run the container:
# docker run --rm -it --network=host -v .:/app simple_client /bin/bash

FROM python:3.11-slim

WORKDIR /app

# Copy the requirements files so we can install dependencies.
# The rest of the project is mounted as a volume, so we don't need to rebuild on changes.
# This strategy is best for development-style usage.
COPY ./examples/simple_client/requirements.txt /tmp/requirements.txt
COPY ./packages/openpi-client/pyproject.toml /tmp/openpi-client/pyproject.toml

# Install python dependencies.
RUN python -m pip install --upgrade pip \
    && python -m pip install -r /tmp/requirements.txt /tmp/openpi-client/pyproject.toml
ENV PYTHONPATH=/app:/app/src:/app/packages/openpi-client/src

CMD /bin/bash -c "python examples/simple_client/main.py $SERVER_ARGS"
