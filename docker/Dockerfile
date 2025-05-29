FROM ubuntu:24.04

RUN apt-get update && apt-get install -y \
    build-essential \
    llvm-dev \
    libboost-graph-dev \
    libpython3-dev \
    curl

# Install uv and add it to PATH
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

ADD . /workspace

WORKDIR /workspace

RUN uv venv && uv pip install .