FROM ubuntu:24.04

RUN apt-get update && apt-get install -y \
    build-essential \
    llvm-dev \
    libboost-graph-dev \
    libpython3-dev \
    curl

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ADD . /workspace

WORKDIR /workspace

RUN uv pip install .