FROM ubuntu:latest
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# Install apt dependencies.
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y \
    git curl build-essential && \
    rm -rf /var/lib/apt/lists/*

# Install pixi
RUN curl -fsSL https://pixi.sh/install.sh | sh
ENV PATH=$PATH:/root/.pixi/bin

# Get this source
COPY . roscon25-workshop
WORKDIR /roscon25-workshop

# Build workspace
RUN pixi run build
