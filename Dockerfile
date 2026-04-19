FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3.10-dev \
    git \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    curl \
    unzip \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/aed-xai

COPY requirements.txt /workspace/aed-xai/requirements.txt
COPY requirements-yolox.txt /workspace/aed-xai/requirements-yolox.txt

RUN python3.10 -m pip install --upgrade pip setuptools wheel && \
    python3.10 -m pip install -r requirements.txt && \
    python3.10 -m pip install --no-build-isolation -r requirements-yolox.txt

COPY . /workspace/aed-xai

ENV PYTHONPATH=/workspace/aed-xai:${PYTHONPATH}
