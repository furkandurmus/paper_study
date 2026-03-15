FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/opt/paper_study/.cache/huggingface \
    TRANSFORMERS_CACHE=/opt/paper_study/.cache/huggingface/transformers \
    HUGGINGFACE_HUB_CACHE=/opt/paper_study/.cache/huggingface/hub

WORKDIR /opt/paper_study

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-venv \
    python3-pip \
    build-essential \
    git \
    curl \
    ca-certificates \
    pkg-config \
    libaio-dev \
    libjpeg-dev \
    zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    python -m pip install --upgrade pip setuptools wheel

COPY requirements.txt ./requirements.txt
RUN python -m pip install -r requirements.txt

COPY . .

RUN mkdir -p /opt/paper_study/.cache/huggingface

CMD ["bash"]
