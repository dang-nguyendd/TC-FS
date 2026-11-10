# xLSTM Large: CUDA 12.4 + PyTorch 2.4.0 + Python 3.11
# Base image includes the CUDA 12.4.1 development toolkit.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    CONDA_ENV=xlstm \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System build tools and utilities.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    git \
    build-essential \
    wget \
    bzip2 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Install Miniforge (Conda).
RUN curl -L \
    https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh \
    -o /tmp/miniforge.sh \
    && bash /tmp/miniforge.sh -b -p /opt/conda \
    && rm /tmp/miniforge.sh

ENV PATH="/opt/conda/bin:${PATH}"

# Recreate the requested Conda environment.
# CUDA toolkit packages are included to match the supplied environment;
# the base image also provides the system CUDA 12.4.1 compiler/toolkit.
RUN conda create -y -n ${CONDA_ENV} \
    -c pytorch -c nvidia -c conda-forge -c defaults \
    python=3.11 \
    pip \
    cuda=12.4 \
    cuda-nvcc=12.4 \
    cuda-toolkit=12.4 \
    cuda-cccl=12.4 \
    gxx_linux-64=13.2.0 \
    pytorch=2.4.0 \
    pytorch-cuda=12.4 \
    cmake \
    ninja \
    && conda clean -afy

ENV PATH="/opt/conda/envs/xlstm/bin:/opt/conda/bin:${PATH}"

# Python dependencies from the supplied environment.
RUN python -m pip install \
    pre-commit \
    ipykernel \
    dacite \
    pyyaml \
    torchmetrics \
    tqdm \
    pytest \
    pytest-xdist \
    "numpy<2.0"

# Install the official NX-AI xLSTM repository and its Large-model kernel dependency.
WORKDIR /workspace
RUN git clone --depth 1 https://github.com/NX-AI/xlstm.git /workspace/xlstm \
    && python -m pip install mlstm_kernels \
    && python -m pip install -e /workspace/xlstm

WORKDIR /workspace/project
EXPOSE 8888

CMD ["/bin/bash"]