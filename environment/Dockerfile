# CPU-only default base image. For GPU tasks: set gpus = 1 in task.toml,
# use a CUDA base image instead, and pin the CUDA build of your ML
# framework in requirements.txt. Examples (pick one, delete the FROM below):
#   FROM pytorch/pytorch:2.4.1-cuda12.4-cudnn9-runtime
#   FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04
# Tasks are validated in single-container, single-GPU sandboxes — do not
# assume more than one GPU.

FROM nvidia/cuda:12.9.1-runtime-ubuntu24.04

ENV PYTHONUNBUFFERED=1
ENV PYTHONHASHSEED=0
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        procps \
        ripgrep \
        python3.12 \
        python3.12-venv \
        python3-pip \
    && rm -rf /var/lib/apt/lists/* \
    && python3.12 -m venv /opt/venv

ENV PATH="/opt/venv/bin:${PATH}"

COPY requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt \
    && python -c "import xgboost as xgb; assert xgb.build_info().get('USE_CUDA')"

COPY task_inputs/ /app/data/

RUN mkdir -p /app/src /app/artifacts /app/reports /app/models \
    && test -f /app/data/train.npz \
    && test -f /app/data/val.npz

ENV OTTER_DATA_PATH=/app/data/train.npz
ENV OTTER_VAL_PATH=/app/data/val.npz
ENV OTTER_DEVICE=cuda
