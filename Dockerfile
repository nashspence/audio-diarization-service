FROM pytorch/pytorch:2.1.1-cuda12.1-cudnn8-runtime

ARG DIARIZEN_REF=d52b8d5e3d96632b1a8a0dc34762bf811471e441
ARG TORCH_VERSION=2.11.0
ARG TORCHAUDIO_VERSION=2.11.0
ARG TORCHCODEC_VERSION=0.11.0
ARG PYTORCH_INDEX_URL=https://download.pytorch.org/whl/cu128

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        ffmpeg \
        git \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN git init /opt/diarizen-src \
    && git -C /opt/diarizen-src remote add origin https://github.com/BUTSpeechFIT/DiariZen \
    && git -C /opt/diarizen-src fetch --depth 1 origin "${DIARIZEN_REF}" \
    && git -C /opt/diarizen-src checkout FETCH_HEAD

COPY requirements.txt /tmp/app-requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --no-cache-dir --upgrade --index-url "${PYTORCH_INDEX_URL}" \
        "torch==${TORCH_VERSION}+cu128" \
        "torchaudio==${TORCHAUDIO_VERSION}+cu128" \
        "torchcodec==${TORCHCODEC_VERSION}+cu128" \
    && python -m pip install --no-cache-dir -r /tmp/app-requirements.txt \
    && python -m pip install --no-cache-dir numpy==1.26.4 scipy toml einops \
    && python -m pip install --no-cache-dir /opt/diarizen-src/pyannote-audio \
    && python -m pip install --no-cache-dir /opt/diarizen-src \
    && rm -rf /root/.cache/pip

COPY app /app/app

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
