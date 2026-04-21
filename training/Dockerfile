FROM python:3.10-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# optional system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY run_online_pipeline.py /app/run_online_pipeline.py

# create writable dirs for tmp/cache/manifest
RUN mkdir -p /app/data /cache/tmp /cache/hf

ENV TMPDIR=/cache/tmp
ENV HF_HOME=/cache/hf
ENV TRANSFORMERS_CACHE=/cache/hf

CMD ["python", "-u", "run_online_pipeline.py"]
