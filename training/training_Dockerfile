FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app/training

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/training/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/training/requirements.txt

COPY . /app/training

CMD ["python3", "train.py", "--help"]
