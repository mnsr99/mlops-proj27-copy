# Python 3.12 to match the training env (python: 3.12.3) captured in the
# MLflow model's python_env.yaml. Pickled transformers objects reference
# internal modules (e.g. transformers.core_model_loading) that only exist
# on matching library versions, so any drift here => load-time import error.
FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl \
    && rm -rf /var/lib/apt/lists/*

# Install torch from the CPU-only wheel index first — this avoids pulling
# the default torch wheel that ships with CUDA runtime (~2GB). The k3s
# node has no GPU, so we'd be wasting disk and image-pull time.
# Training pins only mlflow==2.19.0; torch is unpinned upstream, so let pip
# resolve the latest CPU wheel. Installed from the CPU-only index to avoid
# pulling ~2GB of CUDA runtime the k3s node can't use.
RUN pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      torch

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./app.py

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
