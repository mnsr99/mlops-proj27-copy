FROM python:3.11-slim

WORKDIR /app

# System deps: git for HF downloads (fallback), build-essential is NOT needed
# since torch wheels are prebuilt for linux/amd64.
RUN apt-get update \
    && apt-get install -y --no-install-recommends git curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py ./app.py

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
