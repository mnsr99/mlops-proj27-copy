FROM python:3.11-slim

WORKDIR /app
COPY api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY api/app.py ./app.py
COPY common ./common

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
