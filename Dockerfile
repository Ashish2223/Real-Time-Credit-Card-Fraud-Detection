FROM python:3.11-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# Train at build time so the image is self-contained (uses synthetic data unless data/creditcard.csv is copied in)
COPY data ./data
RUN python -m fraud_detection.train --fast --no-plots

ENV FRAUD_MODEL_DIR=/app/models FRAUD_DB_PATH=/app/data/predictions.db
EXPOSE 8000
CMD ["uvicorn", "fraud_detection.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
