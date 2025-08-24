FROM python:3.10-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \ 
    build-essential libgomp1 &&     rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy minimal first to leverage cache
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src ./src
COPY api ./api
COPY configs ./configs
COPY models ./models

ENV MODEL_DIR=/app/models/v1
EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
