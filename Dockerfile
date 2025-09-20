FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgomp1 curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

ARG MODEL_VERSION=v1
ENV MODEL_VERSION=${MODEL_VERSION}
ENV MODEL_DIR=/app/models/${MODEL_VERSION}

ENV LGBM_DIR=/app/models/v5
ENV XGB_DIR=/app/models/v6
ENV CAT_DIR=/app/models/v7

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY api ./api
COPY configs ./configs

COPY models ./models

RUN mkdir -p /app/models/v5 /app/models/v6 /app/models/v7 && \
    if [ -f "/app/models/${MODEL_VERSION}/lgbm_model.pkl" ]; then \
        cp /app/models/${MODEL_VERSION}/lgbm_model.pkl /app/models/v5/model.pkl && \
        cp /app/models/${MODEL_VERSION}/lgbm_preprocessor.pkl /app/models/v5/preprocessor.pkl; \
    fi && \
    if [ -f "/app/models/${MODEL_VERSION}/xgb_model.pkl" ]; then \
        cp /app/models/${MODEL_VERSION}/xgb_model.pkl /app/models/v6/model.pkl && \
        cp /app/models/${MODEL_VERSION}/xgb_preprocessor.pkl /app/models/v6/preprocessor.pkl; \
    fi && \
    if [ -f "/app/models/${MODEL_VERSION}/cat_model.pkl" ]; then \
        cp /app/models/${MODEL_VERSION}/cat_model.pkl /app/models/v7/model.pkl && \
        cp /app/models/${MODEL_VERSION}/cat_preprocessor.pkl /app/models/v7/preprocessor.pkl; \
    fi

EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--proxy-headers"]