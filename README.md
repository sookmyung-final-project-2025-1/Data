# Fraud Detection API

## 실행
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
# http://127.0.0.1:8000/docs
구조
bash
복사
편집
fraud-service/
├── api/main.py
├── src/fraud/
│   ├── preprocessing.py
│   ├── model.py
│   ├── schema.py
│   ├── explain.py
│   └── config.py
├── scripts/train.py
├── configs/default.yaml
├── models/
└── data/
엔드포인트
POST /predict_raw : 원본 컬럼 입력 → 전처리 → 확률 (+옵션: SHAP)

POST /predict : 전처리된 feature 입력 → 확률

GET /health : 상태 확인

학습 & 결과물
bash
복사
편집
python scripts/train.py --data_dir data/raw --out_dir models/v1
# 생성: preprocessor.pkl, model.pkl, metadata.json
Docker
bash
복사
편집
docker build -t fraud-api:latest .
docker run --rm -p 8000:8000 -e MODEL_DIR=/app/models/v1 fraud-api:latest
배포 TIP
V1: 간단 전처리 + LightGBM

V2: UID 기반 피처 + 시간 기반 CV + 앙상블 + UID 평균 후처리

/predict_raw → SHAP 설명 제공
