
# Fraud Detection 

 사기 거래 탐지 API - 3개 모델 지원 (LGBM, XGBoost, CatBoost)

## 주요 기능

- **3개 모델 적용**: LGBM, XGBoost, CatBoost
- **UID 기반 Feature Engineering**
- **성능**: ROC-AUC 0.96+ 달성
- **RESTful API**: FastAPI 기반 API
- **실시간 예측**: 사기 확률 + 설명 제공

##  모델 성능

| 모델 | ROC-AUC | PR-AUC | F1 Score |
|------|---------|---------|----------|
| **LGBM** | **0.9639** | 0.7307 | 0.6699 |
| **XGBoost** | **0.9137** | 0.5124 | 0.5050 |
| **CatBoost** | **0.8613** | 0.3401 | 0.3799 |

-------
##  API 

### 모델별 예측
- `POST /model/lgbm/predict` - LightGBM 모델
- `POST /model/xgboost/predict` - XGBoost 모델
- `POST /model/catboost/predict` - CatBoost 모델

### 기타
- `GET /models/info` - 모델 상태 확인
- `GET /health` - 헬스체크
- `POST /predict_raw` - 기본 예측 (LGBM 사용)

##  구조

```text
fraud-service/
├── api/               # FastAPI 엔드포인트
├── data/              # 데이터 저장소
├── models/            # 학습된 모델들
│   ├── v5/            # LGBM (최고 성능)
│   ├── v6/            # XGBoost
│   └── v7/            # CatBoost
├── scripts/           # 학습 스크립트
│   ├── train_eval_lgbm.py
│   ├── train_eval_xgb.py
│   └── train_eval_cat.py
├── src/fraud/         # 핵심 모듈
│   ├── preprocessing.py  # UID 특성 공학
│   ├── model.py
│   └── schema.py
└── requirements.txt
```

## 설치 및 실행

### 1. 환경 설정
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 모델 학습 
```bash
# LGBM 학습
PYTHONPATH=. python scripts/train_eval_lgbm.py --data_dir data/raw --out_dir models/v5

# XGBoost 학습
PYTHONPATH=. python scripts/train_eval_xgb.py --data_dir data/raw --out_dir models/v6

# CatBoost 학습
PYTHONPATH=. python scripts/train_eval_cat.py --data_dir data/raw --out_dir models/v7
```

### 3. API 서버 실행
```bash
python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

##  접속 주소

- **개발환경**: http://127.0.0.1:8000/docs
- **운영환경**: https://211.110.155.54/model/docs

## API 사용 예시

```bash
curl -X POST "https://211.110.155.54/model/lgbm/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "payload": {
      "TransactionAmt": 100.0,
      "ProductCD": "W",
      "card1": 1234,
      "addr1": 100,
      "D1": 50
    },
    "explain": true,
    "top_n": 5
  }'
```
