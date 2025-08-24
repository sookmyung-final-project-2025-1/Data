#  Fraud Detection Service (IEEE-CIS 스타일) — Preprocess + Model + FastAPI + Docker

: 전처리 + 모델을 **하나의 파이프라인**으로 묶고, `FastAPI`로 **/predict** , **/predict_raw** (원본 칼럼) 엔드포인트 제공
--

## 0) 

```bash
# 0. 파이썬 3.10 기준 권장
python -V

# 1. 가상환경
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)

# 2. 설치
pip install -U pip
pip install -r requirements.txt

# 3. (선택) 모델/전처리 아티팩트 배치
#    models/v1/ 아래에 preprocessor.pkl, model.pkl, metadata.json 배치
#    (아직 없다면, 아래 '학습 & 내보내기' 실행)

# 4. API 실행
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
# 브라우저에서 http://127.0.0.1:8000/docs
```

## 1) 프로젝트 구조

```
fraud-service/
├── api/
│   └── main.py                # FastAPI 엔드포인트
├── src/
│   └── fraud/
│       ├── __init__.py
│       ├── preprocessing.py   # 전처리 파이프라인(Fit/Transform/One)
│       ├── model.py           # 모델 래퍼(전처리+모델 로드/저장)
│       ├── schema.py          # Pydantic 스키마(요청/응답)
│       ├── explain.py         # SHAP 기반 로컬 설명자
│       └── config.py          # 설정 로딩(YAML/.env)
├── scripts/
│   ├── train.py               # 학습(전처리 fit + 모델 train)
│   └── export_assets.py       # 학습 후 아티팩트 내보내기
├── configs/
│   └── default.yaml           # 기본 설정
├── models/
│   └── README.md              # 아티팩트 배치 안내
├── data/
│   └── README.md              # 데이터 배치 안내
├── requirements.txt
├── Dockerfile
├── .gitignore
├── .dockerignore
└── README.md
```

## 2) 엔드포인트 개요

- `POST /predict_raw` : "raw data 컬럼" 입력 → 전처리 수행 → 확률 + (선택) 중요도 반환
- `POST /predict`     : "feature engineering 된 컬럼" 입력 (`features: list[float]`) → 확률 반환 (기존 호환)
- `GET /health`       : 상태 확인

### 예시: `/predict_raw`

```json
{
  "payload": {
    "TransactionDT": 86400,
    "TransactionAmt": 123.45,
    "ProductCD": "W",
    "card1": 12345,
    "card2": 150,
    "card3": 150,
    "card4": "visa",
    "card5": 226,
    "card6": "debit",
    "addr1": 299,
    "addr2": 87,
    "P_emaildomain": "gmail.com"
    // ... (가능한 한 원본 칼럼들; 누락 가능)
  },
  "explain": true,
  "top_n": 5
}
```

## 3) 학습 & 결과

```bash
# data/raw/ 밑에 kaggle train_transaction.csv, train_identity.csv 배치
python scripts/train.py --data_dir data/raw --out_dir models/v1

# 학습 후 artifacts:
# models/v1/preprocessor.pkl
# models/v1/model.pkl
# models/v1/metadata.json
```

## 4) Docker

```bash
# 이미지 빌드
docker build -t fraud-api:latest .

# 실행
docker run --rm -p 8000:8000 -e MODEL_DIR=/app/models/v1 fraud-api:latest
# http://127.0.0.1:8000/docs
```


## 6) 오늘 배포를 위한 Tip

- **V1**: 우선 간단 전처리(시간 파생, 로그금액, 라벨인코딩, frequency-encoding 일부) + LightGBM으로 서비스.
- **V2**: Kaggle 1등팀 핵심(UID 기반 그룹 집계, 시간 기반 CV, 앙상블, UID 평균 후처리)을 점진 도입.
- **설명가능성**: 규제 대응/디버깅을 위해 SHAP 로컬 설명을 `/predict_raw`에 옵션으로 제공.

---

### 참고: 1등팀 요약(배포 관점)

- **UID** 생성(여러 card/addr/email 조합) 후 **UID별 집계 피처**(최근 N시간 카운트/평균 등) + **GBDT 앙상블(XGB/LGBM/CatBoost)**, 검증은 **시간 순서 그룹 KFold**, 최종은 **UID 평균 후처리**. (NVIDIA 블로그 인용)
