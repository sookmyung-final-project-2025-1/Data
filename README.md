
### Fraud Detection 

- 사기 거래 탐지 API
- 모델 학습, 전처리, API 서버 실행

---

### 구조

```text
fraud-service/
├── api/               # FastAPI 엔드포인트 (main.py)
├── configs/           # 설정 파일 (yaml)
├── data/              # 데이터 저장소
│   └── raw/           # 원본 CSV (train_transaction.csv 등)
├── models/            # 학습된 모델/전처리기 저장 위치
│   └── v1/            # preprocessor.pkl, model.pkl, metadata.json
├── sample_payloads/   # API 테스트용 샘플 JSON
├── scripts/           # 학습 및 유틸리티 스크립트
├── src/               # 전처리, 모델, 설명 모듈
├── tests/             # 간단한 테스트 코드
├── requirements.txt   # 의존성 패키지 목록
└── README.md
```

### 실행방법

1. 가상환경 생성 및 의존성 설치
   ```text
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. 모델 학습
   ```text
   python -m scripts.train --data_dir data/raw --out_dir models/v1
   ```
3. API 실행
   ```text
   python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
   ```
4. 브라우저에서 확인 (로컬)
   ```text
    http://127.0.0.1:8000/docs
    ```
5. 브라우저에서 확인 (운영)
   ```text
    https://211.110.155.54/model/docs
    ```
