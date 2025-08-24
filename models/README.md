# models/
- 학습 후 생성되는 아티팩트 저장 위치입니다.
- 필수 파일
  - `preprocessor.pkl` : 전처리 파이프라인(라벨인코더, frequency map 등 포함)
  - `model.pkl`        : LightGBM 또는 기타 분류 모델
  - `metadata.json`    : 모델 버전/피처리스트/학습일 등 메타
