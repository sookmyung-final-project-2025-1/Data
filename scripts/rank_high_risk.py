# scripts/rank_high_risk.py
import os
import argparse
import pandas as pd
from src.fraud.model import FraudModel

def main(args):
    # 1) 모델 불러오기
    model = FraudModel.load_dir(args.model_dir)

    # 2) 데이터 읽기
    df = pd.read_csv(args.csv)
    print(f"[INFO] Loaded {len(df)} rows from {args.csv}")

    # 3) 전처리 + 예측 (벡터화)
    print("[INFO] Transforming...")
    X = model.pre.transform(df)   # DataFrame 전체 변환
    print("[INFO] Predicting...")
    probs = model.clf.predict_proba(X)[:, 1]

    # 4) 확률 추가 + 정렬
    df["fraud_probability"] = probs
    df_sorted = df.sort_values("fraud_probability", ascending=False)

    # 5) 결과 저장
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    df_sorted.to_csv(args.out_csv, index=False)

    print(f"[OK] Saved sorted results → {args.out_csv}")
    print(df_sorted.head(args.topn))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="입력 CSV (예: data/raw/train_transaction.csv)")
    p.add_argument("--model_dir", default="models/v1")
    p.add_argument("--out_csv", default="results/ranked.csv")
    p.add_argument("--topn", type=int, default=10)
    args = p.parse_args()
    main(args)
