# scripts/train_cat.py
from __future__ import annotations
import argparse, os, json
import pandas as pd
import joblib
from catboost import CatBoostClassifier

from src.fraud.preprocessing import FraudPreprocessor

def main(args):
    # 1) 데이터 로드
    csv_path = os.path.join(args.data_dir, "train_transaction.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)

    if "isFraud" not in df.columns:
        raise ValueError("train_transaction.csv 에 isFraud 컬럼이 필요합니다.")

    y = df["isFraud"].astype(int)
    X_raw = df.drop(columns=["isFraud"])

    # 2) 전처리 학습 & 변환 (y 없이 fit)
    pre = FraudPreprocessor()
    pre.fit(X_raw)
    X = pre.transform(X_raw)

    # 3) 모델 학습
    clf = CatBoostClassifier(
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
        loss_function="Logloss",
        random_seed=42,
        verbose=False,
    )
    clf.fit(X, y)

    # 4) 저장
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    joblib.dump(pre, os.path.join(out_dir, "preprocessor.pkl"))
    joblib.dump(clf, os.path.join(out_dir, "model.pkl"))
    metadata = {
        "model_type": "catboost",
        "n_samples": int(len(y)),
        "features": getattr(pre, "feature_names_", None),
        "threshold": args.threshold,
        "params": {
            "iterations": args.iterations,
            "depth": args.depth,
            "learning_rate": args.learning_rate,
        },
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"[OK] saved to {out_dir}")
    print(" - preprocessor.pkl")
    print(" - model.pkl (CatBoost)")
    print(" - metadata.json")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/raw")
    p.add_argument("--out_dir",  default="models/v4")
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--iterations", type=int, default=500)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--learning_rate", type=float, default=0.05)
    main(p.parse_args())