# scripts/train_xgb.py
from __future__ import annotations
import argparse, os, json
import pandas as pd
import joblib
from xgboost import XGBClassifier

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
    clf = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        reg_alpha=args.reg_alpha,
        reg_lambda=args.reg_lambda,
        eval_metric="logloss",
        tree_method="hist",
        random_state=42,
    )
    clf.fit(X, y)

    # 4) 저장 (preprocessor.pkl, model.pkl, metadata.json)
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    joblib.dump(pre, os.path.join(out_dir, "preprocessor.pkl"))
    joblib.dump(clf, os.path.join(out_dir, "model.pkl"))
    metadata = {
        "model_type": "xgboost",
        "n_samples": int(len(y)),
        "features": getattr(pre, "feature_names_", None),
        "threshold": args.threshold,
        "params": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "reg_alpha": args.reg_alpha,
            "reg_lambda": args.reg_lambda,
        },
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"[OK] saved to {out_dir}")
    print(" - preprocessor.pkl")
    print(" - model.pkl (XGBoost)")
    print(" - metadata.json")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/raw")
    p.add_argument("--out_dir",  default="models/v3")
    p.add_argument("--threshold", type=float, default=0.5)
    # 하이퍼파라미터
    p.add_argument("--n_estimators", type=int, default=500)
    p.add_argument("--max_depth", type=int, default=6)
    p.add_argument("--learning_rate", type=float, default=0.05)
    p.add_argument("--subsample", type=float, default=0.8)
    p.add_argument("--colsample_bytree", type=float, default=0.8)
    p.add_argument("--reg_alpha", type=float, default=0.0)
    p.add_argument("--reg_lambda", type=float, default=0.0)
    main(p.parse_args())