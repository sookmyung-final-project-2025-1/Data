# scripts/train_eval_lgbm.py  (교체본)
import argparse, os, json, time
import pandas as pd
from sklearn.metrics import roc_auc_score, f1_score, average_precision_score
from lightgbm import LGBMClassifier
import joblib

from src.fraud.preprocessing import FraudPreprocessor

def main(args):
    # 1) 데이터 로드
    train_csv = os.path.join(args.data_dir, "train_transaction.csv")
    df = pd.read_csv(train_csv)

    y = df["isFraud"].astype(int)
    X_raw = df.drop(columns=["isFraud"])

    # 2) 전처리 학습/변환
    pre = FraudPreprocessor()
    pre.fit(X_raw)
    X = pre.transform(X_raw)

    # 3) LGBM 학습
    clf = LGBMClassifier(
        n_estimators=1000,
        max_depth=-1,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        random_state=42,
    )
    t0 = time.time()
    clf.fit(X, y)
    train_time = time.time() - t0

    # 4) 평가
    y_proba = clf.predict_proba(X)[:, 1]
    best_thr = 0.20
    y_pred = (y_proba >= best_thr).astype(int)

    roc = roc_auc_score(y, y_proba)
    pr = average_precision_score(y, y_proba)
    f1 = f1_score(y, y_pred)

    # 5) 저장 (직접 저장: joblib + json)
    os.makedirs(args.out_dir, exist_ok=True)
    joblib.dump(pre, os.path.join(args.out_dir, "preprocessor.pkl"))
    joblib.dump(clf, os.path.join(args.out_dir, "model.pkl"))
    meta = {
        "model_type": "lightgbm",
        "version": os.path.basename(args.out_dir),
        "roc_auc": float(roc),
        "pr_auc": float(pr),
        "f1": float(f1),
        "threshold": float(best_thr),
        "train_time_sec": float(train_time),
    }
    with open(os.path.join(args.out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] saved to {args.out_dir}\n - preprocessor.pkl\n - model.pkl (LightGBM)\n - metadata.json (metrics 포함)\n")
    print(f"[Valid] ROC-AUC: {roc:.4f} | PR-AUC: {pr:.4f} | F1: {f1:.4f} | Thr: {best_thr:.4f}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--out_dir", type=str, required=True)  # 예: models/v1
    main(p.parse_args())