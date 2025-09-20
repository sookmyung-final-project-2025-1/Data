# scripts/train_eval_cat.py
from __future__ import annotations
import argparse, os, json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    f1_score, confusion_matrix, classification_report
)
from catboost import CatBoostClassifier

from src.fraud.preprocessing import FraudPreprocessor

def pick_best_threshold(y_true: np.ndarray, p: np.ndarray, mode: str = "f1") -> float:
    if mode == "f1":
        prec, rec, thr = precision_recall_curve(y_true, p)
        f1s = 2 * (prec * rec) / (prec + rec + 1e-12)
        thr_full = np.r_[thr, 1.0]
        return float(thr_full[np.nanargmax(f1s)])
    else:
        from sklearn.metrics import roc_curve
        fpr, tpr, thr = roc_curve(y_true, p)
        j = tpr - fpr
        return float(thr[np.nanargmax(j)])

def evaluate(y_true: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    auc = roc_auc_score(y_true, p)
    ap  = average_precision_score(y_true, p)
    yhat = (p >= threshold).astype(int)

    cm = confusion_matrix(y_true, yhat, labels=[0,1])
    report = classification_report(y_true, yhat, output_dict=True, zero_division=0)

    return {
        "roc_auc": float(auc),
        "pr_auc": float(ap),
        "threshold": float(threshold),
        "confusion_matrix": {
            "tn": int(cm[0,0]),
            "fp": int(cm[0,1]),
            "fn": int(cm[1,0]),
            "tp": int(cm[1,1]),
        },
        "classification_report": report,
        "f1": float(f1_score(y_true, yhat)),
        "precision": float(report["1"]["precision"]),
        "recall": float(report["1"]["recall"]),
    }

def main(args):
    # 1) 데이터 로드
    csv_path = os.path.join(args.data_dir, "train_transaction.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if "isFraud" not in df.columns:
        raise ValueError("train_transaction.csv 에 isFraud 컬럼이 필요합니다.")

    y = df["isFraud"].astype(int).values
    X_raw = df.drop(columns=["isFraud"])

    # 2) 전처리 학습
    pre = FraudPreprocessor()
    pre.fit(X_raw)

    # 3) train/valid 분리
    X_train_raw, X_valid_raw, y_train, y_valid = train_test_split(
        X_raw, y, test_size=args.valid_size, random_state=42, stratify=y
    )
    X_train = pre.transform(X_train_raw)
    X_valid = pre.transform(X_valid_raw)

    # 4) 모델 학습
    clf = CatBoostClassifier(
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
        loss_function="Logloss",
        random_seed=42,
        verbose=False,
    )
    clf.fit(X_train, y_train)

    # 5) 성능평가
    p_tr = clf.predict_proba(X_train)[:, 1]
    p_va = clf.predict_proba(X_valid)[:, 1]

    thr = pick_best_threshold(y_valid, p_va, mode=args.thr_mode)

    metrics = {
        "train": evaluate(y_train, p_tr, thr),
        "valid": evaluate(y_valid, p_va, thr),
        "class_ratio": {
            "train": float(np.mean(y_train)),
            "valid": float(np.mean(y_valid)),
            "all": float(np.mean(y)),
        }
    }

    # 6) 저장
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    joblib.dump(pre, os.path.join(out_dir, "preprocessor.pkl"))
    joblib.dump(clf, os.path.join(out_dir, "model.pkl"))
    metadata = {
        "model_type": "catboost",
        "n_samples": int(len(y)),
        "features": getattr(pre, "feature_names_", None),
        "params": {
            "iterations": args.iterations,
            "depth": args.depth,
            "learning_rate": args.learning_rate,
        },
        "chosen_threshold": float(thr),
        "metrics": metrics,
        "notes": "Hold-out(valid) 성능을 metadata에 포함. API threshold는 chosen_threshold 사용 권장."
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # 7) 콘솔 요약
    print(f"\n[OK] saved to {out_dir}")
    print(" - preprocessor.pkl")
    print(" - model.pkl (CatBoost)")
    print(" - metadata.json (metrics 포함)\n")
    print("[Valid] ROC-AUC: {:.4f} | PR-AUC: {:.4f} | F1: {:.4f} | Thr: {:.4f}".format(
        metrics["valid"]["roc_auc"], metrics["valid"]["pr_auc"],
        metrics["valid"]["f1"], metrics["valid"]["threshold"]
    ))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/raw")
    p.add_argument("--out_dir",  default="models/v4")
    p.add_argument("--valid_size", type=float, default=0.2)
    p.add_argument("--thr_mode", choices=["f1","youden"], default="f1")
    p.add_argument("--iterations", type=int, default=500)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--learning_rate", type=float, default=0.05)
    main(p.parse_args())