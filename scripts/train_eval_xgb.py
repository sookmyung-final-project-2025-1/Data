# scripts/train_eval_xgb.py
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
from xgboost import XGBClassifier

from src.fraud.preprocessing import FraudPreprocessor

def pick_best_threshold(y_true: np.ndarray, p: np.ndarray, mode: str = "f1") -> float:
    """
    PR-curve를 훑어서 F1(또는 J=TPR-FPR)을 최대화하는 임계값을 고른다.
    """
    if mode == "f1":
        prec, rec, thr = precision_recall_curve(y_true, p)
        f1s = 2 * (prec * rec) / (prec + rec + 1e-12)
        # precision_recall_curve는 마지막 점에서 threshold가 없음 -> thr와 길이 맞추기
        thr_full = np.r_[thr, 1.0]
        return float(thr_full[np.nanargmax(f1s)])
    else:  # Youden J
        from sklearn.metrics import roc_curve
        fpr, tpr, thr = roc_curve(y_true, p)
        j = tpr - fpr
        return float(thr[np.nanargmax(j)])

def evaluate(y_true: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    auc = roc_auc_score(y_true, p)
    ap  = average_precision_score(y_true, p)  # PR-AUC
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

    # 3) train/valid 분리 (Stratified hold-out)
    #    - 전처리 후 분리하면 데이터 누수 위험이 있어, 먼저 분리 → 각자 transform 권장.
    X_train_raw, X_valid_raw, y_train, y_valid = train_test_split(
        X_raw, y, test_size=args.valid_size, random_state=42, stratify=y
    )
    X_train = pre.transform(X_train_raw)
    X_valid = pre.transform(X_valid_raw)

    # 4) 모델 학습
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
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    # 5) 성능 평가
    p_tr = clf.predict_proba(X_train)[:, 1]
    p_va = clf.predict_proba(X_valid)[:, 1]

    # 임계값 선택: 검증셋 기반(F1 최대)
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
        "model_type": "xgboost",
        "n_samples": int(len(y)),
        "features": getattr(pre, "feature_names_", None),
        "params": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "reg_alpha": args.reg_alpha,
            "reg_lambda": args.reg_lambda,
        },
        "chosen_threshold": float(thr),
        "metrics": metrics,
        "notes": "Hold-out(valid) 성능을 metadata에 포함. API threshold는 chosen_threshold 사용 권장."
    }
    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    # 7) 콘솔 출력 (요약)
    print(f"\n[OK] saved to {out_dir}")
    print(" - preprocessor.pkl")
    print(" - model.pkl (XGBoost)")
    print(" - metadata.json (metrics 포함)\n")
    print("[Valid] ROC-AUC: {:.4f} | PR-AUC: {:.4f} | F1: {:.4f} | Thr: {:.4f}".format(
        metrics["valid"]["roc_auc"], metrics["valid"]["pr_auc"],
        metrics["valid"]["f1"], metrics["valid"]["threshold"]
    ))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="data/raw")
    p.add_argument("--out_dir",  default="models/v3")
    p.add_argument("--valid_size", type=float, default=0.2)
    p.add_argument("--thr_mode", choices=["f1","youden"], default="f1")
    # 하이퍼파라미터
    p.add_argument("--n_estimators", type=int, default=500)
    p.add_argument("--max_depth", type=int, default=6)
    p.add_argument("--learning_rate", type=float, default=0.05)
    p.add_argument("--subsample", type=float, default=0.8)
    p.add_argument("--colsample_bytree", type=float, default=0.8)
    p.add_argument("--reg_alpha", type=float, default=0.0)
    p.add_argument("--reg_lambda", type=float, default=0.0)
    main(p.parse_args())