from __future__ import annotations
import argparse, os, json, pandas as pd, numpy as np, joblib
from sklearn.model_selection import train_test_split
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from src.fraud.preprocessing import FraudPreprocessor
from src.fraud.config import load_config

def load_kaggle_train(data_dir: str):
    tt = pd.read_csv(os.path.join(data_dir, "train_transaction.csv"))
    ti = pd.read_csv(os.path.join(data_dir, "train_identity.csv"))
    df = tt.merge(ti, on="TransactionID", how="left")
    y = df["isFraud"].astype(int).values
    df = df.drop(columns=["isFraud"])
    return df, y

def main(args):
    cfg = load_config(args.config)
    df, y = load_kaggle_train(args.data_dir)

    # fit preprocessor on train split only
    X_train, X_valid, y_train, y_valid = train_test_split(df, y, test_size=cfg.train["test_size"], random_state=cfg.seed, stratify=y)

    pre = FraudPreprocessor(
        reference_date=cfg.reference_date,
        cat_cols=cfg.features["cats"],
        freq_cols=cfg.features["freqs"],
        num_cols=cfg.features["nums"],
        use_missing_flags=cfg.features["use_missing_flags"]
    )
    pre.fit(X_train)
    Xtr = pre.transform(X_train).values
    Xva = pre.transform(X_valid).values

    clf = LGBMClassifier(**cfg.train["lgbm_params"], random_state=cfg.seed)
    clf.fit(
    Xtr, y_train,
    eval_set=[(Xva, y_valid)],
    eval_metric="auc",
    callbacks=[
        early_stopping(cfg.train["early_stopping_rounds"]),
        log_evaluation(50),   # 50 라운드마다 로그
    ],)


    os.makedirs(args.out_dir, exist_ok=True)
    pre.save(os.path.join(args.out_dir, "preprocessor.pkl"))
    joblib.dump(clf, os.path.join(args.out_dir, "model.pkl"))
    meta = {
        "feature_order": pre.feature_order_,
        "training_rows": int(len(Xtr)),
        "valid_rows": int(len(Xva)),
        "config": cfg.__dict__,
    }
    with open(os.path.join(args.out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data/raw")
    p.add_argument("--out_dir", type=str, default="models/v1")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    main(p.parse_args())
