# scripts/make_payloads.py
import os, json, argparse
import pandas as pd
import numpy as np

EXPECTED_RAW_COLS = [
    "TransactionDT", "TransactionAmt", "ProductCD",
    "card1","card2","card3","card4","card5","card6",
    "addr1","addr2","P_emaildomain"
]

def nan_to_none(x):
    if isinstance(x, float) and (np.isnan(x) or np.isinf(x)):
        return None
    if pd.isna(x):
        return None
    return x

def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    tt = pd.read_csv(os.path.join(args.data_dir, "train_transaction.csv"))
    # identity 병합(있으면)
    id_path = os.path.join(args.data_dir, "train_identity.csv")
    if os.path.exists(id_path):
        ti = pd.read_csv(id_path)
        df = tt.merge(ti, on="TransactionID", how="left")
    else:
        df = tt

    # 샘플링
    if args.random:
        df = df.sample(args.n, random_state=args.seed)
    else:
        df = df.head(args.n)

    # 여러 개 JSON 생성
    made = 0
    for i, row in df.reset_index(drop=True).iterrows():
        payload_raw = {c: nan_to_none(row[c]) for c in EXPECTED_RAW_COLS if c in df.columns}
        if not payload_raw:
            continue
        req = {"payload": payload_raw, "explain": args.explain, "top_n": args.top_n}
        out_path = os.path.join(args.out_dir, f"row_{i:05d}.json")
        with open(out_path, "w") as f:
            json.dump(req, f, ensure_ascii=False, indent=2)
        made += 1

    print(f"[OK] {made} files saved to {args.out_dir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True, help="data/raw 경로")
    p.add_argument("--out_dir", default="sample_payloads/bulk")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--random", action="store_true", help="랜덤 샘플링 여부")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--explain", action="store_true")
    p.add_argument("--top_n", type=int, default=5)
    main(p.parse_args())
