# scripts/batch_predict.py
import os, json, glob, argparse, time
import requests
import pandas as pd

def main(args):
    files = sorted(glob.glob(os.path.join(args.in_dir, "*.json")))
    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)

    rows = []
    for fp in files:
        with open(fp) as f:
            data = json.load(f)
        try:
            r = requests.post(args.host.rstrip("/") + "/predict_raw", json=data, timeout=30)
            r.raise_for_status()
            res = r.json()
            rows.append({
                "file": os.path.basename(fp),
                "fraud_probability": res.get("fraud_probability"),
                "top_features": json.dumps(res.get("top_features", []), ensure_ascii=False)
            })
        except Exception as e:
            rows.append({
                "file": os.path.basename(fp),
                "fraud_probability": None,
                "top_features": None
            })
            print(f"[WARN] {fp}: {e}")

        if args.sleep > 0:
            time.sleep(args.sleep)

    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"[OK] saved: {args.out_csv} ({len(rows)} rows)")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", default="sample_payloads/bulk")
    p.add_argument("--host", default="http://127.0.0.1:8000")
    p.add_argument("--out_csv", default="results/predictions.csv")
    p.add_argument("--sleep", type=float, default=0.0, help="요청 간 대기(초)")
    main(p.parse_args())
