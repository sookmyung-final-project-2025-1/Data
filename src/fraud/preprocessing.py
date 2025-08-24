from __future__ import annotations
import numpy as np, pandas as pd, joblib, json
from datetime import datetime, timedelta
from typing import Dict, List, Any

class FraudPreprocessor:
    """간단하지만 생산 가능한 전처리 파이프라인.
    - TransactionDT -> datetime(+ hour/dayofweek/day)
    - TransactionAmt -> log1p, cents
    - 라벨인코딩(ProductCD, card4, card6, P_emaildomain)
    - frequency encoding(card1, card2, addr1, P_emaildomain)
    - missing flags
    저장되는 자산:
      - label_maps_: dict[str, dict]
      - freq_maps_: dict[str, dict]
      - feature_order_: List[str]
      - ref_date_: datetime
    """
    def __init__(self, reference_date: str = "2017-12-01",
                 cat_cols: List[str] | None = None,
                 freq_cols: List[str] | None = None,
                 num_cols: List[str] | None = None,
                 use_missing_flags: bool = True):
        self.ref_date_ = datetime.fromisoformat(reference_date)
        self.cat_cols = cat_cols or ["ProductCD", "card4", "card6", "P_emaildomain"]
        self.freq_cols = freq_cols or ["card1", "card2", "addr1", "P_emaildomain"]
        self.num_cols = num_cols or ["TransactionAmt", "dist1", "dist2"]
        self.use_missing_flags = use_missing_flags

        self.label_maps_: Dict[str, Dict[Any, int]] = {}
        self.freq_maps_: Dict[str, Dict[Any, float]] = {}
        self.feature_order_: List[str] = []

    # ---------- helpers ----------
    def _to_datetime_parts(self, s: pd.Series) -> pd.DataFrame:
        # s: seconds offset from ref
        dt = self.ref_date_ + pd.to_timedelta(s.fillna(0).astype(float), unit="s")
        out = pd.DataFrame({
            "hour": dt.dt.hour.astype("int16"),
            "dow": dt.dt.dayofweek.astype("int16"),
            "day": dt.dt.day.astype("int16"),
            "is_weekend": dt.dt.dayofweek.isin([5,6]).astype("int8")
        })
        return out

    def _label_encode_fit(self, df: pd.DataFrame, col: str):
        vals = df[col].astype("category").cat.categories.tolist()
        self.label_maps_[col] = {v:i+1 for i,v in enumerate(vals)}  # 0: unseen/NaN
    def _label_encode_apply(self, s: pd.Series, col: str) -> pd.Series:
        m = self.label_maps_.get(col, {})
        return s.map(m).fillna(0).astype("int32")

    def _freq_fit(self, df: pd.DataFrame, col: str):
        vc = df[col].value_counts(dropna=False)
        total = float(vc.sum())
        m = (vc / max(total,1.0)).to_dict()
        self.freq_maps_[col] = m
    def _freq_apply(self, s: pd.Series, col: str) -> pd.Series:
        m = self.freq_maps_.get(col, {})
        return s.map(m).fillna(0.0).astype("float32")

    # ---------- public ----------
    def fit(self, df: pd.DataFrame):
        # fit label/freq maps on training data
        for c in self.cat_cols:
            if c in df.columns:
                self._label_encode_fit(df, c)
        for c in self.freq_cols:
            if c in df.columns:
                self._freq_fit(df, c)
        # define output feature order
        dummy = self.transform(df.head(2), fit_phase=True)
        self.feature_order_ = list(dummy.columns)
        return self

    def transform(self, df: pd.DataFrame, fit_phase: bool=False) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)

        # datetime parts
        if "TransactionDT" in df.columns:
            out = out.join(self._to_datetime_parts(df["TransactionDT"]))
        else:
            out = out.assign(hour=0, dow=0, day=0, is_weekend=0)

        # amount transforms
        amt = df.get("TransactionAmt")
        if amt is not None:
            out["amt"] = amt.fillna(0).astype("float32")
            out["amt_log"] = np.log1p(out["amt"]).astype("float32")
            out["amt_cents"] = (out["amt"] - out["amt"].astype("int32")).astype("float32")
        else:
            out["amt"] = 0.0; out["amt_log"] = 0.0; out["amt_cents"] = 0.0

        # numeric passthrough
        for c in self.num_cols:
            if c == "TransactionAmt": 
                continue
            out[c] = df.get(c, pd.Series(0, index=df.index)).fillna(0).astype("float32")

        # label encoded cats
        for c in self.cat_cols:
            out[f"{c}_le"] = self._label_encode_apply(df.get(c, pd.Series(pd.NA, index=df.index)), c)

        # frequency encodings
        for c in self.freq_cols:
            out[f"{c}_freq"] = self._freq_apply(df.get(c, pd.Series(pd.NA, index=df.index)), c)

        # missing flags
        if self.use_missing_flags:
            for c in set(self.num_cols + self.cat_cols + self.freq_cols + ["TransactionDT"]):
                if c in df.columns:
                    out[f"{c}_isna"] = df[c].isna().astype("int8")
                else:
                    out[f"{c}_isna"] = 1

        # order columns
        if self.feature_order_ and not fit_phase:
            for col in self.feature_order_:
                if col not in out.columns:
                    out[col] = 0
            out = out[self.feature_order_]
        return out.astype("float32")

    def transform_one(self, payload: dict) -> np.ndarray:
        df = pd.DataFrame([payload])
        X = self.transform(df)
        if not self.feature_order_:
            self.feature_order_ = list(X.columns)
        return X[self.feature_order_].astype("float32").values[0]

    # persistence
    def save(self, path: str):
        joblib.dump(
            {
                "ref_date": self.ref_date_,
                "label_maps": self.label_maps_,
                "freq_maps": self.freq_maps_,
                "feature_order": self.feature_order_,
                "cat_cols": self.cat_cols,
                "freq_cols": self.freq_cols,
                "num_cols": self.num_cols,
                "use_missing_flags": self.use_missing_flags,
            }, path
        )

    @classmethod
    def load(cls, path: str) -> "FraudPreprocessor":
        d = joblib.load(path)
        obj = cls(reference_date=d["ref_date"].strftime("%Y-%m-%d"),
                  cat_cols=d["cat_cols"],
                  freq_cols=d["freq_cols"],
                  num_cols=d["num_cols"],
                  use_missing_flags=d["use_missing_flags"])
        obj.label_maps_ = d["label_maps"]
        obj.freq_maps_ = d["freq_maps"]
        obj.feature_order_ = d["feature_order"]
        obj.ref_date_ = d["ref_date"]
        return obj
