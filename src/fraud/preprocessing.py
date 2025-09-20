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
    - UID-based features (card1 + addr1 + D1)
    - missing flags
    저장되는 자산:
      - label_maps_: dict[str, dict]
      - freq_maps_: dict[str, dict]
      - uid_agg_maps_: dict[str, dict] (UID 집계 통계)
      - feature_order_: List[str]
      - ref_date_: datetime
    """
    def __init__(self, reference_date: str = "2017-12-01",
                 cat_cols: List[str] | None = None,
                 freq_cols: List[str] | None = None,
                 num_cols: List[str] | None = None,
                 use_missing_flags: bool = True,
                 use_uid_features: bool = True):
        self.ref_date_ = datetime.fromisoformat(reference_date)
        self.cat_cols = cat_cols or ["ProductCD", "card4", "card6", "P_emaildomain"]
        self.freq_cols = freq_cols or ["card1", "card2", "addr1", "P_emaildomain"]
        self.num_cols = num_cols or ["TransactionAmt", "dist1", "dist2"]
        self.use_missing_flags = use_missing_flags
        self.use_uid_features = use_uid_features

        self.label_maps_: Dict[str, Dict[Any, int]] = {}
        self.freq_maps_: Dict[str, Dict[Any, float]] = {}
        self.uid_agg_maps_: Dict[str, Dict[Any, float]] = {}
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

    def _create_uid(self, df: pd.DataFrame) -> pd.Series:
        """Create UID from card1 + addr1 + D1 (Chris Deotte's method)"""
        card1 = df.get("card1", pd.Series("", index=df.index)).fillna("").astype(str)
        addr1 = df.get("addr1", pd.Series("", index=df.index)).fillna("").astype(str)
        d1 = df.get("D1", pd.Series("", index=df.index)).fillna("").astype(str)
        uid = card1 + "_" + addr1 + "_" + d1
        return uid.replace("__", "_").replace("_$", "")

    def _compute_d1n(self, df: pd.DataFrame) -> pd.Series:
        """D1n = TransactionDay - D1 (normalized D1 per user)"""
        if "TransactionDT" in df.columns and "D1" in df.columns:
            dt = self.ref_date_ + pd.to_timedelta(df["TransactionDT"].fillna(0), unit="s")
            transaction_day = dt.dt.dayofyear
            d1 = df["D1"].fillna(0)
            return (transaction_day - d1).astype("float32")
        return pd.Series(0, index=df.index, dtype="float32")

    def _uid_agg_fit(self, df: pd.DataFrame, uid_col: pd.Series):
        """Fit UID aggregation statistics"""
        temp_df = df.copy()
        temp_df["uid"] = uid_col

        # TransactionAmt aggregations
        if "TransactionAmt" in df.columns:
            amt_stats = temp_df.groupby("uid")["TransactionAmt"].agg(['mean', 'std', 'count']).to_dict()
            self.uid_agg_maps_["TransactionAmt_uid_mean"] = amt_stats['mean']
            self.uid_agg_maps_["TransactionAmt_uid_std"] = amt_stats['std']
            self.uid_agg_maps_["TransactionAmt_uid_count"] = amt_stats['count']

    def _uid_agg_apply(self, uid_col: pd.Series) -> pd.DataFrame:
        """Apply UID aggregation features"""
        out = pd.DataFrame(index=uid_col.index)

        for feature_name, uid_map in self.uid_agg_maps_.items():
            out[feature_name] = uid_col.map(uid_map).fillna(0).astype("float32")

        return out

    # ---------- public ----------
    def fit(self, df: pd.DataFrame):
        # fit label/freq maps on training data
        for c in self.cat_cols:
            if c in df.columns:
                self._label_encode_fit(df, c)
        for c in self.freq_cols:
            if c in df.columns:
                self._freq_fit(df, c)

        # fit UID aggregations
        if self.use_uid_features:
            uid_col = self._create_uid(df)
            self._uid_agg_fit(df, uid_col)

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

        # UID-based features
        if self.use_uid_features:
            uid_col = self._create_uid(df)

            # D1n feature (Chris Deotte's normalization)
            out["D1n"] = self._compute_d1n(df)

            # UID aggregation features
            uid_features = self._uid_agg_apply(uid_col)
            out = out.join(uid_features)

        # missing flags
        if self.use_missing_flags:
            uid_cols = ["card1", "addr1", "D1"] if self.use_uid_features else []
            for c in set(self.num_cols + self.cat_cols + self.freq_cols + ["TransactionDT"] + uid_cols):
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
                "uid_agg_maps": self.uid_agg_maps_,
                "feature_order": self.feature_order_,
                "cat_cols": self.cat_cols,
                "freq_cols": self.freq_cols,
                "num_cols": self.num_cols,
                "use_missing_flags": self.use_missing_flags,
                "use_uid_features": self.use_uid_features,
            }, path
        )
    @classmethod
    def load(cls, path: str) -> "FraudPreprocessor":
        obj: Any = joblib.load(path)

        # 1) 예전/다른 저장 포맷: 이미 객체로 직렬화된 경우
        if isinstance(obj, cls):
            return obj

        # 2) 현재 포맷: dict로 저장된 경우 처리 (필드명은 프로젝트에 맞게 유지)
        if isinstance(obj, dict):
            # ref_date가 datetime일 수도, 문자열일 수도 있으니 방어적으로 처리
            ref_date = obj.get("ref_date", "2017-12-01")
            if hasattr(ref_date, "strftime"):
                reference_date = ref_date.strftime("%Y-%m-%d")
            else:
                reference_date = str(ref_date)

            inst = cls(
                reference_date=reference_date,
                cat_cols=obj.get("cat_cols", ["ProductCD", "card4", "card6", "P_emaildomain"]),
                freq_cols=obj.get("freq_cols", ["card1", "card2", "addr1", "P_emaildomain"]),
                num_cols=obj.get("num_cols", ["TransactionAmt", "dist1", "dist2"]),
                use_missing_flags=obj.get("use_missing_flags", True),
                use_uid_features=obj.get("use_uid_features", True)
            )

            # 저장된 맵들 복원
            inst.label_maps_ = obj.get("label_maps", {})
            inst.freq_maps_ = obj.get("freq_maps", {})
            inst.uid_agg_maps_ = obj.get("uid_agg_maps", {})
            inst.feature_order_ = obj.get("feature_order", [])
            inst.ref_date_ = obj.get("ref_date", inst.ref_date_)

            return inst

        # 3) 그 외 타입은 지원 안 함
        raise TypeError(f"Unsupported preprocessor artifact type: {type(obj)}")
    """
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
"""