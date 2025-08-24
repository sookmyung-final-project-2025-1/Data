from __future__ import annotations
import os, json, joblib, numpy as np
from typing import Dict, Any, List, Optional
from .preprocessing import FraudPreprocessor

class FraudModel:
    def __init__(self, pre: FraudPreprocessor, clf):
        self.pre = pre
        self.clf = clf

    @classmethod
    def load_dir(cls, model_dir: str) -> "FraudModel":
        pre_path = os.path.join(model_dir, "preprocessor.pkl")
        model_path = os.path.join(model_dir, "model.pkl")
        pre = FraudPreprocessor.load(pre_path)
        clf = joblib.load(model_path)
        return cls(pre, clf)

    def predict_proba_raw(self, payload: Dict[str, Any]) -> float:
        x = self.pre.transform_one(payload)
        p = float(self.clf.predict_proba([x])[0][1])
        return p

    def predict_proba_vec(self, vec: List[float]) -> float:
        x = np.array(vec, dtype="float32").reshape(1, -1)
        p = float(self.clf.predict_proba(x)[0][1])
        return p

    def feature_names(self) -> List[str]:
        return self.pre.feature_order_
