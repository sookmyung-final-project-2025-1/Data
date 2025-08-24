from __future__ import annotations
import numpy as np
from typing import List, Dict, Any, Optional

try:
    import shap
except Exception:
    shap = None

def explain_shap(model, feature_names: List[str], x_row: np.ndarray, top_n: int = 5) -> List[Dict[str, float]]:
    if shap is None:
        return []
    try:
        explainer = shap.TreeExplainer(model)  # works for LightGBM/XGB/CatBoost
        sv = explainer.shap_values(x_row.reshape(1, -1))
        # LightGBM returns list for multiclass; binary -> array
        vals = sv if isinstance(sv, np.ndarray) else sv[1]
        vals = vals.reshape(-1)
        idx = np.argsort(np.abs(vals))[::-1][:top_n]
        return [{feature_names[i]: float(vals[i])} for i in idx]
    except Exception:
        return []
