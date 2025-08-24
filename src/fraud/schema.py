from __future__ import annotations
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class PredictRawRequest(BaseModel):
    payload: Dict[str, Any]
    explain: bool = False
    top_n: int = 5

class PredictRawResponse(BaseModel):
    fraud_probability: float
    top_features: Optional[List[Dict[str, float]]] = None

class PredictVectorRequest(BaseModel):
    features: List[float]
