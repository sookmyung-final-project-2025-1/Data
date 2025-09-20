from __future__ import annotations
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.fraud.schema import PredictRawRequest, PredictRawResponse, PredictVectorRequest
from src.fraud.model import FraudModel
from src.fraud.explain import explain_shap

# Multiple model directories
LGBM_DIR = os.environ.get("LGBM_DIR", "models/v5")
XGB_DIR = os.environ.get("XGB_DIR", "models/v6")
CAT_DIR = os.environ.get("CAT_DIR", "models/v7")

app = FastAPI(title="Fraud Detection API", version="1.2.0", root_path="/model")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variables
lgbm_model = None
xgb_model = None
cat_model = None

@app.on_event("startup")
def _load_models():
    global lgbm_model, xgb_model, cat_model
    try:
        lgbm_model = FraudModel.load_dir(LGBM_DIR)
        xgb_model = FraudModel.load_dir(XGB_DIR)
        cat_model = FraudModel.load_dir(CAT_DIR)
        print(f"✅ Models loaded: LGBM({LGBM_DIR}), XGB({XGB_DIR}), CAT({CAT_DIR})")
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        raise

@app.get("/health")
def health():
    return {"status": "ok"}

# Helper function for predictions
def _predict_with_model(model, req: PredictRawRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    prob = model.predict_proba_raw(req.payload)
    resp = {"fraud_probability": round(prob, 6)}

    if req.explain:
        feats = model.feature_names()
        import numpy as np
        x = model.pre.transform_one(req.payload)
        resp["top_features"] = explain_shap(model.clf, feats, x, top_n=req.top_n)

    return resp

# New model-specific endpoints
@app.post("/model/lgbm/predict", response_model=PredictRawResponse)
def predict_lgbm(req: PredictRawRequest):
    return _predict_with_model(lgbm_model, req)

@app.post("/model/xgboost/predict", response_model=PredictRawResponse)
def predict_xgboost(req: PredictRawRequest):
    return _predict_with_model(xgb_model, req)

@app.post("/model/catboost/predict", response_model=PredictRawResponse)
def predict_catboost(req: PredictRawRequest):
    return _predict_with_model(cat_model, req)

# Legacy endpoints (for backward compatibility)
@app.post("/predict_raw", response_model=PredictRawResponse)
def predict_raw(req: PredictRawRequest):
    # Default to LGBM (best performing model)
    return _predict_with_model(lgbm_model, req)

@app.post("/predict")
def predict(vec: PredictVectorRequest):
    # Default to LGBM
    if lgbm_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    prob = lgbm_model.predict_proba_vec(vec.features)
    return {"fraud_probability": round(prob, 6)}

# Model info endpoints
@app.get("/models/info")
def models_info():
    return {
        "lgbm": {"available": lgbm_model is not None, "version": "v5"},
        "xgboost": {"available": xgb_model is not None, "version": "v6"},
        "catboost": {"available": cat_model is not None, "version": "v7"}
    }
