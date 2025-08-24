from __future__ import annotations
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.fraud.schema import PredictRawRequest, PredictRawResponse, PredictVectorRequest
from src.fraud.model import FraudModel
from src.fraud.explain import explain_shap

MODEL_DIR = os.environ.get("MODEL_DIR", "models/v1")

app = FastAPI(title="Fraud Detection API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def _load_model():
    global model
    model = FraudModel.load_dir(MODEL_DIR)

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict_raw", response_model=PredictRawResponse)
def predict_raw(req: PredictRawRequest):
    prob = model.predict_proba_raw(req.payload)
    resp = {"fraud_probability": round(prob, 6)}
    if req.explain:
        feats = model.feature_names()
        import numpy as np
        import numpy
        x = model.pre.transform_one(req.payload)
        resp["top_features"] = explain_shap(model.clf, feats, x, top_n=req.top_n)
    return resp

@app.post("/predict")
def predict(vec: PredictVectorRequest):
    prob = model.predict_proba_vec(vec.features)
    return {"fraud_probability": round(prob, 6)}
