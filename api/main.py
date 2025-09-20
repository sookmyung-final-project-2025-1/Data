from __future__ import annotations
import os
import shutil
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from src.fraud.schema import PredictRawRequest, PredictRawResponse, PredictVectorRequest
from src.fraud.model import FraudModel
from src.fraud.explain import explain_shap

# 환경변수에서 모델 버전 및 디렉토리 설정
MODEL_VERSION = os.environ.get("MODEL_VERSION", "v1")
MODEL_BASE_DIR = os.environ.get("MODEL_DIR", "/app/models")

app = FastAPI(
    title="Fraud Detection API", 
    version="1.2.0",
    # root_path 제거 - nginx 프록시에서 처리
)

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
    
    # 모델 버전 디렉토리 경로
    model_version_dir = os.path.join(MODEL_BASE_DIR, MODEL_VERSION)
    
    try:
        print(f"Loading models from: {model_version_dir}")
        
        # 디렉토리 구조 확인
        if os.path.exists(model_version_dir):
            print("Available files:")
            for root, dirs, files in os.walk(model_version_dir):
                level = root.replace(model_version_dir, '').count(os.sep)
                indent = ' ' * 2 * level
                print(f"{indent}{os.path.basename(root)}/")
                subindent = ' ' * 2 * (level + 1)
                for file in files:
                    print(f"{subindent}{file}")
        
        # 각 모델별 디렉토리에서 로드 시도
        model_types = [
            ("lgbm", "lgbm"),
            ("xgb", "xgboost"), 
            ("cat", "catboost")
        ]
        
        for dir_name, model_name in model_types:
            model_dir = os.path.join(model_version_dir, dir_name)
            if os.path.exists(model_dir):
                try:
                    if model_name == "lgbm":
                        lgbm_model = FraudModel.load_dir(model_dir)
                        print(f"✅ LGBM model loaded from {model_dir}")
                    elif model_name == "xgboost":
                        xgb_model = FraudModel.load_dir(model_dir)
                        print(f"✅ XGBoost model loaded from {model_dir}")
                    elif model_name == "catboost":
                        cat_model = FraudModel.load_dir(model_dir)
                        print(f"✅ CatBoost model loaded from {model_dir}")
                except Exception as e:
                    print(f"⚠️ Failed to load {model_name} model: {e}")
            else:
                print(f"⚠️ {model_name} model directory not found: {model_dir}")
        
        # 최소 하나의 모델은 로드되어야 함
        loaded_models = sum([lgbm_model is not None, xgb_model is not None, cat_model is not None])
        if loaded_models == 0:
            raise Exception("No models could be loaded from the expected file structure")
        
        print(f"✅ Successfully loaded {loaded_models} model(s)")
            
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        raise

@app.get("/health")
def health():
    return {
        "status": "ok",
        "models": {
            "lgbm": lgbm_model is not None,
            "xgboost": xgb_model is not None,
            "catboost": cat_model is not None
        }
    }

@app.get("/")
def root():
    return {
        "message": "Fraud Detection API", 
        "docs": "/docs", 
        "health": "/health",
        "models_info": "/models/info"
    }

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

@app.post("/lgbm/predict", response_model=PredictRawResponse)
def predict_lgbm(req: PredictRawRequest):
    return _predict_with_model(lgbm_model, req)

@app.post("/xgboost/predict", response_model=PredictRawResponse)
def predict_xgboost(req: PredictRawRequest):
    return _predict_with_model(xgb_model, req)

@app.post("/catboost/predict", response_model=PredictRawResponse)
def predict_catboost(req: PredictRawRequest):
    return _predict_with_model(cat_model, req)

# Legacy endpoints (backward compatibility)
@app.post("/predict_raw", response_model=PredictRawResponse)
def predict_raw(req: PredictRawRequest):
    # 우선순위: LGBM > XGBoost > CatBoost
    model = lgbm_model or xgb_model or cat_model
    return _predict_with_model(model, req)

@app.post("/predict")
def predict(vec: PredictVectorRequest):
    model = lgbm_model or xgb_model or cat_model
    if model is None:
        raise HTTPException(status_code=503, detail="No model loaded")
    prob = model.predict_proba_vec(vec.features)
    return {"fraud_probability": round(prob, 6)}

# Model info endpoints
@app.get("/models/info")
def models_info():
    return {
        "lgbm": {"available": lgbm_model is not None, "version": MODEL_VERSION},
        "xgboost": {"available": xgb_model is not None, "version": MODEL_VERSION},
        "catboost": {"available": cat_model is not None, "version": MODEL_VERSION},
        "model_dir": f"{MODEL_BASE_DIR}/{MODEL_VERSION}"
    }