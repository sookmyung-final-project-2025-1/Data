# main.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import uvicorn

model = joblib.load("fraud_model.pkl")

app = FastAPI(title="Fraud Detection API")

class InputFeatures(BaseModel):
    features: list[float]

@app.post("/predict")
def predict(input: InputFeatures):
    data = np.array(input.features).reshape(1, -1)
    probability = model.predict_proba(data)[0][1]  # class 1: fraud일 확률
    return {
        "fraud_probability": round(float(probability), 4)  # 소수점 4자리까지
    }


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
