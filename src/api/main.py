from fastapi import FastAPI
import pandas as pd
import joblib
import os
from .pydantic_models import CreditData, PredictionResponse

app = FastAPI(title="Bati Bank Credit Scoring API")

# Load the model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))

@app.get("/")
def home():
    return {"message": "Bati Bank Credit Scoring API is Online"}

@app.post("/predict", response_model=PredictionResponse)
def predict(data: CreditData):
    df = pd.DataFrame([data.dict()])
    prob = model.predict_proba(df)[0][1]
    pred = int(model.predict(df)[0])
    return {"risk_probability": float(prob), "prediction": pred}