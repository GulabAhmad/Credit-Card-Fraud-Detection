"""Optional REST endpoint (SRS FR-14). Run: uvicorn app.api:app --reload  (from the project root)."""
import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import create_model

ROOT = Path(__file__).resolve().parents[1]
pipe = joblib.load(ROOT / "models" / "best_model.joblib")
meta = json.loads((ROOT / "models" / "threshold.json").read_text())
FEATURES = meta["feature_order"]

Transaction = create_model("Transaction", **{f: (float, ...) for f in FEATURES})

app = FastAPI(title="Credit Card Fraud Detection API")


@app.post("/predict")
def predict(tx: Transaction):
    row = pd.DataFrame([tx.model_dump()], columns=FEATURES)
    proba = float(pipe.predict_proba(row)[0, 1])
    return {
        "fraud_probability": proba,
        "threshold": meta["threshold"],
        "decision": "block" if proba >= meta["threshold"] else "approve",
    }
