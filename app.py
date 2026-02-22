# -*- coding: utf-8 -*-
"""
Created on Sat Feb 21 23:41:23 2026

@author: petro
"""

import os
import hashlib
import joblib
import gdown
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

MODEL_FILE = "rf_house_price_pipe.joblib"

# ✅ Your NEW Google Drive file ID
FILE_ID = "1b7TNc9xyqUaFFh67pWA__68UxS2oZHbg"
MODEL_URL = f"https://drive.google.com/uc?id={FILE_ID}"


def sha256_of_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def download_model_fresh() -> None:
    """
    Always overwrite the model at startup so a new Drive upload takes effect
    after a restart/redeploy. This avoids 'stale model' issues on Render.
    """
    # Remove existing file to force a clean download
    if os.path.exists(MODEL_FILE):
        os.remove(MODEL_FILE)

    print(f"[MODEL] Downloading from: {MODEL_URL}")
    out = gdown.download(MODEL_URL, MODEL_FILE, quiet=False, fuzzy=True)
    if not out or not os.path.exists(MODEL_FILE):
        raise RuntimeError("Model download failed. Check Google Drive sharing / file ID.")

    print(f"[MODEL] Downloaded to: {MODEL_FILE}")
    print(f"[MODEL] SHA256: {sha256_of_file(MODEL_FILE)}")


# ✅ Force model refresh on cold start
download_model_fresh()

# Load pipeline (trained on log1p(Price))
pipe = joblib.load(MODEL_FILE)
print("[MODEL] Pipeline loaded successfully.")


class RowIn(BaseModel):
    PropertyType: str
    NewBuild: str
    Postcode_area: str
    CURRENT_ENERGY_RATING: str
    TOTAL_FLOOR_AREA: float
    NUMBER_HABITABLE_ROOMS: float
    year: int
    quarter: int
    Age: float
    lsoa21cd: str


@app.get("/")
def home():
    return {
        "message": "House Price API is running",
        "model_file": MODEL_FILE,
        "file_id": FILE_ID,
    }


@app.post("/predict")
def predict(row: RowIn):
    X = pd.DataFrame([row.model_dump()])
    pred_log = pipe.predict(X)[0]
    pred_price = float(np.expm1(pred_log))  # back-transform from log1p
    return {"predicted_price": pred_price}