# -*- coding: utf-8 -*-
"""
Created on Sat Feb 21 23:41:23 2026

@author: petro
"""


import os
import joblib
import gdown
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

MODEL_FILE = "rf_house_price_pipe.joblib"
FILE_ID = "19H8XmDUT8z5fwI7ZId-ts_1w8Xki_5_f"

# Download model if missing
if not os.path.exists(MODEL_FILE):
    url = f"https://drive.google.com/uc?id={FILE_ID}"
    gdown.download(url, MODEL_FILE, quiet=False)

pipe = joblib.load(MODEL_FILE)  # trained on log1p(Price)


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

@app.post("/predict")
def predict(row: RowIn):
    X = pd.DataFrame([row.model_dump()])
    pred_log = pipe.predict(X)[0]
    pred_price = float(np.expm1(pred_log))  # convert back to normal price
    return {"predicted_price": pred_price}