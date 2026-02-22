# -*- coding: utf-8 -*-
"""
Created on Sat Feb 21 23:41:23 2026

@author: petro
"""

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib

app = FastAPI()
pipe = joblib.load("rf_house_price_pipe.joblib")  # trained on log1p(Price)

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