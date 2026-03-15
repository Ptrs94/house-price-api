import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from catboost import CatBoostRegressor, Pool
import os

app = FastAPI()

# ----------------------------
# Load models
# ----------------------------

models = {}

for t in ["D", "S", "T"]:
    file = f"catboost_{t}.cbm"
    if os.path.exists(file):
        m = CatBoostRegressor()
        m.load_model(file)
        models[t] = m
        print(f"Loaded model {file}")
    else:
        print(f"WARNING: {file} not found")

# ----------------------------
# Bias correction
# ----------------------------

bins = [200000, 275000, 350000, 425000, 500000]
factors = [1.05, 1.02, 0.98, 0.95]

def bias_correct(price):
    b = np.digitize([price], bins)[0] - 1
    b = np.clip(b, 0, len(factors)-1)
    return price * factors[b]

# ----------------------------
# Input schema
# ----------------------------

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
    msoa21cd: str
    oa21cd: str

# ----------------------------
# Feature engineering
# ----------------------------

def make_features(df):

    df["rooms_per_m2"] = df["NUMBER_HABITABLE_ROOMS"] / df["TOTAL_FLOOR_AREA"]
    df["m2_per_room"] = df["TOTAL_FLOOR_AREA"] / df["NUMBER_HABITABLE_ROOMS"]
    df["time_index"] = df["year"] * 4 + df["quarter"]

    return df

# ----------------------------
# API endpoints
# ----------------------------

@app.get("/")
def home():
    return {"message": "House price model running"}

@app.post("/predict")
def predict(row: RowIn):

    X = pd.DataFrame([row.model_dump()])

    # feature engineering
    X = make_features(X)

    # categorical columns
    cat_cols = [
        "PropertyType",
        "NewBuild",
        "Postcode_area",
        "CURRENT_ENERGY_RATING",
        "lsoa21cd",
        "msoa21cd",
        "oa21cd"
    ]

    for c in cat_cols:
        X[c] = X[c].astype(str)

    # choose correct model
    house_type = X["PropertyType"].iloc[0]

    if house_type not in models:
        return {"error": f"No model for PropertyType {house_type}"}

    model = models[house_type]

    # create CatBoost pool (fixes categorical error)
    pool = Pool(X, cat_features=cat_cols)

    pred_log = float(model.predict(pool)[0])

    price = float(np.expm1(pred_log))

    price = bias_correct(price)

    return {"predicted_price": price}