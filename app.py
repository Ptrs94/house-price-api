import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field
from catboost import CatBoostRegressor
import os

app = FastAPI(title="House Price API with Bias Correction")

# ----------------------------
# Load models from current folder
# ----------------------------
models = {}
bias_bins = {}
bias_factors = {}

# Quartile bins and bias factors from training
quartile_bins = [200000, 275000, 350000, 425000, 500000]
quartile_factors = [1.05, 1.02, 0.98, 0.95]

for house_type in ["D", "S", "T"]:
    model_file = f"catboost_{house_type}.cbm"  # current folder
    if os.path.exists(model_file):
        model = CatBoostRegressor()
        model.load_model(model_file)
        models[house_type] = model
        bias_bins[house_type] = quartile_bins
        bias_factors[house_type] = quartile_factors
        print(f"✅ Loaded model for {house_type} houses: {model_file}")
    else:
        print(f"⚠️ Model file for {house_type} not found. Skipping.")

# ----------------------------
# Input schema
# ----------------------------
class RowIn(BaseModel):
    PropertyType: str = Field(..., description="D/S/T etc")
    NewBuild: str = Field(..., description="Y/N")
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
# Helper functions
# ----------------------------
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rooms_per_m2"] = df["NUMBER_HABITABLE_ROOMS"] / df["TOTAL_FLOOR_AREA"]
    df["m2_per_room"] = df["TOTAL_FLOOR_AREA"] / df["NUMBER_HABITABLE_ROOMS"]
    df["time_index"] = df["year"]*4 + df["quarter"]
    return df

def apply_bias_correction(price: float, house_type: str) -> float:
    bins = bias_bins[house_type]
    factors = bias_factors[house_type]
    bin_number = np.digitize([price], bins)[0] - 1
    bin_number = np.clip(bin_number, 0, len(factors)-1)
    return price * factors[bin_number]

# ----------------------------
# API endpoints
# ----------------------------
@app.get("/")
def home():
    return {"message": "House Price API running with bias correction"}

@app.post("/predict")
def predict(row: RowIn):
    X = pd.DataFrame([row.model_dump()])
    X = compute_features(X)

    cat_cols = [
        "PropertyType", "NewBuild", "Postcode_area", "CURRENT_ENERGY_RATING",
        "lsoa21cd", "msoa21cd", "oa21cd"
    ]
    for c in cat_cols:
        X[c] = X[c].astype(str)

    house_type = X.at[0, "PropertyType"]
    if house_type not in models:
        return {"error": f"No model available for PropertyType '{house_type}'"}

    model = models[house_type]

    pred_log = float(model.predict(X)[0])
    pred_price = float(np.expm1(pred_log))

    pred_price_corrected = apply_bias_correction(pred_price, house_type)

    return {"predicted_price": pred_price_corrected}