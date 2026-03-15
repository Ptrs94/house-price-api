import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from catboost import CatBoostRegressor
import os

app = FastAPI(title="House Price API with Bias Correction (Sheet-Friendly)")

# ----------------------------
# Load models from current folder
# ----------------------------
models = {}
bias_bins = {}
bias_factors = {}

# Quartile bins and bias factors from training (adjust if you have real values)
quartile_bins = [200000, 275000, 350000, 425000, 500000]
quartile_factors = [1.05, 1.02, 0.98, 0.95]

for house_type in ["D", "S", "T"]:
    model_file = f"catboost_{house_type}.cbm"
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
# Expected column order from sheet
# ----------------------------
SHEET_COLUMNS = [
    "PropertyType",
    "NewBuild",
    "Postcode_area",
    "CURRENT_ENERGY_RATING",
    "TOTAL_FLOOR_AREA",
    "NUMBER_HABITABLE_ROOMS",
    "year",
    "quarter",
    "Age",
    "lsoa21cd",
    "msoa21cd",
    "oa21cd"
]

# ----------------------------
# Input schema
# ----------------------------
class RowIn(BaseModel):
    row: str = Field(..., description="Space-separated row from the sheet, 12 columns")

# ----------------------------
# Helper functions
# ----------------------------
def parse_sheet_row(row_str: str) -> pd.DataFrame:
    parts = row_str.strip().split()
    if len(parts) != len(SHEET_COLUMNS):
        raise ValueError(f"Expected {len(SHEET_COLUMNS)} columns, got {len(parts)}")
    # Convert numeric columns
    data = {}
    for i, col in enumerate(SHEET_COLUMNS):
        if col in ["TOTAL_FLOOR_AREA", "NUMBER_HABITABLE_ROOMS"]:
            data[col] = float(parts[i])
        elif col in ["year", "quarter", "Age"]:
            data[col] = int(parts[i])
        else:
            data[col] = parts[i]
    return pd.DataFrame([data])

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
    return {"message": "House Price API (Sheet-friendly) running with bias correction"}

@app.post("/predict")
def predict(input_row: RowIn):
    try:
        X = parse_sheet_row(input_row.row)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    X = compute_features(X)

    cat_cols = [
        "PropertyType", "NewBuild", "Postcode_area", "CURRENT_ENERGY_RATING",
        "lsoa21cd", "msoa21cd", "oa21cd"
    ]
    for c in cat_cols:
        X[c] = X[c].astype(str)

    house_type = X.at[0, "PropertyType"]
    if house_type not in models:
        raise HTTPException(status_code=400, detail=f"No model available for PropertyType '{house_type}'")

    model = models[house_type]

    pred_log = float(model.predict(X)[0])
    pred_price = float(np.expm1(pred_log))

    pred_price_corrected = apply_bias_correction(pred_price, house_type)

    return {"predicted_price": pred_price_corrected}