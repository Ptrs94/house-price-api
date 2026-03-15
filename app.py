import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field
from catboost import CatBoostRegressor

app = FastAPI(title="House Price API")

# ----------------------------
# Load models
# ----------------------------
models = {}
for house_type in ["D", "S", "T"]:
    model_file = f"models/catboost_{house_type}.cbm"
    model = CatBoostRegressor()
    model.load_model(model_file)
    models[house_type] = model
    print(f"✅ Loaded model for {house_type} houses: {model_file}")

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
# Helper function to compute derived features
# ----------------------------
def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rooms_per_m2"] = df["NUMBER_HABITABLE_ROOMS"] / df["TOTAL_FLOOR_AREA"]
    df["m2_per_room"] = df["TOTAL_FLOOR_AREA"] / df["NUMBER_HABITABLE_ROOMS"]
    df["time_index"] = df["year"]*4 + df["quarter"]
    return df

# ----------------------------
# API endpoints
# ----------------------------
@app.get("/")
def home():
    return {"message": "House Price API is running (CatBoost, auto features)"}

@app.post("/predict")
def predict(row: RowIn):
    X = pd.DataFrame([row.model_dump()])

    # Compute derived features
    X = compute_features(X)

    # Ensure categorical columns are strings
    cat_cols = [
        "PropertyType",
        "NewBuild",
        "Postcode_area",
        "CURRENT_ENERGY_RATING",
        "lsoa21cd",
        "msoa21cd",
        "oa21cd",
    ]
    for c in cat_cols:
        X[c] = X[c].astype(str)

    # Select the right model
    house_type = X.at[0, "PropertyType"]
    if house_type not in models:
        return {"error": f"No model available for PropertyType '{house_type}'"}

    model = models[house_type]

    # Predict
    pred_log = float(model.predict(X)[0])
    pred_price = float(np.expm1(pred_log))

    return {"predicted_price": pred_price}