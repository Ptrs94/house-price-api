import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field
from catboost import CatBoostRegressor

app = FastAPI()

MODEL_FILE = "catboost_house_price.cbm"

model = CatBoostRegressor()
model.load_model(MODEL_FILE)
print("✅ CatBoost model loaded:", MODEL_FILE)


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


@app.get("/")
def home():
    return {"message": "House Price API is running (CatBoost, no IMD)"}


@app.post("/predict")
def predict(row: RowIn):
    X = pd.DataFrame([row.model_dump()])

    # ensure categorical columns are strings
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

    pred_log = float(model.predict(X)[0])
    pred_price = float(np.expm1(pred_log))
    return {"predicted_price": pred_price}