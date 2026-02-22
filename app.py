import joblib
import pandas as pd
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Load model directly from repo
MODEL_FILE = "gbr_house_price_pipe.joblib"
pipe = joblib.load(MODEL_FILE)

print("GradientBoosting model loaded successfully.")


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
    return {"message": "House Price API running (GBR model)"}


@app.post("/predict")
def predict(row: RowIn):
    X = pd.DataFrame([row.model_dump()])
    pred_log = pipe.predict(X)[0]
    pred_price = float(np.expm1(pred_log))
    return {"predicted_price": pred_price}