# src/main.py
from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from typing import Dict
from .predict import predict_data

app = FastAPI(title="Wine Classifier API")

# Wine feature schema (13 fields)
class WineData(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: float

class WineResponse(BaseModel):
    response: int  # class id 0..2

@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status": "healthy"}

@app.post("/predict", response_model=WineResponse)
async def predict_wine(payload: WineData):
    try:
        features = [[
            payload.alcohol,
            payload.malic_acid,
            payload.ash,
            payload.alcalinity_of_ash,
            payload.magnesium,
            payload.total_phenols,
            payload.flavanoids,
            payload.nonflavanoid_phenols,
            payload.proanthocyanins,
            payload.color_intensity,
            payload.hue,
            payload.od280_od315_of_diluted_wines,
            payload.proline
        ]]

        prediction = predict_data(features)
        return WineResponse(response=int(prediction[0]))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
