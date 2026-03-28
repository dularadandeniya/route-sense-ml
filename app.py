from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from predictor import predict_traffic_factor

app = FastAPI(title="RouteSense Traffic Forecast Service")

class TrafficPredictionRequest(BaseModel):
    sourceName: str
    destinationName: str
    sourceLat: float
    sourceLon: float
    destinationLat: float
    destinationLon: float
    distanceKm: float
    departureTime: str
    weather: Optional[str] = "Clear"
    roadwork: Optional[str] = "No"

class TrafficPredictionResponse(BaseModel):
    trafficFactor: float
    modelVersion: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict-traffic-factor", response_model=TrafficPredictionResponse)
def predict(req: TrafficPredictionRequest):
    payload = req.model_dump()
    factor = predict_traffic_factor(payload)
    print("ML REQUEST:", payload)
    print("ML PREDICTION:", factor)
    return TrafficPredictionResponse(
        trafficFactor=factor,
        modelVersion="v3.0.0-city-temporal"
    )