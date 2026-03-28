import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path("model/traffic_factor_model.pkl")

bundle = joblib.load(MODEL_PATH)
pipeline = bundle["pipeline"]
feature_columns = bundle["feature_columns"]
city_baselines = bundle["city_baselines"]

KNOWN_CITIES = city_baselines["City"].dropna().astype(str).unique().tolist()

def map_sri_lanka_place_to_city(name: str) -> str:
    """
    Proxy mapping from Sri Lankan place names into one of the trained dataset cities.
    This is a context mapping, not a literal geographic mapping.
    """
    if not name:
        return KNOWN_CITIES[0]

    p = name.lower()

    # heavy urban / capital-like
    if any(x in p for x in ["colombo", "dehiwala", "nugegoda", "maharagama", "moratuwa"]):
        for c in KNOWN_CITIES:
            if "bangalore" in c.lower() or "bengaluru" in c.lower():
                return c

    # secondary city / moderate urban
    if any(x in p for x in ["kandy", "galle", "negombo", "kurunegala"]):
        for c in KNOWN_CITIES:
            if "mumbai" in c.lower():
                return c
        return KNOWN_CITIES[0]

    # hill / intercity / less dense proxy
    if any(x in p for x in ["ratnapura", "badulla", "matale", "anuradhapura"]):
        for c in KNOWN_CITIES:
            if "pune" in c.lower():
                return c
        return KNOWN_CITIES[0]

    return KNOWN_CITIES[0]

def get_city_baseline(city: str) -> dict:
    row = city_baselines[city_baselines["City"] == city]
    if row.empty:
        row = city_baselines.iloc[[0]]
    return row.iloc[0].to_dict()

def predict_traffic_factor(payload: dict) -> float:
    departure_time = pd.to_datetime(payload["departureTime"])
    hour = int(departure_time.hour)
    day_of_week = int(departure_time.dayofweek)
    month = int(departure_time.month)
    is_weekend = 1 if day_of_week in [5, 6] else 0
    is_morning_peak = 1 if 7 <= hour <= 9 else 0
    is_evening_peak = 1 if 17 <= hour <= 19 else 0

    source_name = payload.get("sourceName", "")
    destination_name = payload.get("destinationName", "")
    distance_km = float(payload.get("distanceKm", 10.0) or 10.0)

    # choose context city based on source, and slightly bias by destination
    mapped_city = map_sri_lanka_place_to_city(source_name or destination_name)
    baseline = get_city_baseline(mapped_city)

    row = {
        "City": mapped_city,
        "TrafficIndexLive": baseline.get("TrafficIndexLive", 40.0),
        "TrafficIndexWeekAgo": baseline.get("TrafficIndexWeekAgo", 35.0),
        "TravelTimeHistoricPer10KmsMins": baseline.get("TravelTimeHistoricPer10KmsMins", 18.0),
        "MinsDelay": baseline.get("MinsDelay", 3.0),
        "hour": hour,
        "day_of_week": day_of_week,
        "month": month,
        "is_weekend": is_weekend,
        "is_morning_peak": is_morning_peak,
        "is_evening_peak": is_evening_peak,
    }

    for c in ["JamsCount", "JamsLengthInKms", "JamsDelay"]:
        if c in feature_columns:
            row[c] = baseline.get(c, 0.0)

    X = pd.DataFrame([row])[feature_columns]
    pred = float(pipeline.predict(X)[0])

    # mild route-distance adjustment so longer intercity trips are not over-penalized
    if distance_km > 80:
        pred *= 0.92
    elif distance_km > 40:
        pred *= 0.97

    # stronger peak-hour shaping
    if is_morning_peak or is_evening_peak:
        pred *= 1.08
    elif hour >= 22 or hour <= 5:
        pred *= 0.90

    return max(1.0, pred)