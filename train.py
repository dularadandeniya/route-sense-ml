import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = "ForExport.csv"
MODEL_DIR = Path("model")
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "traffic_factor_model.pkl"

df = pd.read_csv(DATA_PATH)

required_cols = [
    "City",
    "UpdateTimeUTC",
    "TrafficIndexLive",
    "TrafficIndexWeekAgo",
    "TravelTimeLivePer10KmsMins",
    "TravelTimeHistoricPer10KmsMins",
    "MinsDelay",
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df.dropna(subset=required_cols).copy()

df["UpdateTimeUTC"] = pd.to_datetime(df["UpdateTimeUTC"], errors="coerce")
df = df.dropna(subset=["UpdateTimeUTC"])

df["hour"] = df["UpdateTimeUTC"].dt.hour
df["day_of_week"] = df["UpdateTimeUTC"].dt.dayofweek
df["month"] = df["UpdateTimeUTC"].dt.month
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)
df["is_morning_peak"] = df["hour"].between(7, 9).astype(int)
df["is_evening_peak"] = df["hour"].between(17, 19).astype(int)

# target traffic factor
df["traffic_factor"] = (
        df["TravelTimeLivePer10KmsMins"] / df["TravelTimeHistoricPer10KmsMins"]
).clip(lower=1.0)

# optional columns if present
optional_numeric = []
for c in ["JamsCount", "JamsLengthInKms", "JamsDelay"]:
    if c in df.columns:
        optional_numeric.append(c)

feature_columns = [
                      "City",
                      "TrafficIndexLive",
                      "TrafficIndexWeekAgo",
                      "TravelTimeHistoricPer10KmsMins",
                      "MinsDelay",
                      "hour",
                      "day_of_week",
                      "month",
                      "is_weekend",
                      "is_morning_peak",
                      "is_evening_peak",
                  ] + optional_numeric

X = df[feature_columns]
y = df["traffic_factor"]

numeric_features = [
                       "TrafficIndexLive",
                       "TrafficIndexWeekAgo",
                       "TravelTimeHistoricPer10KmsMins",
                       "MinsDelay",
                       "hour",
                       "day_of_week",
                       "month",
                       "is_weekend",
                       "is_morning_peak",
                       "is_evening_peak",
                   ] + optional_numeric

categorical_features = ["City"]

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

model = RandomForestRegressor(
    n_estimators=300,
    max_depth=16,
    random_state=42,
    n_jobs=-1
)

pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("model", model)
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

pipeline.fit(X_train, y_train)

preds = pipeline.predict(X_test)

mae = mean_absolute_error(y_test, preds)
rmse = mean_squared_error(y_test, preds) ** 0.5
r2 = r2_score(y_test, preds)

print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2:   {r2:.4f}")

city_baselines = (
    df.groupby("City")[[
        "TrafficIndexLive",
        "TrafficIndexWeekAgo",
        "TravelTimeHistoricPer10KmsMins",
        "MinsDelay",
        *optional_numeric
    ]]
    .median()
    .reset_index()
)

joblib.dump(
    {
        "pipeline": pipeline,
        "feature_columns": feature_columns,
        "city_baselines": city_baselines,
    },
    MODEL_PATH
)

print(f"Model saved to {MODEL_PATH}")