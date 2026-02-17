from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import sqlite3
import joblib
import os

import os
import joblib

BASE_DIR = os.getcwd()

# -------------------- APP --------------------
app = FastAPI(title="Buy / Wait Prediction API")

# -------------------- LOAD MODEL --------------------
MODEL_PATH = os.path.join(BASE_DIR, "price_history_model.pkl")
model = joblib.load(MODEL_PATH)


# -------------------- REQUEST SCHEMA --------------------
class PredictRequest(BaseModel):
    product_name: str
    price: float

# -------------------- DATABASE --------------------
DB_PATH = os.path.join(BASE_DIR, "price_history.db")


def get_price_history(product_name: str) -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    query = """
    SELECT
        date,
        product_name,
        brand,
        price,
        has_anc,
        anc_level_db,
        has_enc,
        driver_size_mm
    FROM price_history
    WHERE product_name = ?
    ORDER BY date
    """
    df = pd.read_sql(query, conn, params=(product_name,))
    conn.close()
    return df

# -------------------- FEATURE ENGINEERING --------------------
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Time features
    df["day"] = df["date"].dt.day
    df["month"] = df["date"].dt.month
    df["dayofweek"] = df["date"].dt.dayofweek
    df["hour"] = df["date"].dt.hour

    # Lag features
    df["price_lag_1"] = df["price"].shift(1)
    df["price_lag_2"] = df["price"].shift(2)

    # Price change features
    df["price_diff_1"] = df["price"] - df["price_lag_1"]
    df["price_pct_change_1"] = df["price"].pct_change()

    # Rolling stats
    df["rolling_mean_3"] = df["price"].rolling(3).mean()
    df["rolling_min_3"] = df["price"].rolling(3).min()
    df["rolling_max_3"] = df["price"].rolling(3).max()
    df["rolling_std_3"] = df["price"].rolling(3).std()

    # Relative features
    df["price_vs_roll_mean"] = df["price"] - df["rolling_mean_3"]
    df["price_vs_roll_min"] = df["price"] - df["rolling_min_3"]

    return df

# -------------------- CONFIDENCE INTERPRETATION --------------------
def interpret_confidence(confidence: float | None) -> str:
    if confidence is None:
        return "unknown"
    if confidence >= 0.75:
        return "high"
    if confidence >= 0.60:
        return "medium"
    return "low"

# -------------------- ROUTES --------------------
@app.get("/")
def index():
    return {"message": "Buy / Wait Prediction API running ✅"}

@app.get("/health")
def health():
    return {"status": "OK"}

@app.post("/predict")
def predict(req: PredictRequest):
    # 1. Fetch history
    hist_df = get_price_history(req.product_name)

    if hist_df.empty or len(hist_df) < 3:
        return {
            "error": "Not enough price history for prediction (minimum 3 records required)"
        }

    # 2. Append current price as latest observation
    latest = hist_df.iloc[-1].copy()
    latest["date"] = pd.Timestamp.now()
    latest["price"] = req.price

    hist_df = pd.concat(
        [hist_df, pd.DataFrame([latest])],
        ignore_index=True
    )

    # 3. Build features
    feat_df = build_features(hist_df)

    # 4. Select last row for inference
    X = feat_df.iloc[-1:]

    # 5. Predict class
    prediction = model.predict(X)[0]

    # 6. Predict probability (if supported)
    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        class_index = list(model.classes_).index(prediction)
        confidence = float(proba[class_index])

    return {
        "product_name": req.product_name,
        "input_price": req.price,
        "prediction": prediction,
        "confidence": confidence,
        "confidence_level": interpret_confidence(confidence)
    }
