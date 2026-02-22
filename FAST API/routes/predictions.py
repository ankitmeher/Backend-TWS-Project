"""
Prediction Routes
-----------------
Handles ML model inference for Buy/Wait prediction.

Flow:
1. Fetch historical price data
2. Append current price
3. Build engineered features
4. Run model inference
5. Compute confidence
6. Log prediction asynchronously to S3
7. Return response
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
import pandas as pd
import mlflow
import mlflow.sklearn

from schema.request_schema import PredictRequest, PredictionResponse
from config import MLFLOW_TRACKING_URI, MLFLOW_MODEL_URI
from database_utils.db_operations import get_price_history
from features.feature_engineering import build_features
from utils.confidence import interpret_confidence
from routes.log_prediction import log_prediction_to_s3


# --------------------------------------------------
# Router setup
# --------------------------------------------------
router = APIRouter(prefix="/predict", tags=["Predictions"])


# --------------------------------------------------
# Load Production Model (once at startup)
# --------------------------------------------------
try:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model = mlflow.sklearn.load_model(MLFLOW_MODEL_URI)
    print("[Model] Loaded successfully from MLflow")
except Exception as e:
    raise RuntimeError(f"Failed to load MLflow model: {e}")


# Model output mapping
LABEL_MAP = {
    1: "buy",
    0: "wait"
}


# --------------------------------------------------
# Prediction Endpoint
# --------------------------------------------------
@router.post("/", response_model=PredictionResponse)
def predict(req: PredictRequest, background_tasks: BackgroundTasks):
    """
    Predict whether to BUY or WAIT for a product.
    """

    try:
        # ==================================================
        # 1️⃣ Load price history
        # ==================================================
        history_df = get_price_history(req.product_name)

        if history_df.empty or len(history_df) < 3:
            raise HTTPException(
                status_code=400,
                detail=f"Not enough price history. Required=3, Found={len(history_df)}"
            )

        # ==================================================
        # 2️⃣ Append current observation
        # ==================================================
        latest_row = history_df.iloc[-1].copy()
        latest_row["date"] = pd.Timestamp.now()
        latest_row["price"] = req.price

        history_df = pd.concat(
            [history_df, pd.DataFrame([latest_row])],
            ignore_index=True
        )

        # ==================================================
        # 3️⃣ Feature Engineering
        # ==================================================
        features_df = build_features(history_df)
        X = features_df.iloc[-1:]  # only latest row used

        # ==================================================
        # 4️⃣ Model Prediction
        # ==================================================
        raw_prediction = model.predict(X)[0]
        prediction_label = LABEL_MAP.get(int(raw_prediction), str(raw_prediction))

        # ==================================================
        # 5️⃣ Confidence Calculation
        # ==================================================
        confidence = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(X)[0]
            class_index = list(model.classes_).index(raw_prediction)
            confidence = float(probabilities[class_index])

        confidence_level = interpret_confidence(confidence)

        # ==================================================
        # 6️⃣ Async Logging (Non-blocking)
        # ==================================================
        background_tasks.add_task(
            log_prediction_to_s3,
            req.product_name,
            req.price,
            prediction_label,
            confidence,
            confidence_level,
            X.to_dict(orient="records")[0] if not X.empty else None,
        )

        # ==================================================
        # 7️⃣ Response
        # ==================================================
        return PredictionResponse(
            product_name=req.product_name,
            input_price=req.price,
            prediction=prediction_label,
            confidence=confidence,
            confidence_level=confidence_level
        )

    except HTTPException:
        raise

    except Exception as e:
        # Catch unexpected failures safely
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )