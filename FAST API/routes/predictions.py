"""
Prediction routes - ML model inference endpoints
"""
from fastapi import APIRouter, HTTPException
import pandas as pd
import joblib

from schema.request_schema import PredictRequest, PredictionResponse
from config import MODEL_PATH
from database_utils.db_operations import get_price_history
from features.feature_engineering import build_features
from utils.confidence import interpret_confidence

router = APIRouter(prefix="/predict", tags=["Predictions"])

# Load model
try:
    model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    raise RuntimeError(f"Model not found at {MODEL_PATH}. Please ensure the model file exists.")
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")


@router.post("/", response_model=PredictionResponse)
def predict(req: PredictRequest) -> PredictionResponse:
    """
    Main prediction endpoint.
    
    Takes a product name and current price, fetches price history,
    builds features, and returns prediction with confidence level.
    
    Args:
        req (PredictRequest): Request containing product_name and price
        
    Returns:
        PredictionResponse: Prediction result with confidence level
        
    Raises:
        HTTPException: If insufficient price history is available
    """
    try:
        # 1. Fetch historical price data
        hist_df = get_price_history(req.product_name)

        if hist_df.empty or len(hist_df) < 3:
            raise HTTPException(
                status_code=400,
                detail=f"Not enough price history for prediction. Required: 3 records, Found: {len(hist_df)}"
            )

        # 2. Append current price as latest observation
        latest = hist_df.iloc[-1].copy()
        latest["date"] = pd.Timestamp.now()
        latest["price"] = req.price

        hist_df = pd.concat(
            [hist_df, pd.DataFrame([latest])],
            ignore_index=True
        )

        # 3. Build engineered features
        feat_df = build_features(hist_df)

        # 4. Select last row for inference
        X = feat_df.iloc[-1:]

        # 5. Make prediction
        prediction = model.predict(X)[0]

        # 6. Get probability/confidence if model supports it
        confidence = None
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X)[0]
            class_index = list(model.classes_).index(prediction)
            confidence = float(proba[class_index])

        # 7. Return formatted response
        return {
            "product_name": req.product_name,
            "input_price": req.price,
            "prediction": prediction,
            "confidence": confidence,
            "confidence_level": interpret_confidence(confidence)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )
