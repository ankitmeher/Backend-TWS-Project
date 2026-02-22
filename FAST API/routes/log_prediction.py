"""
Prediction Logging - Logs all predictions to AWS S3
Bucket: buy-wait-prediction-logs
"""
import json
import uuid
import boto3
from datetime import datetime, timezone
from botocore.exceptions import BotoCoreError, ClientError

# S3 Configuration
S3_BUCKET = "buy-wait-prediction-log-history"
S3_PREFIX = "predictions"  # logs stored under s3://buy-wait-prediction-logs/predictions/

# Initialize S3 client (uses EC2 IAM role automatically when deployed)
s3_client = boto3.client("s3")


def log_prediction_to_s3(
    product_name: str,
    input_price: float,
    prediction: str,
    confidence: float | None,
    confidence_level: str,
    features: dict | None = None
) -> bool:
    """
    Logs a single prediction to S3 as a JSON file.

    Each prediction is saved as an individual file:
    s3://buy-wait-prediction-logs/predictions/YYYY/MM/DD/<uuid>.json

    Args:
        product_name:     Name of the product queried
        input_price:      Price provided in the request
        prediction:       Model output - "buy" or "wait"
        confidence:       Confidence score (0.0 - 1.0) or None
        confidence_level: Human-readable confidence - "high", "medium", "low"
        features:         Dictionary of engineered features used for this prediction

    Returns:
        True if logged successfully, False otherwise
    """
    now = datetime.now(timezone.utc)

    log_entry = {
        "id": str(uuid.uuid4()),
        "timestamp": now.isoformat(),
        "product_name": product_name,
        "input_price": input_price,
        "prediction": prediction,
        "confidence": confidence,
        "confidence_level": confidence_level,
        "features": features,
    }

    # Organise by date for easy querying: predictions/YYYY/MM/DD/<uuid>.json
    s3_key = (
        f"{S3_PREFIX}/"
        f"{now.strftime('%Y/%m/%d')}/"
        f"{log_entry['id']}.json"
    )

    try:
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=json.dumps(log_entry, indent=2),
            ContentType="application/json"
        )
        print(f"[S3 Log] Prediction logged to s3://{S3_BUCKET}/{s3_key}")
        return True

    except (BotoCoreError, ClientError) as e:
        # Log the error but DO NOT crash the API - prediction still returns to user
        print(f"[S3 Log] WARNING: Failed to log prediction to S3: {e}")
        return False
