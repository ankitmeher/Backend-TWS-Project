import json
import uuid
import boto3
from datetime import datetime, timezone


S3_BUCKET = "buy-wait-prediction-log-history"
S3_PREFIX = "predictions"


def log_prediction_to_s3(
    product_name: str,
    input_price: float,
    prediction: str,
    confidence: float | None,
    confidence_level: str,
    features: dict | None = None,
) -> bool:

    try:
        s3_client = boto3.client("s3", region_name="ap-south-1")

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

        s3_key = (
            f"{S3_PREFIX}/"
            f"{now.strftime('%Y/%m/%d')}/"
            f"{log_entry['id']}.json"
        )

        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=json.dumps(log_entry, default=str),
            ContentType="application/json",
        )

        print(f"[S3 Log] SUCCESS → {s3_key}")
        return True

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"[S3 Log] FAILED: {e}")
        return False