import os
import json
import boto3
import pandas as pd
import mlflow
from typing import List, Dict, Any
from datetime import datetime, timezone

from training.config import DATA_PATH, REGISTERED_MODEL_NAME
from training.train import run_training

# S3 Configuration (Matching FAST API/routes/log_prediction.py)
S3_BUCKET = "buy-wait-prediction-logs"
S3_PREFIX = "predictions"


def fetch_new_logs_from_s3() -> List[Dict[str, Any]]:
    """
    Fetches all JSON prediction logs from S3.
    """
    print(f"Fetching logs from s3://{S3_BUCKET}/{S3_PREFIX}...")
    s3_client = boto3.client("s3")
    logs = []

    try:
        paginator = s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=S3_PREFIX):
            if "Contents" not in page:
                continue

            for obj in page["Contents"]:
                key = obj["Key"]
                if not key.endswith(".json"):
                    continue

                response = s3_client.get_object(Bucket=S3_BUCKET, Key=key)
                content = response["Body"].read().decode("utf-8")
                logs.append(json.loads(content))

        print(f"Successfully fetched {len(logs)} logs from S3.")
    except Exception as e:
        print(f"Error fetching from S3: {e}")

    return logs


def process_logs_to_dataframe(logs: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Converts raw S3 logs into a flat DataFrame matching the training schema.
    """
    if not logs:
        return pd.DataFrame()

    processed_data = []
    for log in logs:
        # Extract base fields and engineered features
        entry = {
            "date": log.get("timestamp"),
            "price": log.get("input_price"),
            "product_name": log.get("product_name"),
            # Map prediction label back to numeric target
            "target_buy_wait": 1 if log.get("prediction") == "buy" else 0
        }

        # Include all pre-engineered features if present
        if "features" in log and log["features"]:
            entry.update(log["features"])

        processed_data.append(entry)

    df = pd.DataFrame(processed_data)
    # Ensure correct column order or at least presence for training
    return df


def get_champion_f1() -> float:
    """
    Retrieves the macro F1 score of the current champion model.
    """
    client = mlflow.tracking.MlflowClient()
    try:
        champion_version = client.get_model_version_by_alias(REGISTERED_MODEL_NAME, "champion")
        run = client.get_run(champion_version.run_id)
        f1 = run.data.metrics.get("f1_score_macro", 0.0)
        print(f"Current Champion Macro F1: {f1:.4f} (Version {champion_version.version})")
        return f1
    except Exception as e:
        print(f"No existing champion found or error: {e}")
        return 0.0


def retrain_and_promote():
    """
    Main retraining workflow.
    """
    # 1. Fetch new data
    new_logs = fetch_new_logs_from_s3()
    if not new_logs:
        print("No new logs found in S3. Skipping retraining.")
        return

    new_df = process_logs_to_dataframe(new_logs)

    # 2. Merge with existing dataset
    if os.path.exists(DATA_PATH):
        existing_df = pd.read_csv(DATA_PATH)
        # Deduplication could be added here based on 'id' if logged
        final_df = pd.concat([existing_df, new_df], ignore_index=True)
        # Prevent duplicate rows if script runs multiple times
        if "id" in final_df.columns:
             final_df = final_df.drop_duplicates(subset=["id"])
    else:
        final_df = new_df

    final_df.to_csv(DATA_PATH, index=False)
    print(f"Dataset updated. Total rows: {len(final_df)}")

    # 3. Get current champion performance
    old_f1 = get_champion_f1()

    # 4. Run training pipeline (without automatic promotion)
    print("\nStarting retraining pipeline...")
    results = run_training(promote=False)

    if not results:
        print("Training failed. No new model to compare.")
        return

    new_f1 = results["f1_score_macro"]
    new_run_id = results["run_id"]

    # 5. Conditional Promotion
    if new_f1 > old_f1:
        print(f"\nIMPROVEMENT DETECTED: New model (F1: {new_f1:.4f}) is better than Champion (F1: {old_f1:.4f}).")
        print("Promoting new model to CHAMPION...")
        from train import promote_model_to_champion
        promote_model_to_champion(new_run_id)
        print("Model transition complete.")
    else:
        print(f"\nNO IMPROVEMENT: New model (F1: {new_f1:.4f}) did not beat Champion (F1: {old_f1:.4f}).")
        print("Retention of existing champion model.")


if __name__ == "__main__":
    retrain_and_promote()
