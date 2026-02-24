import os
import json
import boto3
import pandas as pd
import mlflow
import subprocess
from typing import List, Dict, Any

from training.config import DATA_PATH, REGISTERED_MODEL_NAME, LOG_BUCKET_NAME, LOG_PREFIX
from training.train import run_training, promote_model_to_champion

def fetch_new_logs_from_s3() -> List[Dict[str, Any]]:
    """
    Fetches all JSON prediction logs from the specified S3 bucket.
    """
    print(f"Fetching logs from S3 bucket: {LOG_BUCKET_NAME} with prefix: {LOG_PREFIX}...")
    s3 = boto3.client('s3')
    logs = []

    try:
        paginator = s3.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=LOG_BUCKET_NAME, Prefix=LOG_PREFIX)

        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    if obj['Key'].endswith('.json'):
                        response = s3.get_object(Bucket=LOG_BUCKET_NAME, Key=obj['Key'])
                        content = response['Body'].read().decode('utf-8')
                        logs.append(json.loads(content))
    except Exception as e:
        print(f"Error fetching logs from S3: {e}")

    print(f"Successfully fetched {len(logs)} logs from S3.")
    return logs

def process_logs_to_dataframe(logs: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Converts raw prediction logs into a flat DataFrame matching the training schema.
    """
    if not logs:
        return pd.DataFrame()

    processed_data = []
    for log in logs:
        features = log.get("features", {})
        if not features:
            continue
            
        entry = features.copy()
        entry["target_buy_wait"] = log.get("prediction")
        entry["id"] = log.get("id")
        
        if "date" not in entry:
            entry["date"] = log.get("timestamp")
        if "product_name" not in entry:
            entry["product_name"] = log.get("product_name")

        processed_data.append(entry)

    df = pd.DataFrame(processed_data)
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
        return f1
    except Exception:
        return 0.0

def force_ecs_redeployment():
    """
    Triggers a force redeployment of the ECS service
    so FastAPI reloads the new MLflow champion model.
    """
    cluster = os.getenv("ECS_CLUSTER")
    service = os.getenv("ECS_SERVICE")

    if not cluster or not service:
        print("ECS env vars missing — skipping redeploy")
        return

    print(f"Triggering ECS redeploy → Cluster: {cluster}, Service: {service}")

    ecs = boto3.client("ecs")

    ecs.update_service(
        cluster=cluster,
        service=service,
        forceNewDeployment=True
    )

    print("ECS force deployment triggered successfully.")

def retrain_and_promote():
    """
    Main retraining workflow using S3 data.
    """
    # 1. Fetch new data
    new_logs = fetch_new_logs_from_s3()
    if not new_logs:
        print("No new logs found. Skipping retraining.")
        return

    new_df = process_logs_to_dataframe(new_logs)

    # 2. Merge with existing dataset
    if os.path.exists(DATA_PATH):
        existing_df = pd.read_csv(DATA_PATH)
        
        if "id" in existing_df.columns and "id" in new_df.columns:
            existing_ids = set(existing_df["id"].dropna().unique())
            new_df = new_df[~new_df["id"].isin(existing_ids)]
        
        if len(new_df) > 0:
            print(f"Adding {len(new_df)} new unique records.")
            final_df = pd.concat([existing_df, new_df], ignore_index=True)
            if "id" in final_df.columns:
                mask = final_df["id"].notna()
                df_with_id = final_df[mask].drop_duplicates(subset=["id"])
                df_without_id = final_df[~mask]
                final_df = pd.concat([df_with_id, df_without_id], ignore_index=True).reset_index(drop=True)
            else:
                final_df = final_df.drop_duplicates().reset_index(drop=True)
        else:
            print("No new unique logs found. Skipping retraining.")
            return
    else:
        final_df = new_df

    # Save back to CSV
    final_df.to_csv(DATA_PATH, index=False)
    print(f"Dataset updated. Total rows: {len(final_df)}")

    # 3. Get current champion performance
    old_f1 = get_champion_f1()

    # 4. Run training pipeline
    print("\nStarting retraining pipeline...")
    results = run_training(promote=False)

    if not results:
        print("Training failed.")
        return

    new_f1 = results["f1_score_macro"]
    new_run_id = results["run_id"]

    # 5. Conditional Promotion and Redeployment
    if new_f1 >= old_f1:
        print(f"Promoting new model (F1: {new_f1:.4f}) to CHAMPION...")
        promote_model_to_champion(new_run_id)
        
        # 6. Force Redeployment
        # force_ecs_redeployment() # Commented for manual testing   
    else:
        print(f"New model (F1: {new_f1:.4f}) did not outperform champion (F1: {old_f1:.4f}).")

if __name__ == "__main__":
    retrain_and_promote()
