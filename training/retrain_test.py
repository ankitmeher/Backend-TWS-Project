import os
import json
import pandas as pd
import mlflow
from typing import List, Dict, Any

from training.config import DATA_PATH, REGISTERED_MODEL_NAME
from training.train import run_training

# Local Configuration for testing
DUMMY_LOGS_DIR = os.path.join(os.path.dirname(__file__), "dummy_logs")


def fetch_new_logs_locally() -> List[Dict[str, Any]]:
    """
    Recursively fetches all JSON prediction logs from the local dummy_logs directory.
    """
    print(f"Fetching logs from local directory: {DUMMY_LOGS_DIR}...")
    logs = []

    if not os.path.exists(DUMMY_LOGS_DIR):
        print(f"Warning: Local dummy logs directory {DUMMY_LOGS_DIR} does not exist.")
        return logs

    for root, _, files in os.walk(DUMMY_LOGS_DIR):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        logs.append(json.load(f))
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")

    print(f"Successfully fetched {len(logs)} logs locally.")
    return logs


def process_logs_to_dataframe(logs: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Converts raw prediction logs into a flat DataFrame matching the training schema.
    """
    if not logs:
        return pd.DataFrame()

    processed_data = []
    for log in logs:
        # The user wants to extract the 'features' dict and add 'target_buy_wait'
        # The 'features' dict contains the pre-engineered features
        features = log.get("features", {})
        if not features:
            print(f"Warning: Log {log.get('id')} has no features. Skipping.")
            continue
            
        entry = features.copy()
        
        # Add the target column from the top-level 'prediction' field
        # The training CSV expects 'buy' or 'wait' as strings
        entry["target_buy_wait"] = log.get("prediction")
        
        # Include the unique record ID for deduplication
        entry["id"] = log.get("id")
        
        # Include the unique record ID for deduplication
        
        # Ensure 'date' and 'product_name' are present as they are in the JSON top-level
        # but also typically in 'features' based on user's snippet.
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
        print(f"Current Champion Macro F1: {f1:.4f} (Version {champion_version.version})")
        return f1
    except Exception as e:
        print(f"No existing champion found (starting fresh): {e}")
        return 0.0


def retrain_and_promote():
    """
    Main retraining workflow using local test data.
    """
    # 1. Fetch new data
    new_logs = fetch_new_logs_locally()
    if not new_logs:
        print("No new logs found. Skipping retraining.")
        return

    new_df = process_logs_to_dataframe(new_logs)

    # Merge with existing dataset
    if os.path.exists(DATA_PATH):
        existing_df = pd.read_csv(DATA_PATH)
        print(f"Checking for new unique logs in {len(new_df)} entries...")
        
        # ID-based deduplication
        if "id" in existing_df.columns and "id" in new_df.columns:
            existing_ids = set(existing_df["id"].dropna().unique())
            new_df = new_df[~new_df["id"].isin(existing_ids)]
            print(f"Found {len(new_df)} genuine new records to add.")
        
        if len(new_df) > 0:
            final_df = pd.concat([existing_df, new_df], ignore_index=True)
            # Safer deduplication: only deduplicate rows that actually have an ID.
            if "id" in final_df.columns:
                mask = final_df["id"].notna()
                df_with_id = final_df[mask].drop_duplicates(subset=["id"])
                df_without_id = final_df[~mask]
                final_df = pd.concat([df_with_id, df_without_id], ignore_index=True).reset_index(drop=True)
            else:
                final_df = final_df.drop_duplicates().reset_index(drop=True)
        else:
            print("No new unique logs found. Skipping retraining for today.")
            return
    else:
        final_df = new_df

    # Save back to CSV
    final_df.to_csv(DATA_PATH, index=False)
    print(f"Dataset updated in {DATA_PATH}. Total rows: {len(final_df)}")

    # 3. Get current champion performance
    old_f1 = get_champion_f1()

    # 4. Run training pipeline
    print("\nStarting retraining pipeline...")
    # promote=False means we manually promote based on metric improvement
    results = run_training(promote=False)

    if not results:
        print("Training failed. No new model to compare.")
        return

    new_f1 = results["f1_score_macro"]
    new_run_id = results["run_id"]

    # 5. Conditional Promotion
    if new_f1 >= old_f1:  # Using >= to allow first model to become champion
        print(f"\nIMPROVEMENT (or identical): New model (F1: {new_f1:.4f}) vs Champion (F1: {old_f1:.4f}).")
        print("Promoting new model to CHAMPION...")
        from training.train import promote_model_to_champion
        promote_model_to_champion(new_run_id)
        print("Model transition complete.")
    else:
        print(f"\nNO IMPROVEMENT: New model (F1: {new_f1:.4f}) did not beat Champion (F1: {old_f1:.4f}).")
        print("Retention of existing champion model.")


if __name__ == "__main__":
    retrain_and_promote()
