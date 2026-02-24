import os
import json
import mlflow
import mlflow.sklearn
import mlflow.data
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report

from training.config import DATA_PATH, TARGET, TEST_SIZE, REGISTERED_MODEL_NAME
from training.data_loader import load_data
from training.preprocessing import build_preprocessor
from training.models import get_models
from training.mlflow_utils import setup_mlflow

def run_training(promote: bool = True):
    # 1. Setup MLflow
    print("\n[1/6] Setting up MLflow tracking...")
    setup_mlflow()

    # 2. Load Data
    print("\n[2/6] Loading dataset...")
    try:
        X, y = load_data(DATA_PATH)
        print(f"Successfully loaded {len(X)} rows from {DATA_PATH}")
    except Exception as e:
        print(f"Error: {e}")
        return None

    # 3. Time-based split
    print("\n[3/6] Performing time-based train-test split...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, shuffle=False
    )
    print(f"Training set: {len(X_train)} rows | Test set: {len(X_test)} rows")

    # Log dataset for MLflow UI tracking (populates the Dataset column)
    train_df = X_train.copy()
    train_df[TARGET] = y_train.values
    mlflow_dataset = mlflow.data.from_pandas(train_df, source=DATA_PATH, name="tws_price_history")

    # 4. Build Preprocessor
    print("\n[4/6] Building feature preprocessor...")
    preprocess = build_preprocessor(X_train)
    print("Preprocessor defined for categorical (brand) and numerical features.")

    # 5. Loop through models
    print("\n[5/6] Starting model training loop...")
    models = get_models()
    tscv = TimeSeriesSplit(n_splits=3)
    
    best_f1_macro = 0
    best_run_id = None
    best_model_name = None

    for model_name, (model, param_grid) in models.items():
        print(f"\n--- Running Experiment: {model_name} ---")
        
        with mlflow.start_run(run_name=model_name) as run:
            mlflow.log_input(mlflow_dataset, context="training")
            pipeline = Pipeline([
                ("preprocess", preprocess),
                ("model", model)
            ])

            grid = GridSearchCV(
                pipeline,
                param_grid,
                cv=tscv,
                scoring="accuracy",
                n_jobs=-1
            )

            # Train
            grid.fit(X_train, y_train)
            best_pipeline = grid.best_estimator_

            # Evaluate
            y_pred = best_pipeline.predict(X_test)
            test_acc = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            f1_macro = report["macro avg"]["f1-score"]

            # Log Parameters
            mlflow.log_param("model_type", model_name)
            for k, v in grid.best_params_.items():
                mlflow.log_param(k, v)

            # -------------------------
            # Log Metrics
            # -------------------------
            mlflow.log_metric("accuracy", report.get("accuracy", 0))
            mlflow.log_metric("cv_best_score", grid.best_score_)
            mlflow.log_metric("test_accuracy", test_acc)
            
            # class-wise metrics (Safely handle potentially missing classes)
            for cls in ["0", "1"]:
                if cls in report:
                    mlflow.log_metric(f"recall_class_{cls}", report[cls].get("recall", 0))
                    mlflow.log_metric(f"f1_score_class_{cls}", report[cls].get("f1-score", 0))
            
            # macro metrics
            if "macro avg" in report:
                mlflow.log_metric("f1_score_macro", f1_macro)
                mlflow.log_metric("recall_score_macro", report["macro avg"].get("recall", 0))
            
            # Log classification report (locally only if run directly)
            if __name__ == "__main__":
                report_filename = f"classification_report_{model_name}.json"
                report_dir = os.path.join(os.path.dirname(__file__), "classification_report")
                report_path = os.path.join(report_dir, report_filename)
                os.makedirs(report_dir, exist_ok=True)
                with open(report_path, "w") as f:
                    json.dump(report, f, indent=4)
                mlflow.log_artifact(report_path)

            # Log Model + Register
            mlflow.sklearn.log_model(
                sk_model=best_pipeline,
                artifact_path=model_name,
                registered_model_name=REGISTERED_MODEL_NAME,
            )

            # [NEW] Log the model with its specific name (e.g., random_forest.pkl)
            # We log it INSIDE the same directory as the MLflow model to keep all metadata (conda, requirements) visible
            model_filename = f"{model_name}.pkl"
            joblib.dump(best_pipeline, model_filename)
            mlflow.log_artifact(model_filename, artifact_path=model_name)
            os.remove(model_filename)  # Clean up local file

            print(f"{model_name} Test Accuracy: {test_acc:.4f} | Macro F1: {f1_macro:.4f}")

            # Track best model based on Macro F1
            if f1_macro > best_f1_macro:
                best_f1_macro = f1_macro
                best_run_id = run.info.run_id
                best_model_name = model_name

    print(f"\nBest Model of this run: {best_model_name}")
    print(f"Best Macro F1 of this run: {best_f1_macro:.4f}")
    print(f"Best Run ID: {best_run_id}")

    # =====================================================
    # Promote Best Model as Champion
    # =====================================================
    if best_run_id and promote:
        print("\n[6/6] Promoting best model as CHAMPION...")
        promote_model_to_champion(best_run_id)

    return {"f1_score_macro": best_f1_macro, "run_id": best_run_id}


def promote_model_to_champion(run_id: str):
    """
    Sets the 'champion' alias on the model version generated by the given run_id.
    """
    client = mlflow.tracking.MlflowClient()
    print(f"Target Run ID: {run_id}")
    all_versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
    
    if all_versions:
        best_version = None
        for v in all_versions:
            print(f"Checking Model Version {v.version}: Run ID {v.run_id}")
            if v.run_id == run_id:
                best_version = v.version
                print(f"Matched! Selecting version {best_version}")
                break

        if not best_version:
            print(f"Warning: No version found for run_id {run_id}. Defaulting to version {all_versions[0].version}")
            best_version = all_versions[0].version

        client.set_registered_model_alias(
            name=REGISTERED_MODEL_NAME,
            alias="champion",
            version=best_version
        )
        print(f"Successfully set 'champion' alias on {REGISTERED_MODEL_NAME} version {best_version}.")

if __name__ == "__main__":
    run_training()
