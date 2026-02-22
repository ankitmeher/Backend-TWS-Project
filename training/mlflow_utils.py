import mlflow
from training.config import MLFLOW_URI, EXPERIMENT_NAME

def setup_mlflow():
    """
    Sets the tracking URI and experiment name for MLflow.
    """
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"MLflow tracking URI: {MLFLOW_URI}")
    print(f"Experiment: {EXPERIMENT_NAME}")
