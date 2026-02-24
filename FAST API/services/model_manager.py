import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import threading
import time

from config import MLFLOW_TRACKING_URI, REGISTERED_MODEL_NAME

# Global model state
CURRENT_MODEL = None
CURRENT_VERSION = None

lock = threading.Lock()


def get_champion_version():
    """
    Fetch current champion model version from MLflow registry.
    """
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    mv = client.get_model_version_by_alias(
        REGISTERED_MODEL_NAME,
        "champion"
    )

    return mv.version


def load_model():
    """
    Loads champion model safely.
    """
    global CURRENT_MODEL, CURRENT_VERSION

    try:
        version = get_champion_version()

        if version == CURRENT_VERSION:
            return

        print(f"[MODEL] Loading new champion version: {version}")

        model_uri = f"models:/{REGISTERED_MODEL_NAME}@champion"

        model = mlflow.sklearn.load_model(model_uri)

        with lock:
            CURRENT_MODEL = model
            CURRENT_VERSION = version

        print("[MODEL] Model updated successfully.")

    except Exception as e:
        print(f"[MODEL] Failed to load model: {e}")


def get_model():
    """
    Thread-safe model getter.
    """
    with lock:
        return CURRENT_MODEL


def start_model_watcher(interval_seconds: int = 300):
    """
    Background thread that checks MLflow periodically.
    """

    def watcher():
        print("[MODEL] Champion watcher started.")
        while True:
            load_model()
            time.sleep(interval_seconds)

    thread = threading.Thread(target=watcher, daemon=True)
    thread.start()


if __name__ == "__main__":
    print(MLFLOW_TRACKING_URI)