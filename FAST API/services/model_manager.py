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
    
    print(f"[MODEL] Querying MLflow for alias 'champion' in model '{REGISTERED_MODEL_NAME}'...")
    try:
        mv = client.get_model_version_by_alias(
            REGISTERED_MODEL_NAME,
            "champion"
        )
        print(f"[MODEL] Alias 'champion' found: Version {mv.version}")
        return mv.version
    except Exception as e:
        print(f"[MODEL] Alias 'champion' not found or error occurred: {e}")
        
        # Fallback: Get the latest version instead of failing
        print(f"[MODEL] Fallback: Searching for the latest version of '{REGISTERED_MODEL_NAME}'...")
        versions = client.search_model_versions(f"name='{REGISTERED_MODEL_NAME}'")
        if not versions:
            raise ValueError(f"No versions found for model '{REGISTERED_MODEL_NAME}'")
        
        # Sort by version number descending and pick the newest
        latest_version = sorted([int(v.version) for v in versions], reverse=True)[0]
        print(f"[MODEL] Fallback successful: Using latest version {latest_version}")
        return str(latest_version)


def load_model():
    """
    Loads model safely with fallbacks.
    """
    global CURRENT_MODEL, CURRENT_VERSION
    
    print(f"\n[MODEL] Starting load process (Tracking URI: {MLFLOW_TRACKING_URI})")
    
    try:
        # 1. Discover the version
        version = get_champion_version()

        # 2. Skip loading if version hasn't changed
        if version == CURRENT_VERSION and CURRENT_MODEL is not None:
            return  

        print(f"[MODEL] Transitioning from {CURRENT_VERSION} to {version}...")

        # 3. Construct URI using VERSION number for robustness
        # Loading via '@champion' directly fails in AWS environment due to resolution latency
        model_uri = f"models:/{REGISTERED_MODEL_NAME}/{version}"
        
        print(f"[MODEL] Loading model artifacts from: {model_uri}")

        # Explicitly set tracking URI
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        
        model = mlflow.sklearn.load_model(model_uri)

        with lock:
            CURRENT_MODEL = model
            CURRENT_VERSION = version

        print(f"[MODEL] SUCCESS: Model version {version} is now active.")

    except Exception as e:
        print(f"[MODEL] CRITICAL FAILURE: Failed to load model: {e}")
        import traceback
        traceback.print_exc()


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
            print("Checking for new model")
            load_model()
            time.sleep(interval_seconds)

    thread = threading.Thread(target=watcher, daemon=True)
    thread.start()


if __name__ == "__main__":
    print(MLFLOW_TRACKING_URI)