import os

# =====================================================
# MLflow Configuration
# =====================================================

# --- MLflow Server Address ---
# Use environment variable if set (e.g., in Docker or GitHub Actions), 
# otherwise default to localhost for local development.


# MLFLOW_URI = os.getenv("MLFLOW_URI", "http://13.234.184.77:5000")
MLFLOW_URI = os.getenv("MLFLOW_URI", "http://localhost:5000")

# http://13.234.184.77:5000/ (PRODUCTION) http://localhost:5000

# --- AWS EC2 PRODUCTION (uncomment when deploying) ---
# EC2 hosts the MLflow server with S3 as artifact store:
#   mlflow server \
#     --backend-store-uri sqlite:///mlflow.db \
#     --default-artifact-root s3://mlflow-buy-wait-models/mlflow \
#     --host 0.0.0.0 --port 5000 --workers 1 --allowed-hosts "*"
# Note: EC2 instance must have an IAM role with S3 read/write access


EXPERIMENT_NAME = "tws_buy_wait_experiments"
REGISTERED_MODEL_NAME = "buy_wait_model"

# =====================================================
# Data Configuration
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "database", "processed_price_history_data.csv")
TARGET = "target_buy_wait"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# --- S3 Log Configuration ---
LOG_BUCKET_NAME = "buy-wait-prediction-log-history"
LOG_PREFIX = "predictions/"
