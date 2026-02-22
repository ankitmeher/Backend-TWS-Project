"""
Configuration and constants for the Buy/Wait Prediction API
"""
import os

# Base directory for the application
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model configuration
# OLD: MODEL_PATH = os.path.join(BASE_DIR, "models", "price_history_model.pkl")
MODEL_VERSION = "3.0"  # Now served via MLflow Model Registry

# MLflow Model Registry (uses environment variable if set, e.g. in Docker/ECS)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_URI", "http://13.234.184.77:5000")
MLFLOW_MODEL_URI = "models:/buy_wait_model@champion"
# For EC2 Production, replace MLFLOW_TRACKING_URI with: http://13.234.184.77:5000

# Database configuration
DB_PATH = os.path.join(BASE_DIR, "database", "price_history.db")

# API Configuration
API_TITLE = "Buy / Wait Prediction API"
API_VERSION = "1.0"
API_DESCRIPTION = "Predicts whether to buy or wait for a product based on price history"
