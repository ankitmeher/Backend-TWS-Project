"""
Configuration and constants for the Buy/Wait Prediction API
"""
import os

# Base directory for the application
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Model configuration
MODEL_VERSION = "1.0"
MODEL_PATH = os.path.join(BASE_DIR, "models", "price_history_model.pkl")

# Database configuration
DB_PATH = os.path.join(BASE_DIR, "database", "price_history.db")

# API Configuration
API_TITLE = "Buy / Wait Prediction API"
API_VERSION = "1.0"
API_DESCRIPTION = "Predicts whether to buy or wait for a product based on price history"
