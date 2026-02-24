"""
Main API entry point for Buy/Wait Prediction API
Uses modular components for better organization and maintainability
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config import MODEL_VERSION, API_TITLE
from services.model_manager import start_model_watcher, load_model
from routes import index, health, predictions


# ================== APP INITIALIZATION ==================
app = FastAPI(
    title=API_TITLE,
    version=MODEL_VERSION,
    description="Predicts whether to buy or wait for a product based on price history"
)


# ================== MIDDLEWARE ==================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (change to specific domain in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup_event():
    print("[APP] Starting model watcher...")
    load_model()              # initial load
    start_model_watcher(interval_seconds=60)     # background refresh

# ================== INCLUDE ROUTES ==================
app.include_router(index.router)
app.include_router(health.router)
app.include_router(predictions.router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
