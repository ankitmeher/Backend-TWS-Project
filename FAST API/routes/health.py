"""
Health check routes
"""
from fastapi import APIRouter
from schema.request_schema import HealthResponse
from config import MODEL_VERSION

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
def health():
    """
    Health check endpoint - confirms API is running and model is loaded
    """
    return {
        "status": "OK",
        "model_version": MODEL_VERSION
    }
