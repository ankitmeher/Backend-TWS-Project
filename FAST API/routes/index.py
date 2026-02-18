"""
Index/status route
"""
from fastapi import APIRouter
from schema.request_schema import IndexResponse

router = APIRouter(tags=["Status"])


@router.get("/", response_model=IndexResponse)
def index():
    """
    Root endpoint - returns API status message
    """
    return {
        "message": "Buy / Wait Prediction API running with CI/CD pipeline added ALB",
        "Improvements": ["1. Maintain Modularity approach by separating files."]
    }
