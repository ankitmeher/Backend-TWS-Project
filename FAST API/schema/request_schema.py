"""
Request and Response schemas for the API
"""
from pydantic import BaseModel
from typing import Optional


class PredictRequest(BaseModel):
    """Schema for prediction request"""
    product_name: str
    price: float

    class Config:
        json_schema_extra = {
            "example": {
                "product_name": "Sony WH-1000XM4",
                "price": 348.99
            }
        }


class PredictionResponse(BaseModel):
    """Schema for prediction response"""
    product_name: str
    input_price: float
    prediction: str
    confidence: Optional[float] = None
    confidence_level: str

    class Config:
        json_schema_extra = {
            "example": {
                "product_name": "Sony WH-1000XM4",
                "input_price": 348.99,
                "prediction": "buy",
                "confidence": 0.85,
                "confidence_level": "high"
            }
        }


class HealthResponse(BaseModel):
    """Schema for health check response"""
    status: str
    model_version: str


class IndexResponse(BaseModel):
    """Schema for index response"""
    message: str
    Improvements: Optional[list] = None

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Buy / Wait Prediction API running with CI/CD pipeline added ALB",
                "Improvements": ["1. Maintain Modularity approach by separating files."]
            }
        }
