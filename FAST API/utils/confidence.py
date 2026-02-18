"""
Utility functions for confidence scoring and interpretation
"""
from typing import Optional


def interpret_confidence(confidence: Optional[float]) -> str:
    """
    Interpret model confidence score as a human-readable level.
    
    Args:
        confidence (Optional[float]): Model confidence score between 0 and 1,
                                     or None if not available
        
    Returns:
        str: Confidence level as one of: "high", "medium", "low", or "unknown"
        
    Confidence thresholds:
        - high: >= 0.75
        - medium: >= 0.60 and < 0.75
        - low: < 0.60
        - unknown: None
    """
    if confidence is None:
        return "unknown"
    
    if confidence >= 0.75:
        return "high"
    
    if confidence >= 0.60:
        return "medium"
    
    return "low"
