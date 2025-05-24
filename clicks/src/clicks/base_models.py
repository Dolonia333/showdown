from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class ParsedPrediction:
    x: int
    y: int
    raw_response: str

@dataclass 
class EvaluationResult:
    id: str
    instruction: str
    image_path: str
    gt_x1: int
    gt_y1: int  
    gt_x2: int
    gt_y2: int
    width: int
    height: int
    prediction: Optional[ParsedPrediction] = None
    error: Optional[str] = None
    latency_seconds: Optional[float] = None
    is_in_bbox: Optional[bool] = None
    visualization_path: Optional[str] = None
