from typing import Optional

from pydantic import BaseModel


class Coordinate(BaseModel):
  x: Optional[int] = None
  y: Optional[int] = None


class ClickAction(BaseModel):
  kind: str = 'left_click'
  coordinate: Coordinate


class Action(BaseModel):
  kind: str
  coordinate: Optional[Coordinate] = None
  text: Optional[str] = None


class AcePrediction(BaseModel):
  action: Action
  raw_response: Optional[str] = None


class ParsedPrediction(BaseModel):
  pred_x: Optional[int] = None
  pred_y: Optional[int] = None
  raw_response: Optional[str] = None


class GroundTruth(BaseModel):
  gt_x1: Optional[int] = None
  gt_y1: Optional[int] = None
  gt_x2: Optional[int] = None
  gt_y2: Optional[int] = None


class EvaluationMetrics(BaseModel):
  total_processed: int
  total_correct: int
  accuracy: float
  ci: float
  accuracy_ci_low: Optional[float] = None
  accuracy_ci_high: Optional[float] = None


class EvaluationItem(BaseModel):
  id: str
  recording_id: str
  instruction: str
  image: str
  x1: Optional[int] = None
  y1: Optional[int] = None
  x2: Optional[int] = None
  y2: Optional[int] = None
  width: Optional[int] = None
  height: Optional[int] = None


# Moved to base_models.py


class ModelConfig(BaseModel):
  api_endpoint: str = ''


class AceModelConfig(ModelConfig):
  model: str = 'ace-control-medium'
  api_key: Optional[str] = None


class ClaudeModelConfig(ModelConfig):
  model: str = 'claude-3-7-sonnet-20250219'
  api_key: Optional[str] = None
  thinking_budget: Optional[int] = 1024
  tool_version: str = '20250124'


class QwenModelConfig(ModelConfig):
  model: str = 'qwen2.5-vl-72b-instruct'
  api_key: Optional[str] = None
  max_tokens: int = 4096
  use_smart_resize: bool = True
  resize_factor: int = 28
  min_pixels: int = 3136
  max_pixels: int = 12845056


class OpenAIModelConfig(ModelConfig):
  model: str = 'o1'
  api_key: Optional[str] = None
  max_tokens: int = 4096
  reasoning_effort: str = 'medium'
  environment: str = 'mac'


class GeminiModelConfig(ModelConfig):
  model: str = 'gemini-2.0-flash'
  api_key: Optional[str] = None
  max_tokens: int = 4096
  temperature: float = 0.0


class OmniParserModelConfig(ModelConfig):
  model: str = 'gpt-4o-2024-05-13'
  temperature: float = 0.7
