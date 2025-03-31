import os
import time
from typing import Any, Dict, Optional

from clicks.api_client_base import AbstractAPIClient
from clicks.evaluate.models import (
  EvaluationItem,
  EvaluationResult,
  OpenAIModelConfig,
  ParsedPrediction,
)
from clicks.evaluate.utils import (
  check_prediction_in_bbox,
  print_colored_result,
  visualize_prediction,
)

from ..common import encode_image_to_base64_uri
from .client import DEFAULT_ENVIRONMENT, DEFAULT_MODEL, OpenAICUAClient


class OpenAICUAAPIClient(AbstractAPIClient):
  def __init__(
    self,
    config: Optional[OpenAIModelConfig] = None,
  ):
    self.config = config or OpenAIModelConfig(
      api_endpoint=os.environ.get('OPENAI_API_ENDPOINT', 'https://api.openai.com/v1')
    )

    self.api_key = self.config.api_key
    self.api_endpoint = self.config.api_endpoint
    self.model = self.config.model
    self.max_tokens = self.config.max_tokens
    self.environment = getattr(self.config, 'environment', DEFAULT_ENVIRONMENT)
    self.client_type = 'openai_cua'

  def predict(
    self, image_data_uri: str, prompt: str, model: Optional[str] = None
  ) -> Dict[str, Any] | None:
    client = OpenAICUAClient(
      api_key=self.api_key,
      api_endpoint=self.api_endpoint,
      model=model or self.model,
      max_tokens=self.max_tokens,
      environment=self.environment,
    )

    return client.predict(image_data_uri, prompt)

  def parse_prediction(self, prediction: Dict[str, Any]) -> ParsedPrediction:
    client = OpenAICUAClient(
      api_key=self.api_key,
      api_endpoint=self.api_endpoint,
      model=self.model,
      max_tokens=self.max_tokens,
      environment=self.environment,
    )

    parsed = client.parse_prediction(prediction)

    return ParsedPrediction(
      pred_x=parsed['pred_x'],
      pred_y=parsed['pred_y'],
      raw_response=parsed['raw_response'],
    )

  def process_single_item(
    self,
    item: Dict[str, Any],
    frames_dir: str,
    run_id: str,
  ) -> EvaluationResult:
    eval_item = EvaluationItem(
      id=item['id'],
      recording_id=item['recording_id'],
      instruction=item['instruction'],
      image=item['image'],
      x1=item['x1'],
      y1=item['y1'],
      x2=item['x2'],
      y2=item['y2'],
    )

    image_path = os.path.join(frames_dir, eval_item.image)

    if not os.path.exists(image_path):
      raise FileNotFoundError(f'Image file not found at {image_path}')

    image_data_uri = encode_image_to_base64_uri(image_path)
    if image_data_uri is None:
      raise ValueError(f'Failed to encode image at {image_path}')

    start_time = time.time()
    prediction = self.predict(image_data_uri, eval_item.instruction)
    end_time = time.time()
    latency = end_time - start_time

    if prediction:
      parsed_prediction = self.parse_prediction(prediction)

      is_in_bbox = check_prediction_in_bbox(
        pred_x=parsed_prediction.pred_x,
        pred_y=parsed_prediction.pred_y,
        gt_x1=eval_item.x1,
        gt_y1=eval_item.y1,
        gt_x2=eval_item.x2,
        gt_y2=eval_item.y2,
      )

      print_colored_result(
        item_id=eval_item.id,
        instruction=eval_item.instruction,
        pred_x=parsed_prediction.pred_x,
        pred_y=parsed_prediction.pred_y,
        latency=latency,
        is_in_bbox=is_in_bbox,
      )

      visualization_path = visualize_prediction(
        image_path=image_path,
        pred_x=parsed_prediction.pred_x,
        pred_y=parsed_prediction.pred_y,
        item_id=eval_item.id,
        recording_id=eval_item.recording_id,
        instruction=eval_item.instruction,
        model_name='openai_cua',
        run_id=run_id,
        gt_x1=eval_item.x1,
        gt_y1=eval_item.y1,
        gt_x2=eval_item.x2,
        gt_y2=eval_item.y2,
        is_in_bbox=is_in_bbox,
      )

      result = EvaluationResult(
        id=eval_item.id,
        recording_id=eval_item.recording_id,
        instruction=eval_item.instruction,
        image_path=image_path,
        gt_x1=eval_item.x1,
        gt_y1=eval_item.y1,
        gt_x2=eval_item.x2,
        gt_y2=eval_item.y2,
        pred_x=parsed_prediction.pred_x,
        pred_y=parsed_prediction.pred_y,
        is_in_bbox=is_in_bbox,
        latency_seconds=latency,
        raw_response=parsed_prediction.raw_response,
        visualization_path=visualization_path,
      )

      return result
    else:
      raise ValueError('Prediction is None')


def get_openai_cua_api_client(
  api_key: Optional[str] = None,
  api_endpoint: Optional[str] = None,
  model: str = DEFAULT_MODEL,
  max_tokens: int = 4096,
  environment: str = DEFAULT_ENVIRONMENT,
) -> OpenAICUAAPIClient:
  # Create a custom config that includes CUA-specific parameters
  config = OpenAIModelConfig(
    api_endpoint=api_endpoint or os.environ.get('OPENAI_API_ENDPOINT', 'https://api.openai.com/v1'),
    model=model,
    api_key=api_key,
    max_tokens=max_tokens,
  )

  # Add CUA-specific attributes
  setattr(config, 'environment', environment)

  return OpenAICUAAPIClient(config=config)
