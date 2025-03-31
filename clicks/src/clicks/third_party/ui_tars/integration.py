import os
import time
from typing import Any, Dict

from clicks.api_client_base import AbstractAPIClient
from clicks.evaluate.models import (
  EvaluationItem,
  EvaluationResult,
  ParsedPrediction,
)
from clicks.evaluate.utils import (
  check_prediction_in_bbox,
  print_colored_result,
  visualize_prediction,
)

from ..common import encode_image_to_base64_uri
from .client import UITarsClient


class UITarsAPIClient(AbstractAPIClient):
  def __init__(
    self,
    api_url: str,
    api_key: str = 'super-secret-key',
    max_tokens: int = 128,
    temperature: float = 0.0,
    frequency_penalty: float = 1.0,
    model_name: str = 'bytedance-research/UI-TARS-72B-SFT',
  ):
    self.client = UITarsClient(
      api_url=api_url,
      api_key=api_key,
      max_tokens=max_tokens,
      temperature=temperature,
      frequency_penalty=frequency_penalty,
      model_name=model_name,
    )
    self.model_name = model_name
    self.client_type = 'ui_tars'

  def predict(self, image_data_uri: str, prompt: str) -> Dict[str, Any]:
    return self.client.predict(image_data_uri=image_data_uri, prompt=prompt)

  def parse_prediction(self, prediction: Dict[str, Any]) -> ParsedPrediction:
    parsed_data = self.client.parse_prediction(prediction)

    return ParsedPrediction(
      pred_x=parsed_data.get('pred_x'),
      pred_y=parsed_data.get('pred_y'),
      raw_response=parsed_data.get('raw_responses', '{}'),
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

    image_data_uri = encode_image_to_base64_uri(image_path)
    if not image_data_uri:
      raise ValueError(f'Failed to encode image: {image_path}')

    print(f'Processing item {eval_item.id}: {eval_item.instruction}')
    start_time = time.time()
    prediction = self.predict(image_data_uri=image_data_uri, prompt=eval_item.instruction)
    end_time = time.time()
    latency = end_time - start_time

    parsed_prediction = self.parse_prediction(prediction)

    error = prediction.get('error')
    if error:
      print(f'Error: {error}')
      raise ValueError(error)

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
      model_name=self.model_name,
      run_id=run_id,
      gt_x1=eval_item.x1,
      gt_y1=eval_item.y1,
      gt_x2=eval_item.x2,
      gt_y2=eval_item.y2,
      is_in_bbox=is_in_bbox,
    )

    return EvaluationResult(
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


def get_ui_tars_api_client(
  api_url: str,
  api_key: str = 'super-secret-key',
  max_tokens: int = 128,
  temperature: float = 0.0,
  frequency_penalty: float = 1.0,
  model_name: str = 'bytedance-research/UI-TARS-72B-SFT',
  **kwargs,
) -> UITarsAPIClient:
  client = UITarsAPIClient(
    api_url=api_url,
    api_key=api_key,
    max_tokens=max_tokens,
    temperature=temperature,
    frequency_penalty=frequency_penalty,
    model_name=model_name,
  )
  return client
