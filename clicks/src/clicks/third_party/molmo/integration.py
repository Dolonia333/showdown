import os
import time
from typing import Any, Dict, Optional

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
from .client import MolmoClient


class MolmoAPIClient(AbstractAPIClient):
  def __init__(
    self,
    api_url: str,
    api_key: Optional[str] = None,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    top_p: float = 0.9,
    top_k: int = 50,
  ):
    self.client = MolmoClient(
      api_url=api_url,
      api_key=api_key,
      max_tokens=max_tokens,
      temperature=temperature,
      top_p=top_p,
      top_k=top_k,
    )
    self.client_type = 'molmo'

  def predict(self, image_data_uri: str, prompt: str) -> Dict[str, Any]:
    return self.client.predict(image_data_uri=image_data_uri, prompt=prompt)

  def parse_prediction(self, prediction: Dict[str, Any]) -> ParsedPrediction:
    parsed = self.client.parse_prediction(prediction)

    return ParsedPrediction(
      pred_x=parsed.get('pred_x'),
      pred_y=parsed.get('pred_y'),
      raw_response=parsed.get('raw_responses', '{}'),
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

    image_path = eval_item.image

    local_path = os.path.join(frames_dir, image_path)
    if not os.path.exists(local_path):
      raise FileNotFoundError(f'Local image not found at {local_path}')

    image_data_uri = encode_image_to_base64_uri(local_path)
    if image_data_uri is None:
      raise ValueError(f'Failed to encode image at {local_path}')

    try:
      start_time = time.time()
      prediction = self.predict(image_data_uri, eval_item.instruction)
      latency = time.time() - start_time

      pred_result = self.parse_prediction(prediction)
      pred_x = pred_result.pred_x
      pred_y = pred_result.pred_y

      is_in_bbox = check_prediction_in_bbox(
        pred_x, pred_y, eval_item.x1, eval_item.y1, eval_item.x2, eval_item.y2
      )

      print_colored_result(
        item_id=eval_item.id,
        instruction=eval_item.instruction,
        pred_x=pred_x,
        pred_y=pred_y,
        latency=latency,
        is_in_bbox=is_in_bbox,
      )

      visualization_path = visualize_prediction(
        image_path=local_path,
        pred_x=pred_x,
        pred_y=pred_y,
        item_id=eval_item.id,
        recording_id=eval_item.recording_id,
        instruction=eval_item.instruction,
        model_name='molmo',
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
        image_path=local_path,
        gt_x1=eval_item.x1,
        gt_y1=eval_item.y1,
        gt_x2=eval_item.x2,
        gt_y2=eval_item.y2,
        pred_x=pred_x,
        pred_y=pred_y,
        is_in_bbox=is_in_bbox,
        latency_seconds=latency,
        raw_response=prediction.get('raw_response', '{}'),
        visualization_path=visualization_path,
      )

      return result

    except Exception as e:
      print(f'Error processing item {eval_item.id}: {str(e)}')
      raise e


def get_molmo_api_client(
  api_url: str,
  api_key: Optional[str] = None,
  max_tokens: int = 4096,
  temperature: float = 0.0,
  top_p: float = 0.9,
  top_k: int = 50,
) -> MolmoAPIClient:
  return MolmoAPIClient(
    api_url=api_url,
    api_key=api_key,
    max_tokens=max_tokens,
    temperature=temperature,
    top_p=top_p,
    top_k=top_k,
  )
