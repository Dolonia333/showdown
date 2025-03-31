import os
import time
from typing import Any, Dict, Optional

from clicks.api_client_base import AbstractAPIClient
from clicks.evaluate.models import (
  ClaudeModelConfig,
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
from .client import DEFAULT_MODEL, ClaudeComputerUseClient


class ClaudeComputerUseAPIClient(AbstractAPIClient):
  def __init__(
    self,
    config: Optional[ClaudeModelConfig] = None,
  ):
    if config is None:
      config = ClaudeModelConfig(
        api_endpoint='https://api.anthropic.com/v1/messages',
        model=DEFAULT_MODEL,
        api_key=os.environ.get('ANTHROPIC_API_KEY'),
        thinking_budget=1024,
        tool_version='20250124',
      )

    self.claude_client = ClaudeComputerUseClient(
      api_key=config.api_key,
      api_endpoint=config.api_endpoint,
      model=config.model,
      thinking_budget=config.thinking_budget,
      tool_version=config.tool_version,
    )
    self.config = config
    self.client_type = 'claude'

  def predict(
    self, image_data_uri: str, prompt: str, model: Optional[str] = None
  ) -> Dict[str, Any] | None:
    return self.claude_client.predict(image_data_uri, prompt)

  def parse_prediction(self, prediction: Dict[str, Any]) -> ParsedPrediction:
    result_dict = self.claude_client.parse_prediction(prediction)

    return ParsedPrediction(
      pred_x=result_dict.get('pred_x'),
      pred_y=result_dict.get('pred_y'),
      raw_response=result_dict.get('raw_response'),
    )

  def process_single_item(
    self,
    item: Dict[str, Any],
    frames_dir: str,
    run_id: str,
  ) -> EvaluationResult:
    try:
      eval_item = item if isinstance(item, EvaluationItem) else EvaluationItem.model_validate(item)

      item_id = eval_item.id
      recording_id = eval_item.recording_id
      instruction = eval_item.instruction
      image_path = eval_item.image

      gt_x1 = eval_item.x1
      gt_y1 = eval_item.y1
      gt_x2 = eval_item.x2
      gt_y2 = eval_item.y2

      local_path = os.path.join(frames_dir, image_path)
      if not os.path.exists(local_path):
        raise FileNotFoundError(f'Image file not found: {local_path}')

      image_data_uri = encode_image_to_base64_uri(local_path)
      print(f'Processing item {item_id} with instruction: {instruction}')

      start_time = time.time()
      prediction = self.predict(image_data_uri, instruction)
      end_time = time.time()
      latency = end_time - start_time

      if prediction is None:
        print(f'Claude returned None for item {item_id}')
        pred_result = ParsedPrediction()
        pred_x = None
        pred_y = None
      else:
        pred_result = self.parse_prediction(prediction)
        pred_x = pred_result.pred_x
        pred_y = pred_result.pred_y

      is_in_bbox = check_prediction_in_bbox(pred_x, pred_y, gt_x1, gt_y1, gt_x2, gt_y2)

      print_colored_result(
        item_id,
        instruction,
        pred_x,
        pred_y,
        latency,
        is_in_bbox,
      )

      visualization_path = visualize_prediction(
        local_path,
        pred_x,
        pred_y,
        item_id,
        recording_id,
        instruction,
        self.config.model,
        run_id,
        gt_x1,
        gt_y1,
        gt_x2,
        gt_y2,
        is_in_bbox,
      )

      result = EvaluationResult(
        id=item_id,
        recording_id=recording_id,
        instruction=instruction,
        image_path=local_path,
        gt_x1=gt_x1,
        gt_y1=gt_y1,
        gt_x2=gt_x2,
        gt_y2=gt_y2,
        pred_x=pred_x,
        pred_y=pred_y,
        is_in_bbox=is_in_bbox,
        latency_seconds=latency,
        raw_response=pred_result.raw_response if pred_result else None,
        visualization_path=visualization_path,
      )

      return result

    except Exception as e:
      print(f'API request failed: {str(e)}')
      raise e


def get_claude_api_client(
  api_key: Optional[str] = None,
  api_endpoint: Optional[str] = None,
  model: str = DEFAULT_MODEL,
  thinking_budget: Optional[int] = 1024,
  tool_version: str = '20250124',
) -> ClaudeComputerUseAPIClient:
  config = ClaudeModelConfig(
    api_endpoint=api_endpoint or 'https://api.anthropic.com/v1/messages',
    model=model,
    api_key=api_key or os.environ.get('ANTHROPIC_API_KEY'),
    thinking_budget=thinking_budget,
    tool_version=tool_version,
  )
  return ClaudeComputerUseAPIClient(config)
