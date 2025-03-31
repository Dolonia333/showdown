import os
import time
from typing import Any, Dict, Optional

from clicks.api_client_base import AbstractAPIClient
from clicks.evaluate.models import (
  EvaluationItem,
  EvaluationResult,
  ParsedPrediction,
  QwenModelConfig,
)
from clicks.evaluate.utils import (
  check_prediction_in_bbox,
  print_colored_result,
  visualize_prediction,
)

from ..common import encode_image_to_base64_uri
from .client import DEFAULT_MODEL, QwenVLClient


class QwenVLAPIClient(AbstractAPIClient):
  def __init__(
    self,
    config: Optional[QwenModelConfig] = None,
  ):
    if config is None:
      config = QwenModelConfig(
        api_endpoint='https://dashscope.aliyuncs.com/compatible-mode/v1',
        model=DEFAULT_MODEL,
        api_key=os.environ.get('DASHSCOPE_API_KEY'),
        max_tokens=4096,
        use_smart_resize=True,
        resize_factor=28,
        min_pixels=3136,
        max_pixels=12845056,
      )

    self.qwen_client = QwenVLClient(
      api_key=config.api_key,
      api_endpoint=config.api_endpoint,
      model=config.model,
      max_tokens=config.max_tokens,
      use_smart_resize=config.use_smart_resize,
      resize_factor=config.resize_factor,
      min_pixels=config.min_pixels,
      max_pixels=config.max_pixels,
    )
    self.config = config
    self.client_type = 'qwen'

  def predict(
    self, image_data_uri: str, prompt: str, model: Optional[str] = None
  ) -> Dict[str, Any] | None:
    return self.qwen_client.predict(image_data_uri, prompt)

  def parse_prediction(self, prediction: Dict[str, Any]) -> ParsedPrediction:
    result_dict = self.qwen_client.parse_prediction(prediction)

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

      item_id = eval_item.id
      recording_id = eval_item.recording_id
      instruction = eval_item.instruction
      image_path = eval_item.image

      local_path = os.path.join(frames_dir, image_path)
      if not os.path.exists(local_path):
        raise FileNotFoundError(f'Image file not found: {local_path}')

      image_data_uri = encode_image_to_base64_uri(local_path)
      if image_data_uri is None:
        raise ValueError(f'Failed to encode image: {local_path}')

      print(f'Processing item {item_id} with instruction: {instruction}')
      start_time = time.time()
      prediction = self.predict(image_data_uri, instruction)
      latency = time.time() - start_time

      if prediction is None:
        print(f'Qwen returned None for item {item_id}')
        pred_result = ParsedPrediction()
        pred_x = None
        pred_y = None
      else:
        pred_result = self.parse_prediction(prediction)
        pred_x = pred_result.pred_x
        pred_y = pred_result.pred_y

      is_in_bbox = check_prediction_in_bbox(
        pred_x, pred_y, eval_item.x1, eval_item.y1, eval_item.x2, eval_item.y2
      )

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
        eval_item.x1,
        eval_item.y1,
        eval_item.x2,
        eval_item.y2,
        is_in_bbox,
      )

      result = EvaluationResult(
        id=item_id,
        recording_id=recording_id,
        instruction=instruction,
        image_path=local_path,
        gt_x1=eval_item.x1,
        gt_y1=eval_item.y1,
        gt_x2=eval_item.x2,
        gt_y2=eval_item.y2,
        pred_x=pred_x,
        pred_y=pred_y,
        is_in_bbox=is_in_bbox,
        latency_seconds=latency,
        raw_response=pred_result.raw_response,
        visualization_path=visualization_path,
      )

      return result

    except Exception as e:
      print(f'Error processing item: {e}')
      raise e


def get_qwen_api_client(
  api_key: Optional[str] = None,
  api_endpoint: Optional[str] = None,
  model: str = DEFAULT_MODEL,
  max_tokens: int = 4096,
  use_smart_resize: bool = True,
  resize_factor: int = 28,
  min_pixels: int = 3136,
  max_pixels: int = 12845056,
) -> QwenVLAPIClient:
  config = QwenModelConfig(
    api_endpoint=api_endpoint or 'https://dashscope.aliyuncs.com/compatible-mode/v1',
    model=model,
    api_key=api_key or os.environ.get('DASHSCOPE_API_KEY'),
    max_tokens=max_tokens,
    use_smart_resize=use_smart_resize,
    resize_factor=resize_factor,
    min_pixels=min_pixels,
    max_pixels=max_pixels,
  )
  return QwenVLAPIClient(config)
