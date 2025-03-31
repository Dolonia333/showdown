import os
import time
from typing import Any, Dict, Optional

from clicks.api_client_base import AbstractAPIClient
from clicks.evaluate.models import (
  AceModelConfig,
  AcePrediction,
  EvaluationItem,
  EvaluationResult,
  ParsedPrediction,
)
from clicks.evaluate.utils import (
  check_prediction_in_bbox,
  print_colored_result,
  visualize_prediction,
)
from generalagents import Agent
from PIL import Image


class AceAPIClient(AbstractAPIClient):
  def __init__(self, config: AceModelConfig):
    self.config = config
    self.client_type = 'ace'
    self.agent = Agent(model=self.config.model, api_key=self.config.api_key or '')

  def predict(
    self,
    image_data_uri: str,
    prompt: str,
    model: Optional[str] = None,
  ) -> Dict[str, Any]:
    raise NotImplementedError('Ace API client does not support predict method')

  def predict_ace(
    self,
    image: Image.Image,
    prompt: str,
  ) -> ParsedPrediction:
    session = self.agent.start(prompt)
    action = session.plan(image)

    if action.kind == 'left_click':
      pred_x = action.coordinate.x
      pred_y = action.coordinate.y
    else:
      pred_x, pred_y = None, None

    return ParsedPrediction(
      pred_x=pred_x,
      pred_y=pred_y,
      raw_response=str(action),
    )

  def parse_prediction(self, prediction: Dict[str, Any]) -> ParsedPrediction:
    ace_prediction = AcePrediction.model_validate(prediction)
    if ace_prediction.action.kind == 'left_click':
      if ace_prediction.action.coordinate:
        pred_x = ace_prediction.action.coordinate.x
        pred_y = ace_prediction.action.coordinate.y
      else:
        pred_x, pred_y = None, None
    else:
      pred_x, pred_y = None, None

    return ParsedPrediction(
      pred_x=pred_x,
      pred_y=pred_y,
      raw_response=ace_prediction.raw_response,
    )

  def process_single_item(
    self,
    item: Dict[str, Any],
    frames_dir: str,
    run_id: str,
  ) -> EvaluationResult:
    eval_item = item if isinstance(item, EvaluationItem) else EvaluationItem.model_validate(item)

    image_path = eval_item.image
    local_path = os.path.join(frames_dir, image_path)
    if not os.path.exists(local_path):
      raise FileNotFoundError(f'Local image not found at {local_path}')
    image = Image.open(local_path)

    try:
      start_time = time.time()
      prediction = self.predict_ace(image, eval_item.instruction)
      latency = time.time() - start_time

      pred_x = prediction.pred_x
      pred_y = prediction.pred_y

      gt_x1 = eval_item.x1
      gt_y1 = eval_item.y1
      gt_x2 = eval_item.x2
      gt_y2 = eval_item.y2

      is_in_bbox = check_prediction_in_bbox(pred_x, pred_y, gt_x1, gt_y1, gt_x2, gt_y2)

      visualization_path = visualize_prediction(
        local_path,
        pred_x,
        pred_y,
        eval_item.id,
        eval_item.recording_id,
        eval_item.instruction,
        self.config.model,
        run_id,
        gt_x1,
        gt_y1,
        gt_x2,
        gt_y2,
        is_in_bbox,
      )

      print_colored_result(
        eval_item.id,
        eval_item.instruction,
        pred_x,
        pred_y,
        latency,
        is_in_bbox,
      )

      result = EvaluationResult(
        id=eval_item.id,
        recording_id=eval_item.recording_id,
        instruction=eval_item.instruction,
        image_path=local_path,
        gt_x1=gt_x1,
        gt_y1=gt_y1,
        gt_x2=gt_x2,
        gt_y2=gt_y2,
        pred_x=pred_x,
        pred_y=pred_y,
        is_in_bbox=is_in_bbox,
        latency_seconds=latency,
        raw_response=prediction.raw_response,
        visualization_path=visualization_path,
      )

      return result

    except Exception as e:
      print(f'API request failed for {eval_item.id}: {str(e)}')
      raise e


def get_api_client(api_key: Optional[str] = None, model: Optional[str] = None) -> AceAPIClient:
  api_key = api_key or os.environ.get('GENERALAGENTS_API_KEY', '')
  config = AceModelConfig(api_key=api_key, model=model or 'ace-control-small')
  return AceAPIClient(config=config)
