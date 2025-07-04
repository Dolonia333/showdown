import os
import time
from typing import Any, Dict, Optional

from clicks.api_client_base import AbstractAPIClient
from clicks.base_models import EvaluationResult, ParsedPrediction
from clicks.evaluate.models import (
  AceModelConfig,
  AcePrediction,
  EvaluationItem,
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

    if action.kind == 'left_click' and action.coordinate and action.coordinate.x is not None and action.coordinate.y is not None:
      return ParsedPrediction(
        x=action.coordinate.x,
        y=action.coordinate.y,
        raw_response=str(action)
      )
    else:
      raise ValueError("No valid click coordinates found in action")

  def parse_prediction(self, prediction: Dict[str, Any]) -> ParsedPrediction:
    ace_prediction = AcePrediction.model_validate(prediction)
    if ace_prediction.action.kind == 'left_click' and ace_prediction.action.coordinate:
      x = ace_prediction.action.coordinate.x
      y = ace_prediction.action.coordinate.y
      if x is not None and y is not None:
        return ParsedPrediction(
          x=x,
          y=y,
          raw_response=ace_prediction.raw_response or ""
        )
    raise ValueError("No valid coordinates found in prediction")
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

      if prediction is None or not isinstance(prediction, ParsedPrediction):
        raise ValueError("Invalid prediction returned")

      gt_x1 = int(eval_item.x1) if eval_item.x1 is not None else 0
      gt_y1 = int(eval_item.y1) if eval_item.y1 is not None else 0
      gt_x2 = int(eval_item.x2) if eval_item.x2 is not None else image.width
      gt_y2 = int(eval_item.y2) if eval_item.y2 is not None else image.height

      is_in_bbox = check_prediction_in_bbox(prediction.x, prediction.y, gt_x1, gt_y1, gt_x2, gt_y2)

      visualization_path = visualize_prediction(
        image_path=local_path,
        pred_x=prediction.x,
        pred_y=prediction.y,
        item_id=str(eval_item.id),
        recording_id=str(eval_item.id),  # using item_id as recording_id
        instruction=eval_item.instruction,
        model_name=str(self.config.model),
        run_id=str(run_id) if run_id else None,
        gt_x1=int(gt_x1) if gt_x1 is not None else None,
        gt_y1=int(gt_y1) if gt_y1 is not None else None,
        gt_x2=int(gt_x2) if gt_x2 is not None else None,
        gt_y2=int(gt_y2) if gt_y2 is not None else None,
        is_in_bbox=is_in_bbox
      )
      # If you have an integer run_id, convert it to string before passing to visualize_prediction
      # Example:
      # run_id_int = 123
      # visualization_path = visualize_prediction(..., run_id=str(run_id_int), ...)
      print_colored_result(
        str(eval_item.id),
        eval_item.instruction,
        prediction.x,
        prediction.y,
        latency,
        is_in_bbox,
      )

      result = EvaluationResult(
        id=str(eval_item.id),
        instruction=str(eval_item.instruction),
        image_path=str(local_path),
        gt_x1=gt_x1,
        gt_y1=gt_y1,
        gt_x2=gt_x2,
        gt_y2=gt_y2,
        width=image.width,
        height=image.height,
        prediction=prediction,
        is_in_bbox=is_in_bbox,
        latency_seconds=latency,
      )

      return result

    except Exception as e:
      print(f'API request failed for {eval_item.id}: {str(e)}')
      raise e


def get_api_client(api_key: Optional[str] = None, model: Optional[str] = None) -> AceAPIClient:
  api_key = api_key or os.environ.get('GENERALAGENTS_API_KEY', '')
  config = AceModelConfig(api_key=api_key, model=model or 'ace-control-small')
  return AceAPIClient(config=config)