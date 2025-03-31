from abc import ABC, abstractmethod
from typing import Any, Dict

from clicks.evaluate.models import EvaluationResult, ParsedPrediction


class AbstractAPIClient(ABC):
  client_type: str
  model_name: str = ''

  @abstractmethod
  def predict(self, image_data_uri: str, prompt: str) -> Dict[str, Any] | None:
    pass

  @abstractmethod
  def parse_prediction(self, prediction: Dict[str, Any]) -> ParsedPrediction:
    pass

  @abstractmethod
  def process_single_item(
    self, item: Dict[str, Any], frames_dir: str, run_id: str
  ) -> EvaluationResult:
    pass
