import base64
import io
import json
from typing import Any, Dict

import requests
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

OMNIPARSER_API_ENDPOINT = 'https://omniparser-api-omniparser-api.modal.run'
DEFAULT_MODEL = 'gpt-4o-2024-05-13'


class OmniParserClient:
  def __init__(
    self,
    api_endpoint: str = OMNIPARSER_API_ENDPOINT,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
  ):
    self.api_endpoint = api_endpoint
    self.model = model
    self.temperature = temperature

  def _convert_to_base64(self, image_path: str) -> str:
    """Convert image file to base64 string."""
    try:
      with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
      print(f'Error converting image to base64: {e}')
      raise

  def _extract_image_dimensions(self, base64_data: str) -> tuple[int, int]:
    try:
      image_data = base64.b64decode(base64_data)
      image = Image.open(io.BytesIO(image_data))
      width, height = image.size
      return width, height
    except Exception as e:
      print(f'Error extracting image dimensions: {e}')
      return 1024, 768

  @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=15))
  def predict(self, image_data_uri: str, prompt: str) -> Dict[str, Any]:
    try:
      if ',' in image_data_uri:
        base64_data = image_data_uri.split(',')[1]
      else:
        base64_data = image_data_uri

      width, height = self._extract_image_dimensions(base64_data)

      payload = {
        'image': base64_data,
        'instruction': prompt,
        'model_name': self.model,
        'temperature': self.temperature,
        'width': width,
        'height': height,
      }

      response = requests.post(
        f'{self.api_endpoint}/computer_use',
        json=payload,
        headers={'Content-Type': 'application/json'},
      )

      response.raise_for_status()
      result = response.json()
      result['model'] = self.model
      result['width'] = width
      result['height'] = height

      return result

    except Exception as e:
      print(f'Error making prediction: {e}')
      return {'error': str(e)}

  def parse_prediction(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
    if not prediction or 'error' in prediction:
      return {
        'pred_x': None,
        'pred_y': None,
        'raw_response': json.dumps(prediction) if prediction else None,
      }

    if 'point' in prediction and prediction['point'] and len(prediction['point']) == 2:
      pred_x, pred_y = prediction['point']
      pred_x = int(pred_x * prediction['width'])
      pred_y = int(pred_y * prediction['height'])
    else:
      pred_x, pred_y = None, None

    return {
      'pred_x': pred_x,
      'pred_y': pred_y,
      'raw_response': json.dumps(prediction, default=str),
    }
