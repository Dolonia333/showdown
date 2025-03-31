import base64
import io
import json
import re
import time
from typing import Any, Dict, Optional

import requests
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential


class MolmoClient:
  def __init__(
    self,
    api_url: str,
    api_key: Optional[str] = None,
    max_tokens: int = 4096,
    temperature: float = 0.0,
    top_p: float = 0.9,
    top_k: int = 50,
  ):
    self.api_url = api_url.rstrip('/')
    self.api_key = api_key
    self.max_tokens = max_tokens
    self.temperature = temperature
    self.top_p = top_p
    self.top_k = top_k

  def _encode_image(self, image_data_uri: str) -> str:
    if ',' in image_data_uri:
      base64_data = image_data_uri.split(',')[1]
    else:
      base64_data = image_data_uri
    return base64_data

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
    base64_image = self._encode_image(image_data_uri)

    prompt_template = (
      f"""Point at the UI element to click to achieve the following action: {prompt}"""
    )

    width, height = self._extract_image_dimensions(base64_image)
    request_data = {
      'images': [base64_image],
      'text': prompt_template,
      'max_new_tokens': self.max_tokens,
      'temperature': self.temperature,
      'top_p': self.top_p,
      'top_k': self.top_k,
    }

    headers = {'Content-Type': 'application/json'}
    if self.api_key:
      headers['Authorization'] = f'Bearer {self.api_key}'

    try:
      start_time = time.time()

      response = requests.post(
        f'{self.api_url}/generate',
        json=request_data,
        headers=headers,
        timeout=3600,
      )

      end_time = time.time()
      latency = end_time - start_time

      if response.status_code == 200:
        result = response.json()
        return {
          'raw_response': json.dumps(result),
          'content': result.get('generated_text', ''),
          'latency_seconds': latency,
          'width': width,
          'height': height,
        }
      else:
        error_text = response.text
        try:
          error_json = response.json()
          error_text = json.dumps(error_json)
        except:
          pass

        return {
          'error': f'HTTP Error {response.status_code}',
          'error_details': error_text,
          'latency_seconds': latency,
        }

    except Exception as e:
      return {
        'error': f'API Error: {str(e)}',
        'latency_seconds': 0,
      }

  def parse_prediction(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
    if 'error' in prediction:
      return {
        'pred_x': None,
        'pred_y': None,
        'error': prediction.get('error'),
        'error_details': prediction.get('error_details', ''),
        'raw_responses': prediction.get('raw_response', '{}'),
      }

    content = prediction.get('content', '')
    point_match = re.search(r'<point x="([\d.]+)" y="([\d.]+)"', content)

    pred_x = None
    pred_y = None

    if point_match:
      rel_x = float(point_match.group(1))
      rel_y = float(point_match.group(2))

      width = prediction.get('width', 0)
      height = prediction.get('height', 0)

      if width > 0 and height > 0:
        pred_x = int(rel_x * width / 100)
        pred_y = int(rel_y * height / 100)

    return {
      'pred_x': pred_x,
      'pred_y': pred_y,
      'content': content,
      'raw_responses': prediction.get('raw_response', '{}'),
    }
