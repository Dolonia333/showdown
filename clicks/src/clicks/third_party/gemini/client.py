import base64
import io
import json
import os
import time
from typing import Any, Dict, Optional

from google import genai
from google.genai.types import GenerateContentConfig, SafetySetting
from PIL import Image
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

DEFAULT_MODEL = 'gemini-2.0-flash'


class Point(BaseModel):
  point: list[int]
  label: str


class GeminiClient:
  def __init__(
    self,
    api_key: Optional[str] = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4096,
    temperature: float = 0.0,
  ):
    self.api_key = api_key or os.environ.get('GEMINI_API_KEY')
    if not self.api_key:
      raise ValueError(
        'API key must be provided either as an argument or through the GEMINI_API_KEY environment variable'
      )

    self.model = model
    self.max_tokens = max_tokens
    self.temperature = temperature

  def _extract_image_dimensions(self, base64_data: str) -> tuple[int, int]:
    try:
      image_data = base64.b64decode(base64_data)
      image = Image.open(io.BytesIO(image_data))
      width, height = image.size
      return width, height
    except Exception as e:
      print(f'Error extracting image dimensions: {e}')
      return 1024, 768

  def _encode_image(self, image_path: str) -> str:
    try:
      with open(image_path, 'rb') as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
      print(f'Error encoding image: {e}')
      return ''

  @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=15))
  def predict(self, image_data_uri: str, prompt: str, **kwargs) -> Dict[str, Any]:
    try:
      if ',' in image_data_uri:
        base64_data = image_data_uri.split(',')[1]
        mime_type = image_data_uri.split(',')[0]
      else:
        base64_data = image_data_uri
        mime_type = 'data:image/jpeg;base64'

      image_data = base64.b64decode(base64_data)
      image_pil = Image.open(io.BytesIO(image_data))
      width, height = image_pil.size

      start_time = time.time()

      client = genai.Client(api_key=self.api_key)

      config_localize = GenerateContentConfig(
        temperature=0.0,
      )

      prompt_localize = f"""You are an AI assistant that helps users with their tasks.
You are given an image and a task. Your job is to return a description of the UI element that should be clicked on to advance or complete the task.
There will always be an UI element that can be clicked to advance or complete the task. Do not question this.
The description should not be a question, or an action. For example, "the "-" button in the Target Membership section." is good, but "click the "-" button in the Target Membership section." is bad.
You MUST remember that you are not describing the actions you will take, but the UI element that should be clicked. You should just describe the UI element.
The task is: `{prompt}`.
Return nothing else but the singular description."""

      result_localize = client.models.generate_content(
        model=self.model,
        contents=[
          image_pil,
          prompt_localize,
        ],
        config=config_localize,
      )

      print(f'Localize: {result_localize.text}')

      prompt = (
        """Point to the UI element matching the description: `"""
        + (result_localize.text or prompt)
        + """`, with no more than 1 item. The answer should follow the json format: [{'point': <point>, "label": <label1>}, ...]. The points are in [y, x] format normalized to 0-1000."""
      )

      config = GenerateContentConfig(
        temperature=0.5,
        safety_settings=[
          SafetySetting(
            category='HARM_CATEGORY_DANGEROUS_CONTENT',  # type: ignore
            threshold='BLOCK_ONLY_HIGH',  # type: ignore
          ),
        ],
        response_mime_type='application/json',
        response_schema=list[Point],
      )

      response = client.models.generate_content(
        model=self.model,
        contents=[
          image_pil,
          prompt,
        ],
        config=config,
      )

      latency = time.time() - start_time

      return {
        'response': response,
        'latency': latency,
        'width': width,
        'height': height,
      }

    except Exception as e:
      print(f'API Error: {str(e)}')
      return {'response': None, 'latency': 0, 'width': 0, 'height': 0, 'error': str(e)}

  def parse_prediction(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
    if not prediction or 'response' not in prediction:
      return {
        'pred_x': None,
        'pred_y': None,
        'raw_response': json.dumps(prediction) if prediction else None,
      }

    response = prediction['response']

    pred_x = None
    pred_y = None

    width = prediction['width']
    height = prediction['height']

    try:
      content = response.parsed

      point = content[0]

      y, x = point.point
      y = int(y / 1000 * height)
      x = int(x / 1000 * width)

      pred_x = x
      pred_y = y

      pred_x = int(pred_x)
      pred_y = int(pred_y)
    except Exception as e:
      print(f'Error parsing prediction: {e}')

    return {
      'pred_x': pred_x,
      'pred_y': pred_y,
      'raw_response': json.dumps(prediction, default=str),
    }
