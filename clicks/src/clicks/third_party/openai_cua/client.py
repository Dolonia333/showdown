import base64
import io
import json
import os
import time
import warnings
from typing import Any, Dict, Optional, Tuple

from openai import OpenAI
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

warnings.filterwarnings('ignore')

OPENAI_API_ENDPOINT = 'https://api.openai.com/v1'
DEFAULT_MODEL = 'computer-use-preview'
DEFAULT_DISPLAY_WIDTH = 1024
DEFAULT_DISPLAY_HEIGHT = 768
DEFAULT_ENVIRONMENT = 'mac'


class OpenAICUAClient:
  def __init__(
    self,
    api_key: Optional[str] = None,
    api_endpoint: str = OPENAI_API_ENDPOINT,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4096,
    environment: str = DEFAULT_ENVIRONMENT,
  ):
    self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
    if not self.api_key:
      raise ValueError(
        'API key must be provided either as an argument or through the OPENAI_API_KEY environment variable'
      )

    self.api_endpoint = api_endpoint
    self.model = model
    self.max_tokens = max_tokens
    self.environment = environment
    self.client = OpenAI(api_key=self.api_key)

  def _extract_image_dimensions(self, image_data_uri: str) -> Tuple[int, int]:
    try:
      if ',' in image_data_uri:
        base64_data = image_data_uri.split(',')[1]
      else:
        base64_data = image_data_uri

      image_data = base64.b64decode(base64_data)
      image_pil = Image.open(io.BytesIO(image_data))
      width, height = image_pil.size
      return width, height
    except Exception as e:
      print(f'Error extracting image dimensions: {e}')
      return DEFAULT_DISPLAY_WIDTH, DEFAULT_DISPLAY_HEIGHT

  @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=15))
  def predict(self, image_data_uri: str, prompt: str) -> Dict[str, Any]:
    try:
      base64_data = image_data_uri.split(',')[1] if ',' in image_data_uri else image_data_uri
      width, height = self._extract_image_dimensions(base64_data)

      start_time = time.time()

      input_data = [
        {
          'role': 'user',
          'content': [
            {
              'type': 'input_text',
              'text': f'Perform the following task on the screen: {prompt} by clicking on a UI element. Do not ask for confirmation or clarifications. You have all the information you need to complete the task.',
            },
            {
              'type': 'input_image',
              'image_url': f'data:image/png;base64,{base64_data}',
            },
          ],
        }
      ]

      response = self.client.responses.create(
        model=self.model,
        tools=[
          {
            'type': 'computer_use_preview',  # type: ignore
            'display_width': width,
            'display_height': height,
            'environment': self.environment,
          }
        ],
        input=input_data,  # type: ignore
        truncation='auto',
      )

      print(response)
      print(f'Response: {response.output}')

      iteration = 0
      previous_response_id = response.id

      computer_call = None

      while iteration < 10:
        iteration += 1

        computer_calls = [item for item in response.output if item.type == 'computer_call']
        print(f'Computer calls: {computer_calls}')
        if len(computer_calls) > 0:
          computer_call = computer_calls[0]
          if computer_call.action.type == 'click':
            print(f'Click action found: {computer_call.action}')
            break
          else:
            print(f'Non-click action found: {computer_call.action}. Continuing...')

        if len(computer_calls) > 0:
          computer_call = computer_calls[0]
          call_id = computer_call.call_id

          pending_safety_checks = getattr(computer_call, 'pending_safety_checks', [])
          acknowledged_safety_checks = []

          if pending_safety_checks:
            print(f'Safety checks detected: {pending_safety_checks}')
            acknowledged_safety_checks = pending_safety_checks
        else:
          elapsed_time = time.time() - start_time
          return {
            'elapsed_time': elapsed_time,
            'response': response.model_dump(),
            'computer_call': None,
          }

        input_data = [
          {
            'call_id': call_id,
            'type': 'computer_call_output',
            'output': {
              'type': 'input_image',
              'image_url': f'data:image/png;base64,{base64_data}',
            },
          }
        ]

        if acknowledged_safety_checks:
          input_data[0]['acknowledged_safety_checks'] = acknowledged_safety_checks

        response = self.client.responses.create(
          model=self.model,
          previous_response_id=previous_response_id,
          tools=[
            {
              'type': 'computer_use_preview',  # type: ignore
              'display_width': width,
              'display_height': height,
              'environment': self.environment,
            }
          ],
          input=input_data,  # type: ignore
          truncation='auto',
        )

        print(f'[CUA Loop Iteration {iteration}] Response: {response.output}')

        previous_response_id = response.id

      elapsed_time = time.time() - start_time
      print(f'API call completed in {elapsed_time:.2f} seconds')

      return {
        'elapsed_time': elapsed_time,
        'response': response.model_dump(),
        'computer_call': computer_call.model_dump() if computer_call else None,
      }
    except Exception as e:
      print(f'Error in CUA prediction: {e}')
      raise

  def parse_prediction(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
    if not prediction or 'computer_call' not in prediction or not prediction['computer_call']:
      return {
        'pred_x': None,
        'pred_y': None,
        'raw_response': json.dumps(prediction, default=str) if prediction else None,
      }

    computer_call = prediction['computer_call']

    pred_x = None
    pred_y = None

    if computer_call and 'action' in computer_call and computer_call['action']['type'] == 'click':
      pred_x = computer_call['action']['x']
      pred_y = computer_call['action']['y']

    return {
      'pred_x': pred_x,
      'pred_y': pred_y,
      'raw_response': json.dumps(prediction, default=str),
    }
