import base64
import io
import json
import os
import time
from typing import Any, Dict, Optional

from openai import OpenAI
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

OPENAI_API_ENDPOINT = 'https://api.openai.com/v1'
DEFAULT_MODEL = 'gpt-4o'


class OpenAIClient:
  def __init__(
    self,
    api_key: Optional[str] = None,
    api_endpoint: str = OPENAI_API_ENDPOINT,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4096,
    reasoning_effort: str = 'medium',
  ):
    self.api_key = api_key or os.environ.get('OPENAI_API_KEY')
    if not self.api_key:
      raise ValueError(
        'API key must be provided either as an argument or through the OPENAI_API_KEY environment variable'
      )

    self.api_endpoint = api_endpoint
    self.model = model
    self.max_tokens = max_tokens
    self.reasoning_effort = reasoning_effort
    self.client = OpenAI(api_key=self.api_key)

  def _extract_image_dimensions(self, image_data_uri: str) -> tuple[int, int]:
    try:
      image_data = base64.b64decode(image_data_uri)
      image_pil = Image.open(io.BytesIO(image_data))
      width, height = image_pil.size
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

      messages = [
        {
          'role': 'system',
          'content': (
            'You are an AI assistant that can see and interact with a computer screen. '
            'You will be shown a screenshot and given an instruction. '
            'The instruction always requires you to interact with a UI element on the screen. You can assume that the UI element is always visible on the screen.'
            'Return the position of the UI element that can be interacted with to advance the action specified in the instruction. '
            f'The screen resolution is {width}x{height}. '
            'Your response must be in JSON format.'
          ),
        },
        {
          'role': 'user',
          'content': [
            {
              'type': 'image_url',
              'image_url': {'url': f'data:image/jpeg;base64,{base64_data}', 'detail': 'high'},
            },
            {'type': 'text', 'text': prompt},
          ],
        },
      ]

      start_time = time.time()

      params = {
        'model': self.model,
        'messages': messages,
        'max_completion_tokens': self.max_tokens,
        'response_format': {'type': 'json_object'},
        'seed': 42,
        'tools': [
          {
            'type': 'function',
            'function': {
              'name': 'click_on_element',
              'description': 'Click on the specified UI element',
              'parameters': {
                'type': 'object',
                'properties': {
                  'x': {'type': 'integer', 'description': 'The x-coordinate for the click action'},
                  'y': {'type': 'integer', 'description': 'The y-coordinate for the click action'},
                },
                'required': ['x', 'y'],
              },
            },
          }
        ],
        'tool_choice': 'auto',
      }

      if self.model == 'o1':
        params['reasoning_effort'] = self.reasoning_effort

      response = self.client.chat.completions.create(**params)

      print(response)

      end_time = time.time()
      latency = end_time - start_time

      result = {
        'model': self.model,
        'latency': latency,
        'response': response,
      }

      return result

    except Exception as e:
      print(f'Error making prediction: {e}')
      return {'error': str(e)}

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

    if hasattr(response, 'choices') and len(response.choices) > 0:
      choice = response.choices[0]

      if (
        hasattr(choice, 'message')
        and hasattr(choice.message, 'tool_calls')
        and choice.message.tool_calls
      ):
        tool_call = choice.message.tool_calls[0]
        if hasattr(tool_call, 'function') and hasattr(tool_call.function, 'arguments'):
          try:
            args = json.loads(tool_call.function.arguments)
            pred_x = args.get('x')
            pred_y = args.get('y')
          except Exception as e:
            print(f'Error parsing tool call arguments: {e}')

      elif (
        hasattr(choice, 'message') and hasattr(choice.message, 'content') and choice.message.content
      ):
        try:
          content = json.loads(choice.message.content)
          pred_x = content.get('x')
          pred_y = content.get('y')
        except Exception as e:
          print(f'Error parsing message content: {e}')

    return {
      'pred_x': pred_x,
      'pred_y': pred_y,
      'raw_response': json.dumps(prediction, default=str),
    }
