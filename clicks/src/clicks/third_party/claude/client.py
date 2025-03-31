import base64
import io
import json
import os
import time
from typing import Any, Dict, List, Optional

import requests
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

CLAUDE_API_ENDPOINT = 'https://api.anthropic.com/v1/messages'
DEFAULT_MODEL = 'claude-3-7-sonnet-20250219'


class ClaudeComputerUseClient:
  def __init__(
    self,
    api_key: Optional[str] = None,
    api_endpoint: str = CLAUDE_API_ENDPOINT,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4096,
    thinking_budget: Optional[int] = 1024,
    tool_version: str = '20250124',
  ):
    self.api_key = api_key or os.environ.get('ANTHROPIC_API_KEY')
    if not self.api_key:
      raise ValueError(
        'API key must be provided either as an argument or through the ANTHROPIC_API_KEY environment variable'
      )

    self.api_endpoint = api_endpoint
    self.model = model
    self.max_tokens = max_tokens
    self.thinking_budget = thinking_budget
    self.tool_version = tool_version
    self.beta_flag = (
      'computer-use-2025-01-24' if '20250124' in tool_version else 'computer-use-2024-10-22'
    )
    self.display_width = None
    self.display_height = None

  def _extract_image_dimensions(self, base64_data: str) -> tuple[int, int]:
    try:
      image_data = base64.b64decode(base64_data)
      image = Image.open(io.BytesIO(image_data))
      width, height = image.size
      return width, height
    except Exception as e:
      print(f'Error extracting image dimensions: {e}')
      return 1024, 768

  def _create_tools(self) -> List[Dict[str, Any]]:
    width = self.display_width or 1024
    height = self.display_height or 768

    return [
      {
        'type': f'computer_{self.tool_version}',
        'name': 'computer',
        'display_width_px': width,
        'display_height_px': height,
        'display_number': 1,
      },
    ]

  def _create_thinking_config(self) -> Optional[Dict[str, Any]]:
    if self.thinking_budget is None:
      return None

    return {'type': 'enabled', 'budget_tokens': self.thinking_budget}

  @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=15))
  def predict(self, image_data_uri: str, prompt: str) -> Dict[str, Any] | None:
    headers = {
      'Content-Type': 'application/json',
      'x-api-key': self.api_key,
      'anthropic-version': '2023-06-01',
      'anthropic-beta': self.beta_flag,
    }

    if ',' in image_data_uri:
      base64_data = image_data_uri.split(',')[1]
    else:
      base64_data = image_data_uri

    self.display_width, self.display_height = self._extract_image_dimensions(base64_data)

    # Note: it is unclear if the Claude computer use agent expects the screenshot to be in the very first user message,
    # or is it elicited by the screenshot request tool call. During testing, the behavior was inconsistent.
    # The default behavior here is to not include the screenshot in the first user message, and then the same image
    # is sent in the tool result. You may wonder why we don't include the screenshot in the first user message.
    # During testing on the dev set, the evals are almost the same whether we include the screenshot in the first
    # user message or not.
    payload = {
      'model': self.model,
      'max_tokens': self.max_tokens,
      'messages': [
        {
          'role': 'user',
          'content': [
            {
              'type': 'text',
              'text': prompt,
            },
          ],
        }
      ],
      'tools': self._create_tools(),
    }

    thinking = self._create_thinking_config()
    if thinking:
      payload['thinking'] = thinking

    start_time = time.time()
    response = requests.post(
      self.api_endpoint,
      headers=headers,
      json=payload,
    )

    if response.status_code != 200:
      print(f'API Error: {response.status_code} - {response.text}')
      response.raise_for_status()

    result = response.json()

    raw_response = json.dumps(result)

    print(result)

    tool_use = None
    tool_use_id = None
    for content_item in result.get('content', []):
      if content_item.get('type') == 'tool_use' and content_item.get('name') == 'computer':
        tool_use = content_item.get('input', {})
        tool_use_id = content_item.get('id')
        break

    if not tool_use:
      print('No computer tool call found in the response')
      return None
    if tool_use.get('action') == 'screenshot':
      print('Claude requested a screenshot. Sending the same image again...')

      payload = {
        'model': self.model,
        'max_tokens': self.max_tokens,
        'messages': [
          {
            'role': 'user',
            'content': [
              {
                'type': 'text',
                'text': prompt,
              },
            ],
          },
          {
            'role': 'assistant',
            'content': result.get('content', []),
          },
          {
            'role': 'user',
            'content': [
              {
                'type': 'tool_result',
                'tool_use_id': tool_use_id,
                'content': [
                  {
                    'type': 'image',
                    'source': {
                      'type': 'base64',
                      'media_type': 'image/jpeg',
                      'data': base64_data,
                    },
                  }
                ],
              }
            ],
          },
        ],
        'tools': self._create_tools(),
      }

      if thinking:
        payload['thinking'] = thinking

      response = requests.post(
        self.api_endpoint,
        headers=headers,
        json=payload,
      )

      if response.status_code != 200:
        print(f'API Error: {response.status_code} - {response.text}')
        response.raise_for_status()

      result = response.json()

      raw_response_second = json.dumps(result)

      print('Second response after screenshot:')
      print(result)

      tool_use = None
      for content_item in result.get('content', []):
        if content_item.get('type') == 'tool_use' and content_item.get('name') == 'computer':
          tool_use = content_item.get('input', {})
          break

      if not tool_use:
        print('No computer tool call found in the second response')
        return None

      tool_use['raw_responses'] = [raw_response, raw_response_second]
    else:
      tool_use['raw_responses'] = [raw_response]

    tool_use['latency'] = time.time() - start_time
    tool_use['model'] = self.model

    if 'thinking' in result:
      tool_use['thinking'] = result['thinking']

    return tool_use

  def parse_prediction(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
    try:
      assert isinstance(prediction, dict)

      action_kind = prediction.get('action', {})

      if action_kind == 'left_click':
        coordinate = prediction.get('coordinate', {})
        pred_x, pred_y = coordinate
        pred_type = 'left_click'
        pred_text = None
      elif action_kind == 'type':
        pred_x, pred_y = None, None
        pred_type = 'type'
        pred_text = prediction.get('text')
      elif action_kind == 'screenshot':
        pred_x, pred_y = None, None
        pred_type = 'screenshot'
        pred_text = None
      else:
        pred_x, pred_y = None, None
        pred_type = action_kind
        pred_text = None

      result = {
        'pred_type': pred_type,
        'pred_x': pred_x,
        'pred_y': pred_y,
        'pred_text': pred_text,
      }

      if 'raw_responses' in prediction:
        result['raw_responses'] = prediction['raw_responses']

      if 'thinking' in prediction:
        result['thinking'] = prediction['thinking']

      return result

    except Exception as e:
      print(f'Error parsing prediction: {e}')
      return {
        'pred_type': None,
        'pred_x': None,
        'pred_y': None,
        'pred_text': None,
      }
