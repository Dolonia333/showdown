import base64
import io
import json
import re
import time
from typing import Any, Dict

import requests
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential


class UITarsClient:
  def __init__(
    self,
    api_url: str,
    api_key: str = 'super-secret-key',
    max_tokens: int = 128,
    temperature: float = 0.0,
    frequency_penalty: float = 1.0,
    model_name: str = 'bytedance-research/UI-TARS-72B-SFT',
  ):
    self.api_url = api_url.rstrip('/')
    self.api_key = api_key
    self.max_tokens = max_tokens
    self.temperature = temperature
    self.frequency_penalty = frequency_penalty
    self.model_name = model_name

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
    width, height = self._extract_image_dimensions(base64_image)

    # Note: UI-TARS is not a generalist VLM: prompting it with plain English will cause the model to severely collapse.
    # Hence, it is unclear how to change the given computer use prompt, so we just use the default one provided in the UI-TARS repo.
    # drag, right_single, hotkey, type, scroll, wait, finished, call_user are here but we won't use them and will treat it as a failure.
    prompt_template = f"""You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task. 

## Output Format
```\nThought: ...
Action: ...\n```

## Action Space

click(start_box='<|box_start|>(x1,y1)<|box_end|>')
left_double(start_box='<|box_start|>(x1,y1)<|box_end|>')
right_single(start_box='<|box_start|>(x1,y1)<|box_end|>')
drag(start_box='<|box_start|>(x1,y1)<|box_end|>', end_box='<|box_start|>(x3,y3)<|box_end|>')
hotkey(key='')
type(content='') #If you want to submit your input, use \"\
\" at the end of `content`.
scroll(start_box='<|box_start|>(x1,y1)<|box_end|>', direction='down or up or right or left')
wait() #Sleep for 5s and take a screenshot to check for any changes.
finished()
call_user() # Submit the task and call the user when the task is unsolvable, or when you need the user's help.


## Note
- Use Chinese in `Thought` part.
- Summarize your next action (with its target element) in one sentence in `Thought` part.

## User Instruction
{prompt}"""

    # Prepare the multimodal message
    multimodal_message = {
      'role': 'user',
      'content': [
        {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{base64_image}'}},
        {'type': 'text', 'text': prompt_template},
      ],
    }

    request_data = {
      'messages': [multimodal_message],
      'model': self.model_name,
      'max_tokens': self.max_tokens,
      'temperature': self.temperature,
      'frequency_penalty': self.frequency_penalty,
    }

    headers = {
      'Content-Type': 'application/json',
      'Authorization': f'Bearer {self.api_key}',
    }

    try:
      start_time = time.time()

      response = requests.post(
        f'{self.api_url}/v1/chat/completions',
        json=request_data,
        headers=headers,
        timeout=7200,
      )

      end_time = time.time()
      latency = end_time - start_time

      if response.status_code == 200:
        result = response.json()

        content = result.get('choices', [{}])[0].get('message', {}).get('content', '')

        print(content)
        return {
          'raw_response': json.dumps(result),
          'content': content,
          'latency_seconds': latency,
          'width': width,
          'height': height,
        }
      else:
        print(response.text)
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
      print(f'API Error: {str(e)}')
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
    width = prediction.get('width', 1920)  # Default to 1920 if width not provided
    height = prediction.get('height', 1080)  # Default to 1080 if height not provided

    action_match = re.search(r'Action:\s*(.*?)(?:\n|$)', content, re.DOTALL)
    action_text = action_match.group(1).strip() if action_match else content

    click_match = re.search(
      r"click\(start_box='<\|box_start\|>\((\d+),\s*(\d+)\)<\|box_end\|>'\)", action_text
    )
    if click_match:
      rel_x = int(click_match.group(1))
      rel_y = int(click_match.group(2))
      pred_x = round(width * rel_x / 1000)
      pred_y = round(height * rel_y / 1000)
      return {
        'pred_x': pred_x,
        'pred_y': pred_y,
        'content': content,
        'raw_responses': prediction.get('raw_response', '{}'),
      }

    # Process double click action
    double_click_match = re.search(
      r"left_double\(start_box='<\|box_start\|>\((\d+),\s*(\d+)\)<\|box_end\|>'\)", action_text
    )
    if double_click_match:
      rel_x = int(double_click_match.group(1))
      rel_y = int(double_click_match.group(2))
      pred_x = round(width * rel_x / 1000)
      pred_y = round(height * rel_y / 1000)
      return {
        'pred_x': pred_x,
        'pred_y': pred_y,
        'content': content,
        'raw_responses': prediction.get('raw_response', '{}'),
      }

    # Process generic coordinate pattern
    coord_match = re.search(r'\((\d+),\s*(\d+)\)', content)
    if coord_match:
      rel_x = int(coord_match.group(1))
      rel_y = int(coord_match.group(2))
      pred_x = round(width * rel_x / 1000)
      pred_y = round(height * rel_y / 1000)
      return {
        'pred_x': pred_x,
        'pred_y': pred_y,
        'content': content,
        'raw_responses': prediction.get('raw_response', '{}'),
      }

    # Process x=X, y=Y format
    x_match = re.search(r'x\s*=\s*(\d+)', content, re.IGNORECASE)
    y_match = re.search(r'y\s*=\s*(\d+)', content, re.IGNORECASE)
    if x_match and y_match:
      rel_x = int(x_match.group(1))
      rel_y = int(y_match.group(1))
      pred_x = round(width * rel_x / 1000)
      pred_y = round(height * rel_y / 1000)
      return {
        'pred_x': pred_x,
        'pred_y': pred_y,
        'content': content,
        'raw_responses': prediction.get('raw_response', '{}'),
      }

    # Process box format
    box_match = re.search(r'<\|box_start\|>\((\d+),\s*(\d+)\)<\|box_end\|>', content)
    if box_match:
      rel_x = int(box_match.group(1))
      rel_y = int(box_match.group(2))
      pred_x = round(width * rel_x / 1000)
      pred_y = round(height * rel_y / 1000)
      return {
        'pred_x': pred_x,
        'pred_y': pred_y,
        'content': content,
        'raw_responses': prediction.get('raw_response', '{}'),
      }

    return {
      'pred_x': None,
      'pred_y': None,
      'content': content,
      'raw_responses': prediction.get('raw_response', '{}'),
      'error': 'No coordinates found in response',
    }
