import base64
import io
import json
import math
import os
import re
import time
from typing import Any, Dict, Optional, Tuple

from openai import OpenAI
from PIL import Image
from tenacity import retry, stop_after_attempt, wait_exponential

DEFAULT_MODEL = 'qwen2.5-vl-72b-instruct'
DASHSCOPE_API_ENDPOINT = 'https://dashscope.aliyuncs.com/compatible-mode/v1'

# Qwen performs better with this slightly modified prompt adapted from the official cookbook
QWEN_ACTION_SPACE = {
  'type': 'function',
  'function': {
    'name_for_human': 'computer_use',
    'name': 'computer_use',
    'description': "Use a mouse and keyboard to interact with a computer, and take screenshots.\\n* This is an interface to a desktop GUI. You do not have access to a terminal or applications menu. You must click on desktop icons to start applications.\\n* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions. E.g. if you click on Firefox and a window doesn't open, try wait and taking another screenshot.\\n* The screen's resolution is {RES_WIDTH}x{RES_HEIGHT}.\\n* Whenever you intend to move the cursor to click on an element like an icon, you should consult a screenshot to determine the coordinates of the element before moving the cursor.\\n* If you tried clicking on a program or link but it failed to load, even after waiting, try adjusting your cursor position so that the tip of the cursor visually falls on the element that you want to click.\\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.",
    'parameters': {
      'properties': {
        'action': {
          'description': 'The action to perform. The available actions are:\\n* `key`: Performs key down presses on the arguments passed in order, then performs key releases in reverse order.\\n* `type`: Type a string of text on the keyboard.\\n* `mouse_move`: Move the cursor to a specified (x, y) pixel coordinate on the screen.\\n* `left_click`: Click the left mouse button.\\n* `left_click_drag`: Click and drag the cursor from a start coordinate to an end coordinate on the screen.\\n* `right_click`: Click the right mouse button.\\n* `double_click`: Double-click the left mouse button.\\n* `scroll`: Performs a scroll of the mouse scroll wheel.\\n* `wait`: Wait for the change to happen.\\n* `terminate`: Terminate the current task when it is completed.',
          'enum': [
            'key',
            'type',
            'mouse_move',
            'left_click',
            'left_click_drag',
            'right_click',
            'double_click',
            'scroll',
            'wait',
            'terminate',
          ],
          'type': 'string',
        },
        'keys': {'description': 'Required only by `action=key`.', 'type': 'array'},
        'text': {'description': 'Required only by `action=type`.', 'type': 'string'},
        'start_coordinate': {
          'description': '(x, y): The starting x (pixels from the left edge) and y (pixels from the top edge) coordinates. Required only by `action=left_click_drag`.',
          'type': 'array',
        },
        'end_coordinate': {
          'description': '(x, y): The ending x (pixels from the left edge) and y (pixels from the top edge) coordinates. Required only by `action=left_click_drag`.',
          'type': 'array',
        },
        'coordinate': {
          'description': '(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required by `action=mouse_move, action=left_click, action=right_click, action=double_click`.',
          'type': 'array',
        },
        'pixels': {
          'description': 'The amount of scrolling to perform. Positive values scroll up, negative values scroll down. Required only by `action=scroll`.',
          'type': 'number',
        },
      },
      'required': ['action'],
      'type': 'object',
    },
    'args_format': 'Format the arguments as a JSON object.',
  },
}

BASE_PROMPT_TEMPLATE = (
  """# Tools

You MUST call a single function to assist with the user query. Do not call multiple functions, and do not answer the user's query without calling a function.

You are provided with function signatures within <tools></tools> XML tags:
<tools>"""
  + json.dumps(QWEN_ACTION_SPACE)
  + """</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>"""
)


def smart_resize(
  height: int, width: int, factor: int = 28, min_pixels: int = 3136, max_pixels: int = 12845056
):
  if height < factor or width < factor:
    raise ValueError(f'height:{height} or width:{width} must be larger than factor:{factor}')
  elif max(height, width) / min(height, width) > 200:
    raise ValueError(
      f'absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}'
    )
  h_bar = round(height / factor) * factor
  w_bar = round(width / factor) * factor
  if h_bar * w_bar > max_pixels:
    beta = math.sqrt((height * width) / max_pixels)
    h_bar = math.floor(height / beta / factor) * factor
    w_bar = math.floor(width / beta / factor) * factor
  elif h_bar * w_bar < min_pixels:
    beta = math.sqrt(min_pixels / (height * width))
    h_bar = math.ceil(height * beta / factor) * factor
    w_bar = math.ceil(width * beta / factor) * factor
  return h_bar, w_bar


class QwenVLClient:
  def __init__(
    self,
    api_key: Optional[str] = None,
    api_endpoint: str = DASHSCOPE_API_ENDPOINT,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4096,
    use_smart_resize: bool = True,
    resize_factor: int = 28,
    min_pixels: int = 3136,
    max_pixels: int = 12845056,
  ):
    self.api_key = api_key or os.environ.get('DASHSCOPE_API_KEY')
    if not self.api_key:
      raise ValueError(
        'API key must be provided either as an argument or through the DASHSCOPE_API_KEY environment variable'
      )

    self.api_endpoint = api_endpoint
    self.model = model
    self.max_tokens = max_tokens
    self.display_width = None
    self.display_height = None
    self.original_width = None
    self.original_height = None
    self.use_smart_resize = use_smart_resize
    self.resize_factor = resize_factor
    self.min_pixels = min_pixels
    self.max_pixels = max_pixels

  def _create_client(self) -> OpenAI:
    return OpenAI(
      api_key=self.api_key,
      base_url=self.api_endpoint,
    )

  def _extract_image_dimensions(self, base64_data: str) -> tuple[int, int]:
    try:
      image_data = base64.b64decode(base64_data)
      image = Image.open(io.BytesIO(image_data))
      width, height = image.size
      return width, height
    except Exception as e:
      print(f'Error extracting image dimensions: {e}')
      return 1024, 768

  def _resize_image(self, base64_data: str) -> Tuple[str, int, int]:
    try:
      image_data = base64.b64decode(base64_data)
      image = Image.open(io.BytesIO(image_data))
      self.original_width, self.original_height = image.size

      new_height, new_width = smart_resize(
        self.original_height,
        self.original_width,
        factor=self.resize_factor,
        min_pixels=self.min_pixels,
        max_pixels=self.max_pixels,
      )

      resized_image = image.resize((new_width, new_height), resample=2)

      buffer = io.BytesIO()
      resized_image.save(buffer, format=image.format or 'JPEG')
      buffer.seek(0)
      new_base64_data = base64.b64encode(buffer.getvalue()).decode('utf-8')

      return new_base64_data, new_width, new_height
    except Exception as e:
      print(f'Error resizing image: {e}')
      width = self.original_width or 1024
      height = self.original_height or 768
      return base64_data, width, height

  def _translate_coordinates(
    self, x: Optional[int], y: Optional[int]
  ) -> Tuple[Optional[int], Optional[int]]:
    if x is None or y is None:
      return x, y

    orig_width = self.original_width or 1024
    orig_height = self.original_height or 768
    disp_width = self.display_width or 1024
    disp_height = self.display_height or 768

    x_scale = orig_width / disp_width
    y_scale = orig_height / disp_height

    original_x = round(x * x_scale)
    original_y = round(y * y_scale)

    return original_x, original_y

  def _create_system_prompt(self) -> str:
    width = self.display_width or 1024
    height = self.display_height or 768

    prompt = BASE_PROMPT_TEMPLATE.replace('{RES_WIDTH}', str(width)).replace(
      '{RES_HEIGHT}', str(height)
    )

    return prompt

  @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=15))
  def predict(self, image_data_uri: str, prompt: str) -> Dict[str, Any] | None:
    if ',' in image_data_uri:
      base64_data = image_data_uri.split(',')[1]
      mime_type = image_data_uri.split(',')[0]
    else:
      base64_data = image_data_uri
      mime_type = 'data:image/jpeg;base64'

    self.original_width, self.original_height = self._extract_image_dimensions(base64_data)

    if self.use_smart_resize:
      base64_data, new_width, new_height = self._resize_image(base64_data)
      self.display_width, self.display_height = new_width, new_height
    else:
      self.display_width, self.display_height = self.original_width, self.original_height

    image_data_uri = f'{mime_type},{base64_data}'

    system_prompt = self._create_system_prompt()

    try:
      client = self._create_client()

      start_time = time.time()

      response = client.chat.completions.create(
        model=self.model,
        temperature=0.0,
        max_tokens=self.max_tokens,
        messages=[
          {
            'role': 'system',
            'content': [
              {
                'type': 'text',
                'text': 'You are a helpful assistant.',
              },
              {'type': 'text', 'text': system_prompt},
            ],
          },
          {
            'role': 'user',
            'content': [
              {'type': 'image_url', 'image_url': {'url': image_data_uri}},
              {'type': 'text', 'text': prompt},
            ],
          },
        ],
      )

      result = response.model_dump()

      raw_response = json.dumps(result)

      assistant_message = result.get('choices', [{}])[0].get('message', {})
      content = assistant_message.get('content', '')

      return {
        'raw_response': raw_response,
        'content': content,
        'latency': time.time() - start_time,
        'original_width': self.original_width,
        'original_height': self.original_height,
        'display_width': self.display_width,
        'display_height': self.display_height,
      }

    except Exception as e:
      print(f'API Error: {str(e)}')
      return None

  def parse_prediction(self, prediction: Dict[str, Any]) -> Dict[str, Any]:
    if not prediction or 'content' not in prediction:
      return {
        'pred_x': None,
        'pred_y': None,
        'raw_responses': prediction.get('raw_response', '{}'),
      }

    content = prediction['content']

    print(f'Content: {content}')

    tool_call_match = re.search(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', content, flags=re.DOTALL)
    if not tool_call_match:
      return {
        'pred_x': None,
        'pred_y': None,
        'raw_responses': prediction.get('raw_response', '{}'),
      }

    try:
      json_text = tool_call_match.group(1)
      data = json.loads(json_text)

      if 'arguments' not in data:
        return {
          'pred_x': None,
          'pred_y': None,
          'raw_responses': prediction.get('raw_response', '{}'),
        }

      args = data['arguments']
      action_str = args.get('action')

      if not action_str:
        return {
          'pred_x': None,
          'pred_y': None,
          'raw_responses': prediction.get('raw_response', '{}'),
        }

      pred_x = None
      pred_y = None
      if (
        'coordinate' in args
        and isinstance(args['coordinate'], list)
        and len(args['coordinate']) == 2
      ):
        pred_x = int(args['coordinate'][0])
        pred_y = int(args['coordinate'][1])

        if self.use_smart_resize and pred_x is not None and pred_y is not None:
          pred_x, pred_y = self._translate_coordinates(pred_x, pred_y)

      return {
        'pred_x': pred_x,
        'pred_y': pred_y,
        'raw_responses': prediction.get('raw_response', '{}'),
        'original_width': self.original_width,
        'original_height': self.original_height,
        'display_width': self.display_width,
        'display_height': self.display_height,
      }

    except Exception as e:
      print(f'Error parsing prediction: {e}')
      return {
        'pred_x': None,
        'pred_y': None,
        'raw_responses': prediction.get('raw_response', '{}'),
      }
