import base64
from io import BytesIO

from PIL import Image


def encode_image_to_base64_uri(image_path: str) -> str:
  try:
    with Image.open(image_path) as img:
      if img.mode != 'RGB':
        img = img.convert('RGB')

      buffer = BytesIO()
      img.save(buffer, format='JPEG', quality=95)
      buffer.seek(0)

      img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

      return f'data:image/jpeg;base64,{img_base64}'
  except Exception as e:
    raise ValueError(f'Error encoding image to base64: {e}')
