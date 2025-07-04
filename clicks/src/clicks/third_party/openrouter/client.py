import json
import os
import time
from typing import Any, Dict, Optional

import requests
from ...base_models import ParsedPrediction, EvaluationResult
from ...api_client_base import AbstractAPIClient
from ..common import encode_image_to_base64_uri

class OpenRouterClient(AbstractAPIClient):
    client_type = "openrouter"
    
    def __init__(self, api_key: Optional[str] = None, model: str = "anthropic/claude-3-opus", max_tokens: int = 4096):
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OpenRouter API key must be provided either through constructor or OPENROUTER_API_KEY environment variable")
        self.model = model
        self.max_tokens = max_tokens
        self.endpoint = "https://openrouter.ai/api/v1/chat/completions"
        
    def predict(self, image_data_uri: str, prompt: str) -> Dict[str, Any] | None:
        system_prompt = """You are a computer control agent. You must analyze the screenshot and 
        determine where to click to fulfill the given instruction. Return ONLY a JSON object like {"x": 100, "y": 200} 
        where x and y are integer coordinates for where to click. Do not include any other text or explanation."""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Instruction: {prompt}\n\nScreenshot: {image_data_uri}"}
        ]
        
        try:
            response = requests.post(
                self.endpoint,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "HTTP-Referer": "https://github.com/generalagents/showdown",
                    "X-Title": "Showdown Computer Control Evaluation"
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                    "temperature": 0.1,
                    "stream": False
                }
            )
            response.raise_for_status()
            data = response.json()
            return {"response": data.get("choices", [{}])[0].get("message", {}).get("content", "")}
        except Exception as e:
            print(f"Error querying OpenRouter: {str(e)}")
            return None
            
    def parse_prediction(self, prediction: Dict[str, Any]) -> ParsedPrediction:
        try:
            response = prediction.get('response', '')
            # Try to find JSON in the response
            start = response.find('{')
            end = response.rfind('}')
            if start >= 0 and end > start:
                json_str = response[start:end+1]
                try:
                    coords = json.loads(json_str)
                    if isinstance(coords, dict) and 'x' in coords and 'y' in coords:
                        x = int(coords['x'])
                        y = int(coords['y'])
                        return ParsedPrediction(
                            x=x,
                            y=y,
                            raw_response=response
                        )
                except (json.JSONDecodeError, ValueError, TypeError):
                    pass
            # Try to find numbers in the response if JSON parsing fails
            import re
            numbers = re.findall(r'x[=:]?\s*(\d+)[,\s]+y[=:]?\s*(\d+)', response.lower())
            if numbers:
                x, y = map(int, numbers[0])
                return ParsedPrediction(
                    x=x,
                    y=y,
                    raw_response=response
                )
            raise ValueError("No valid coordinates found in response")
        except Exception as e:
            raise ValueError(f"Failed to parse prediction: {str(e)}")
            
    def process_single_item(
        self, item: Dict[str, Any], frames_dir: str, run_id: str
    ) -> EvaluationResult:
        start_time = time.time()
        
        result = EvaluationResult(
            id=item['id'],
            instruction=item['instruction'],
            image_path=item['image'],
            gt_x1=int(item['x1']),
            gt_y1=int(item['y1']),
            gt_x2=int(item['x2']),
            gt_y2=int(item['y2']),
            width=int(item['width']),
            height=int(item['height'])
        )
        
        try:
            image_path = os.path.join(frames_dir, item['image'])
            if not os.path.exists(image_path):
                raise ValueError(f"Image not found: {image_path}")
                
            image_data_uri = encode_image_to_base64_uri(image_path)
            
            prediction = self.predict(image_data_uri, item['instruction'])
            if not prediction:
                raise ValueError("No prediction returned from model")
                
            parsed = self.parse_prediction(prediction)
            result.prediction = parsed
            
            result.is_in_bbox = (
                result.gt_x1 <= parsed.x <= result.gt_x2 and
                result.gt_y1 <= parsed.y <= result.gt_y2
            )
            
        except Exception as e:
            result.error = str(e)
            
        result.latency_seconds = time.time() - start_time
        
        return result

def get_api_client(api_key: str, model: str = "anthropic/claude-3-opus", max_tokens: int = 4096) -> AbstractAPIClient:
    return OpenRouterClient(api_key=api_key, model=model, max_tokens=max_tokens)
