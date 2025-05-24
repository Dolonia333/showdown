import speech_recognition as sr
import pyttsx3
import json
import requests
import time
from datetime import datetime
import os

def get_models():
    openrouter_api_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_api_key and os.path.exists(".env"):
        with open(".env") as f:
            for line in f:
                if line.startswith("OPENROUTER_API_KEY="):
                    openrouter_api_key = line.strip().split("=", 1)[1]
    return [
        {
            "name": "Dolphin 3 (LM Studio)",
            "url": "http://localhost:1234/v1/chat",
            "type": "openai",
            "headers": {},
            "body_fn": lambda prompt: {"model": "dolphin-3", "messages": [{"role": "user", "content": prompt}]}
        },
        {
            "name": "DeepSeek-R1:14B (Ollama)",
            "url": "http://localhost:11434/api/chat",
            "type": "ollama",
            "headers": {},
            "body_fn": lambda prompt: {"model": "deepseek-coder:14b", "messages": [{"role": "user", "content": prompt}]}
        },
        {
            "name": "OpenRouter (Devstral Small)",
            "url": "https://openrouter.ai/api/v1/chat",
            "type": "openai",
            "headers": {"Authorization": f"Bearer {openrouter_api_key}"},
            "body_fn": lambda prompt: {"model": "mistralai/devstral-small", "messages": [{"role": "user", "content": prompt}]}
        }
    ]

def query_model(model, prompt):
    try:
        body = model["body_fn"](prompt)
        response = requests.post(model["url"], headers=model["headers"], json=body, timeout=30)
        response.raise_for_status()
        data = response.json()
        if model["type"] == "openai":
            return data["choices"][0]["message"]["content"]
        elif model["type"] == "ollama":
            if "response" in data:
                return data["response"]
            elif isinstance(data, list) and data and "response" in data[0]:
                return data[0]["response"]
            else:
                return str(data)
        else:
            return str(data)
    except Exception as e:
        return f"[ERROR] {e}"

def summarize_responses(responses):
    # Simple concatenation; you can improve this with an LLM call
    return "\n".join([f"{k}: {v}" for k, v in responses.items()])

def get_voice_input():
    # Initialize the recognizer
    recognizer = sr.Recognizer()
    
    # Configure the recognizer
    recognizer.energy_threshold = 4000
    recognizer.dynamic_energy_threshold = True
    
    try:
        # Use the default microphone as the audio source
        with sr.Microphone() as source:
            print("Listening...")
            # Listen for the first phrase and extract audio data from it
            audio = recognizer.listen(source)
            
            try:
                # Recognize speech using Google Speech Recognition
                text = recognizer.recognize_google(audio, language="en-US")
                print(f"Google Speech Recognition: {text}")
                return text
            except sr.RequestError:
                # Try using offline recognition with PocketSphinx if Google fails
                try:
                    text = recognizer.recognize_sphinx(audio)
                    print(f"Sphinx Recognition: {text}")
                    return text
                except Exception as e:
                    print(f"Sphinx error: {str(e)}")
                    return None
            except sr.UnknownValueError:
                print("Could not understand audio")
                return None
            
    except Exception as e:
        print(f"Error capturing audio: {str(e)}")
        return None

def main():
    tts = pyttsx3.init()
    models = get_models()
    print("Say a prompt. Say 'exit', 'quit', or 'stop' to end.")
    while True:
        prompt = get_voice_input()
        if prompt is None:
            continue
        if prompt.lower() in ["exit", "quit", "stop"]:
            print("Exiting.")
            tts.say("Goodbye.")
            tts.runAndWait()
            break
        responses = {}
        for model in models:
            print(f"  Querying {model['name']}...")
            result = query_model(model, prompt)
            print(f"    {model['name']} response:\n{result}\n")
            responses[model["name"]] = result
            time.sleep(1)
        summary = summarize_responses(responses)
        print("Summary:")
        print(summary)
        tts.say(summary)
        tts.runAndWait()

if __name__ == "__main__":
    main()
