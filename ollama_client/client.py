import requests
from typing import Optional


class OllamaClient:
    def __init__(self, host, port, model):
        self.host = host
        self.port = port
        self.model = model
        self.model_loaded = False
        self.pull()

    def generate(
        self,
        prompt: str,
        stream: Optional[bool] = False,
    ):
        url = f"http://{self.host}:{self.port}/api/generate"
        data = {
            "model": self.model,
            "prompt": prompt,
            "stream": stream,
        }
        response = requests.post(url, json=data, stream=stream)
        for line in response.iter_lines():
            if line:
                yield line

    def pull(self, stream: Optional[bool] = False):
        url = f"http://{self.host}:{self.port}/api/pull"
        data = {
            "name": self.model,
            "stream": stream,
        }
        self.model_loaded = True
        response = requests.post(url, json=data)
        return response.json()
