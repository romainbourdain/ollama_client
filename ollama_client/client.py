import requests
import json

from ollama_client._types import Option, Message, Image, Format
from typing import List, Optional


class OllamaClient:
    def __init__(self, base_url):
        self.base_url = base_url

    def generate(
        self,
        model: str = "",
        prompt: str = "",
        images: Optional[Image] = None,
        format: Optional[Format] = None,
        options: Optional[Option] = None,
        system: str = "",
        template: str = "",
        context: List[int] = None,
        stream: bool = False,
        raw: bool = False,
        keep_alive: str = "5m",
    ):
        """
        Generate a response for a given prompt with a provided model. This is a streaming endpoint, so there will be a series of responses. The final response object will include statistics and additional data from the request.
        """

        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "images": images,
            "format": format,
            "options": options,
            "system": system,
            "template": template,
            "context": context,
            "stream": stream,
            "raw": raw,
            "keep_alive": keep_alive,
        }
        response = requests.post(url, json=payload, stream=stream)
        if stream:
            return self._handle_stream(response)

        return response.json()

    def chat(
        self,
        model: str,
        messages: List[Message],
        format: Optional[Format] = Format.JSON,
        options: Optional[Option] = None,
        stream: bool = False,
        keep_alive: str = "5m",
    ):
        """
        Generate the next message in a chat with a provided model. This is a streaming endpoint, so there will be a series of responses.
        """

        url = f"{self.base_url}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "format": format,
            "options": options,
            "stream": stream,
            "keep_alive": keep_alive,
        }
        response = requests.post(url, json=payload, stream=stream)
        if stream:
            return self._handle_stream(response)

        return response.json()

    def create(
        self,
        name: str,
        modelfile: Optional[str] = None,
        stream: bool = False,
        path: Optional[str] = None,
    ):
        """
        Create a model from a Modelfile. It is recommended to set modelfile to the content of the Modelfile rather than just set path. This is a requirement for remote create. Remote model creation must also create any file blobs, fields such as FROM and ADAPTER, explicitly with the server using Create a Blob and the value to the path indicated in the response.
        """

        url = f"{self.base_url}/api/create"
        payload = {"name": name, "modelfile": modelfile, "stream": stream, "path": path}
        response = requests.post(url, json=payload, stream=stream)
        if stream:
            return self._handle_stream(response)

        return response.json()

    def tags(self):
        """
        List models that are available locally.
        """

        url = f"{self.base_url}/api/tags"
        response = requests.get(url)
        return response.json()

    def show(
        self,
        name: str,
    ):
        """
        Show information about a model including details, modelfile, template, parameters, license, and system prompt.
        """

        url = f"{self.base_url}/api/show"
        payload = {"name": name}

        response = requests.post(url, json=payload)
        return response.json()

    def copy(
        self,
        source: str,
        destination: str,
    ):
        """
        Copy a model. Creates a model with another name from an existing model.
        """

        url = f"{self.base_url}/api/copy"
        payload = {"source": source, "destination": destination}

        response = requests.post(url, json=payload)
        return response.json()

    def delete(
        self,
        name: str,
    ):
        """
        Delete a model and its data.
        """

        url = f"{self.base_url}/api/delete"
        payload = {"name": name}

        response = requests.delete(url, json=payload)
        return response.json()

    def pull(
        self,
        name: str,
        insecure: Optional[bool] = None,
        stream: bool = False,
    ):
        """
        Download a model from the ollama library. Cancelled pulls are resumed from where they left off, and multiple calls will share the same download progress.
        """

        url = f"{self.base_url}/api/pull"
        payload = {"name": name, "insecure": insecure, "stream": stream}
        response = requests.post(url, json=payload, stream=stream)
        if stream:
            return self._handle_stream(response)

        return response.json()

    def push(
        self,
        name: str,
        insecure: Optional[bool] = None,
        stream: bool = False,
    ):
        """
        Upload a model to a model library. Requires registering for ollama.ai and adding a public key first.
        """

        url = f"{self.base_url}/api/push"
        payload = {"name": name, "insecure": insecure, "stream": stream}

        response = requests.post(url, json=payload, stream=stream)

        if stream:
            return self._handle_stream(response)

        return response.json()

    def embeddings(
        self,
        model: str,
        prompt: str,
        options: Optional[Option] = None,
        keep_alive: str = "5m",
    ):
        """
        Generate embeddings from a model
        """

        url = f"{self.base_url}/api/embeddings"
        payload = {
            "model": model,
            "prompt": prompt,
            "options": options,
            "keep_alive": keep_alive,
        }

        response = requests.post(url, json=payload)
        return response.json()

    def _handle_stream(self, response):
        """Handle streaming responses."""
        for line in response.iter_lines():
            if line:
                yield json.loads(line)
