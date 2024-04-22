import requests
import json
import logging

from ollama_client._types import Option, Message, Image, Format
from typing import List, Optional

logging.basicConfig(level=logging.INFO)


class OllamaAPIError(Exception):
    def __init__(self, message, status_code=None):
        super().__init__(message)
        self.status_code = status_code


class OllamaClient:
    def __init__(self, base_url):
        if not base_url:
            raise ValueError("base_url is required")
        self.base_url = base_url
        self.headers = {"Content-Type": "application/json"}

    def generate(
        self,
        model: str,
        prompt: str,
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
        self._validate_required_parameters(model=model, prompt=prompt)
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
        return self._request("post", "api/generate", payload, stream)

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
        self._validate_required_parameters(model=model, messages=messages)
        payload = {
            "model": model,
            "messages": messages,
            "format": format,
            "options": options,
            "stream": stream,
            "keep_alive": keep_alive,
        }
        return self._request("post", "api/chat", payload, stream)

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
        self._validate_required_parameters(name=name)
        payload = {"name": name, "modelfile": modelfile, "stream": stream, "path": path}
        return self._request("post", "api/create", payload, stream)

    def tags(self):
        """
        List models that are available locally.
        """
        return self._request("get", "api/tags")

    def show(
        self,
        name: str,
    ):
        """
        Show information about a model including details, modelfile, template, parameters, license, and system prompt.
        """
        self._validate_required_parameters(name=name)
        payload = {"name": name}
        return self._request("post", "api/show", payload)

    def copy(
        self,
        source: str,
        destination: str,
    ):
        """
        Copy a model. Creates a model with another name from an existing model.
        """
        self._validate_required_parameters(source=source, destination=destination)
        payload = {"source": source, "destination": destination}
        return self._request("post", "api/copy", payload)

    def delete(
        self,
        name: str,
    ):
        """
        Delete a model and its data.
        """
        self._validate_required_parameters(name=name)
        payload = {"name": name}
        return self._request("delete", "api/delete", payload)

    def pull(
        self,
        name: str,
        insecure: Optional[bool] = None,
        stream: bool = False,
    ):
        """
        Download a model from the ollama library. Cancelled pulls are resumed from where they left off, and multiple calls will share the same download progress.
        """
        self._validate_required_parameters(name=name)
        payload = {"name": name, "insecure": insecure, "stream": stream}
        return self._request("post", "api/pull", payload, stream)

    def push(
        self,
        name: str,
        insecure: Optional[bool] = None,
        stream: bool = False,
    ):
        """
        Upload a model to a model library. Requires registering for ollama.ai and adding a public key first.
        """
        self._validate_required_parameters(name=name)
        payload = {"name": name, "insecure": insecure, "stream": stream}
        return self._request("post", "api/push", payload, stream)

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
        self._validate_required_parameters(model=model, prompt=prompt)
        payload = {
            "model": model,
            "prompt": prompt,
            "options": options,
            "keep_alive": keep_alive,
        }
        return self._request("post", "api/embeddings", payload)

    def _validate_required_parameters(self, **params):
        """
        Validate that required parameters are present.
        """
        for param_name, value in params.items():
            if value is None or (isinstance(value, str) and not value.strip()):
                raise ValueError(f"{param_name} is required")

    def _request(self, method, endpoint, payload=None, stream=False):
        """
        Make a request to the Ollama API.
        """
        url = f"{self.base_url}/{endpoint}"
        logging.info(f"Making {method} request to {url} with payload: {payload}")
        try:
            response = requests.request(
                method,
                url,
                json=payload,
                headers=self.headers,
                stream=stream,
                timeout=10,
            )
            if not response.ok:
                logging.error(f"API error: {response.text}")
                raise OllamaAPIError(
                    f"Ollama API error: {response.text}",
                    status_code=response.status_code,
                )
            if stream:
                return self._handle_stream(response)
            return response.json()
        except requests.exceptions.Timeout:
            logging.error("Request timed out")
            raise OllamaAPIError("Request timed out")
        except requests.exceptions.RequestException as e:
            logging.error(f"Request error: {str(e)}")
            raise OllamaAPIError(f"Request error: {str(e)}")
        except Exception as e:
            logging.error(f"Unknown error: {str(e)}")
            raise OllamaAPIError(f"Unknown error: {str(e)}")

    def _handle_stream(self, response):
        """Handle streaming responses."""
        try:
            for line in response.iter_lines():
                if line:
                    yield json.loads(line)
        except json.JSONDecodeError as e:
            raise OllamaAPIError(f"Failed to decode JSON: {str(e)}")
        except requests.exceptions.RequestException as e:
            raise OllamaAPIError(f"Request error: {str(e)}")
        except Exception as e:
            raise OllamaAPIError(f"Unknown error: {str(e)}")
