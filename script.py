from ollama_client.client import OllamaClient
import json

client = OllamaClient("192.168.10.4", "11435", "llama3:70b")

res = client.generate("Why the sky is blue ?", stream=True)

for line in res:
    data = json.loads(line.decode())
    print(data["response"])
