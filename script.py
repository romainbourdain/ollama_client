from ollama_client.api import OllamaAPI

api = OllamaAPI("http://192.168.10.4:11435")

api.pull("llama2")

response = api.generate("llama2", "Why is the sky blue", stream=True)

for message in response:
    print(message["response"], end="")
print("")
