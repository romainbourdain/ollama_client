from ollama_client.client import OllamaClient

api = OllamaClient("http://192.168.10.4:11435")

res = api.pull("llama3:70b", stream=True)
for part in res:
    print(part)

response = api.generate("llama3:70b", "Why is the sky blue", stream=True)

for message in response:
    print(message["response"], end="")
print("")
