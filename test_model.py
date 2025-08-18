import requests

url = "http://localhost:11434/api/generate"
payload = {
    "model": "gemma3:1b",
    "prompt": "幫我寫一首三行詩，主題是貓",
    "stream": False
}

response = requests.post(url, json=payload)
print(response.json()["response"])