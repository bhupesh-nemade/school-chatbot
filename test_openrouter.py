import os
import time
import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENROUTER_API_KEY")

print("API key loaded:", bool(api_key))

start = time.perf_counter()

response = requests.post(
    "https://openrouter.ai/api/v1/embeddings",
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    },
    json={
        "model": "baai/bge-m3",
        "input": "What documents are required for admission?",
        "encoding_format": "float",
    },
    timeout=60,
)

latency = (time.perf_counter() - start) * 1000

print("Status:", response.status_code)
print("Latency:", round(latency, 2), "ms")

data = response.json()

if response.status_code == 200:
    embedding = data["data"][0]["embedding"]

    print("Dimensions:", len(embedding))
    print("First 5 values:", embedding[:5])
else:
    print("Error:", data)
    