from dotenv import load_dotenv
import os

from langchain_openai import ChatOpenAI

# Load variables from .env
load_dotenv()

MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

if not MISTRAL_API_KEY:
    raise ValueError("MISTRAL_API_KEY not found in .env")

llm = ChatOpenAI(
    api_key=MISTRAL_API_KEY,
    base_url="https://api.mistral.ai/v1",
    model="mistral-small-latest",
    temperature=0,
)

response = llm.invoke("Say 'Hello from Mistral API' in one sentence.")

print(response.content)