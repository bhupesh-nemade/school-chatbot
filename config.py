import os
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")


def validate_env():
    missing = []

    if not PINECONE_API_KEY:
        missing.append("PINECONE_API_KEY")

    if not OPENROUTER_API_KEY:
        missing.append("OPENROUTER_API_KEY")

    if not HF_TOKEN:
        missing.append("HF_TOKEN")

    if missing:
        raise ValueError(f"Missing environment variables: {missing}")

    print("All environment variables loaded successfully.")