from pinecone import Pinecone, ServerlessSpec
from config import PINECONE_API_KEY

INDEX_NAME = "school-chatbot-index"

pc = Pinecone(api_key=PINECONE_API_KEY)

existing_indexes = [index["name"] for index in pc.list_indexes()]

if INDEX_NAME not in existing_indexes:
    pc.create_index(
        name=INDEX_NAME,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print("Pinecone index created successfully.")
else:
    print("Index already exists.")