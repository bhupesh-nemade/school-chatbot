from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from config import OPENROUTER_API_KEY, PINECONE_API_KEY
from chatbot.retriever import get_embedding_model, get_vectorstore
INDEX_NAME = "school-chatbot-index"


def get_embedding_model():
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1"
    )
    return embeddings


def store_chunks_in_pinecone(chunks, batch_size=50):
    vectorstore = get_vectorstore()

    total = len(chunks)

    print(f"Total chunks to upload: {total}")

    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]

        print(f"Uploading batch {i//batch_size + 1} ({len(batch)} chunks)...")

        vectorstore.add_documents(batch)

    print("Chunks stored successfully in Pinecone.")