from functools import lru_cache
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from config import OPENROUTER_API_KEY, PINECONE_API_KEY

INDEX_NAME = "school-chatbot-index"


@lru_cache(maxsize=1)
def get_embedding_model():
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=OPENROUTER_API_KEY,
        openai_api_base="https://openrouter.ai/api/v1"
    )


@lru_cache(maxsize=1)
def get_vectorstore():
    embeddings = get_embedding_model()

    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)

    return PineconeVectorStore(
        index=index,
        embedding=embeddings
    )


@lru_cache(maxsize=1)
def get_retriever():
    vectorstore = get_vectorstore()

    return vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3,
            "fetch_k": 8
        }
    )