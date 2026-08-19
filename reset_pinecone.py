from pinecone import Pinecone
from config import PINECONE_API_KEY,PINECONE_INDEX_NAME

INDEX_NAME = PINECONE_INDEX_NAME

pc = Pinecone(api_key=PINECONE_API_KEY)

if INDEX_NAME in [idx["name"] for idx in pc.list_indexes()]:
    pc.delete_index(INDEX_NAME)
    print("Old index deleted.")
else:
    print("Index not found.")