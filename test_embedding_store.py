from ingestion.load_pdfs import load_all_pdfs
from ingestion.chunk_docs import chunk_documents
from ingestion.embed_store import store_chunks_in_pinecone

docs = load_all_pdfs()
chunks = chunk_documents(docs)

store_chunks_in_pinecone(chunks)