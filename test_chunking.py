from ingestion.load_pdfs import load_all_pdfs
from ingestion.chunk_docs import chunk_documents

docs = load_all_pdfs()
chunks = chunk_documents(docs)

print(f"Original documents: {len(docs)}")
print(f"Chunked documents: {len(chunks)}")

if chunks:
    print("\nFirst chunk preview:\n")
    print(chunks[0].page_content[:700])

    print("\nChunk metadata:")
    print(chunks[0].metadata)